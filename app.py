import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pickle
import streamlit as st
import anthropic
from groq import Groq

# Constants
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
GROQ_MODEL = "mixtral-8x7b-32768"
CHUNK_SIZE = 1000  # Number of characters per chunk
OVERLAP = 200      # Overlap between chunks

class DocumentProcessor:
    def __init__(self, base_dir: str = None):
        """Initialize the document processor with directory setup."""
        self.base_dir = base_dir or os.getcwd()
        self.data_dir = Path(self.base_dir) / 'data'
        self.docs_dir = self.data_dir / 'docs'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        for dir_path in [self.data_dir, self.docs_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize document storage
        self.documents: Dict[str, str] = {}  # filename -> content
        self.chunks: Dict[int, str] = {}     # chunk_id -> content
        self.embeddings = None
        
        # Load the embedding model
        self.embedding_model = self.load_embedding_model()
        
    @st.cache_resource
    def load_embedding_model(self):
        """Load and cache the embedding model."""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_embedding_model(self):
        """Load and cache the embedding model."""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + CHUNK_SIZE
            
            # If we're not at the end of the text, try to break at a sentence
            if end < len(text):
                # Look for the last period in the overlap region
                last_period = text[end-OVERLAP:end].rfind('.')
                if last_period != -1:
                    end = end - OVERLAP + last_period + 1
            
            chunks.append(text[start:end].strip())
            start = end - OVERLAP
            
            # If we're near the end, just add the remaining text
            if len(text) - start < CHUNK_SIZE:
                chunks.append(text[start:].strip())
                break
                
        return chunks
    
    def process_file(self, file_path: Path) -> Tuple[bool, str]:
        """Process a single file and return success status and message."""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                return False, f"Failed to read {file_path.name}: {str(e)}"
        
        # Store the full document
        self.documents[file_path.name] = content
        
        # Create and store chunks
        chunks = self.chunk_text(content)
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks, start_idx):
            self.chunks[i] = chunk
            
        return True, f"Successfully processed {file_path.name}"
    
    def process_documents(self) -> Tuple[int, str]:
        """Process all documents in the docs directory."""
        if not self.docs_dir.exists():
            return 0, "Documents directory not found"
        
        files = list(self.docs_dir.glob('*.txt'))
        if not files:
            return 0, "No text files found in documents directory"
        
        processed_count = 0
        failed_files = []
        
        for file_path in files:
            success, message = self.process_file(file_path)
            if success:
                processed_count += 1
            else:
                failed_files.append(message)
        
        # Generate embeddings for all chunks
        if self.chunks:
            self.embeddings = self.embedding_model.encode(list(self.chunks.values()))
            self.save_state()
            
        status = f"Processed {processed_count} files into {len(self.chunks)} chunks"
        if failed_files:
            status += f"\nErrors: {'; '.join(failed_files)}"
            
        return len(self.chunks), status
    
    def save_state(self) -> Tuple[bool, str]:
        """Save the current state to disk."""
        try:
            state = {
                'documents': self.documents,
                'chunks': self.chunks
            }
            
            # Save state dictionary
            with open(self.processed_dir / 'state.pkl', 'wb') as f:
                pickle.dump(state, f)
                
            # Save embeddings separately due to size
            if self.embeddings is not None:
                np.save(self.processed_dir / 'embeddings.npy', self.embeddings)
                
            return True, "State saved successfully"
        except Exception as e:
            return False, f"Failed to save state: {str(e)}"
    
    def load_state(self) -> Tuple[bool, str]:
        """Load the previously saved state."""
        try:
            # Load main state
            state_path = self.processed_dir / 'state.pkl'
            embeddings_path = self.processed_dir / 'embeddings.npy'
            
            if not state_path.exists():
                return False, "No saved state found"
                
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.documents = state['documents']
                self.chunks = state['chunks']
            
            # Load embeddings if they exist
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                
            return True, f"Loaded {len(self.documents)} documents with {len(self.chunks)} chunks"
        except Exception as e:
            return False, f"Failed to load state: {str(e)}"
    
    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Find the most relevant chunks for a query."""
        if not self.embeddings is not None:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = [1 - cosine(query_embedding, doc_embedding) 
                      for doc_embedding in self.embeddings]
        
        # Get top_k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]


def main():
    st.title("Document Processing and Analysis")

    # Sidebar for API Key Input
    st.sidebar.title("API Key Configuration")
    anthropic_api_key = st.sidebar.text_input(
        "Enter your Anthropic API Key",
        type="password",
        placeholder="Enter your Anthropic API Key here..."
    )
    groq_api_key = st.sidebar.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="Enter your Groq API Key here..."
    )
    
    if anthropic_api_key and groq_api_key:
        try:
            # Initialize API clients
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            groq_client = Groq(api_key=groq_api_key)
            st.sidebar.success("API clients initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"Error initializing API clients: {e}")
            return
    else:
        st.sidebar.warning("Please enter both API keys to proceed.")
        return

    # Initialize document processor
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
        # Try to load existing state
        success, message = st.session_state.doc_processor.load_state()
        if not success and "No saved state found" not in message:
            st.warning(message)

    # Sidebar for document management
    st.sidebar.title("Document Management")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Document", type=['txt'])
    if uploaded_file:
        # Save uploaded file
        save_path = st.session_state.doc_processor.docs_dir / uploaded_file.name
        try:
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error saving file: {str(e)}")
    
    # Process documents button
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            num_chunks, status = st.session_state.doc_processor.process_documents()
            if num_chunks > 0:
                st.sidebar.success(status)
            else:
                st.sidebar.warning(status)
    
    # Main area for interaction
    st.write("### Ask Questions")
    question = st.text_input("Enter your question about the documents")
    
    if question and st.button("Get Answer"):
        if not st.session_state.doc_processor.chunks:
            st.warning("Please process some documents first")
            return
            
        with st.spinner("Finding answer..."):
            # Get relevant chunks
            relevant_chunks = st.session_state.doc_processor.find_relevant_chunks(question)
            
            if not relevant_chunks:
                st.warning("No relevant content found")
                return
            
            # Format prompt with context
            context = "\n\n".join(relevant_chunks)
            prompt = f"""Context:
            {context}
            
            Question: {question}
            
            Please answer the question based only on the provided context."""
            
            try:
                # Get response from Claude
                response = anthropic_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.write("### Answer")
                st.write(response.content)
                
                # Show context used
                with st.expander("View relevant context"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"\nChunk {i}:")
                        st.write(chunk)
                        
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

                
if __name__ == "__main__":
    main()
