from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self):
        # Use a free local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents):
        print("Creating embeddings and vector store...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"✓ Vector store created with {self.vectorstore.index.ntotal} vectors")
        return self.vectorstore
    
    def save_vectorstore(self, path: str = "./vectorstore"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save")
    
    def load_vectorstore(self, path: str = "./vectorstore"):
        """Load vector store from disk"""
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {path}")
            return self.vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def add_documents(self, new_documents):
        if self.vectorstore:
            self.vectorstore.add_documents(new_documents)
            print(f"Added {len(new_documents)} new documents")
        else:
            print("No existing vector store found")


# Import and run document loading first
from document_load import DocumentProcessor

# Create processor and load documents
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.load_documents_from_directory('./documents')
document_chunks = processor.split_documents(documents)

print(f"Loaded {len(documents)} documents into {len(document_chunks)} chunks")

# Create vector store with local embeddings
vector_manager = VectorStoreManager()
vectorstore = vector_manager.create_vectorstore(document_chunks)
vector_manager.save_vectorstore("./my_vectorstore_local") 