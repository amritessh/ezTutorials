
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HuggingFaceVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.model_name = model_name
        print(f"Initialized Hugging Face embeddings with {model_name}")
    
    def create_vectorstore(self, documents):
        print("Creating embeddings and vector store...")
        print(f"Processing {len(documents)} document chunks...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"Vector store created with {self.vectorstore.index.ntotal} vectors")
        return self.vectorstore
    
    def save_vectorstore(self, path: str = "./vectorstore"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            with open(f"{path}/model_info.txt", "w") as f:
                f.write(self.model_name)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save")
    
    def load_vectorstore(self, path: str = "./vectorstore"):

        try:
            with open(f"{path}/model_info.txt", "r") as f:
                saved_model_name = f.read().strip()
            
            if saved_model_name != self.model_name:
                print(f"Warning: Saved model ({saved_model_name}) differs from current model ({self.model_name})")
            
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
    
    def similarity_search_with_score(self, query: str, k: int = 5):

        if self.vectorstore:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return docs_and_scores
        return []

def load_and_split_documents(directory_path: str = "./documents", chunk_size: int = 1000, chunk_overlap: int = 200):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.lower().endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            
            print(f"Loaded: {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    

    chunks = text_splitter.split_documents(documents)
    

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_length': len(chunk.page_content),
            'source_name': os.path.basename(chunk.metadata.get('source', ''))
        })
    
    print(f"Created {len(chunks)} document chunks")
    return chunks


if __name__ == "__main__":

    document_chunks = load_and_split_documents()
    

    vector_manager = HuggingFaceVectorStore("sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = vector_manager.create_vectorstore(document_chunks)
    vector_manager.save_vectorstore("./my_vectorstore")