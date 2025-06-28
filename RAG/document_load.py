from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_documents_from_directory(self, directory_path: str):
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            try:
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif filename.lower().endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif filename.lower().endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                
                print(f"Loaded: {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return documents
    
    def split_documents(self, documents):

        chunks = self.text_splitter.split_documents(documents)
        

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_length': len(chunk.page_content),
                'source_name': os.path.basename(chunk.metadata.get('source', ''))
            })
        
        return chunks


processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.load_documents_from_directory('./documents')
document_chunks = processor.split_documents(documents)

print(f"Processed {len(documents)} documents into {len(document_chunks)} chunks")

