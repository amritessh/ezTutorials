from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def create_vector_store(documents, store_type="faiss"):
    if store_type == "faiss":
        vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store


documents = text_splitter.split_documents(docs_file_path)
vector_store = create_vector_store(documents)


query = "Relevant questions to ask about the document"
relevant_docs = vector_store.similarity_search(query, k=4)

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:")
    print(doc.page_content[:200] + "...")
    print("-" * 50)

