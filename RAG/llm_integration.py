from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import time
from typing import List

from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class HuggingFaceRAGEngine:
    def __init__(self, vectorstore, model_name: str = "google/flan-t5-base"):
        self.vectorstore = vectorstore
        self.model_name = model_name
        
        print(f"Loading Hugging Face model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            truncation=True,
            max_length=1024
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        self.prompt_template = """Answer the question using only the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 1}
            ),
            chain_type_kwargs={"prompt": self.PROMPT},
            return_source_documents=True
        )
        
        print("Hugging Face RAG engine initialized")
    
    def query(self, question: str):
        print(f"Processing question: {question}")
        start_time = time.time()
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            # Print the retrieved context for debugging
            if result["source_documents"]:
                print("\n--- Retrieved Context ---\n", result["source_documents"][0].page_content, "\n------------------------\n")
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": [],
                "processing_time": round(time.time() - start_time, 2),
                "model_used": self.model_name
            }
            
            for doc in result["source_documents"]:
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get('source_name', 'Unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', 'N/A')
                }
                response["sources"].append(source_info)
            
            return response
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "processing_time": round(time.time() - start_time, 2),
                "model_used": self.model_name
            }
    
    def batch_query(self, questions: List[str]):
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
            print(f"Processed: {question[:50]}...")
        
        return results

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        vectorstore = FAISS.load_local(
            "./my_vectorstore", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please run embeddings_vector_store.py first to create the vector store")
        exit(1)
    
    rag_engine = HuggingFaceRAGEngine(vectorstore, "google/flan-t5-base")

    result = rag_engine.query("List the main benefits of Retrieval-Augmented Generation (RAG).")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents used")
    print(f"Processing time: {result['processing_time']} seconds")