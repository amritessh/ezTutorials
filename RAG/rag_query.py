from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import time
from typing import List

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class RAGQueryEngine:
    def __init__(self, vectorstore, openai_api_key: str):
        self.vectorstore = vectorstore
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=500
        )
        
        # Create custom prompt template
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
        Always cite the source document when possible.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 similar chunks
            ),
            chain_type_kwargs={"prompt": self.PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str):
        """Process a question and return answer with sources"""
        print(f"Processing question: {question}")
        start_time = time.time()
        
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": [],
                "processing_time": round(time.time() - start_time, 2)
            }
            
            # Extract source information
            for doc in result["source_documents"]:
                source_info = {
                    "content": doc.page_content[:200] + "...",  # First 200 chars
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
                "processing_time": round(time.time() - start_time, 2)
            }
    
    def batch_query(self, questions: List[str]):
        """Process multiple questions"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
            print(f"✓ Processed: {question[:50]}...")
        
        return results

# Usage example
rag_engine = RAGQueryEngine(vectorstore, openai_api_key)

# Single query
result = rag_engine.query("What are the main benefits of renewable energy?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents used")
print(f"Processing time: {result['processing_time']} seconds")