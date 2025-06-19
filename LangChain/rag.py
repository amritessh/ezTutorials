from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_stores_embeddings import vector_store

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)

def create_rag_system(vector_store):
    rag_prompt = PromptTemplate(
        template="""Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context: {context}

Question: {question}

Answer: """,
        input_variables=["context", "question"]
    )
    


def create_modern_rag_chain(vector_store):
    
    retriever = vector_store.as_retriever()
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer questions based on the provided context. Be accurate and cite your sources."),
        ("human", """Context: {context}
        
Question: {question}""")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Usage
rag_system = create_modern_rag_chain(vector_store)
answer = rag_system.invoke("What are the environmental benefits of solar energy?")
print(answer)