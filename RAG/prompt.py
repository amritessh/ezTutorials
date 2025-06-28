from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class RAGPromptOptimizer:
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.qa_prompt = self._create_qa_prompt()
    
    def _create_system_prompt(self):
        """Create system prompt template"""
        template = """You are an expert AI assistant with access to a comprehensive knowledge base. 
        Your role is to provide accurate, detailed, and contextually relevant answers based on the retrieved information.

        Guidelines:
        1. Use ONLY the provided context to answer questions
        2. If information is insufficient, clearly state limitations
        3. Cite specific sources when possible
        4. Maintain professional and helpful tone
        5. Structure responses clearly with relevant details

        Context: {context}

        Question: {question}

        Provide a comprehensive answer based on the context above:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_prompt(self):
        """Create question-answering prompt template"""
        template = """Given the following context and question, provide a detailed answer.
        If you cannot answer based on the context, explain what information would be needed.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

class ProductionRAGChain:
    def __init__(self, llm, retriever, prompt_optimizer):
        self.llm = llm
        self.retriever = retriever
        self.prompt_optimizer = prompt_optimizer
        self.qa_chain = self._build_qa_chain()
    
    def _build_qa_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,  
            chain_type_kwargs={
                "prompt": self.prompt_optimizer.qa_prompt
            },
            return_source_documents=True,
            verbose=True
        )
    
    def query(self, question):
        result = self.qa_chain.invoke({"query": question}) 
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]],
            "confidence": self._calculate_confidence(result)
        }
    
    def _calculate_confidence(self, result):
        """Calculate confidence score based on source relevance"""
        source_count = len(result["source_documents"])
        avg_score = sum([doc.metadata.get("score", 0.5) 
                        for doc in result["source_documents"]]) / max(source_count, 1)
        return min(avg_score * source_count / 3, 1.0)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("./my_vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.3,
    do_sample=True,
    truncation=True,
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)


prompt_optimizer = RAGPromptOptimizer()
rag_chain = ProductionRAGChain(llm, retriever, prompt_optimizer)