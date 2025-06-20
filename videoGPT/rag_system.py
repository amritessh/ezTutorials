from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor

class AdvancedRAGSystem:
    """Advanced Retrieval-Augmented Generation system."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger, metrics: VideoGPTMetrics):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.setup_components()
    
    def setup_components(self):
        """Initialize RAG components."""
        # Initialize LLM and embeddings
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.get("model_name"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=self.config.get("temperature"),
            max_output_tokens=self.config.get("max_tokens")
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.get("embedding_model"),
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        self.vector_stores = {}
        self.retrievers = {}
        
        self.logger.info("RAG system initialized", "RAG_SYSTEM")
    
    def create_vector_stores(self, documents: List[Document]) -> Dict[str, Any]:
        """Create multiple vector stores for different retrieval strategies."""
        try:
            self.logger.info(f"Creating vector stores from {len(documents)} documents", "RAG_SYSTEM")
            
            # Main vector store
            self.vector_stores['main'] = FAISS.from_documents(documents, self.embeddings)
            
            # Create hierarchical stores with different chunk sizes
            coarse_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            fine_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            
            # Get original documents (before chunking)
            original_content = "\n\n".join([doc.page_content for doc in documents])
            original_doc = Document(page_content=original_content)
            
            coarse_chunks = coarse_splitter.split_documents([original_doc])
            fine_chunks = fine_splitter.split_documents([original_doc])
            
            self.vector_stores['coarse'] = FAISS.from_documents(coarse_chunks, self.embeddings)
            self.vector_stores['fine'] = FAISS.from_documents(fine_chunks, self.embeddings)
            
            self.logger.info(f"Created {len(self.vector_stores)} vector stores", "RAG_SYSTEM")
            
            return {
                'main_chunks': len(documents),
                'coarse_chunks': len(coarse_chunks),
                'fine_chunks': len(fine_chunks),
                'stores_created': list(self.vector_stores.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create vector stores: {str(e)}", "RAG_SYSTEM")
            raise
    
    def create_advanced_retrievers(self) -> Dict[str, Any]:
        """Create various types of retrievers."""
        try:
            self.logger.info("Creating advanced retrievers", "RAG_SYSTEM")
            
            # Base retrievers
            base_retriever = self.vector_stores['main'].as_retriever(search_kwargs={"k": 6})
            coarse_retriever = self.vector_stores['coarse'].as_retriever(search_kwargs={"k": 4})
            fine_retriever = self.vector_stores['fine'].as_retriever(search_kwargs={"k": 8})
            
            # Contextual compression retriever
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            # Multi-query retriever
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.llm
            )
            
            # Ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[compression_retriever, multi_query_retriever],
                weights=[0.6, 0.4]
            )
            
            # Store retrievers
            self.retrievers = {
                'base': base_retriever,
                'coarse': coarse_retriever,
                'fine': fine_retriever,
                'compression': compression_retriever,
                'multi_query': multi_query_retriever,
                'ensemble': ensemble_retriever
            }
            
            self.logger.info(f"Created {len(self.retrievers)} retrievers", "RAG_SYSTEM")
            
            return {
                'retrievers_created': list(self.retrievers.keys()),
                'default_retriever': 'compression'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create retrievers: {str(e)}", "RAG_SYSTEM")
            raise
    
    def retrieve_context(self, query: str, retriever_type: str = "compression", k: int = 4) -> List[Document]:
        """Retrieve relevant context for a query - SYNCHRONOUS VERSION."""
        try:
            self.logger.debug(f"Retrieving context for query: {query[:50]}...", "RAG_SYSTEM")
            
            if retriever_type not in self.retrievers:
                retriever_type = "compression"  # fallback
            
            retriever = self.retrievers[retriever_type]
            
            # FIXED: Use invoke() instead of aget_relevant_documents()
            docs = retriever.invoke(query)
            
            # Limit results
            docs = docs[:k]
            
            self.logger.debug(f"Retrieved {len(docs)} documents", "RAG_SYSTEM")
            return docs
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {str(e)}", "RAG_SYSTEM")
            return []
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            source_info = doc.metadata.get('source_url', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', i-1)
            
            context_parts.append(f"[Context {i} - Chunk {chunk_id}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    async def setup_rag_system(self, documents: List[Document]) -> Dict[str, Any]:
        """Complete RAG system setup."""
        try:
            self.logger.info("Setting up complete RAG system", "RAG_SYSTEM")
            
            # Create vector stores
            vector_info = self.create_vector_stores(documents)
            
            # Create retrievers
            retriever_info = self.create_advanced_retrievers()
            
            setup_result = {
                'setup_success': True,
                'vector_stores': vector_info,
                'retrievers': retriever_info,
                'available_retriever_types': list(self.retrievers.keys())
            }
            
            self.logger.info("RAG system setup completed", "RAG_SYSTEM")
            return setup_result
            
        except Exception as e:
            self.logger.error(f"RAG system setup failed: {str(e)}", "RAG_SYSTEM")
            return {
                'setup_success': False,
                'error': str(e)
            }

# Test the RAG System
async def test_rag_system():
    """Test RAG system functionality."""
    print("🧪 Testing Advanced RAG System...")
    
    config = VideoGPTConfig()
    logger = VideoGPTLogger(config)
    metrics = VideoGPTMetrics()
    
    rag_system = AdvancedRAGSystem(config, logger, metrics)
    
    # Create some test documents
    test_docs = [
        Document(page_content="This is about machine learning and AI technologies.", 
                metadata={'chunk_id': 0}),
        Document(page_content="Deep learning uses neural networks for pattern recognition.", 
                metadata={'chunk_id': 1}),
        Document(page_content="Natural language processing helps computers understand text.", 
                metadata={'chunk_id': 2})
    ]
    
    try:
        # Setup RAG system
        setup_result = await rag_system.setup_rag_system(test_docs)
        
        if setup_result['setup_success']:
            print(f"✅ RAG system setup successful")
            print(f"✅ Available retrievers: {setup_result['available_retriever_types']}")
            
            # Test retrieval
            query = "What is machine learning?"
            docs = await rag_system.retrieve_context(query)
            context = rag_system.format_context(docs)
            
            print(f"✅ Retrieved context for '{query}':")
            print(context[:200] + "...")
            
        else:
            print(f"❌ RAG setup failed: {setup_result['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_rag_system())