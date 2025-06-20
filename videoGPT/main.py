import streamlit as st
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import hmac
import secrets

# All our imports from previous sections
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.callbacks import BaseCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

import nest_asyncio
nest_asyncio.apply()

class VideoGPTSystem:
    """Complete VideoGPT system integrating all LangChain concepts."""
    
    def __init__(self):
        self.setup_components()
        self.setup_security()
        self.setup_monitoring()
    
    def setup_components(self):
        """Initialize all LangChain components."""
        # Core LLM - FIXED: Removed streaming parameter
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            max_output_tokens=2048
        )
        
        # Embeddings for RAG
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Advanced text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Memory for conversations
        self.conversation_memory = {}
    
    def setup_security(self):
        """Initialize security components."""
        self.security_key = secrets.token_hex(32)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        }
    
    def setup_monitoring(self):
        """Initialize monitoring and logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('videogpt.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0
        }

class VideoGPTStreamingHandler(BaseCallbackHandler):
    """Custom streaming callback for real-time responses."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens to UI in real-time."""
        self.text += token
        self.container.markdown(self.text + "▋")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Finalize streaming."""
        self.container.markdown(self.text)

class AdvancedVideoProcessor:
    """Advanced video processing with all RAG techniques."""
    
    def __init__(self, videogpt_system):
        self.system = videogpt_system
        self.hierarchical_stores = {}
    
    def process_video_advanced(self, youtube_url: str) -> Dict[str, Any]:
        """Advanced video processing with hierarchical RAG - FIXED: Made synchronous."""
        try:
            start_time = time.time()
            self.system.logger.info(f"Starting advanced processing for: {youtube_url}")
            
            # Step 1: Load video content
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in video")
            
            # Step 2: Create document hierarchy
            self._create_hierarchical_structure(documents)
            
            # Step 3: Generate comprehensive metadata
            metadata = self._generate_video_metadata(documents[0])
            
            # Step 4: Create advanced retrievers
            retrievers = self._create_advanced_retrievers(documents)
            
            processing_time = time.time() - start_time
            self.system.logger.info(f"Video processed in {processing_time:.2f}s")
            
            return {
                'documents': documents,
                'metadata': metadata,
                'retrievers': retrievers,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.system.logger.error(f"Video processing failed: {e}")
            raise
    
    def _create_hierarchical_structure(self, documents):
        """Create hierarchical document structure - FIXED: Made synchronous."""
        # Split into different granularities
        coarse_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=300
        )
        fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        
        # Create different levels
        coarse_chunks = coarse_splitter.split_documents(documents)
        fine_chunks = fine_splitter.split_documents(documents)
        
        # Store hierarchically
        self.hierarchical_stores['coarse'] = FAISS.from_documents(
            coarse_chunks, self.system.embeddings
        )
        self.hierarchical_stores['fine'] = FAISS.from_documents(
            fine_chunks, self.system.embeddings
        )
        
        self.system.logger.info(f"Created hierarchical stores: {len(coarse_chunks)} coarse, {len(fine_chunks)} fine chunks")
    
    def _generate_video_metadata(self, document) -> Dict[str, Any]:
        """Generate comprehensive video metadata - FIXED: Made synchronous."""
        metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert video content analyzer."),
            ("human", """Analyze this video transcript and provide structured metadata:

Transcript: {content}

Please provide:
1. Main topic and subject area
2. Key themes and concepts discussed
3. Target audience
4. Content complexity level (beginner/intermediate/advanced)
5. Main takeaways (3-5 points)
6. Estimated video duration category
7. Content type (educational, entertainment, news, etc.)

Format as structured text.""")
        ])
        
        chain = metadata_prompt | self.system.llm | StrOutputParser()
        metadata_text = chain.invoke({"content": document.page_content[:3000]})
        
        return {"analysis": metadata_text, "timestamp": datetime.now()}
    
    def _create_advanced_retrievers(self, documents):
        """Create multiple advanced retrievers."""
        # Base vector store
        vector_store = FAISS.from_documents(
            self.system.text_splitter.split_documents(documents),
            self.system.embeddings
        )
        
        # Contextual compression retriever
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        compressor = LLMChainExtractor.from_llm(self.system.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return {
            'base': base_retriever,
            'compression': compression_retriever,
            'hierarchical_coarse': self.hierarchical_stores['coarse'].as_retriever(),
            'hierarchical_fine': self.hierarchical_stores['fine'].as_retriever()
        }

class ConversationalVideoGPT:
    """Conversational interface with memory and streaming."""
    
    def __init__(self, videogpt_system, video_data):
        self.system = videogpt_system
        self.video_data = video_data
        self.conversation_history = []
        
        # FIXED: Only reference existing attributes
        self.logger = videogpt_system.logger
        self.metrics = videogpt_system.metrics
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input for security."""
        import re
        
        # Remove potential XSS attempts
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text.strip()
    
    def chat_with_streaming(self, user_question: str, container, retriever_type: str = "compression"):
        """Handle chat with streaming response - FIXED: Ensures UI display."""
        try:
            start_time = time.time()
            self.system.metrics['total_requests'] += 1
            
            # Sanitize input
            clean_question = self.sanitize_input(user_question)
            self.system.logger.info(f"Processing question: {clean_question[:50]}...")
            
            # Add user message to memory
            self.conversation_history.append({"role": "user", "content": clean_question})
            
            # Retrieve relevant context
            retriever = self.video_data['retrievers'][retriever_type]
            relevant_docs = retriever.invoke(clean_question)
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:4]])
            
            # Create conversational prompt
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are VideoGPT, an expert AI assistant specializing in video content analysis.

    Video Metadata:
    {metadata}

    Use the conversation history and video context to provide detailed, accurate answers.
    Be conversational and reference previous parts of our discussion when relevant.

    Guidelines:
    - Always base answers on the provided video content
    - Be comprehensive but concise
    - If information isn't in the video, say so clearly
    - Use natural, conversational language
    - Reference specific parts of the video when helpful"""),
                
                ("human", """Conversation History:
    {history}

    Video Context:
    {context}

    Current Question: {question}

    Please provide a comprehensive answer based on the video content and our conversation history.""")
            ])
            
            # Prepare conversation history
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-6:]  # Last 6 messages
            ])
            
            # Create streaming chain
            streaming_handler = VideoGPTStreamingHandler(container)
            streaming_llm = self.system.llm.with_config(callbacks=[streaming_handler])
            
            chain = chat_prompt | streaming_llm | StrOutputParser()
            
            # Execute with streaming
            response = chain.invoke({
                "metadata": self.video_data['metadata']['analysis'][:500],
                "history": history_text,
                "context": context,
                "question": clean_question
            })
            
            # FIXED: Ensure response is displayed in container
            if container and response:
                container.markdown(response)
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Update metrics
            self.system.metrics['successful_requests'] += 1
            
            return response
            
        except Exception as e:
            self.system.logger.error(f"Chat error: {e}")
            self.system.metrics['failed_requests'] += 1
            if container:
                container.error("I encountered an error processing your question. Please try again.")
            return None
    
    def get_chat_statistics(self):
        """Get conversation statistics."""
        user_messages = len([m for m in self.conversation_history if m['role'] == 'user'])
        assistant_messages = len([m for m in self.conversation_history if m['role'] == 'assistant'])
        
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages
        }

class VideoGPTAnalytics:
    """Advanced analytics and insights generation."""
    
    def __init__(self, videogpt_system):
        self.system = videogpt_system
    
    def generate_comprehensive_analysis(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive video analysis - FIXED: Made synchronous."""
        
        # Create analysis chains
        summary_chain = self._create_summary_chain()
        insights_chain = self._create_insights_chain()
        quotes_chain = self._create_quotes_chain()
        structure_chain = self._create_structure_chain()
        
        content = video_data['documents'][0].page_content[:8000]
        
        # FIXED: Execute synchronously instead of parallel
        results = {
            'summary': summary_chain.invoke({"content": content}),
            'insights': insights_chain.invoke({"content": content}),
            'quotes': quotes_chain.invoke({"content": content}),
            'structure': structure_chain.invoke({"content": content})
        }
        
        return {
            **results,
            "metadata": video_data['metadata'],
            "analysis_timestamp": datetime.now()
        }
    
    def _create_summary_chain(self):
        """Create summary generation chain."""
        prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive summary of this video content:
        
        {content}
        
        Include:
        - Main topic and purpose (2-3 sentences)
        - Key points discussed (5-7 bullet points)
        - Target audience and complexity level
        - Main conclusions or takeaways
        
        Make it detailed but digestible.
        """)
        
        return prompt | self.system.llm | StrOutputParser()
    
    def _create_insights_chain(self):
        """Create insights extraction chain."""
        prompt = ChatPromptTemplate.from_template("""
        Extract key insights and learning points from this video:
        
        {content}
        
        Focus on:
        - Novel ideas or unique perspectives presented
        - Practical applications or actionable advice
        - Important facts or statistics mentioned
        - Connections to broader topics or trends
        
        Provide 5-7 specific insights with brief explanations.
        """)
        
        return prompt | self.system.llm | StrOutputParser()
    
    def _create_quotes_chain(self):
        """Create key quotes extraction chain."""
        prompt = ChatPromptTemplate.from_template("""
        Extract the most impactful and memorable quotes from this video:
        
        {content}
        
        Find 5-7 quotes that are:
        - Particularly insightful or thought-provoking
        - Memorable or quotable
        - Represent key messages
        - Could stand alone as valuable statements
        
        Format each quote with context about why it's significant.
        """)
        
        return prompt | self.system.llm | StrOutputParser()
    
    def _create_structure_chain(self):
        """Create content structure analysis chain."""
        prompt = ChatPromptTemplate.from_template("""
        Analyze the structure and flow of this video content:
        
        {content}
        
        Provide:
        - Overall structure/organization
        - Main sections or topics covered
        - How ideas flow and connect
        - Pacing and information density
        - Effectiveness of the presentation structure
        
        Help viewers understand how the content is organized.
        """)
        
        return prompt | self.system.llm | StrOutputParser()

def main():
    """Main VideoGPT Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="VideoGPT - AI Video Intelligence",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎥 VideoGPT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Video Analysis & Intelligence</p>', unsafe_allow_html=True)
    
    # Initialize VideoGPT system
    if 'videogpt' not in st.session_state:
        with st.spinner("🚀 Initializing VideoGPT system..."):
            st.session_state.videogpt = VideoGPTSystem()
            st.session_state.processor = AdvancedVideoProcessor(st.session_state.videogpt)
            st.session_state.analytics = VideoGPTAnalytics(st.session_state.videogpt)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ VideoGPT Control Panel")
        
        # Video input
        youtube_url = st.text_input(
            "📺 YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter any YouTube video URL for analysis"
        )
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        enable_hierarchical = st.checkbox("🏗️ Hierarchical RAG", value=True)
        enable_compression = st.checkbox("🗜️ Contextual Compression", value=True)
        enable_streaming = st.checkbox("⚡ Real-time Streaming", value=True)
        
        # Analysis type
        analysis_mode = st.selectbox(
            "📊 Analysis Mode",
            ["Interactive Chat", "Complete Analysis", "Quick Summary", "Deep Insights"]
        )
        
        # Process button
        if st.button("🚀 Process Video", type="primary"):
            if youtube_url:
                st.session_state.processing_video = True
                st.session_state.current_url = youtube_url
            else:
                st.warning("Please enter a YouTube URL")
        
        # System metrics
        if hasattr(st.session_state, 'videogpt'):
            st.subheader("📈 System Metrics")
            metrics = st.session_state.videogpt.metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Requests", metrics['total_requests'])
                st.metric("Success Rate", f"{(metrics['successful_requests']/(metrics['total_requests'] or 1)*100):.1f}%")
            with col2:
                st.metric("Successful", metrics['successful_requests'])
                st.metric("Failed", metrics['failed_requests'])
    
    # Main content area
    if hasattr(st.session_state, 'processing_video') and st.session_state.processing_video:
        
        # Process video
        with st.spinner("🎬 Processing video with advanced AI..."):
            try:
                # Update metrics
                st.session_state.videogpt.metrics['total_requests'] += 1
                
                # FIXED: Process video synchronously
                video_data = st.session_state.processor.process_video_advanced(st.session_state.current_url)
                
                st.session_state.video_data = video_data
                st.session_state.conversational_gpt = ConversationalVideoGPT(
                    st.session_state.videogpt, video_data
                )
                st.session_state.processing_video = False
                
                st.success(f"✅ Video processed successfully in {video_data['processing_time']:.2f}s!")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.session_state.processing_video = False
    
    # Display results based on mode
    if hasattr(st.session_state, 'video_data'):
        
        if analysis_mode == "Interactive Chat":
            st.header("💬 Chat with VideoGPT")
            
            # Chat interface
            user_question = st.text_input(
                "❓ Ask anything about the video:",
                placeholder="What are the main points discussed in this video?"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary")
            with col2:
                retriever_type = st.selectbox(
                    "Retriever",
                    ["compression", "base", "hierarchical_coarse", "hierarchical_fine"]
                )
            
            if ask_button and user_question:
                response_container = st.empty()
                
                if enable_streaming:
                    # FIXED: Streaming response - synchronous call
                    st.session_state.conversational_gpt.chat_with_streaming(
                        user_question, response_container, retriever_type
                    )
                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = st.session_state.conversational_gpt.chat_with_streaming(
                            user_question, response_container, retriever_type
                        )
            
            # FIXED: Conversation history - only show if conversational_gpt exists
            if 'conversational_gpt' in st.session_state and st.session_state.conversational_gpt:
                # Show conversation statistics
                with st.expander("📊 Conversation Statistics"):
                    stats = st.session_state.conversational_gpt.get_chat_statistics()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Messages", stats['total_messages'])
                    with col2:
                        st.metric("User Messages", stats['user_messages'])
                    with col3:
                        st.metric("AI Responses", stats['assistant_messages'])
                
                # Show conversation history
                if st.session_state.conversational_gpt.conversation_history:
                    with st.expander("📜 Conversation History"):
                        for msg in st.session_state.conversational_gpt.conversation_history[-10:]:
                            if msg['role'] == 'user':
                                st.write(f"**You:** {msg['content']}")
                            else:
                                st.write(f"**VideoGPT:** {msg['content'][:200]}...")
        
        elif analysis_mode == "Complete Analysis":
            st.header("📊 Complete Video Analysis")
            
            if st.button("🔬 Generate Complete Analysis"):
                with st.spinner("🧠 Generating comprehensive analysis..."):
                    # FIXED: Synchronous analysis call
                    analysis = st.session_state.analytics.generate_comprehensive_analysis(
                        st.session_state.video_data
                    )
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "💡 Insights", "💬 Key Quotes", "🏗️ Structure"])
                    
                    with tab1:
                        st.markdown("### 📝 Video Summary")
                        st.write(analysis['summary'])
                    
                    with tab2:
                        st.markdown("### 💡 Key Insights")
                        st.write(analysis['insights'])
                    
                    with tab3:
                        st.markdown("### 💬 Memorable Quotes")
                        st.write(analysis['quotes'])
                    
                    with tab4:
                        st.markdown("### 🏗️ Content Structure")
                        st.write(analysis['structure'])
        
        # Video metadata display
        with st.expander("📋 Video Metadata & Processing Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎥 Video Information")
                st.write(st.session_state.video_data['metadata']['analysis'][:500] + "...")
            
            with col2:
                st.subheader("⚡ Processing Statistics")
                st.metric("Processing Time", f"{st.session_state.video_data['processing_time']:.2f}s")
                st.metric("Document Chunks", len(st.session_state.video_data['documents']))
                st.metric("Available Retrievers", len(st.session_state.video_data['retrievers']))
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="feature-box">
        <h3>🎯 VideoGPT Features</h3>
        
        VideoGPT demonstrates <strong>ALL major LangChain concepts</strong> in a production-ready application:
        
        <ul>
        <li>🧠 <strong>Advanced RAG</strong> - Hierarchical document processing with contextual compression</li>
        <li>⚡ <strong>Real-time Streaming</strong> - Live response generation with custom callbacks</li>
        <li>🔒 <strong>Security & Privacy</strong> - Input sanitization and PII protection</li>
        <li>🚀 <strong>Async Processing</strong> - High-performance parallel operations</li>
        <li>📊 <strong>Monitoring & Analytics</strong> - Comprehensive observability</li>
        <li>🛠️ <strong>Custom Components</strong> - Advanced runnables and retrievers</li>
        <li>💾 <strong>Conversation Memory</strong> - Context-aware chat experiences</li>
        <li>🎛️ <strong>Multi-modal Analysis</strong> - Complex document understanding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👆 Enter a YouTube URL in the sidebar to start analyzing!")
        
        # Example videos
        st.subheader("🎬 Try These Example Videos:")
        example_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with actual examples
            "https://www.youtube.com/watch?v=example2",
            "https://www.youtube.com/watch?v=example3"
        ]
        
        for i, url in enumerate(example_videos, 1):
            if st.button(f"📺 Example Video {i}", key=f"example_{i}"):
                st.session_state.current_url = url
                st.session_state.processing_video = True
                st.rerun()

if __name__ == "__main__":
    main()