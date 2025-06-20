from langchain_core.callbacks import BaseCallbackHandler

class StreamingCallback(BaseCallbackHandler):
    """Custom callback for streaming responses."""
    
    def __init__(self, container=None):
        self.container = container
        self.text = ""
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        self.text += token
        
        # Update container if provided (for Streamlit)
        if self.container:
            self.container.markdown(self.text + "▋")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Handle LLM completion."""
        if self.container:
            self.container.markdown(self.text)

class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_context(self, last_n: int = 6) -> str:
        """Get formatted conversation history."""
        recent_messages = self.messages[-last_n:]
        
        context_lines = []
        for msg in recent_messages:
            role = msg['role'].capitalize()
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        self.session_start = datetime.now()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = datetime.now() - self.session_start
        user_messages = len([msg for msg in self.messages if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in self.messages if msg['role'] == 'assistant'])
        
        return {
            'total_messages': len(self.messages),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'session_duration_minutes': round(duration.total_seconds() / 60, 2)
        }

class ConversationalAgent:
    """Main conversational agent with streaming and memory."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger, 
                 metrics: VideoGPTMetrics, rag_system: AdvancedRAGSystem):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.rag_system = rag_system
        self.memory = ConversationMemory()
        self.setup_agent()
    
    def setup_agent(self):
        """Initialize the conversational agent."""
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.get("model_name"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=self.config.get("temperature"),
            max_output_tokens=self.config.get("max_tokens"),
        )
        
        # Create conversational prompt
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are VideoGPT, an expert AI assistant specializing in video content analysis.

Your capabilities:
- Analyze and discuss video content in detail
- Answer questions based on video transcripts
- Provide insights and summaries
- Maintain conversational context
- Reference specific parts of videos

Guidelines:
- Always base answers on the provided video context
- Be conversational and engaging
- Reference previous parts of our conversation when relevant
- If information isn't in the video content, clearly state that
- Provide comprehensive but focused responses
- Use natural, friendly language

Video Context Available: {has_context}
Conversation History: {has_history}"""),
            
            ("human", """Conversation History:
{conversation_history}

Video Context:
{video_context}

Current Question: {user_question}

Please provide a comprehensive answer based on the video content and our conversation history.""")
        ])
        
        self.logger.info("Conversational agent initialized", "CHAT_AGENT")
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input for security."""
        import re
        
        # Remove potential harmful content
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        # Limit length
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    def chat_with_streaming(self, user_question: str, container, retriever_type: str = "compression"):
        """Handle chat with streaming response."""
        try:
            start_time = time.time()
            # FIXED: Use system references instead of self
            self.system.metrics.increment('total_requests')
            
            # Sanitize input
            clean_question = self.sanitize_input(user_question)
            self.system.logger.debug(f"Processing question: {clean_question[:50]}...", "CHAT_AGENT")
            
            # Add user message to memory
            self.conversation_history.append({"role": "user", "content": clean_question})
            
            # Retrieve relevant context
            retriever = self.video_data['retrievers'][retriever_type]
            relevant_docs = retriever.invoke(clean_question)
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt
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
                "metadata": str(self.video_data['metadata']['analysis'])[:500] if 'metadata' in self.video_data else "No metadata available",
                "history": history_text,
                "context": context,
                "question": clean_question
            })
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Update metrics
            self.system.metrics.increment('successful_requests')
            
            return response
            
        except Exception as e:
            # FIXED: Use system references
            self.system.logger.error(f"Chat error: {e}")
            self.system.metrics.increment('failed_requests')
            container.error("I encountered an error processing your question. Please try again.")
            return None
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        session_stats = self.memory.get_session_stats()
        
        return {
            **session_stats,
            'questions_answered': self.metrics.metrics.get('questions_answered', 0),
            'avg_response_time': round(self.metrics.metrics.get('avg_response_time', 0), 2)
        }
    
    def reset_conversation(self):
        """Reset conversation memory."""
        self.memory.clear_history()
        self.logger.info("Conversation history cleared", "CHAT_AGENT")

# Test the Conversational Agent
async def test_conversational_agent():
    """Test conversational agent functionality."""
    print("🧪 Testing Conversational Agent...")
    
    config = VideoGPTConfig()
    logger = VideoGPTLogger(config)
    metrics = VideoGPTMetrics()
    
    # Setup RAG system first
    rag_system = AdvancedRAGSystem(config, logger, metrics)
    test_docs = [
        Document(page_content="This video discusses artificial intelligence and machine learning concepts.", 
                metadata={'chunk_id': 0})
    ]
    await rag_system.setup_rag_system(test_docs)
    
    # Create conversational agent
    agent = ConversationalAgent(config, logger, metrics, rag_system)
    
    try:        
        # Test conversation
        questions = [
            "What does this video discuss?",
            "Can you tell me more about AI?",
            "What did we talk about before?"
        ]
        
        for question in questions:
            print(f"\n👤 User: {question}")
            response = await agent.chat_with_streaming(question)
            print(f"🤖 VideoGPT: {response[:100]}...")
        
        # Show statistics
        stats = agent.get_chat_statistics()
        print(f"\n📊 Chat Statistics: {stats}")
        
        print("✅ Conversational agent test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_conversational_agent())