from langchain_core.runnables import RunnableParallel

class AnalyticsEngine:
    """Advanced analytics and insights generation engine."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger, metrics: VideoGPTMetrics):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.setup_analytics()
    
    def setup_analytics(self):
        """Initialize analytics components."""
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.get("model_name"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_output_tokens=self.config.get("max_tokens")
        )
        
        self.analysis_chains = self._create_analysis_chains()
        self.logger.info("Analytics engine initialized", "ANALYTICS")
    
    def _create_analysis_chains(self) -> Dict[str, Any]:
        """Create specialized analysis chains."""
        
        # Summary chain
        summary_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive summary of this video content:
        
        {content}
        
        Provide:
        1. **Main Topic**: What is this video primarily about? (2-3 sentences)
        2. **Key Points**: The most important points discussed (5-7 bullet points)
        3. **Target Audience**: Who is this content designed for?
        4. **Complexity Level**: Beginner, Intermediate, or Advanced
        5. **Main Takeaways**: Core lessons or insights (3-5 points)
        6. **Content Type**: Educational, Entertainment, News, Tutorial, etc.
        
        Make it detailed but digestible and well-structured.
        """)
        
        # Insights chain
        insights_prompt = ChatPromptTemplate.from_template("""
        Extract key insights and learning points from this video:
        
        {content}
        
        Focus on:
        - **Novel Ideas**: Unique perspectives or innovative concepts presented
        - **Practical Applications**: Actionable advice or real-world applications
        - **Important Facts**: Significant statistics, data, or factual information
        - **Expert Opinions**: Notable viewpoints or expert commentary
        - **Trends & Connections**: Links to broader topics, trends, or movements
        
        Provide 6-8 specific insights with brief explanations for each.
        """)
        
        # Structure analysis chain
        structure_prompt = ChatPromptTemplate.from_template("""
        Analyze the structure and organization of this video content:
        
        {content}
        
        Examine:
        - **Overall Structure**: How is the content organized?
        - **Main Sections**: What are the primary topics/sections covered?
        - **Information Flow**: How do ideas connect and build upon each other?
        - **Presentation Style**: Teaching method, storytelling approach, etc.
        - **Pacing**: Information density and flow throughout
        - **Effectiveness**: How well is the content structured for understanding?
        
        Help viewers understand the content's organization and flow.
        """)
        
        # Key quotes chain
        quotes_prompt = ChatPromptTemplate.from_template("""
        Extract the most impactful and memorable quotes from this video:
        
        {content}
        
        Find 6-8 quotes that are:
        - **Thought-provoking**: Ideas that make you think differently
        - **Memorable**: Lines that stick with you
        - **Key Messages**: Statements that capture main points
        - **Quotable**: Could stand alone as valuable insights
        - **Impactful**: Have emotional or intellectual impact
        
        For each quote:
        Quote: "exact quote here"
        Context: Why this quote is significant and what it reveals
        """)
        
        # Create chains
        chains = {
            'summary': summary_prompt | self.llm | StrOutputParser(),
            'insights': insights_prompt | self.llm | StrOutputParser(),
            'structure': structure_prompt | self.llm | StrOutputParser(),
            'quotes': quotes_prompt | self.llm | StrOutputParser()
        }
        
        return chains
    
    async def generate_summary(self, content: str) -> str:
        """Generate comprehensive video summary."""
        try:
            self.logger.debug("Generating video summary", "ANALYTICS")
            
            # Limit content length to avoid token limits
            limited_content = content[:8000] if len(content) > 8000 else content
            
            summary = await self.analysis_chains['summary'].ainvoke({"content": limited_content})
            
            self.logger.info("Summary generated successfully", "ANALYTICS")
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}", "ANALYTICS")
            return "Failed to generate summary. Please try again."
    
    async def extract_insights(self, content: str) -> str:
        """Extract key insights from video content."""
        try:
            self.logger.debug("Extracting key insights", "ANALYTICS")
            
            limited_content = content[:8000] if len(content) > 8000 else content
            insights = await self.analysis_chains['insights'].ainvoke({"content": limited_content})
            
            self.logger.info("Insights extracted successfully", "ANALYTICS")
            return insights
            
        except Exception as e:
            self.logger.error(f"Insights extraction failed: {str(e)}", "ANALYTICS")
            return "Failed to extract insights. Please try again."
    
    async def analyze_structure(self, content: str) -> str:
        """Analyze content structure and organization."""
        try:
            self.logger.debug("Analyzing content structure", "ANALYTICS")
            
            limited_content = content[:8000] if len(content) > 8000 else content
            structure = await self.analysis_chains['structure'].ainvoke({"content": limited_content})
            
            self.logger.info("Structure analysis completed", "ANALYTICS")
            return structure
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {str(e)}", "ANALYTICS")
            return "Failed to analyze structure. Please try again."
    
    async def extract_quotes(self, content: str) -> str:
        """Extract key quotes from video content."""
        try:
            self.logger.debug("Extracting key quotes", "ANALYTICS")
            
            limited_content = content[:8000] if len(content) > 8000 else content
            quotes = await self.analysis_chains['quotes'].ainvoke({"content": limited_content})
            
            self.logger.info("Quotes extracted successfully", "ANALYTICS")
            return quotes
            
        except Exception as e:
            self.logger.error(f"Quote extraction failed: {str(e)}", "ANALYTICS")
            return "Failed to extract quotes. Please try again."
    
    async def generate_complete_analysis(self, content: str) -> Dict[str, Any]:
        """Generate complete analysis using all chains in parallel."""
        try:
            start_time = time.time()
            self.logger.info("Starting complete analysis", "ANALYTICS")
            
            # Limit content length
            limited_content = content[:8000] if len(content) > 8000 else content
            
            # Create parallel analysis chain
            parallel_analysis = RunnableParallel(
                summary=self.analysis_chains['summary'],
                insights=self.analysis_chains['insights'],
                structure=self.analysis_chains['structure'],
                quotes=self.analysis_chains['quotes']
            )
            
            # Execute all analyses in parallel
            results = await parallel_analysis.ainvoke({"content": limited_content})
            
            # Add metadata
            analysis_result = {
                **results,
                'analysis_metadata': {
                    'content_length': len(content),
                    'processing_time': time.time() - start_time,
                    'analysis_timestamp': datetime.now(),
                    'content_truncated': len(content) > 8000
                }
            }
            
            processing_time = time.time() - start_time
            self.metrics.record_processing_time(processing_time)
            
            self.logger.info(f"Complete analysis generated in {processing_time:.2f}s", "ANALYTICS")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Complete analysis failed: {str(e)}", "ANALYTICS")
            return {
                'error': str(e),
                'analysis_metadata': {
                    'analysis_timestamp': datetime.now(),
                    'success': False
                }
            }
    
    def format_analysis_for_display(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Format analysis results for better display."""
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        formatted = {}
        
        # Format each section
        sections = ['summary', 'insights', 'structure', 'quotes']
        section_titles = {
            'summary': '📝 Video Summary',
            'insights': '💡 Key Insights',
            'structure': '🏗️ Content Structure',
            'quotes': '💬 Memorable Quotes'
        }
        
        for section in sections:
            if section in analysis:
                title = section_titles.get(section, section.title())
                content = analysis[section]
                formatted[title] = content
        
        return formatted

# Test the Analytics Engine
async def test_analytics_engine():
    """Test analytics engine functionality."""
    print("🧪 Testing Analytics Engine...")
    
    config = VideoGPTConfig()
    logger = VideoGPTLogger(config)
    metrics = VideoGPTMetrics()
    
    analytics = AnalyticsEngine(config, logger, metrics)
    
    # Test content
    test_content = """
    This video explores the fascinating world of artificial intelligence and machine learning.
    We begin by understanding what AI really means and how it differs from traditional programming.
    
    The speaker explains that AI systems learn from data rather than being explicitly programmed.
    Machine learning, a subset of AI, uses algorithms to find patterns in large datasets.
    
    Key applications discussed include:
    - Natural language processing for chatbots
    - Computer vision for image recognition
    - Recommendation systems for personalized content
    
    The video concludes with thoughts on the future of AI and its potential impact on society.
    As the speaker notes: "AI is not about replacing humans, but augmenting human capabilities."
    """
    
    try:
        # Test individual analyses
        print("Testing individual analysis functions...")
        
        summary = await analytics.generate_summary(test_content)
        print(f"✅ Summary generated: {len(summary)} characters")
        
        insights = await analytics.extract_insights(test_content)
        print(f"✅ Insights extracted: {len(insights)} characters")
        
        # Test complete analysis
        print("\nTesting complete parallel analysis...")
        complete_analysis = await analytics.generate_complete_analysis(test_content)
        
        if 'error' not in complete_analysis:
            print("✅ Complete analysis successful!")
            metadata = complete_analysis['analysis_metadata']
            print(f"📊 Processing time: {metadata['processing_time']:.2f}s")
            
            # Test formatting
            formatted = analytics.format_analysis_for_display(complete_analysis)
            print(f"✅ Formatted {len(formatted)} sections for display")
            
        else:
            print(f"❌ Complete analysis failed: {complete_analysis['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


        def generate_quick_summary(self, video_data: Dict[str, Any]) -> str:
            """Generate quick summary only."""
            summary_chain = self._create_summary_chain()
            content = video_data['documents'][0].page_content[:8000]
            return summary_chain.invoke({"content": content})

        def generate_deep_insights(self, video_data: Dict[str, Any]) -> str:
            """Generate deep insights only."""
            insights_chain = self._create_insights_chain()
            content = video_data['documents'][0].page_content[:8000]
            return insights_chain.invoke({"content": content})

if __name__ == "__main__":
    asyncio.run(test_analytics_engine())