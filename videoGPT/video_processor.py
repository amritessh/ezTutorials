from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class VideoProcessor:
    """Handles YouTube video loading and content processing."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger, metrics: VideoGPTMetrics):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.setup_components()
    
    def setup_components(self):
        """Initialize processing components."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size"),
            chunk_overlap=self.config.get("chunk_overlap"),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger.info("Video processor initialized", "VIDEO_PROCESSOR")
    
    def load_video_transcript(self, youtube_url: str) -> List[Document]:
        """Load transcript from YouTube video."""
        try:
            start_time = time.time()
            self.logger.info(f"Loading video: {youtube_url}", "VIDEO_PROCESSOR")
            
            # Load video transcript
            loader = YoutubeLoader.from_youtube_url(youtube_url)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No transcript found for this video")
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_url': youtube_url,
                    'processed_at': datetime.now().isoformat(),
                    'processor_version': '1.0'
                })
            
            processing_time = time.time() - start_time
            self.metrics.record_processing_time(processing_time)
            self.metrics.increment('videos_processed')
            
            self.logger.info(f"Video loaded successfully in {processing_time:.2f}s", "VIDEO_PROCESSOR")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load video: {str(e)}", "VIDEO_PROCESSOR")
            self.metrics.increment('failed_requests')
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'total_chunks': len(chunks)
                })
            
            self.logger.info(f"Created {len(chunks)} chunks", "VIDEO_PROCESSOR")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk documents: {str(e)}", "VIDEO_PROCESSOR")
            raise
    
    def extract_video_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract metadata from video content."""
        if not documents:
            return {}
        
        doc = documents[0]
        content = doc.page_content
        
        # Basic metadata extraction
        metadata = {
            'content_length': len(content),
            'estimated_duration': len(content.split()) / 150,  # Rough estimate: 150 words per minute
            'source_url': doc.metadata.get('source_url', ''),
            'has_content': bool(content.strip())
        }
        
        self.logger.debug(f"Extracted metadata: {metadata}", "VIDEO_PROCESSOR")
        return metadata
    
    async def process_video_complete(self, youtube_url: str) -> Dict[str, Any]:
        """Complete video processing pipeline."""
        try:
            self.metrics.increment('total_requests')
            
            # Load transcript
            documents = self.load_video_transcript(youtube_url)
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Extract metadata
            metadata = self.extract_video_metadata(documents)
            
            # Prepare result
            result = {
                'original_documents': documents,
                'chunked_documents': chunks,
                'metadata': metadata,
                'processing_success': True,
                'processed_at': datetime.now()
            }
            
            self.metrics.increment('successful_requests')
            self.logger.info(f"Video processing completed successfully", "VIDEO_PROCESSOR")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}", "VIDEO_PROCESSOR")
            self.metrics.increment('failed_requests')
            
            return {
                'processing_success': False,
                'error': str(e),
                'processed_at': datetime.now()
            }

# Test the Video Processor
async def test_video_processor():
    """Test video processor functionality."""
    print("🧪 Testing Video Processor...")
    
    config = VideoGPTConfig()
    logger = VideoGPTLogger(config)
    metrics = VideoGPTMetrics()
    
    processor = VideoProcessor(config, logger, metrics)
    
    # Test with a sample video (replace with actual URL for testing)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with real URL
    
    try:
        result = await processor.process_video_complete(test_url)
        
        if result['processing_success']:
            print(f"✅ Processed {len(result['chunked_documents'])} chunks")
            print(f"✅ Metadata: {result['metadata']}")
        else:
            print(f"❌ Processing failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"📊 Metrics: {metrics.get_summary()}")

if __name__ == "__main__":
    asyncio.run(test_video_processor())