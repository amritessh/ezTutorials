import os
import logging
import time
import secrets
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

class VideoGPTConfig:
    """Configuration management for VideoGPT."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.setup_config()
    
    def setup_config(self):
        """Load environment-specific settings."""
        base_config = {
            "model_name": "gemini-2.0-flash-exp",
            "embedding_model": "models/embedding-001",
            "temperature": 0.7,
            "max_tokens": 2048,
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        if self.environment == "production":
            self.config = {
                **base_config,
                "rate_limit": 60,  # requests per minute
                "enable_logging": True,
                "log_level": "INFO",
                "enable_security": True,
                "enable_monitoring": True
            }
        else:  # development
            self.config = {
                **base_config,
                "rate_limit": 300,
                "enable_logging": True,
                "log_level": "DEBUG",
                "enable_security": False,
                "enable_monitoring": False
            }
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)

class VideoGPTLogger:
    """Centralized logging for VideoGPT."""
    
    def __init__(self, config: VideoGPTConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging based on environment."""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('videogpt.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('VideoGPT')
        self.logger.info(f"VideoGPT initialized in {self.config.environment} mode")
    
    def info(self, message: str, component: str = "SYSTEM"):
        self.logger.info(f"[{component}] {message}")
    
    def error(self, message: str, component: str = "SYSTEM"):
        self.logger.error(f"[{component}] {message}")
    
    def debug(self, message: str, component: str = "SYSTEM"):
        self.logger.debug(f"[{component}] {message}")

class VideoGPTMetrics:
    """Performance and usage metrics tracking."""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0,
            'avg_response_time': 0,
            'videos_processed': 0,
            'questions_answered': 0
        }
        self.start_time = datetime.now()
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric."""
        self.metrics[metric] = self.metrics.get(metric, 0) + value
    
    def record_processing_time(self, duration: float):
        """Record processing time and update averages."""
        self.metrics['total_processing_time'] += duration
        total_requests = max(1, self.metrics['total_requests'])
        self.metrics['avg_response_time'] = self.metrics['total_processing_time'] / total_requests
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime = datetime.now() - self.start_time
        success_rate = 0
        if self.metrics['total_requests'] > 0:
            success_rate = (self.metrics['successful_requests'] / self.metrics['total_requests']) * 100
        
        return {
            **self.metrics,
            'success_rate': round(success_rate, 2),
            'uptime_hours': round(uptime.total_seconds() / 3600, 2)
        }

# Test the foundation
def test_foundation():
    """Test the foundation components."""
    print("🧪 Testing VideoGPT Foundation...")
    
    # Test config
    config = VideoGPTConfig("development")
    print(f"✅ Config loaded: {config.get('model_name')}")
    
    # Test logger
    logger = VideoGPTLogger(config)
    logger.info("Foundation test successful", "TEST")
    
    # Test metrics
    metrics = VideoGPTMetrics()
    metrics.increment('total_requests')
    metrics.record_processing_time(1.5)
    print(f"✅ Metrics working: {metrics.get_summary()}")
    
    print("🎉 Foundation components ready!")

if __name__ == "__main__":
    test_foundation()