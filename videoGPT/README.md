# 🎥 VideoGPT - AI-Powered Video Intelligence

![VideoGPT Banner](https://img.shields.io/badge/VideoGPT-AI%20Video%20Analysis-blue?style=for-the-badge&logo=youtube&logoColor=white)

**VideoGPT** is a sophisticated AI-powered application that transforms YouTube videos into interactive, intelligent conversations. Built with cutting-edge LangChain technology, it demonstrates advanced RAG (Retrieval-Augmented Generation) techniques, real-time streaming, and production-ready AI architecture.

## 🚀 Features

### 🧠 **Advanced AI Capabilities**
- **Hierarchical RAG**: Multi-level document processing with coarse and fine-grained retrieval
- **Contextual Compression**: Smart content filtering for more relevant responses
- **Real-time Streaming**: Live AI responses with custom callback handlers
- **Conversation Memory**: Context-aware chat that remembers previous interactions

### 📊 **Comprehensive Analysis**
- **Video Summarization**: AI-generated summaries of video content
- **Key Insights Extraction**: Identifies novel ideas and actionable advice
- **Memorable Quotes**: Extracts impactful and quotable statements
- **Content Structure Analysis**: Breaks down video organization and flow

### 🛡️ **Production-Ready Features**
- **Input Sanitization**: XSS protection and PII detection
- **Performance Monitoring**: Real-time metrics and system health checks
- **Error Handling**: Robust exception management and recovery
- **Modular Architecture**: Clean, maintainable, and scalable codebase

### 🎛️ **Interactive Interface**
- **Multiple Retrieval Methods**: Choose from compression, base, hierarchical retrievers
- **Processing Options**: Configurable RAG techniques and streaming
- **System Metrics**: Live performance statistics and success rates
- **Conversation History**: Track and review chat interactions

## 🏗️ Architecture

VideoGPT is built with a modular architecture consisting of 7 main components:

```
📱 Streamlit UI (User Interface)
    ↓
🎬 VideoProcessor (Content Loading & Processing)  
    ↓
🧠 AdvancedRAG (Retrieval & Knowledge Management)
    ↓
💬 ConversationalAgent (Chat & Memory)
    ↓
📊 AnalyticsEngine (Insights & Analysis)
    ↓
🔒 SecurityMonitor (Safety & Observability)
    ↓
🎯 MainApplication (Orchestration & UI)
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- Internet connection for YouTube API access

### API Keys Required
- **Google Gemini API Key** (for LLM and embeddings)
- Get your key at: [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/videogpt.git
cd videogpt
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## 📦 Dependencies

### Core Dependencies
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.20
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
youtube-transcript-api>=0.6.0
nest-asyncio>=1.5.6
typing-extensions>=4.8.0
```

## 🎯 Usage Guide

### Step 1: Process a Video
1. Enter a YouTube URL in the sidebar
2. Configure processing options (Hierarchical RAG, Contextual Compression, Real-time Streaming)
3. Select analysis mode (Interactive Chat, Complete Analysis, Quick Summary, Deep Insights)
4. Click "🚀 Process Video"

### Step 2: Interactive Chat
- Ask questions about the video content
- Choose different retrieval methods for varied response styles
- View conversation history and statistics

### Step 3: Comprehensive Analysis
- Generate complete video analysis with summaries, insights, quotes, and structure
- Export results for further use
- Review processing statistics and metadata

## 🔧 Configuration Options

### Retrieval Methods
- **Compression**: Uses contextual compression for focused responses
- **Base**: Standard vector similarity search
- **Hierarchical Coarse**: Broader context chunks (2000 characters)
- **Hierarchical Fine**: Detailed context chunks (500 characters)

### Analysis Modes
- **Interactive Chat**: Real-time Q&A with the video content
- **Complete Analysis**: Comprehensive breakdown of all aspects
- **Quick Summary**: Fast overview of main points
- **Deep Insights**: Detailed extraction of key learnings

## 📊 System Monitoring

VideoGPT includes built-in monitoring and metrics:

- **Total Requests**: Number of API calls made
- **Success Rate**: Percentage of successful operations
- **Processing Time**: Average response time
- **System Health**: Real-time health status

## 🐛 Troubleshooting

### Common Issues

#### 1. "No transcript found for this video"
- **Cause**: Video doesn't have captions/transcript
- **Solution**: Try a different video with available captions

#### 2. "GEMINI_API_KEY not found"
- **Cause**: Missing or incorrect API key
- **Solution**: Check your `.env` file and API key validity

#### 3. "Event loop is closed" error
- **Cause**: Async/sync conflicts (should be fixed in current version)
- **Solution**: Restart the application

#### 4. High memory usage
- **Cause**: Large video transcripts
- **Solution**: Process shorter videos or increase system RAM

### Performance Optimization

1. **Use shorter videos** (< 1 hour) for better performance
2. **Enable hierarchical RAG** for better context management
3. **Choose appropriate retrieval method** based on use case
4. **Monitor system metrics** to identify bottlenecks

## 🔒 Security & Privacy

VideoGPT implements several security measures:

- **Input Sanitization**: Removes XSS attempts and malicious code
- **PII Detection**: Identifies and masks personal information
- **Rate Limiting**: Prevents API abuse
- **Error Handling**: Graceful failure without exposing system details

### Data Privacy
- No video content is stored permanently
- Conversation history is session-based only
- API keys are stored securely in environment variables
- User interactions are anonymized in logs

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Write comprehensive error handling

### Testing
```bash
# Run individual component tests
python -c "from main import VideoGPTSystem; VideoGPTSystem()"

# Test video processing
python -c "from main import AdvancedVideoProcessor, VideoGPTSystem; processor = AdvancedVideoProcessor(VideoGPTSystem()); print('✅ Components working')"
```

## 📈 Roadmap

### Upcoming Features
- [ ] **Multi-language Support**: Process videos in different languages
- [ ] **Batch Processing**: Handle multiple videos simultaneously
- [ ] **Custom Models**: Support for different LLM providers
- [ ] **API Endpoint**: REST API for programmatic access
- [ ] **Video Timestamps**: Link responses to specific video segments
- [ ] **Export Options**: PDF, Word, and JSON export formats

### Performance Improvements
- [ ] **Caching Layer**: Redis integration for faster responses
- [ ] **Database Storage**: Persistent conversation history
- [ ] **Load Balancing**: Support for high-traffic scenarios
- [ ] **GPU Acceleration**: CUDA support for faster processing

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

### Getting Help
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/videogpt/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/videogpt/discussions)
- **Email**: support@videogpt.ai

### FAQ

**Q: Can I use VideoGPT with private videos?**
A: Currently, VideoGPT only works with public YouTube videos that have available transcripts.

**Q: How much does it cost to run VideoGPT?**
A: Costs depend on your Gemini API usage. Typical video processing costs $0.01-0.10 per video.

**Q: Can I deploy VideoGPT to production?**
A: Yes! VideoGPT is built with production-ready features. Consider using a robust hosting platform like AWS, GCP, or Azure.

**Q: Does VideoGPT work offline?**
A: No, VideoGPT requires internet access for YouTube API and Gemini API calls.

## 🙏 Acknowledgments

- **LangChain**: For the amazing framework that powers our RAG system
- **Google Gemini**: For providing the LLM and embedding capabilities
- **Streamlit**: For the intuitive web framework
- **FAISS**: For efficient vector similarity search
- **YouTube API**: For transcript access

## 🌟 Star History

If you find VideoGPT helpful, please consider giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/videogpt&type=Date)](https://star-history.com/#yourusername/videogpt&Date)

---


*Transform your video consumption with AI-powered intelligence.*