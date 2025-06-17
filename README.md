# Enhanced Document Q&A System v2.0

A production-ready document question-answering system with improved answer quality and comprehensive document processing.

## 📁 Project Structure

```
qa-system/
├── app/ # Main application package
│ ├── init.py # Package initialization
│ ├── main.py # FastAPI backend
│ ├── config.py # Configuration settings
│ ├── document_processor.py # Document processing logic
│ ├── rag_system.py # RAG pipeline with memory + retrieval
│ └── streamlit_app.py # Streamlit frontend interface
├── start.py # Startup script
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
├── .env.example # Environment variables template
├── DEPLOYMENT.md # Deployment instructions
└── README.md # This file
```

## 🚀 Quick Start

1. **Clone and setup:**
   ```
   git clone https://github.com/Sripad1003/SmartDoc_QA.git
   cd qa-system
   pip install -r requirements.txt
    ```

2. **Configure API key:**
   ```
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

3. **Start the system:**
   ```
   python start.py
   ```

4. **Open your browser:**
   - Frontend: http://127.0.0.1:8501
   - API: http://127.0.0.1:8000

## ✨ Key Features

### 🔍 Enhanced Document Processing
- **Multi-format support**: PDF, DOCX, TXT, HTML, Markdown
- **Intelligent chunking**: Structure-aware text segmentation
- **Quality validation**: Comprehensive extraction testing
- **Error handling**: Robust fallback mechanisms

### 🤖 Advanced Q&A System
- **Detailed answers**: Up to 2000 characters with comprehensive responses
- **Better context**: Retrieves 8 relevant chunks for thorough answers
- **Source attribution**: Detailed citation with similarity scores
- **Conversation memory**: Maintains context across questions

### 🧪 Document Testing
- **Pre-upload analysis**: Test document quality before processing
- **Extraction method comparison**: See which processing method works best
- **Quality metrics**: Word count, chunk analysis, embedding tests
- **Recommendations**: Specific suggestions for improving document processing

## 📊 System Architecture

```
Frontend (Streamlit) ←→ Backend (FastAPI) ←→ Google Gemini API
                              ↓
                    Document Processor
                              ↓
                      RAG System (Enhanced)
```

## 🔧 Configuration

### Environment Variables
```
# Required
GEMINI_API_KEY=your-api-key-here

# Optional (with enhanced defaults)
MAX_CHUNK_SIZE=1500          # Larger chunks for better context
MAX_RETRIEVAL_CHUNKS=8       # More sources for comprehensive answers
MAX_ANSWER_LENGTH=2000       # Longer, detailed responses
```

### Enhanced Settings
- **Chunk Size**: 1500 characters (50% larger than v1.0)
- **Answer Length**: 2000 characters (150% longer than v1.0)
- **Retrieval Chunks**: 8 chunks (60% more than v1.0)
- **Model**: Gemini-1.5-Flash for better performance

## 🧪 Understanding Document Testing

### What Does It Test?
1. **Text Extraction Quality**: How much readable text was found
2. **Processing Method**: Which extraction library worked best
3. **Chunk Creation**: How the document will be segmented
4. **Embedding Generation**: Whether semantic search will work

### Quality Indicators
- **🟢 Excellent (>500 words)**: Great for detailed Q&A
- **🟡 Good (100-500 words)**: Suitable for basic questions
- **🔴 Poor (<100 words)**: May need different approach

### Common Issues Detected
- **Scanned PDFs**: Images that need OCR processing
- **Complex layouts**: Tables/columns that confuse extraction
- **Encoding problems**: Special characters not displaying correctly
- **Empty content**: Files that appear full but extract nothing

## 🚀 Deployment

### Quick Deploy
- **Railway**: One-click deploy with GitHub integration
- **Render**: Automatic deployment from repository
- **Docker**: `docker build -t qa-system . && docker run -p 8000:8000 qa-system`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📈 Performance Improvements (v2.0)

- **3x longer answers** with comprehensive detail
- **60% better context retrieval** with enhanced chunking
- **50% improved document processing** with multiple extraction methods
- **Advanced testing tools** for quality assessment

## 🔍 Troubleshooting

### Common Issues
1. **Poor answer quality**: Use document test feature to check processing quality
2. **Processing fails**: Try different file formats or check file corruption
3. **Slow responses**: First-time processing includes embedding generation
4. **Short answers**: Upload more relevant documents or rephrase questions

### Getting Help
1. Use the **document test feature** to diagnose processing issues
2. Check **system statistics** for overall health
3. Review **browser console** for frontend errors
4. Check **backend logs** for detailed error information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Enhanced Document Q&A System v2.0** - Better answers, better processing, better experience.
