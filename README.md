# 📄 Enhanced Document Q&A System v1.0

A production-ready document question-answering system with improved answer quality and comprehensive document processing.

## 📁 Project Structure

\`\`\`
qa-system/
├── app/ # Main application package
│ ├── init.py # Package initialization
│ ├── main.py # FastAPI backend
│ ├── config.py # Configuration settings
│ ├── document_processor.py # Document processing logic
│ ├── rag_system.py # RAG pipeline with memory + retrieval
│ ├── evaluation_system.py # SQUAD evaluation system
│ └── streamlit_app.py # Streamlit frontend interface
├── scripts/ # Evaluation and utility scripts
├── start.py # Startup script
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
└── README.md # This file
\`\`\`

## 🚀 Quick Start

1. **Clone and Install Dependencies**
    ```bash
    git clone https://github.com/Sripad1003/SmartDoc_QA.git
    cd qa-system
    pip install -r requirements.txt
    ```

2. **Configure Environment**
    ```bash
    cp .env.example .env
    # Edit .env to add your Gemini API key and adjust settings as needed
    ```

3. **Run the System**
    ```bash
    python start.py
    ```

4. **Access the Interfaces**
    - **Frontend (Streamlit):** [http://127.0.0.1:8501](http://127.0.0.1:8501)
    - **API (FastAPI):** [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🗂️ Project Structure

```
qa-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   ├── config.py            # Configuration settings
│   ├── document_processor.py# Document processing logic
│   ├── rag_system.py        # RAG pipeline (memory + retrieval)
│   ├── evaluation_system.py # SQUAD evaluation
│   └── streamlit_app.py     # Streamlit frontend
├── scripts/                 # Utility & evaluation scripts
├── start.py                 # Startup script
├── requirements.txt         # Dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

---

## ✨ Features Overview

### 🔍 Enhanced Document Processing
- **Multi-format support:** PDF, DOCX, TXT, HTML, Markdown
- **Intelligent chunking:** Structure-aware segmentation for better context
- **Robust validation:** Extraction quality, method comparison, and fallback logic
- **Error detection:** Scanned images, encoding issues, empty content, and more

### 🤖 Advanced Q&A System
- **Comprehensive answers:** Up to 2000 characters, well-cited & detailed
- **Deep context retrieval:** Fetches 8 most relevant chunks for high-quality answers
- **Source attribution:** Citations with similarity scores
- **Conversation memory:** Maintains chat context for follow-ups

### 🧪 Document Testing & Analysis
- **Pre-upload quality analysis:** Test before processing
- **Library comparison:** See which extraction method performs best
- **Quality metrics:** Word count, chunk distribution, embedding checks
- **Actionable recommendations:** Specific tips to improve document processing

### 📊 SQUAD Evaluation System
- **Automated SQUAD 2.0 evaluation:** F1 & Exact Match scoring
- **Performance benchmarking:** Reports on speed, accuracy, and readiness
- **Comprehensive reporting:** Detailed evaluation summaries

---

## 🏗️ System Architecture

```
flowchart LR
    A[Frontend (Streamlit)] <--> B[Backend (FastAPI)]
    B --> C[Google Gemini API]
    B --> D[Document Processor]
    D --> E[RAG System (Enhanced)]
    E --> F[Evaluation System (SQUAD)]
```

---

## ⚙️ Configuration

### Required Environment Variables (`.env`)
```
GEMINI_API_KEY=your-api-key-here
```

### Optional Enhanced Defaults
```
MAX_CHUNK_SIZE=1500          # Larger context per chunk
MAX_RETRIEVAL_CHUNKS=8       # Fetch more sources for answers
MAX_ANSWER_LENGTH=2000       # Longer, in-depth responses
```

- **Model:** Uses Gemini-1.5-Flash for fast, high-quality answers

---

## 🧪 Document Testing: How It Works

**Quality Indicators:**
- 🟢 **Excellent** (>500 words): Ideal for Q&A
- 🟡 **Good** (100-500 words): Sufficient for basic use
- 🔴 **Poor** (<100 words): Needs improvement

**Common Issues Detected:**
- Scanned PDFs needing OCR
- Complex layouts (tables/columns)
- Encoding or special character problems
- Empty or corrupted files

**Testing Steps:**
1. **Text Extraction:** Measures readable content
2. **Method Comparison:** Chooses best extraction library
3. **Chunk Analysis:** Segments and evaluates structure
4. **Embedding Check:** Ensures searchability

---

## 📈 Performance Highlights (v2.0)

- **3× longer answers** for comprehensive details
- **60% more context** retrieved for better accuracy
- **50% more robust document processing** via multi-method extraction
- **Advanced testing tools** for document analysis
- **SQUAD benchmark system** for answer quality

---

## 📊 SQUAD Evaluation

**Metrics:**
| Metric         | Target (Production) |
| -------------- |:------------------:|
| F1 Score       | > 0.75             |
| Exact Match    | > 0.65             |
| Response Time  | < 5s per query     |
| Error Rate     | < 1%               |

---

## 🛠️ Troubleshooting

| Issue                | Solutions                                           |
|----------------------|----------------------------------------------------|
| Poor answer quality  | Use document test, check extraction quality        |
| Processing fails     | Try different file formats, check file integrity   |
| Slow responses       | First-time processing generates embeddings         |
| Short answers        | Add more relevant documents, rephrase questions    |

- **Check logs:** Backend for errors, browser console for frontend issues

---

## 🤝 Contributing

1. **Fork & branch**: Fork the repo and create a feature branch
2. **Develop & test**: Implement and test your changes
3. **Pull request**: Submit your PR for review

---
