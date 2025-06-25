import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import DocumentProcessor
from app.rag_system import RAGSystem
from app.evaluation_system import EvaluationSystem
from app.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run SQUAD evaluation and report results"""
    try:
        print("🚀 Starting SQUAD Evaluation System...")
        
        # Initialize components
        doc_processor = DocumentProcessor()
        rag_system = RAGSystem()
        evaluation_system = EvaluationSystem(rag_system, doc_processor)
        
        print("✅ Components initialized successfully")
        
        # Add sample content for evaluation
        print("📄 Adding sample content...")
        sample_content = """# AI/ML Exercise: Intelligent Document Q&A System

## Project Overview
Build a production-ready document question-answering system that learns from user interactions and improves over time. This system should handle multiple document formats, maintain conversation history, and adapt based on user feedback.

## Technical Requirements

### Phase 1: Document Processing Pipeline (30-40 minutes)
**Objective**: Create a robust document ingestion system

**Key Components**:
1. **Multi-format Document Support**
   - PDF processing with multiple extraction methods
   - DOCX document handling
   - Plain text and HTML support
   - Markdown file processing
   - Error handling for corrupted files

2. **Intelligent Text Chunking**
   - Semantic chunking based on document structure
   - Overlap management for context preservation
   - Metadata extraction and storage
   - Chunk size optimization for embeddings

### Phase 2: RAG System Implementation (40-50 minutes)
**Objective**: Implement retrieval-augmented generation

**Core Features**:
1. **Vector Database Setup**
   - Document embedding generation using Google Gemini
   - Efficient similarity search implementation
   - Metadata indexing for source tracking
   - Caching mechanisms for performance

2. **Query Processing**
   - Query understanding and expansion
   - Multi-modal retrieval strategies
   - Context ranking and selection
   - Relevance scoring algorithms

### Phase 3: System Integration (30-40 minutes)
**Objective**: Create a cohesive user experience

### Phase 4: Evaluation and Testing (30-40 minutes)
**Objective**: Comprehensive system evaluation

**Evaluation Datasets**:
1. **Stanford Question Answering Dataset (SQUAD 2.0)**
   - 150,000+ questions on 500+ Wikipedia articles
   - Includes unanswerable questions for robustness testing
   - F1 and Exact Match scoring

2. **COQA - Conversational Question Answering**
   - 127,000+ questions with answers from 8,000+ conversations
   - Multi-turn conversational structure
   - Coherence and context retention evaluation

3. **Natural Questions (NQ) Dataset**
   - Real Google search queries with Wikipedia answers
   - Long-form and short-form answer annotations

**Evaluation Metrics**:
- **F1 Score**: Token overlap between predicted and expected answers
- **Exact Match**: Binary score for perfect answer matches
- **Response Time**: System performance benchmarking
- **Coherence Score**: Conversational consistency measurement
- **Memory Usage**: Resource utilization profiling

**Performance Targets**:
- Document processing: < 30 seconds per document
- Query response time: < 5 seconds
- Answer accuracy: > 80% user satisfaction (F1 > 0.8)
- System uptime: > 99% availability

## Recommended Technology Stack

### Backend Technologies
- **Framework**: FastAPI for high-performance API development
- **LLM Integration**: Google Gemini Pro for text generation
- **Embeddings**: Google Gemini Embedding Model
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup4
- **Evaluation**: Custom SQUAD/COQA evaluation pipeline

### Evaluation and Testing
- **Automated Testing Suite**: SQUAD evaluation metrics (F1, Exact Match)
- **Conversational Testing**: COQA for conversational coherence
- **Performance Benchmarking**: Response time and memory profiling
- **Production Readiness**: API documentation, error handling, rate limiting

## Success Metrics

### Accuracy Benchmarks
- **SQUAD 2.0 F1 Score**: Target > 0.75
- **SQUAD 2.0 Exact Match**: Target > 0.65
- **COQA F1 Score**: Target > 0.70
- **Conversational Coherence**: Target > 0.80

### Performance Benchmarks
- **Average Response Time**: < 3 seconds
- **P95 Response Time**: < 8 seconds
- **Throughput**: > 10 requests/second
- **Memory Usage**: < 500MB per request
- **Error Rate**: < 1%

This comprehensive evaluation framework ensures the system meets production-quality standards for accuracy, performance, and reliability."""

        # Add content to document processor
        doc_id = doc_processor.add_manual_content(
            "AI_ML_Exercise_with_Evaluation.md",
            sample_content
        )
        
        # Get chunks and index them
        chunks = doc_processor.get_document_chunks(doc_id)
        print(f"📊 Created {len(chunks)} chunks from document")
        
        # Index documents
        print("🔗 Indexing documents with embeddings...")
        await rag_system.index_documents(chunks)
        print(f"✅ Indexed {len(rag_system.chunk_embeddings)} chunks")
        
        # Run comprehensive evaluation
        print("🧪 Running comprehensive SQUAD evaluation...")
        results = await evaluation_system.run_comprehensive_evaluation(doc_id)
        
        # Extract and display results
        print("\n" + "="*60)
        print("📊 SQUAD EVALUATION RESULTS")
        print("="*60)
        
        if "squad_results" in results:
            squad = results["squad_results"]
            print(f"📋 Total Questions Evaluated: {squad.get('total_questions', 0)}")
            print(f"🎯 Exact Match Score: {squad.get('exact_match_score', 0):.3f}")
            print(f"📈 F1 Score: {squad.get('average_f1', 0):.3f}")
            print(f"⏱️  Average Response Time: {squad.get('average_response_time', 0):.3f}s")
            print(f"❌ Errors: {len(squad.get('errors', []))}")
            
            # Performance assessment
            f1_score = squad.get('average_f1', 0)
            em_score = squad.get('exact_match_score', 0)
            
            print(f"\n🎯 ACCURACY ASSESSMENT:")
            if f1_score >= 0.8:
                print("   F1 Score: A+ (Excellent) - Exceeds target of 0.75")
            elif f1_score >= 0.75:
                print("   F1 Score: A (Very Good) - Meets target of 0.75")
            elif f1_score >= 0.7:
                print("   F1 Score: B (Good) - Close to target")
            else:
                print("   F1 Score: C (Needs Improvement) - Below target")
            
            if em_score >= 0.65:
                print("   Exact Match: A (Excellent) - Meets target of 0.65")
            elif em_score >= 0.6:
                print("   Exact Match: B (Good) - Close to target")
            else:
                print("   Exact Match: C (Needs Improvement) - Below target")
        
        if "performance_results" in results:
            perf = results["performance_results"]
            print(f"\n⚡ PERFORMANCE RESULTS:")
            print(f"   Average Response Time: {perf.get('average_response_time', 0):.3f}s")
            print(f"   P95 Response Time: {perf.get('p95_response_time', 0):.3f}s")
            print(f"   Throughput: {perf.get('throughput', 0):.2f} requests/second")
            print(f"   Error Rate: {perf.get('error_rate', 0):.3f}")
            print(f"   Memory Usage: {perf.get('average_memory_usage', 0):.2f} MB")
            
            # Performance assessment
            avg_time = perf.get('average_response_time', 0)
            if avg_time < 3.0:
                print("   Performance Grade: A (Excellent) - Meets target < 3s")
            elif avg_time < 5.0:
                print("   Performance Grade: B (Good) - Meets target < 5s")
            else:
                print("   Performance Grade: C (Needs Improvement) - Above target")
        
        # Generate detailed report
        print(f"\n📋 GENERATING DETAILED REPORT...")
        report = evaluation_system.generate_evaluation_report(results)
        
        # Save report to file
        with open("evaluation_report.txt", "w") as f:
            f.write(report)
        
        print("✅ Evaluation completed successfully!")
        print("📄 Detailed report saved to 'evaluation_report.txt'")
        
        # Summary
        print(f"\n🎉 SUMMARY:")
        if "squad_results" in results:
            squad = results["squad_results"]
            f1 = squad.get('average_f1', 0)
            em = squad.get('exact_match_score', 0)
            
            if f1 >= 0.75 and em >= 0.65:
                print("   ✅ System meets production-quality SQUAD benchmarks!")
            elif f1 >= 0.7 or em >= 0.6:
                print("   ⚠️  System shows good performance but could be improved")
            else:
                print("   ❌ System needs significant improvement to meet benchmarks")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = asyncio.run(main())
