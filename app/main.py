from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import asyncio
from datetime import datetime
import logging
import uuid
import time

# Import our modules
from .document_processor import DocumentProcessor
from .rag_system import RAGSystem
from .config import Config
from .evaluation_system import EvaluationSystem

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Document Q&A System",
    version="2.0.0",
    description="Enhanced document Q&A system with comprehensive evaluation capabilities"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    context_limit: int = 5

class EvaluationRequest(BaseModel):
    dataset_type: str = "squad"  # "squad", "coqa", "custom"
    sample_size: Optional[int] = None
    include_performance: bool = True
    doc_id: Optional[str] = None

# Initialize system components
doc_processor = DocumentProcessor()
rag_system = RAGSystem()
evaluation_system = EvaluationSystem(rag_system, doc_processor)

# Simple session storage
sessions = {}

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 Starting Enhanced Document Q&A System with Evaluation")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    
    if Config.validate():
        logger.info("✅ Configuration validated successfully")
    else:
        logger.warning("⚠️  Configuration validation failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Document Q&A System API with Evaluation",
        "version": "2.0.0",
        "features": [
            "Enhanced answer generation",
            "Comprehensive document processing",
            "SQUAD 2.0 evaluation support",
            "Conversational Q&A evaluation",
            "Performance benchmarking",
            "Production-ready metrics"
        ],
        "endpoints": {
            "upload": "/upload-documents",
            "ask": "/ask-question", 
            "test": "/test-document",
            "evaluate": "/evaluate",
            "benchmark": "/benchmark",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "documents_processed": len(doc_processor.processed_documents),
        "chunks_indexed": len(rag_system.chunk_embeddings),
        "gemini_api": "configured" if Config.GEMINI_API_KEY else "not_configured",
        "evaluation_ready": True,
        "config": {
            "max_chunk_size": Config.MAX_CHUNK_SIZE,
            "max_retrieval_chunks": Config.MAX_RETRIEVAL_CHUNKS,
            "max_answer_length": Config.MAX_ANSWER_LENGTH
        }
    }

@app.post("/evaluate")
async def run_evaluation(request: EvaluationRequest):
    """Run comprehensive system evaluation"""
    try:
        if request.dataset_type not in ["squad", "coqa", "custom"]:
            raise HTTPException(status_code=400, detail="Invalid dataset type")

        # Run the comprehensive evaluation suite with document context if provided
        doc_id = request.doc_id
        if not doc_id:
            raise HTTPException(status_code=400, detail="Document ID is required for evaluation. Please upload documents first.")

        results = await evaluation_system.run_comprehensive_evaluation(doc_id=doc_id)
        evaluation_system.evaluation_results = results

        return {
            "message": "Comprehensive evaluation completed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation-report")
async def get_evaluation_report():
    """Get detailed evaluation report"""
    try:
        # Use the latest evaluation results stored in evaluation_system
        evaluation_results = getattr(evaluation_system, "evaluation_results", {})
        report = evaluation_system.generate_evaluation_report(evaluation_results)
        return {
            "message": "Evaluation report generated",
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating evaluation report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
async def run_performance_benchmark():
    """Run performance benchmark"""
    try:
        results = await evaluation_system.run_benchmark()
        grade = _get_performance_grade(results)
        return {
            "message": "Performance benchmark completed",
            "results": results,
            "grade": grade
        }
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_performance_grade(results: dict) -> str:
    """Get performance grade based on benchmark results"""
    avg_time = results.get("average_response_time", float('inf'))
    error_rate = results.get("error_rate", 1.0)
    
    if avg_time < 2.0 and error_rate < 0.01:
        return "A+ (Excellent)"
    elif avg_time < 3.0 and error_rate < 0.05:
        return "A (Very Good)"
    elif avg_time < 5.0 and error_rate < 0.1:
        return "B (Good)"
    elif avg_time < 8.0 and error_rate < 0.2:
        return "C (Acceptable)"
    else:
        return "D (Needs Improvement)"

@app.post("/test-document")
async def test_document_processing(file: UploadFile = File(...)):
    """Enhanced document processing test with evaluation metrics"""
    try:
        content = await file.read()
        
        # Detailed extraction analysis
        test_processor = DocumentProcessor()
        
        # Test extraction
        text = await test_processor._extract_text(content, file.filename, file.content_type or "text/plain")
        cleaned_text = test_processor._clean_text(text) if hasattr(test_processor, '_clean_text') else text
        
        # Analyze text quality
        text_analysis = {
            "total_characters": len(text),
            "cleaned_characters": len(cleaned_text),
            "word_count": len(cleaned_text.split()) if cleaned_text else 0,
            "paragraph_count": len([p for p in cleaned_text.split('\n\n') if p.strip()]) if cleaned_text else 0,
            "readability_score": _calculate_readability_score(cleaned_text)
        }
        
        # Test chunking
        chunks = test_processor._intelligent_chunking(cleaned_text, file.filename) if cleaned_text else []
        
        # Test embedding generation (small sample)
        embedding_test = None
        if chunks:
            try:
                sample_text = chunks[0]["text"][:500]
                test_embeddings = await rag_system.generate_embeddings([sample_text])
                embedding_test = {
                    "success": True,
                    "embedding_dimensions": len(test_embeddings[0]) if test_embeddings else 0,
                    "generation_time": time.time()
                }
            except Exception as e:
                embedding_test = {
                    "success": False,
                    "error": str(e)
                }
        
        # Quality assessment
        quality_score = _assess_document_quality(text_analysis, chunks, embedding_test)
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": len(content),
            "text_analysis": text_analysis,
            "chunk_analysis": {
                "total_chunks": len(chunks),
                "average_chunk_size": sum(len(chunk["text"]) for chunk in chunks) / len(chunks) if chunks else 0,
                "optimal_chunk_count": len(chunks) >= 3 and len(chunks) <= 20
            },
            "embedding_test": embedding_test,
            "quality_assessment": quality_score,
            "evaluation_ready": quality_score["overall_score"] > 0.6,
            "recommendations": _generate_processing_recommendations(text_analysis, chunks)
        }
    
    except Exception as e:
        logger.error(f"Error in test document processing: {str(e)}")
        return {
            "filename": file.filename,
            "error": str(e),
            "success": False
        }

def _calculate_readability_score(text: str) -> float:
    """Calculate simple readability score"""
    if not text:
        return 0.0
    
    words = text.split()
    sentences = text.split('.')
    
    if not words or not sentences:
        return 0.0
    
    avg_words_per_sentence = len(words) / len(sentences)
    avg_chars_per_word = sum(len(word) for word in words) / len(words)
    
    # Simple readability score (0-1, higher is better)
    readability = max(0, min(1, 1 - (avg_words_per_sentence - 15) / 50 - (avg_chars_per_word - 5) / 10))
    return round(readability, 3)

def _assess_document_quality(text_analysis: dict, chunks: list, embedding_test: dict) -> dict:
    """Assess overall document quality for Q&A"""
    scores = {
        "text_quality": 0,
        "chunk_quality": 0,
        "embedding_quality": 0,
        "overall_score": 0
    }
    
    # Text quality (0-1)
    word_count = text_analysis.get("word_count", 0)
    if word_count > 500:
        scores["text_quality"] = 1.0
    elif word_count > 100:
        scores["text_quality"] = 0.7
    elif word_count > 50:
        scores["text_quality"] = 0.4
    else:
        scores["text_quality"] = 0.1
    
    # Chunk quality (0-1)
    chunk_count = len(chunks)
    if 5 <= chunk_count <= 15:
        scores["chunk_quality"] = 1.0
    elif 3 <= chunk_count <= 20:
        scores["chunk_quality"] = 0.8
    elif chunk_count > 0:
        scores["chunk_quality"] = 0.5
    else:
        scores["chunk_quality"] = 0.0
    
    # Embedding quality (0-1)
    if embedding_test and embedding_test.get("success"):
        scores["embedding_quality"] = 1.0
    else:
        scores["embedding_quality"] = 0.0
    
    # Overall score
    scores["overall_score"] = (
        scores["text_quality"] * 0.4 +
        scores["chunk_quality"] * 0.3 +
        scores["embedding_quality"] * 0.3
    )
    
    return scores

def _generate_processing_recommendations(text_analysis: dict, chunks: list) -> List[str]:
    """Generate recommendations based on processing results"""
    recommendations = []
    
    word_count = text_analysis.get("word_count", 0)
    if word_count < 100:
        recommendations.append("Document contains very little text - consider using a different file or OCR for image-based PDFs")
    
    if len(chunks) == 0:
        recommendations.append("No chunks created - text might be too short or extraction failed")
    elif len(chunks) > 25:
        recommendations.append("Many chunks created - consider using larger documents or combining related content")
    
    readability = text_analysis.get("readability_score", 0)
    if readability < 0.5:
        recommendations.append("Text might be difficult to process - consider documents with simpler structure")
    
    return recommendations

# Keep existing endpoints (upload, ask-question, etc.)
@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Enhanced document upload with evaluation readiness check"""
    try:
        processed_docs = []
        total_chunks = 0
        evaluation_ready_docs = 0
        
        for file in files:
            content = await file.read()
            
            try:
                start_time = time.time()
                doc_id = await doc_processor.process_document(
                    content, file.filename, file.content_type or "text/plain"
                )
                processing_time = time.time() - start_time
                
                doc_info = doc_processor.get_document_info(doc_id)
                chunks = doc_processor.get_document_chunks(doc_id)
                
                # Index the document chunks
                await rag_system.index_documents(chunks)
                total_chunks += len(chunks)
                
                # Check evaluation readiness
                evaluation_ready = len(chunks) >= 3 and doc_info.get("text_length", 0) > 500
                if evaluation_ready:
                    evaluation_ready_docs += 1
                
                processed_docs.append({
                    "filename": file.filename,
                    "doc_id": doc_id,
                    "status": "processed",
                    "chunks": len(chunks),
                    "text_length": doc_info.get("text_length", 0),
                    "processing_time": round(processing_time, 2),
                    "evaluation_ready": evaluation_ready
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                processed_docs.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e),
                    "chunks": 0,
                    "evaluation_ready": False
                })
        
        successful_count = len([d for d in processed_docs if d['status'] == 'processed'])
        
        return {
            "documents": processed_docs,
            "summary": {
                "total_files": len(files),
                "successful": successful_count,
                "failed": len(files) - successful_count,
                "total_chunks_created": total_chunks,
                "evaluation_ready_docs": evaluation_ready_docs,
                "ready_for_evaluation": evaluation_ready_docs > 0
            },
            "message": f"Processed {successful_count}/{len(files)} documents successfully"
        }
    
    except Exception as e:
        logger.error(f"Error in upload_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Enhanced question answering with evaluation metrics"""
    try:
        start_time = time.time()
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = sessions.get(session_id, [])
        
        # Generate answer
        answer_data = await rag_system.generate_answer(
            request.question, conversation_history[-request.context_limit:]
        )
        
        # Store interaction
        interaction = {
            "question": request.question,
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(interaction)
        
        # Keep only last 20 interactions
        if len(sessions[session_id]) > 20:
            sessions[session_id] = sessions[session_id][-20:]
        
        response_time = time.time() - start_time
        
        # Calculate answer quality metrics
        answer_quality = {
            "length_score": min(len(answer_data["answer"]) / 500, 1.0),
            "source_diversity": len(set(s.get("source", "") for s in answer_data["sources"])),
            "confidence_score": answer_data["confidence"]
        }
        
        return {
            "answer": answer_data["answer"],
            "sources": answer_data["sources"],
            "confidence": answer_data["confidence"],
            "session_id": session_id,
            "response_time": round(response_time, 3),
            "retrieved_chunks": answer_data["retrieved_chunks"],
            "answer_quality": answer_quality,
            "evaluation_metrics": {
                "response_time_grade": "A" if response_time < 3 else "B" if response_time < 5 else "C",
                "confidence_grade": "A" if answer_data["confidence"] > 0.8 else "B" if answer_data["confidence"] > 0.6 else "C"
            }
        }
    
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history with evaluation metrics"""
    try:
        history = sessions.get(session_id, [])
        
        # Calculate conversation metrics
        if history:
            avg_confidence = sum(h.get("confidence", 0) for h in history) / len(history)
            total_questions = len(history)
            avg_answer_length = sum(len(h.get("answer", "")) for h in history) / len(history)
        else:
            avg_confidence = 0
            total_questions = 0
            avg_answer_length = 0
        
        return {
            "session_id": session_id,
            "history": history,
            "conversation_metrics": {
                "total_interactions": total_questions,
                "average_confidence": round(avg_confidence, 3),
                "average_answer_length": round(avg_answer_length, 1),
                "conversation_quality": "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Low"
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-stats")
async def get_system_stats():
    """Get comprehensive system statistics with evaluation readiness"""
    try:
        # Get processing stats
        processing_stats = doc_processor.get_processing_stats()
        
        return {
            "documents": processing_stats,
            "sessions": {
                "active_sessions": len(sessions),
                "total_interactions": sum(len(history) for history in sessions.values())
            },
            "configuration": {
                "max_chunk_size": Config.MAX_CHUNK_SIZE,
                "max_retrieval_chunks": Config.MAX_RETRIEVAL_CHUNKS,
                "max_answer_length": Config.MAX_ANSWER_LENGTH,
                "generation_model": Config.GENERATION_MODEL
            },
            "cache": {
                "query_cache_size": len(rag_system.query_cache),
                "embedding_cache_size": len(rag_system.embedding_cache)
            },
            "evaluation_readiness": {
                "documents_indexed": len(rag_system.chunk_embeddings) > 0,
                "evaluation_system_ready": True,
                "supported_evaluations": ["SQUAD", "COQA", "Performance Benchmark"],
                "ready_for_production": len(rag_system.chunk_embeddings) > 10
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
