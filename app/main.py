from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging,uuid,time,uvicorn,random

# Import our modules - FIXED IMPORTS
from .document_processor import DocumentProcessor
from .rag_system import RAGSystem
from .config import Config
from .simple_evaluator import SimpleEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Document Q&A System",
    version="1.0.0",
    description="Clean document Q&A system with F1 evaluation"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# question endpoint — validates incoming JSON.
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

# Initialize system components
doc_processor = DocumentProcessor()
rag_system = RAGSystem()
evaluator = SimpleEvaluator(rag_system, doc_processor)

# Simple session storage
sessions = {}

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 Starting Simple Document Q&A System")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple Document Q&A System",
        "version": "1.0.0",
        "features": [
            "Document upload and processing",
            "Question answering",
            "Simple F1 evaluation",
            "Question generation from documents"
        ],
        "endpoints": {
            "upload": "/upload-documents",
            "ask": "/ask-question",
            "evaluate": "/evaluate/{doc_id}",
            "questions": "/generate-questions/{doc_id}",
            "documents": "/list-documents",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "documents_processed": len(doc_processor.processed_documents),
        "chunks_indexed": len(rag_system.chunk_embeddings)
    }

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        processed_docs = []
        
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
                
                processed_docs.append({
                    "filename": file.filename,
                    "doc_id": doc_id,
                    "status": "processed",
                    "chunks": len(chunks),
                    "text_length": doc_info.get("text_length", 0),
                    "processing_time": round(processing_time, 2)
                })
                
                logger.info(f"Processed {file.filename} with {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                processed_docs.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        successful_count = len([d for d in processed_docs if d['status'] == 'processed'])
        
        return {
            "documents": processed_docs,
            "summary": {
                "total_files": len(files),
                "successful": successful_count,
                "failed": len(files) - successful_count
            },
            "message": f"Processed {successful_count}/{len(files)} documents successfully"
        }
    
    except Exception as e:
        logger.error(f"Error in upload_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = []
        for doc_id, doc_info in doc_processor.processed_documents.items():
            chunks = doc_processor.get_document_chunks(doc_id)
            documents.append({
                "doc_id": doc_id,
                "filename": doc_info.get("filename", "Unknown"),
                "processed_at": doc_info.get("processed_at", "Unknown"),
                "chunk_count": len(chunks),
                "text_length": doc_info.get("text_length", 0)
            })
        
        return {
            "total_documents": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-questions/{doc_id}")
async def generate_questions(doc_id: str, max_questions: int = random.randint(5, 8)):
    """Generate questions from a document"""
    try:
        if doc_id not in doc_processor.processed_documents:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        doc_info = doc_processor.get_document_info(doc_id)
        questions_data = await evaluator.generate_questions_from_document(doc_id, max_questions)
        
        if not questions_data:
            raise HTTPException(status_code=400, detail="No questions could be generated from this document")
        
        questions = [item["question"] for item in questions_data]
        
        return {
            "doc_id": doc_id,
            "filename": doc_info.get("filename", "Unknown"),
            "questions_generated": len(questions),
            "questions": questions,
            "questions_data": questions_data  # Full data for evaluation
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/{doc_id}")
async def evaluate_document(doc_id: str):
    """Run F1 evaluation on a document"""
    try:
        if doc_id not in doc_processor.processed_documents:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        results = await evaluator.run_simple_evaluation(doc_id)
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "message": "Evaluation completed",
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Ask a question"""
    try:
        start_time = time.time()
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = sessions.get(session_id, [])
        
        # Generate answer
        answer_data = await rag_system.generate_answer(
            request.question, conversation_history[-5:]
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
        
        # Keep only last 10 interactions
        if len(sessions[session_id]) > 10:
            sessions[session_id] = sessions[session_id][-10:]
        
        response_time = time.time() - start_time
        
        return {
            "answer": answer_data["answer"],
            "sources": answer_data["sources"],
            "confidence": answer_data["confidence"],
            "session_id": session_id,
            "response_time": round(response_time, 3)
        }
    
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history"""
    try:
        history = sessions.get(session_id, [])
        
        return {
            "session_id": session_id,
            "history": history,
            "total_interactions": len(history)
        }
    
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        processing_stats = doc_processor.get_processing_stats()
        
        return {
            "documents": processing_stats,
            "sessions": {
                "active_sessions": len(sessions),
                "total_interactions": sum(len(history) for history in sessions.values())
            },
            "system": {
                "documents_processed": len(doc_processor.processed_documents),
                "chunks_indexed": len(rag_system.chunk_embeddings),
                "cache_size": len(rag_system.query_cache)
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
    )
