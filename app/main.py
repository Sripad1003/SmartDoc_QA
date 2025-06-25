import logging
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.document_processor import DocumentProcessor
from core.rag import RAGSystem
from .simple_evaluator import SimpleEvaluator  # Import SimpleEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration (adjust as needed for production)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "*",  # WARNING: This allows all origins.  Remove for production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Settings(BaseModel):
    pinecone_api_key: str
    pinecone_environment: str
    openai_api_key: str
    anthropic_api_key: str
    embedding_model: str = "openai"
    llm_model: str = "openai"
    index_name: str = "quickstart"
    chunk_size: int = 512
    chunk_overlap: int = 50


@app.on_event("startup")
async def startup_event():
    try:
        # Load settings from environment variables
        settings = Settings(
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            pinecone_environment=os.environ["PINECONE_ENVIRONMENT"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        # Initialize RAG system and document processor
        app.rag_system = RAGSystem(
            pinecone_api_key=settings.pinecone_api_key,
            pinecone_environment=settings.pinecone_environment,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
            index_name=settings.index_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        await app.rag_system.init_vectorstore()  # Initialize the vectorstore

        app.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        app.evaluator = SimpleEvaluator(app.rag_system, app.document_processor)

        logger.info("Application startup complete.")

    except KeyError as e:
        logger.error(f"Missing environment variable: {e}")
        raise  # Re-raise the exception to prevent the app from starting

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise  # Re-raise to prevent the app from starting


@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG API"}


@app.post("/upload")
async def upload_document(url: str):
    try:
        doc_id = await app.rag_system.ingest_document(url)
        return {"message": "Document uploaded successfully", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/{doc_id}")
async def query_document(doc_id: str, query: str):
    try:
        response = await app.rag_system.query_document(doc_id, query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-questions/{doc_id}")
async def generate_questions(doc_id: str, max_questions: int = 8):
    try:
        questions_data = await app.evaluator.generate_questions_from_document(doc_id, max_questions)
        
        if not questions_data:
            raise HTTPException(status_code=404, detail="No questions could be generated from this document")
        
        # Extract just the questions for display
        questions = [q_data["question"] for q_data in questions_data]
        
        return {
            "questions": questions,
            "questions_data": questions_data,
            "questions_generated": len(questions_data),
            "doc_id": doc_id
        }
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/{doc_id}")
async def evaluate_document(doc_id: str):
    try:
        results = await app.evaluator.run_simple_evaluation(doc_id)
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "results": results,
            "doc_id": doc_id,
            "evaluation_completed": True
        }
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
