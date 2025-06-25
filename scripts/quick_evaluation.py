import asyncio
import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv

from core.doc_processor import DocumentProcessor
from core.rag import RAGSystem
from core.embedding import OpenAIEmbeddingModel, EmbeddingModel
from core.llm import OpenAIModel, LLMModel

load_dotenv()

# Define paths
DOC_DIR = "documents"
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
LLM_MODEL_NAME = "gpt-3.5-turbo"
INDEX_NAME = "my_index"
SQUAD_FILE = "squad_dev.json"
MAX_DOCS = 20 # Maximum number of documents to process
MAX_CHUNKS = 500 # Maximum number of chunks to index

# Load SQUAD data
def load_squad_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        squad_data = json.load(f)
    return squad_data['data']

# Extract question-answer pairs from SQUAD data
def extract_qas(squad_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    qas = []
    for article in squad_data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [answer['text'] for answer in qa['answers']]
                # For simplicity, use only the first answer as the ground truth
                if answers:
                    qas.append({'question': question, 'answer': answers[0]})
    return qas

async def run_squad_evaluation(doc_processor: DocumentProcessor, rag_system: RAGSystem, squad_file: str, max_questions: int = 10):
    """
    Evaluates the RAG system using the SQUAD dataset.
    """
    print("📝 Running SQUAD evaluation...")

    # Load SQUAD data and extract question-answer pairs
    squad_data = load_squad_data(squad_file)
    qas = extract_qas(squad_data)

    correct_predictions = 0
    total_questions = min(len(qas), max_questions)

    start_time = time.time()

    # Check for existing documents
    if not doc_processor.processed_documents:
        print("❌ No documents found!")
        print("📄 Please upload documents first using:")
        print("   python start.py")
        print("   Then upload via the web interface")
        return None
    
    # Get existing chunks
    all_chunks = doc_processor.get_all_chunks()
    if not all_chunks:
        print("❌ No chunks found!")
        return None
        
    print(f"📊 Using {len(all_chunks)} existing chunks")
    
    # Index documents
    print("🔗 Indexing documents...")
    await rag_system.index_documents(all_chunks)
    print(f"✅ Indexed {len(rag_system.chunk_embeddings)} chunks")

    for i in range(total_questions):
        qa = qas[i]
        question = qa['question']
        ground_truth = qa['answer']

        # Query the RAG system
        response = await rag_system.query(question)
        predicted_answer = response.answer.strip()

        # Evaluate the response (simple string matching)
        if ground_truth.lower() in predicted_answer.lower() or predicted_answer.lower() in ground_truth.lower():
            correct_predictions += 1
            print(f"✅ Question {i+1}: Correct!")
        else:
            print(f"❌ Question {i+1}: Incorrect.")
            print(f"  Question: {question}")
            print(f"  Predicted Answer: {predicted_answer}")
            print(f"  Ground Truth: {ground_truth}")

    end_time = time.time()
    evaluation_time = end_time - start_time

    accuracy = (correct_predictions / total_questions) * 100
    print(f"\n🎯 Accuracy: {accuracy:.2f}%")
    print(f"⏱️  Evaluation Time: {evaluation_time:.2f} seconds")

async def main():
    # Initialize components
    embedding_model: EmbeddingModel = OpenAIEmbeddingModel(EMBEDDING_MODEL_NAME)
    llm: LLMModel = OpenAIModel(LLM_MODEL_NAME)
    doc_processor = DocumentProcessor(DOC_DIR, embedding_model, MAX_DOCS, MAX_CHUNKS)
    rag_system = RAGSystem(embedding_model, llm, INDEX_NAME)

    # Run SQUAD evaluation
    await run_squad_evaluation(doc_processor, rag_system, SQUAD_FILE)

if __name__ == "__main__":
    asyncio.run(main())
