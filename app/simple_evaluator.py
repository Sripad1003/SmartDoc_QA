import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import logging
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, rag_system, document_processor):
        self.rag_system = rag_system
        self.document_processor = document_processor
    
    async def generate_questions_from_document(self, doc_id: str, max_questions: int = 5) -> List[Dict]:
        """Generate questions from uploaded document"""
        questions_data = []
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return questions_data
            
            # Select diverse chunks for question generation
            selected_chunks = chunks[:max_questions]
            
            for i, chunk in enumerate(selected_chunks):
                context_text = chunk.get("text", "")
                if len(context_text) < 100:  # Skip very short chunks
                    continue
                
                try:
                    # Generate a clear, answerable question
                    prompt = f"""Based on this text, create ONE specific question that can be answered directly from the content:

Text: {context_text[:600]}

Generate only the question (no extra text):"""
                    
                    question = await self.rag_system._generate_with_gemini(prompt)
                    question = question.strip()
                    
                    # Clean up the question
                    if not question.endswith('?'):
                        question += '?'
                    question = question.replace("Question:", "").replace("Q:", "").strip()
                    
                    # Get the expected answer from the context
                    answer_prompt = f"""Answer this question based on the provided text:

Question: {question}
Text: {context_text}

Provide a concise, direct answer:"""
                    
                    expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
                    expected_answer = expected_answer.strip()
                    
                    if question and expected_answer and len(expected_answer) > 10:
                        questions_data.append({
                            "question": question,
                            "expected_answer": expected_answer,
                            "context": context_text[:300] + "..."
                        })
                        
                        logger.info(f"Generated Q{i+1}: {question[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error generating question {i}: {str(e)}")
                    continue
            
            logger.info(f"Generated {len(questions_data)} questions from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
        
        return questions_data
    
    async def evaluate_with_f1(self, questions_data: List[Dict]) -> Dict[str, Any]:
        """Simple F1 score evaluation"""
        logger.info(f"Starting F1 evaluation with {len(questions_data)} questions")
        
        results = {
            "total_questions": len(questions_data),
            "f1_scores": [],
            "response_times": [],
            "predictions": [],
            "errors": []
        }
        
        for i, item in enumerate(questions_data):
            try:
                question = item["question"]
                expected_answer = item["expected_answer"]
                
                # Measure response time
                start_time = time.time()
                
                # Get prediction from our system
                answer_data = await self.rag_system.generate_answer(question)
                predicted_answer = answer_data["answer"]
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                # Calculate F1 score
                f1_score = self._calculate_f1_score(predicted_answer, expected_answer)
                results["f1_scores"].append(f1_score)
                
                results["predictions"].append({
                    "question": question,
                    "predicted": predicted_answer,
                    "expected": expected_answer,
                    "f1_score": f1_score,
                    "response_time": response_time,
                    "confidence": answer_data.get("confidence", 0)
                })
                
                logger.info(f"Q{i+1} F1: {f1_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating question {i}: {str(e)}")
                results["errors"].append({
                    "question_index": i,
                    "error": str(e)
                })
        
        # Calculate final metrics
        results["average_f1"] = statistics.mean(results["f1_scores"]) if results["f1_scores"] else 0
        results["average_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        results["evaluation_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Evaluation completed: Average F1={results['average_f1']:.3f}")
        return results
    
    def _calculate_f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score using token overlap"""
        predicted_tokens = self._tokenize(predicted)
        expected_tokens = self._tokenize(expected)
        
        if not expected_tokens:
            return 1.0 if not predicted_tokens else 0.0
        
        if not predicted_tokens:
            return 0.0
        
        # Calculate token overlap
        common_tokens = Counter(predicted_tokens) & Counter(expected_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(predicted_tokens)
        recall = num_common / len(expected_tokens)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.lower()
        answer = ''.join(char for char in answer if char not in string.punctuation)
        answer = ' '.join(answer.split())
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        return answer.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for F1 calculation"""
        normalized = self._normalize_answer(text)
        return normalized.split()
    
    async def run_simple_evaluation(self, doc_id: str) -> Dict[str, Any]:
        """Run complete simple evaluation"""
        try:
            # Generate questions from document
            logger.info(f"Generating questions from document: {doc_id}")
            questions_data = await self.generate_questions_from_document(doc_id, max_questions=6)
            
            if not questions_data:
                return {
                    "error": "No questions could be generated from the document",
                    "doc_id": doc_id
                }
            
            # Evaluate with F1 scores
            logger.info("Running F1 evaluation...")
            evaluation_results = await self.evaluate_with_f1(questions_data)
            
            # Add document info
            evaluation_results["doc_id"] = doc_id
            evaluation_results["questions_generated"] = len(questions_data)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in simple evaluation: {str(e)}")
            return {
                "error": str(e),
                "doc_id": doc_id
            }
