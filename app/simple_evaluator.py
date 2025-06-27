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
    
    async def generate_questions_from_document(self, doc_id: str, max_questions: int = None) -> List[Dict]:
        """Generate questions from uploaded document"""
        questions_data = []
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return []
            
            logger.info(f"Found {len(chunks)} chunks for document {doc_id}")
            
            # Simple logic: Default 5, scale up for larger documents
            if max_questions is None:
                if len(chunks) >= 25:
                    max_questions = 8
                elif len(chunks) >= 15:
                    max_questions = 7
                elif len(chunks) >= 8:
                    max_questions = 6
                else:
                    max_questions = 5
            
            logger.info(f"Target: {max_questions} questions from {len(chunks)} chunks")
            
            # Use chunks with reasonable content
            usable_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if len(text) >= 20:
                    usable_chunks.append(chunk)
            
            if not usable_chunks:
                logger.error("No usable chunks found")
                return []
            
            logger.info(f"Using {len(usable_chunks)} usable chunks")
            
            generated_questions = set()
            
            # Strategy 1: Template questions
            for i, chunk in enumerate(usable_chunks[:max_questions]):
                if len(questions_data) >= max_questions:
                    break
                
                try:
                    question_data = await self._generate_template_question(chunk, i)
                    if question_data and question_data['question'].lower() not in generated_questions:
                        questions_data.append(question_data)
                        generated_questions.add(question_data['question'].lower())
                        logger.info(f"Generated Q{len(questions_data)}: {question_data['question'][:50]}...")
                except Exception as e:
                    logger.error(f"Error in template question {i}: {str(e)}")
                    continue
            
            # Strategy 2: Basic questions if needed
            if len(questions_data) < 3:
                logger.info("Not enough questions, trying basic approach...")
                for i, chunk in enumerate(usable_chunks):
                    if len(questions_data) >= max_questions:
                        break
                    
                    try:
                        question_data = await self._generate_basic_question(chunk, i)
                        if question_data and question_data['question'].lower() not in generated_questions:
                            questions_data.append(question_data)
                            generated_questions.add(question_data['question'].lower())
                            logger.info(f"Generated basic Q{len(questions_data)}: {question_data['question'][:50]}...")
                    except Exception as e:
                        logger.error(f"Error in basic question {i}: {str(e)}")
                        continue
            
            # Strategy 3: Simple fallback questions
            if len(questions_data) < 2:
                logger.info("Using fallback questions...")
                simple_questions = [
                    "What is the main topic discussed?",
                    "What information is provided in the document?",
                    "What does the text describe?",
                    "What are the key points mentioned?",
                    "What is explained in the content?"
                ]
                
                for i, simple_q in enumerate(simple_questions[:max_questions]):
                    if len(questions_data) >= max_questions:
                        break
                    
                    if simple_q.lower() not in generated_questions:
                        chunk = usable_chunks[0] if usable_chunks else chunks[0]
                        context_text = chunk.get("text", "")[:300]
                        
                        questions_data.append({
                            "question": simple_q,
                            "expected_answer": f"Information from the document about {simple_q.lower().replace('?', '').replace('what ', '')}",
                            "context": context_text + "...",
                            "question_type": "simple",
                            "chunk_index": i
                        })
                        generated_questions.add(simple_q.lower())
                        logger.info(f"Generated simple Q{len(questions_data)}: {simple_q}")
            
            logger.info(f"Successfully generated {len(questions_data)} questions from document {doc_id}")
            return questions_data
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []
    
    async def _generate_template_question(self, chunk: Dict, index: int) -> Dict:
        """Generate question using simple templates"""
        context_text = chunk.get("text", "").strip()
        
        templates = [
            "What is mentioned about",
            "What does the text say about", 
            "What information is provided about",
            "What is described regarding",
            "What details are given about"
        ]
        
        template = templates[index % len(templates)]
        
        try:
            prompt = f"Complete this question based on the text: '{template} ___?'\n\nText: {context_text[:200]}\n\nComplete question:"
            
            question = await self.rag_system._generate_with_gemini(prompt)
            question = self._clean_question(question)
            
            if not question or len(question) < 10:
                question = f"{template} the main topic?"
            
            answer_prompt = f"Answer this question in 1-2 sentences based on the text:\n\nQuestion: {question}\nText: {context_text[:400]}\n\nAnswer:"
            expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
            
            if not expected_answer:
                expected_answer = "Information from the document"
            
            return {
                "question": question,
                "expected_answer": expected_answer.strip(),
                "context": context_text[:200] + "...",
                "question_type": "template",
                "chunk_index": index
            }
            
        except Exception as e:
            logger.error(f"Error in template question generation: {str(e)}")
            return None
    
    async def _generate_basic_question(self, chunk: Dict, index: int) -> Dict:
        """Generate very basic question"""
        context_text = chunk.get("text", "").strip()
        
        try:
            prompt = f"Create a simple question about this text:\n\n{context_text[:300]}\n\nQuestion:"
            
            question = await self.rag_system._generate_with_gemini(prompt)
            question = self._clean_question(question)
            
            if not question:
                question = "What is discussed in this text?"
            
            expected_answer = context_text[:100] + "..." if len(context_text) > 100 else context_text
            
            return {
                "question": question,
                "expected_answer": expected_answer,
                "context": context_text[:200] + "...",
                "question_type": "basic",
                "chunk_index": index
            }
            
        except Exception as e:
            logger.error(f"Error in basic question generation: {str(e)}")
            return None
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the question"""
        if not question:
            return ""
        
        question = question.strip()
        
        prefixes_to_remove = [
            "question:", "q:", "here's a question:", "based on the text:",
            "from the text:", "a question could be:", "one question is:",
            "the question is:", "question -", "q -", "here is a question:",
            "a good question would be:", "the question could be:",
            "question about the text:", "text question:", "complete question:",
            "completed question:", "answer:"
        ]
        
        question_lower = question.lower()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                question = question[len(prefix):].strip()
                break
        
        question = question.strip('"\'`')
        
        if question and not question.endswith('?'):
            question += '?'
        
        if question:
            question = question[0].upper() + question[1:]
        
        if len(question) < 5 or len(question) > 300:
            return ""
        
        return question
    
    def _calculate_improved_f1_score(self, predicted: str, expected: str) -> float:
        """Improved F1 score calculation"""
        expected_clean = self._normalize_answer(expected)
        predicted_clean = self._normalize_answer(predicted)
        
        if expected_clean in predicted_clean:
            return 0.85
        
        expected_words = set(expected_clean.split())
        predicted_words = set(predicted_clean.split())
        
        if not expected_words:
            return 1.0 if not predicted_words else 0.0
        
        overlap = len(expected_words & predicted_words)
        overlap_ratio = overlap / len(expected_words)
        
        if overlap_ratio >= 0.8:
            return 0.8
        elif overlap_ratio >= 0.6:
            return 0.6
        elif overlap_ratio >= 0.4:
            return 0.4
        else:
            return self._calculate_token_f1(predicted, expected)
    
    def _calculate_token_f1(self, predicted: str, expected: str) -> float:
        """Standard token-based F1 calculation"""
        predicted_tokens = self._tokenize(predicted)
        expected_tokens = self._tokenize(expected)
        
        if not expected_tokens:
            return 1.0 if not predicted_tokens else 0.0
        
        if not predicted_tokens:
            return 0.0
        
        common_tokens = Counter(predicted_tokens) & Counter(expected_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(predicted_tokens)
        recall = num_common / len(expected_tokens)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def _calculate_semantic_similarity(self, predicted: str, expected: str) -> float:
        """Enhanced semantic similarity check"""
        pred_words = set(self._tokenize(predicted.lower()))
        exp_words = set(self._tokenize(expected.lower()))
        
        if exp_words.issubset(pred_words):
            return 1.0
        
        intersection = len(pred_words & exp_words)
        union = len(pred_words | exp_words)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_contains_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer contains the expected answer"""
        pred_clean = self._normalize_answer(predicted)
        exp_clean = self._normalize_answer(expected)
        
        if exp_clean in pred_clean:
            return True
        
        exp_words = set(exp_clean.split())
        pred_words = set(pred_clean.split())
        
        if len(exp_words) > 0:
            overlap_ratio = len(exp_words & pred_words) / len(exp_words)
            return overlap_ratio >= 0.6
        
        return False

    async def evaluate_with_f1(self, questions_data: List[Dict]) -> Dict[str, Any]:
        """Enhanced evaluation with improved F1 calculation"""
        logger.info(f"Starting evaluation with {len(questions_data)} questions")
        
        results = {
            "total_questions": len(questions_data),
            "f1_scores": [],
            "semantic_scores": [],
            "contains_answer": [],
            "response_times": [],
            "predictions": [],
            "errors": [],
            "question_types": {}
        }
        
        for i, item in enumerate(questions_data):
            try:
                question = item["question"]
                expected_answer = item["expected_answer"]
                question_type = item.get("question_type", "general")
                
                if question_type not in results["question_types"]:
                    results["question_types"][question_type] = 0
                results["question_types"][question_type] += 1
                
                start_time = time.time()
                
                answer_data = await self.rag_system.generate_answer(question)
                predicted_answer = answer_data["answer"]
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                f1_score = self._calculate_improved_f1_score(predicted_answer, expected_answer)
                semantic_score = self._calculate_semantic_similarity(predicted_answer, expected_answer)
                contains_answer = self._calculate_contains_answer(predicted_answer, expected_answer)
                
                results["f1_scores"].append(f1_score)
                results["semantic_scores"].append(semantic_score)
                results["contains_answer"].append(contains_answer)
                
                results["predictions"].append({
                    "question": question,
                    "predicted": predicted_answer,
                    "expected": expected_answer,
                    "f1_score": f1_score,
                    "semantic_score": semantic_score,
                    "contains_answer": contains_answer,
                    "response_time": response_time,
                    "confidence": answer_data.get("confidence", 0),
                    "question_type": question_type,
                    "chunk_index": item.get("chunk_index", i)
                })
                
                logger.info(f"Q{i+1} ({question_type}) - F1: {f1_score:.3f}, Semantic: {semantic_score:.3f}, Contains: {contains_answer}")
                
            except Exception as e:
                logger.error(f"Error evaluating question {i}: {str(e)}")
                results["errors"].append({
                    "question_index": i,
                    "error": str(e),
                    "question": item.get("question", "Unknown")
                })
        
        results["average_f1"] = statistics.mean(results["f1_scores"]) if results["f1_scores"] else 0
        results["average_semantic"] = statistics.mean(results["semantic_scores"]) if results["semantic_scores"] else 0
        results["accuracy_rate"] = sum(results["contains_answer"]) / len(results["contains_answer"]) if results["contains_answer"] else 0
        results["average_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        results["evaluation_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Evaluation completed: F1={results['average_f1']:.3f}, Semantic={results['average_semantic']:.3f}, Accuracy={results['accuracy_rate']:.3f}")
        
        return results
    
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
            logger.info(f"Generating questions from document: {doc_id}")
            questions_data = await self.generate_questions_from_document(doc_id)
            
            if not questions_data:
                return {
                    "error": "No questions could be generated from the document",
                    "doc_id": doc_id
                }
            
            logger.info("Running F1 evaluation...")
            evaluation_results = await self.evaluate_with_f1(questions_data)
            
            evaluation_results["doc_id"] = doc_id
            evaluation_results["questions_generated"] = len(questions_data)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in simple evaluation: {str(e)}")
            return {
                "error": str(e),
                "doc_id": doc_id
            }
