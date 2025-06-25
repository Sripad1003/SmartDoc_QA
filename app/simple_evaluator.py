import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import logging
import re
import string
from collections import Counter
import random

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, rag_system, document_processor):
        self.rag_system = rag_system
        self.document_processor = document_processor
    
    async def generate_questions_from_document(self, doc_id: str, max_questions: int = None) -> List[Dict]:
        """Generate questions from uploaded document - simplified and reliable"""
        questions_data = []
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return questions_data
            
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
            
            logger.info(f"Generating {max_questions} questions from {len(chunks)} chunks")
            
            # Use all chunks that have reasonable content
            usable_chunks = [
                chunk for chunk in chunks 
                if len(chunk.get("text", "").strip()) >= 50  # Lower threshold
            ]
            
            if not usable_chunks:
                logger.warning("No usable chunks found, using all chunks")
                usable_chunks = chunks
            
            logger.info(f"Using {len(usable_chunks)} usable chunks")
            
            # Simple approach: Generate one question per selected chunk
            selected_chunks = self._select_chunks_evenly(usable_chunks, max_questions)
            
            generated_questions = set()
            
            for i, chunk in enumerate(selected_chunks):
                if len(questions_data) >= max_questions:
                    break
                
                context_text = chunk.get("text", "").strip()
                if len(context_text) < 30:  # Very minimal threshold
                    continue
                
                try:
                    # Simple, reliable question generation
                    question = await self._generate_simple_reliable_question(context_text, i)
                    
                    if question and question.lower() not in generated_questions:
                        # Generate expected answer
                        expected_answer = await self._generate_simple_answer(question, context_text)
                        
                        if expected_answer and len(expected_answer.strip()) > 3:
                            questions_data.append({
                                "question": question,
                                "expected_answer": expected_answer,
                                "context": context_text[:200] + "...",
                                "question_type": "general",
                                "chunk_index": i
                            })
                            
                            generated_questions.add(question.lower())
                            logger.info(f"Generated Q{len(questions_data)}: {question[:60]}...")
                
                except Exception as e:
                    logger.error(f"Error generating question {i}: {str(e)}")
                    continue
            
            # If we don't have enough questions, try a backup approach
            if len(questions_data) < 3:  # Minimum 3 questions
                logger.info("Not enough questions generated, trying backup approach...")
                backup_questions = await self._generate_backup_questions(usable_chunks, max_questions - len(questions_data), generated_questions)
                questions_data.extend(backup_questions)
            
            logger.info(f"Successfully generated {len(questions_data)} questions from document {doc_id}")
            return questions_data
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []
    
    def _select_chunks_evenly(self, chunks: List[Dict], count: int) -> List[Dict]:
        """Select chunks evenly distributed across the document"""
        if len(chunks) <= count:
            return chunks
        
        # Select chunks at regular intervals
        step = len(chunks) / count
        selected = []
        
        for i in range(count):
            index = int(i * step)
            if index < len(chunks):
                selected.append(chunks[index])
        
        return selected
    
    async def _generate_simple_reliable_question(self, context_text: str, attempt: int) -> str:
        """Generate a simple, reliable question"""
        
        # Use different approaches based on attempt number
        approaches = [
            f"Based on this text, create a simple question that starts with 'What':\n\n{context_text[:400]}\n\nQuestion:",
            f"Create a question asking about the main topic in this text:\n\n{context_text[:400]}\n\nQuestion:",
            f"What question can be answered from this text? Make it simple:\n\n{context_text[:400]}\n\nQuestion:",
            f"Create a 'How' or 'Why' question from this text:\n\n{context_text[:400]}\n\nQuestion:",
            f"Ask about something specific mentioned in this text:\n\n{context_text[:400]}\n\nQuestion:"
        ]
        
        prompt = approaches[attempt % len(approaches)]
        
        try:
            question = await self.rag_system._generate_with_gemini(prompt)
            return self._clean_question(question)
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return ""
    
    async def _generate_simple_answer(self, question: str, context_text: str) -> str:
        """Generate a simple, direct answer"""
        prompt = f"""Answer this question based on the text. Keep the answer concise (1-3 sentences):

Question: {question}

Text: {context_text}

Answer:"""
        
        try:
            answer = await self.rag_system._generate_with_gemini(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return ""
    
    async def _generate_backup_questions(self, chunks: List[Dict], needed: int, existing_questions: set) -> List[Dict]:
        """Backup question generation method"""
        backup_questions = []
        
        # Simple templates that usually work
        templates = [
            "What is mentioned in the text?",
            "What does the text describe?",
            "What information is provided?",
            "What is the main topic?",
            "What is explained in the text?"
        ]
        
        for i, chunk in enumerate(chunks[:needed * 2]):  # Try more chunks
            if len(backup_questions) >= needed:
                break
            
            context_text = chunk.get("text", "").strip()
            if len(context_text) < 30:
                continue
            
            try:
                # Use a simple template
                template = templates[i % len(templates)]
                question = f"{template}"
                
                if question.lower() not in existing_questions:
                    # Generate simple answer
                    answer_prompt = f"Based on this text, answer: {question}\n\nText: {context_text[:300]}\n\nAnswer:"
                    expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
                    
                    if expected_answer and len(expected_answer.strip()) > 3:
                        backup_questions.append({
                            "question": question,
                            "expected_answer": expected_answer.strip(),
                            "context": context_text[:200] + "...",
                            "question_type": "backup",
                            "chunk_index": i
                        })
                        
                        existing_questions.add(question.lower())
                        logger.info(f"Generated backup Q: {question}")
            
            except Exception as e:
                logger.error(f"Error in backup question generation: {str(e)}")
                continue
        
        return backup_questions
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the question"""
        if not question:
            return ""
        
        question = question.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Question:", "Q:", "Here's a question:", "Based on the text:",
            "From the text:", "A question could be:", "One question is:",
            "The question is:", "Question -", "Q -", "Here is a question:",
            "A good question would be:", "The question could be:",
            "Question about the text:", "Text question:"
        ]
        
        for prefix in prefixes_to_remove:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        # Remove quotes
        question = question.strip('"\'')
        
        # Ensure it ends with a question mark
        if question and not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        # Basic validation
        if len(question) < 10 or len(question) > 200:
            return ""
        
        return question
    
    def _calculate_improved_f1_score(self, predicted: str, expected: str) -> float:
        """Improved F1 score calculation"""
        # If expected answer is found in predicted answer, give high score
        expected_clean = self._normalize_answer(expected)
        predicted_clean = self._normalize_answer(predicted)
        
        # Direct containment check
        if expected_clean in predicted_clean:
            return 0.85  # High score for containing the answer
        
        # Check word overlap
        expected_words = set(expected_clean.split())
        predicted_words = set(predicted_clean.split())
        
        if not expected_words:
            return 1.0 if not predicted_words else 0.0
        
        # Calculate overlap ratio
        overlap = len(expected_words & predicted_words)
        overlap_ratio = overlap / len(expected_words)
        
        if overlap_ratio >= 0.8:  # 80% of expected words found
            return 0.8
        elif overlap_ratio >= 0.6:  # 60% of expected words found
            return 0.6
        elif overlap_ratio >= 0.4:  # 40% of expected words found
            return 0.4
        else:
            # Fall back to standard F1
            return self._calculate_token_f1(predicted, expected)
    
    def _calculate_token_f1(self, predicted: str, expected: str) -> float:
        """Standard token-based F1 calculation"""
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
    
    def _calculate_semantic_similarity(self, predicted: str, expected: str) -> float:
        """Enhanced semantic similarity check"""
        pred_words = set(self._tokenize(predicted.lower()))
        exp_words = set(self._tokenize(expected.lower()))
        
        # Check if all expected words are in predicted
        if exp_words.issubset(pred_words):
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(pred_words & exp_words)
        union = len(pred_words | exp_words)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_contains_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer contains the expected answer"""
        pred_clean = self._normalize_answer(predicted)
        exp_clean = self._normalize_answer(expected)
        
        # Direct containment
        if exp_clean in pred_clean:
            return True
        
        # Check if most words from expected are in predicted
        exp_words = set(exp_clean.split())
        pred_words = set(pred_clean.split())
        
        if len(exp_words) > 0:
            overlap_ratio = len(exp_words & pred_words) / len(exp_words)
            return overlap_ratio >= 0.6  # 60% of expected words found
        
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
                
                # Track question types
                if question_type not in results["question_types"]:
                    results["question_types"][question_type] = 0
                results["question_types"][question_type] += 1
                
                # Measure response time
                start_time = time.time()
                
                # Get prediction from our system
                answer_data = await self.rag_system.generate_answer(question)
                predicted_answer = answer_data["answer"]
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                # Calculate multiple metrics with improved F1
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
        
        # Calculate final metrics
        results["average_f1"] = statistics.mean(results["f1_scores"]) if results["f1_scores"] else 0
        results["average_semantic"] = statistics.mean(results["semantic_scores"]) if results["semantic_scores"] else 0
        results["accuracy_rate"] = sum(results["contains_answer"]) / len(results["contains_answer"]) if results["contains_answer"] else 0
        results["average_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        results["evaluation_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Evaluation completed: F1={results['average_f1']:.3f}, Semantic={results['average_semantic']:.3f}, Accuracy={results['accuracy_rate']:.3f}")
        logger.info(f"Question types generated: {results['question_types']}")
        
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
            # Generate questions from document
            logger.info(f"Generating questions from document: {doc_id}")
            questions_data = await self.generate_questions_from_document(doc_id)
            
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
