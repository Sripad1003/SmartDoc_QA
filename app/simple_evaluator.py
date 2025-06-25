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
        """Generate diverse questions from uploaded document"""
        questions_data = []
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return questions_data
            
            # Determine number of questions based on document size
            if max_questions is None:
                if len(chunks) >= 15:
                    max_questions = 10
                elif len(chunks) >= 10:
                    max_questions = 8
                elif len(chunks) >= 5:
                    max_questions = 6
                else:
                    max_questions = min(5, len(chunks))
            
            logger.info(f"Generating {max_questions} questions from {len(chunks)} chunks")
            
            # Select diverse chunks - spread across the document
            selected_chunks = self._select_diverse_chunks(chunks, max_questions)
            
            # Different question types to ensure variety
            question_types = [
                "factual",      # What is X?
                "definition",   # How is X defined?
                "explanation",  # Why does X happen?
                "comparison",   # How does X differ from Y?
                "process",      # How does X work?
                "purpose",      # What is the purpose of X?
                "example",      # What are examples of X?
                "relationship"  # How are X and Y related?
            ]
            
            generated_questions = set()  # Track to avoid duplicates
            
            for i, chunk in enumerate(selected_chunks):
                context_text = chunk.get("text", "")
                if len(context_text) < 50:  # Skip very short chunks
                    continue
                
                # Try different question types for variety
                question_type = question_types[i % len(question_types)]
                
                for attempt in range(3):  # Try up to 3 times per chunk
                    try:
                        question = await self._generate_question_by_type(context_text, question_type, attempt)
                        
                        if question and question not in generated_questions and len(question) > 10:
                            # Get expected answer
                            expected_answer = await self._generate_expected_answer(question, context_text)
                            
                            if expected_answer and len(expected_answer) > 5:
                                questions_data.append({
                                    "question": question,
                                    "expected_answer": expected_answer,
                                    "context": context_text[:300] + "...",
                                    "question_type": question_type,
                                    "chunk_index": chunk.get("chunk_id", i)
                                })
                                
                                generated_questions.add(question)
                                logger.info(f"Generated Q{len(questions_data)}: {question[:60]}...")
                                break  # Success, move to next chunk
                    
                    except Exception as e:
                        logger.error(f"Error generating question {i}, attempt {attempt}: {str(e)}")
                        continue
                
                # Stop if we have enough questions
                if len(questions_data) >= max_questions:
                    break
            
            # If we don't have enough questions, try generating more from random chunks
            if len(questions_data) < max_questions:
                logger.info(f"Only generated {len(questions_data)} questions, trying to generate more...")
                additional_needed = max_questions - len(questions_data)
                
                # Shuffle chunks and try again
                random.shuffle(chunks)
                for chunk in chunks[:additional_needed * 2]:  # Try more chunks
                    if len(questions_data) >= max_questions:
                        break
                    
                    context_text = chunk.get("text", "")
                    if len(context_text) < 50:
                        continue
                    
                    try:
                        # Use a simple, general question format
                        question = await self._generate_simple_question(context_text)
                        
                        if question and question not in generated_questions and len(question) > 10:
                            expected_answer = await self._generate_expected_answer(question, context_text)
                            
                            if expected_answer and len(expected_answer) > 5:
                                questions_data.append({
                                    "question": question,
                                    "expected_answer": expected_answer,
                                    "context": context_text[:300] + "...",
                                    "question_type": "general",
                                    "chunk_index": chunk.get("chunk_id", len(questions_data))
                                })
                                
                                generated_questions.add(question)
                                logger.info(f"Generated additional Q{len(questions_data)}: {question[:60]}...")
            
            logger.info(f"Successfully generated {len(questions_data)} unique questions from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
        
        return questions_data
    
    def _select_diverse_chunks(self, chunks: List[Dict], max_questions: int) -> List[Dict]:
        """Select diverse chunks spread across the document"""
        if len(chunks) <= max_questions:
            return chunks
        
        # Select chunks evenly distributed across the document
        step = len(chunks) // max_questions
        selected = []
        
        for i in range(0, len(chunks), step):
            if len(selected) >= max_questions:
                break
            selected.append(chunks[i])
        
        # If we need more, add some random ones
        if len(selected) < max_questions:
            remaining = [c for c in chunks if c not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:max_questions - len(selected)])
        
        return selected[:max_questions]
    
    async def _generate_question_by_type(self, context_text: str, question_type: str, attempt: int) -> str:
        """Generate a question based on specific type"""
        
        # Different prompts for different question types
        prompts = {
            "factual": [
                f"Based on this text, create a factual question that asks 'What is...' or 'What are...':\n\n{context_text[:500]}\n\nQuestion:",
                f"Create a question asking for a specific fact from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"What factual question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "definition": [
                f"Create a question asking for a definition from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask 'How is [something] defined?':\n\n{context_text[:500]}\n\nQuestion:",
                f"What definition question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "explanation": [
                f"Create a 'Why' or 'How' question from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask for an explanation:\n\n{context_text[:500]}\n\nQuestion:",
                f"What explanation question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "comparison": [
                f"Create a comparison question from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask about differences or similarities:\n\n{context_text[:500]}\n\nQuestion:",
                f"What comparison can be made from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "process": [
                f"Create a question about a process or procedure from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask 'How does [something] work?':\n\n{context_text[:500]}\n\nQuestion:",
                f"What process question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "purpose": [
                f"Create a question about purpose or function from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask about the purpose of something:\n\n{context_text[:500]}\n\nQuestion:",
                f"What purpose question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "example": [
                f"Create a question asking for examples from this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask 'What are examples of...':\n\n{context_text[:500]}\n\nQuestion:",
                f"What example question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ],
            "relationship": [
                f"Create a question about relationships between concepts in this text:\n\n{context_text[:500]}\n\nQuestion:",
                f"Based on this text, ask how things are related:\n\n{context_text[:500]}\n\nQuestion:",
                f"What relationship question can be answered from this text?\n\n{context_text[:500]}\n\nQuestion:"
            ]
        }
        
        # Get the appropriate prompt
        type_prompts = prompts.get(question_type, prompts["factual"])
        prompt = type_prompts[attempt % len(type_prompts)]
        
        question = await self.rag_system._generate_with_gemini(prompt)
        question = self._clean_question(question)
        
        return question
    
    async def _generate_simple_question(self, context_text: str) -> str:
        """Generate a simple, general question"""
        prompt = f"""Based on this text, create ONE clear, specific question that can be answered directly from the content.

Text: {context_text[:600]}

Create a question that:
1. Is specific and clear
2. Can be answered from the text
3. Is different from common questions
4. Starts with What, How, Why, When, Where, or Who

Question:"""
        
        question = await self.rag_system._generate_with_gemini(prompt)
        return self._clean_question(question)
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the question"""
        question = question.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Question:", "Q:", "Here's a question:", "Based on the text:",
            "From the text:", "A question could be:", "One question is:",
            "The question is:", "Question -", "Q -"
        ]
        
        for prefix in prefixes_to_remove:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    async def _generate_expected_answer(self, question: str, context_text: str) -> str:
        """Generate expected answer for the question"""
        prompt = f"""Answer this question based ONLY on the provided text. Give a concise, direct answer.

Question: {question}

Text: {context_text}

Provide a clear, concise answer (1-3 sentences):"""
        
        answer = await self.rag_system._generate_with_gemini(prompt)
        return answer.strip()
    
    def _calculate_semantic_similarity(self, predicted: str, expected: str) -> float:
        """Simple semantic similarity check"""
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
        
        return exp_clean in pred_clean

    async def evaluate_with_f1(self, questions_data: List[Dict]) -> Dict[str, Any]:
        """Enhanced evaluation with multiple metrics"""
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
                
                # Calculate multiple metrics
                f1_score = self._calculate_f1_score(predicted_answer, expected_answer)
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
    
    def _extract_key_answer(self, full_answer: str, question: str) -> str:
        """Extract the key answer from a long response"""
        # Look for direct answers in the first few sentences
        sentences = full_answer.split('.')[:3]  # First 3 sentences
        
        # Common answer patterns
        patterns = [
            r'is (?:a )?(?:subfield of |part of |branch of )?([^.]+)',
            r'(?:answer is |it is |that is )([^.]+)',
            r'^([^.]{10,50})\.',  # First sentence if reasonable length
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            for pattern in patterns:
                import re
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Fallback: return first 50 characters
        return full_answer[:50].strip()

    def _calculate_f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score using token overlap with answer extraction"""
        # Extract key answer from long response
        if len(predicted.split()) > 20:  # If answer is long, extract key part
            predicted = self._extract_key_answer(predicted, "")
        
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
