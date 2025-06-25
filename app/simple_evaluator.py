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
        """Generate diverse questions from uploaded document with improved logic"""
        questions_data = []
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return questions_data
            
            # Determine number of questions based on document size
            if max_questions is None:
                if len(chunks) >= 20:
                    max_questions = 10
                elif len(chunks) >= 15:
                    max_questions = 8
                elif len(chunks) >= 10:
                    max_questions = 6
                elif len(chunks) >= 5:
                    max_questions = 5
                else:
                    max_questions = max(3, len(chunks))
            
            logger.info(f"Generating {max_questions} questions from {len(chunks)} chunks")
            
            # Filter chunks with substantial content
            substantial_chunks = [
                chunk for chunk in chunks 
                if len(chunk.get("text", "").strip()) >= 100
            ]
            
            if not substantial_chunks:
                substantial_chunks = chunks  # Fallback to all chunks
            
            logger.info(f"Using {len(substantial_chunks)} substantial chunks")
            
            # Generate questions using multiple strategies
            generated_questions = set()
            
            # Strategy 1: Direct factual extraction
            questions_data.extend(await self._generate_factual_questions(substantial_chunks, max_questions // 2, generated_questions))
            
            # Strategy 2: Definition and concept questions
            questions_data.extend(await self._generate_concept_questions(substantial_chunks, max_questions // 3, generated_questions))
            
            # Strategy 3: Simple what/how/why questions
            questions_data.extend(await self._generate_simple_questions(substantial_chunks, max_questions, generated_questions))
            
            # Remove duplicates and limit to max_questions
            unique_questions = []
            seen_questions = set()
            
            for q_data in questions_data:
                question_key = q_data['question'].lower().strip()
                if question_key not in seen_questions and len(unique_questions) < max_questions:
                    unique_questions.append(q_data)
                    seen_questions.add(question_key)
            
            logger.info(f"Successfully generated {len(unique_questions)} unique questions from document {doc_id}")
            return unique_questions
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []
    
    async def _generate_factual_questions(self, chunks: List[Dict], target_count: int, existing_questions: set) -> List[Dict]:
        """Generate factual questions that can be answered with short, specific answers"""
        questions_data = []
        
        for chunk in chunks[:target_count * 2]:  # Try more chunks than needed
            if len(questions_data) >= target_count:
                break
                
            context_text = chunk.get("text", "").strip()
            if len(context_text) < 100:
                continue
            
            try:
                # Look for specific facts, names, numbers, definitions
                prompt = f"""Based on this text, create a simple factual question that has a SHORT, SPECIFIC answer (1-5 words).

Text: {context_text[:800]}

Examples of good questions:
- "What is [specific term]?"
- "Who developed [something]?"
- "When did [event] happen?"
- "Where is [place] located?"

Create ONE question that can be answered with a short, specific phrase from the text:"""
                
                question = await self.rag_system._generate_with_gemini(prompt)
                question = self._clean_question(question)
                
                if question and question.lower() not in existing_questions and len(question) > 10:
                    # Generate a SHORT expected answer
                    answer_prompt = f"""Answer this question with a SHORT, SPECIFIC answer (1-5 words) based on the text:

Question: {question}
Text: {context_text}

Give only the essential answer (no explanations):"""
                    
                    expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
                    expected_answer = expected_answer.strip()
                    
                    # Ensure answer is short
                    if len(expected_answer.split()) <= 8:  # Max 8 words
                        questions_data.append({
                            "question": question,
                            "expected_answer": expected_answer,
                            "context": context_text[:300] + "...",
                            "question_type": "factual",
                            "chunk_index": chunk.get("chunk_id", len(questions_data))
                        })
                        
                        existing_questions.add(question.lower())
                        logger.info(f"Generated factual Q: {question[:50]}... -> {expected_answer}")
                
            except Exception as e:
                logger.error(f"Error generating factual question: {str(e)}")
                continue
        
        return questions_data
    
    async def _generate_concept_questions(self, chunks: List[Dict], target_count: int, existing_questions: set) -> List[Dict]:
        """Generate concept and definition questions"""
        questions_data = []
        
        for chunk in chunks[:target_count * 2]:
            if len(questions_data) >= target_count:
                break
                
            context_text = chunk.get("text", "").strip()
            if len(context_text) < 100:
                continue
            
            try:
                # Look for key concepts and terms
                prompt = f"""Find the main concept or term in this text and create a definition question:

Text: {context_text[:800]}

Create a question like:
- "What is [main concept]?"
- "How is [term] defined?"
- "What does [term] mean?"

Focus on the MAIN concept in the text:"""
                
                question = await self.rag_system._generate_with_gemini(prompt)
                question = self._clean_question(question)
                
                if question and question.lower() not in existing_questions and len(question) > 10:
                    # Generate concise definition
                    answer_prompt = f"""Provide a concise definition (1-2 sentences) for this question:

Question: {question}
Text: {context_text}

Give a clear, brief definition:"""
                    
                    expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
                    expected_answer = expected_answer.strip()
                    
                    if len(expected_answer.split()) <= 20:  # Max 20 words for definitions
                        questions_data.append({
                            "question": question,
                            "expected_answer": expected_answer,
                            "context": context_text[:300] + "...",
                            "question_type": "definition",
                            "chunk_index": chunk.get("chunk_id", len(questions_data))
                        })
                        
                        existing_questions.add(question.lower())
                        logger.info(f"Generated concept Q: {question[:50]}... -> {expected_answer[:30]}...")
                
            except Exception as e:
                logger.error(f"Error generating concept question: {str(e)}")
                continue
        
        return questions_data
    
    async def _generate_simple_questions(self, chunks: List[Dict], max_total: int, existing_questions: set) -> List[Dict]:
        """Generate simple questions to fill remaining slots"""
        questions_data = []
        remaining_needed = max_total - len(existing_questions)
        
        if remaining_needed <= 0:
            return questions_data
        
        # Shuffle chunks for variety
        shuffled_chunks = chunks.copy()
        random.shuffle(shuffled_chunks)
        
        for chunk in shuffled_chunks:
            if len(questions_data) >= remaining_needed:
                break
                
            context_text = chunk.get("text", "").strip()
            if len(context_text) < 80:
                continue
            
            try:
                # Simple question templates
                templates = [
                    "What is mentioned about",
                    "How is described",
                    "What does the text say about",
                    "According to the text, what is",
                    "What information is provided about"
                ]
                
                template = random.choice(templates)
                
                prompt = f"""Create a simple question using this template: "{template} [something]?"

Text: {context_text[:600]}

Make it specific to the content. Question:"""
                
                question = await self.rag_system._generate_with_gemini(prompt)
                question = self._clean_question(question)
                
                if question and question.lower() not in existing_questions and len(question) > 10:
                    # Generate direct answer
                    answer_prompt = f"""Answer this question directly from the text (keep it under 15 words):

Question: {question}
Text: {context_text}

Direct answer:"""
                    
                    expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
                    expected_answer = expected_answer.strip()
                    
                    if len(expected_answer.split()) <= 15:
                        questions_data.append({
                            "question": question,
                            "expected_answer": expected_answer,
                            "context": context_text[:300] + "...",
                            "question_type": "simple",
                            "chunk_index": chunk.get("chunk_id", len(questions_data))
                        })
                        
                        existing_questions.add(question.lower())
                        logger.info(f"Generated simple Q: {question[:50]}...")
                
            except Exception as e:
                logger.error(f"Error generating simple question: {str(e)}")
                continue
        
        return questions_data
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the question"""
        question = question.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Question:", "Q:", "Here's a question:", "Based on the text:",
            "From the text:", "A question could be:", "One question is:",
            "The question is:", "Question -", "Q -", "Here is a question:",
            "A good question would be:", "The question could be:"
        ]
        
        for prefix in prefixes_to_remove:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        # Remove quotes
        question = question.strip('"\'')
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _calculate_improved_f1_score(self, predicted: str, expected: str) -> float:
        """Improved F1 score calculation that handles long answers better"""
        # First, try to extract key information from long predicted answers
        if len(predicted.split()) > 15:
            # Look for the expected answer within the predicted answer
            expected_clean = self._normalize_answer(expected)
            predicted_clean = self._normalize_answer(predicted)
            
            # If expected answer is contained in predicted, give high score
            if expected_clean in predicted_clean:
                return 0.9  # High score for containing the right answer
            
            # Try to extract the most relevant sentence
            sentences = predicted.split('.')[:3]  # First 3 sentences
            best_score = 0
            
            for sentence in sentences:
                sentence_score = self._calculate_token_f1(sentence.strip(), expected)
                best_score = max(best_score, sentence_score)
            
            return best_score
        
        # For shorter answers, use standard token F1
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
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Boost score if most important words match
        if intersection >= len(exp_words) * 0.7:  # 70% of expected words found
            jaccard = min(1.0, jaccard + 0.2)
        
        return jaccard

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
            return overlap_ratio >= 0.7  # 70% of expected words found
        
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
