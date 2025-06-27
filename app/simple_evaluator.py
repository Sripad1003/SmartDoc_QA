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
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, rag_system, document_processor):
        self.rag_system = rag_system
        self.document_processor = document_processor
        # Add randomization seed based on current time
        self.question_seed = int(time.time() * 1000) % 10000
    
    async def generate_questions_from_document(self, doc_id: str, max_questions: int = None) -> List[Dict]:
        """Generate diverse questions from uploaded document - different each time"""
        questions_data = []
        
        # Update seed for different questions each time
        self.question_seed = int(time.time() * 1000) % 10000
        random.seed(self.question_seed)
        
        try:
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return []
            
            logger.info(f"Found {len(chunks)} chunks for document {doc_id} (seed: {self.question_seed})")
            
            # Ensure minimum 5 questions, respect max_questions from API
            if max_questions is None:
                if len(chunks) >= 25:
                    max_questions = 8
                elif len(chunks) >= 15:
                    max_questions = 7
                elif len(chunks) >= 8:
                    max_questions = 6
                else:
                    max_questions = 5
            
            # Ensure minimum of 5 questions
            if max_questions < 5:
                max_questions = 5
            
            logger.info(f"Target: {max_questions} questions (minimum 5) from {len(chunks)} chunks")
            
            # Use ALL chunks, even small ones
            usable_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if len(text) >= 20:  # Very low threshold
                    usable_chunks.append(chunk)
            
            if not usable_chunks:
                logger.error("No usable chunks found - all chunks too short")
                return []
            
            # Shuffle chunks for variety each time
            random.shuffle(usable_chunks)
            logger.info(f"Using {len(usable_chunks)} usable chunks (shuffled)")
            
            # Try multiple strategies to ensure we get diverse questions
            generated_questions = set()
            
            # Strategy 1: Diverse template questions - randomized selection
            template_count = min(max_questions // 2 + 1, len(usable_chunks))
            selected_chunks = random.sample(usable_chunks, min(template_count, len(usable_chunks)))
            
            for i, chunk in enumerate(selected_chunks):
                if len(questions_data) >= max_questions:
                    break
                
                try:
                    question_data = await self._generate_diverse_template_question(chunk, i)
                    if question_data and question_data['question'].lower() not in generated_questions:
                        questions_data.append(question_data)
                        generated_questions.add(question_data['question'].lower())
                        logger.info(f"Generated diverse Q{len(questions_data)}: {question_data['question'][:50]}...")
                except Exception as e:
                    logger.error(f"Error in diverse template question {i}: {str(e)}")
                    continue
            
            # Strategy 2: Contextual questions - different approach each time
            if len(questions_data) < max_questions:
                remaining_needed = max_questions - len(questions_data)
                logger.info(f"Need {remaining_needed} more questions, trying contextual approach...")
                
                # Use different chunks for contextual questions
                remaining_chunks = [c for c in usable_chunks if c not in selected_chunks]
                if remaining_chunks:
                    contextual_chunks = random.sample(remaining_chunks, min(remaining_needed, len(remaining_chunks)))
                    
                    for i, chunk in enumerate(contextual_chunks):
                        if len(questions_data) >= max_questions:
                            break
                        
                        try:
                            question_data = await self._generate_contextual_question(chunk, i)
                            if question_data and question_data['question'].lower() not in generated_questions:
                                questions_data.append(question_data)
                                generated_questions.add(question_data['question'].lower())
                                logger.info(f"Generated contextual Q{len(questions_data)}: {question_data['question'][:50]}...")
                        except Exception as e:
                            logger.error(f"Error in contextual question {i}: {str(e)}")
                            continue
            
            # Strategy 3: Analytical questions - for variety
            if len(questions_data) < max_questions:
                remaining_needed = max_questions - len(questions_data)
                logger.info(f"Need {remaining_needed} more questions, trying analytical approach...")
                
                for i in range(remaining_needed):
                    if len(questions_data) >= max_questions:
                        break
                    
                    try:
                        # Use random chunk for analytical questions
                        chunk = random.choice(usable_chunks)
                        question_data = await self._generate_analytical_question(chunk, i)
                        if question_data and question_data['question'].lower() not in generated_questions:
                            questions_data.append(question_data)
                            generated_questions.add(question_data['question'].lower())
                            logger.info(f"Generated analytical Q{len(questions_data)}: {question_data['question'][:50]}...")
                    except Exception as e:
                        logger.error(f"Error in analytical question {i}: {str(e)}")
                        continue
            
            # Strategy 4: Guaranteed fallback with variety - ensure minimum 5 questions
            if len(questions_data) < 5:
                logger.info(f"Only have {len(questions_data)} questions, using varied fallback to reach minimum 5...")
                
                # More diverse fallback questions
                varied_questions = [
                    "What is the main topic discussed in this document?",
                    "What key information is provided?",
                    "What important details are mentioned?",
                    "What concepts are explained?",
                    "What facts or data are presented?",
                    "What processes or methods are described?",
                    "What conclusions can be drawn?",
                    "What examples are given?",
                    "What problems or solutions are discussed?",
                    "What recommendations are made?",
                    "What benefits or advantages are mentioned?",
                    "What challenges or issues are addressed?"
                ]
                
                # Shuffle for variety
                random.shuffle(varied_questions)
                
                needed = max(5 - len(questions_data), 0)
                for i, varied_q in enumerate(varied_questions[:needed]):
                    if varied_q.lower() not in generated_questions:
                        # Use different chunks for variety
                        chunk_index = i % len(usable_chunks)
                        chunk = usable_chunks[chunk_index]
                        context_text = chunk.get("text", "")[:300]
                        
                        questions_data.append({
                            "question": varied_q,
                            "expected_answer": f"Based on the document content: {varied_q.lower().replace('?', '').replace('what ', '')}",
                            "context": context_text + "...",
                            "question_type": "varied_fallback",
                            "chunk_index": chunk_index
                        })
                        generated_questions.add(varied_q.lower())
                        logger.info(f"Generated varied fallback Q{len(questions_data)}: {varied_q}")
            
            logger.info(f"Successfully generated {len(questions_data)} diverse questions from document {doc_id}")
            
            # Final check - ensure we have at least 5 questions
            if len(questions_data) < 5:
                logger.warning(f"Only generated {len(questions_data)} questions, expected minimum 5")
            
            return questions_data
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []
    
    async def _generate_diverse_template_question(self, chunk: Dict, index: int) -> Dict:
        """Generate question using diverse templates with randomization"""
        context_text = chunk.get("text", "").strip()
        
        # Expanded diverse templates
        template_categories = {
            "factual": [
                "What specific information is provided about",
                "What details are mentioned regarding",
                "What facts are stated about",
                "What data is given concerning"
            ],
            "analytical": [
                "How does the text explain",
                "What approach is described for",
                "What method is outlined for",
                "How is the concept of"
            ],
            "descriptive": [
                "What characteristics are described for",
                "What features are highlighted about",
                "What aspects are covered regarding",
                "What properties are mentioned for"
            ],
            "comparative": [
                "What differences are noted about",
                "What similarities are discussed regarding",
                "How does the text compare",
                "What contrasts are made concerning"
            ]
        }
        
        # Randomly select category and template
        category = random.choice(list(template_categories.keys()))
        template = random.choice(template_categories[category])
        
        try:
            # More sophisticated prompt with variety
            prompt = f"""Based on this text, complete the question using the template: "{template} ___?"

Text: {context_text[:250]}

Create a specific, relevant question that can be answered from the text. Complete the question:"""
            
            question = await self.rag_system._generate_with_gemini(prompt)
            question = self._clean_question(question)
            
            if not question or len(question) < 10:
                # Fallback with category-specific question
                fallback_questions = {
                    "factual": f"{template} the main subject?",
                    "analytical": f"{template} discussed in the text?",
                    "descriptive": f"{template} the topic?",
                    "comparative": f"{template} mentioned in the content?"
                }
                question = fallback_questions.get(category, f"{template} the main topic?")
            
            # Generate contextual answer
            answer_prompt = f"""Answer this question in 2-3 sentences based on the text:

Question: {question}
Text: {context_text[:400]}

Provide a clear, specific answer:"""
            
            expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
            
            if not expected_answer:
                expected_answer = f"Information from the document about {category} aspects"
            
            return {
                "question": question,
                "expected_answer": expected_answer.strip(),
                "context": context_text[:200] + "...",
                "question_type": f"diverse_{category}",
                "chunk_index": index
            }
            
        except Exception as e:
            logger.error(f"Error in diverse template question generation: {str(e)}")
            return None
    
    async def _generate_contextual_question(self, chunk: Dict, index: int) -> Dict:
        """Generate contextual questions based on content analysis"""
        context_text = chunk.get("text", "").strip()
        
        try:
            # Analyze content for better question generation
            analysis_prompt = f"""Analyze this text and create a thoughtful question that requires understanding of the content:

Text: {context_text[:300]}

Generate a question that:
1. Requires comprehension of the text
2. Has a clear answer in the content
3. Is specific and meaningful

Question:"""
            
            question = await self.rag_system._generate_with_gemini(analysis_prompt)
            question = self._clean_question(question)
            
            if not question:
                # Contextual fallback
                contextual_fallbacks = [
                    "What is the significance of the information presented?",
                    "What can be understood from this content?",
                    "What important point is being made?",
                    "What is the purpose of this information?",
                    "What insight does this text provide?"
                ]
                question = random.choice(contextual_fallbacks)
            
            # Generate comprehensive answer
            expected_answer = context_text[:150] + "..." if len(context_text) > 150 else context_text
            
            return {
                "question": question,
                "expected_answer": expected_answer,
                "context": context_text[:200] + "...",
                "question_type": "contextual",
                "chunk_index": index
            }
            
        except Exception as e:
            logger.error(f"Error in contextual question generation: {str(e)}")
            return None
    
    async def _generate_analytical_question(self, chunk: Dict, index: int) -> Dict:
        """Generate analytical questions for deeper understanding"""
        context_text = chunk.get("text", "").strip()
        
        try:
            # Analytical question types
            analytical_types = [
                "Why is this information important?",
                "What does this suggest about the topic?",
                "How does this relate to the main theme?",
                "What implications can be drawn?",
                "What is the underlying meaning?",
                "What purpose does this serve?",
                "What can be inferred from this?",
                "What is the significance of this content?"
            ]
            
            question = random.choice(analytical_types)
            
            # Generate analytical answer
            answer_prompt = f"""Provide an analytical answer to this question based on the text:

Question: {question}
Text: {context_text[:400]}

Give a thoughtful, analytical response:"""
            
            expected_answer = await self.rag_system._generate_with_gemini(answer_prompt)
            
            if not expected_answer:
                expected_answer = "This content provides important insights that contribute to understanding the overall topic and its implications."
            
            return {
                "question": question,
                "expected_answer": expected_answer.strip(),
                "context": context_text[:200] + "...",
                "question_type": "analytical",
                "chunk_index": index
            }
            
        except Exception as e:
            logger.error(f"Error in analytical question generation: {str(e)}")
            return None
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the question - more robust"""
        if not question:
            return ""
        
        question = question.strip()
        
        # Remove common prefixes - more comprehensive
        prefixes_to_remove = [
            "question:", "q:", "here's a question:", "based on the text:",
            "from the text:", "a question could be:", "one question is:",
            "the question is:", "question -", "q -", "here is a question:",
            "a good question would be:", "the question could be:",
            "question about the text:", "text question:", "complete question:",
            "completed question:", "answer:", "generate a question:",
            "create a question:", "thoughtful question:"
        ]
        
        question_lower = question.lower()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                question = question[len(prefix):].strip()
                break
        
        # Remove quotes and extra characters
        question = question.strip('"\'`')
        
        # Ensure it ends with a question mark
        if question and not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        # Basic validation - be more lenient
        if len(question) < 5 or len(question) > 300:
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
