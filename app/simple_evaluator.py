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
import google.generativeai as genai
from .config import Config

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, rag_system, document_processor):
        """Initialize the evaluator with RAG system and document processor"""
        self.rag_system = rag_system
        self.document_processor = document_processor
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure Gemini for question generation
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.question_generator = genai.GenerativeModel(
            Config.GENERATION_MODEL,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,  # Higher temperature for more creative questions
                top_p=0.9,
                top_k=40
            )
        )
        
        # Fallback questions for when Gemini fails
        self.fallback_questions = [
            "What is the main topic discussed in this document?",
            "What are the key points mentioned in the text?",
            "What important information is provided in this document?",
            "What are the primary themes covered?",
            "What significant details are highlighted?",
            "What core concepts are explained?",
            "What essential information is shared?",
            "What fundamental ideas are presented?"
        ]
        
        # Add randomization seed based on current time
        self.question_seed = int(time.time() * 1000) % 10000
    
        logger.info("SimpleEvaluator initialized with Gemini-powered question generation")

    async def generate_questions_from_document(self, doc_id: str, max_questions: int = None) -> List[Dict]:
        """Generate diverse questions using Gemini AI - different each time"""
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
            
            # Determine target number of questions
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
            
            logger.info(f"Target: {max_questions} questions from {len(chunks)} chunks using Gemini AI")
            
            # Filter usable chunks
            usable_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if len(text) >= 30:  # Minimum content for meaningful questions
                    usable_chunks.append(chunk)
            
            if not usable_chunks:
                logger.error("No usable chunks found - all chunks too short")
                return await self._generate_fallback_questions(max_questions)
            
            # Shuffle chunks for variety
            random.shuffle(usable_chunks)
            logger.info(f"Using {len(usable_chunks)} usable chunks for Gemini generation")
            
            # Strategy 1: Generate questions using Gemini AI
            generated_questions = set()
            
            # Select diverse chunks for question generation
            selected_chunks = self._select_diverse_chunks(usable_chunks, max_questions)
            
            for i, chunk in enumerate(selected_chunks):
                if len(questions_data) >= max_questions:
                    break
                
                try:
                    # Generate questions using Gemini
                    chunk_questions = await self._generate_gemini_questions(chunk, i)
                    
                    for question_data in chunk_questions:
                        if len(questions_data) >= max_questions:
                            break
                        
                        question_lower = question_data['question'].lower()
                        if question_lower not in generated_questions:
                            questions_data.append(question_data)
                            generated_questions.add(question_lower)
                            logger.info(f"Generated Gemini Q{len(questions_data)}: {question_data['question'][:60]}...")
                
                except Exception as e:
                    logger.error(f"Error generating Gemini questions for chunk {i}: {str(e)}")
                    # Try fallback for this chunk
                    try:
                        fallback_q = await self._generate_fallback_for_chunk(chunk, i)
                        if fallback_q and fallback_q['question'].lower() not in generated_questions:
                            questions_data.append(fallback_q)
                            generated_questions.add(fallback_q['question'].lower())
                            logger.info(f"Generated fallback Q{len(questions_data)}: {fallback_q['question'][:60]}...")
                    except Exception as fe:
                        logger.error(f"Fallback also failed for chunk {i}: {str(fe)}")
                        continue
            
            # Strategy 2: Ensure minimum questions with intelligent fallbacks
            if len(questions_data) < 5:
                logger.info(f"Only have {len(questions_data)} questions, generating more with fallbacks...")
                
                remaining_needed = max(5 - len(questions_data), 0)
                fallback_questions = await self._generate_intelligent_fallbacks(
                    usable_chunks, remaining_needed, generated_questions
                )
                questions_data.extend(fallback_questions)
            
            logger.info(f"Successfully generated {len(questions_data)} questions using Gemini AI")
            
            return questions_data
            
        except Exception as e:
            logger.error(f"Error in Gemini question generation: {str(e)}")
            # Complete fallback to template-based questions
            return await self._generate_fallback_questions(max_questions or 5)
    
    def _select_diverse_chunks(self, chunks: List[Dict], target_count: int) -> List[Dict]:
        """Select diverse chunks for question generation"""
        if len(chunks) <= target_count:
            return chunks
        
        # Sort chunks by length and select diverse ones
        chunks_with_length = [(chunk, len(chunk.get("text", ""))) for chunk in chunks]
        chunks_with_length.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        step = max(1, len(chunks_with_length) // target_count)
        
        for i in range(0, len(chunks_with_length), step):
            if len(selected) >= target_count:
                break
            selected.append(chunks_with_length[i][0])
        
        # Fill remaining slots randomly
        remaining_chunks = [c for c, _ in chunks_with_length if c not in selected]
        while len(selected) < target_count and remaining_chunks:
            selected.append(remaining_chunks.pop(random.randint(0, len(remaining_chunks) - 1)))
        
        return selected[:target_count]
    
    async def _generate_gemini_questions(self, chunk: Dict, chunk_index: int) -> List[Dict]:
        """Generate questions for a chunk using Gemini AI"""
        context_text = chunk.get("text", "").strip()
        
        if len(context_text) < 30:
            return []
        
        # Create a comprehensive prompt for question generation
        prompt = f"""You are an expert question generator. Based on the following text content, generate 2-3 diverse, thoughtful questions that would test someone's understanding of the material.

TEXT CONTENT:
{context_text}

REQUIREMENTS:
1. Generate 2-3 questions of different types (factual, analytical, conceptual)
2. Questions should be clear, specific, and answerable from the text
3. Vary the question styles (What, How, Why, etc.)
4. Make questions that require understanding, not just memorization
5. Each question should be on a separate line
6. Do not include question numbers or bullets
7. Each question must end with a question mark

EXAMPLE FORMAT:
What specific approach does the text describe for handling this situation?
How does the author explain the relationship between these concepts?
Why is this particular method considered effective according to the text?

GENERATE QUESTIONS:"""

        try:
            # Generate questions using Gemini
            response = await self._generate_with_gemini(prompt)
            
            if not response:
                return []
            
            # Parse the response to extract questions
            questions = self._parse_gemini_questions(response, context_text, chunk_index)
            
            logger.info(f"Gemini generated {len(questions)} questions for chunk {chunk_index}")
            return questions
            
        except Exception as e:
            logger.error(f"Error in Gemini question generation: {str(e)}")
            return []
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.question_generator.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def _parse_gemini_questions(self, response: str, context_text: str, chunk_index: int) -> List[Dict]:
        """Parse Gemini response to extract clean questions"""
        questions_data = []
        
        # Split response into lines and clean
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and non-questions
            if not line or not line.endswith('?'):
                continue
            
            # Clean the question
            question = self._clean_gemini_question(line)
            
            if question and len(question) >= 10:
                # Determine question type based on starting word
                question_type = self._classify_question_type(question)
                
                # Generate expected answer from context
                expected_answer = self._generate_expected_answer(question, context_text)
                
                questions_data.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                    "question_type": f"gemini_{question_type}",
                    "chunk_index": chunk_index
                })
        
        return questions_data
    
    def _clean_gemini_question(self, question: str) -> str:
        """Clean and format Gemini-generated questions"""
        if not question:
            return ""
        
        # Remove common prefixes that Gemini might add
        prefixes_to_remove = [
            "question:", "q:", "here's a question:", "based on the text:",
            "from the text:", "a question could be:", "one question is:",
            "the question is:", "question -", "q -", "here is a question:",
            "a good question would be:", "the question could be:",
            "1.", "2.", "3.", "4.", "5.", "•", "-", "*",
            "question about the text:", "text question:"
        ]
        
        question = question.strip()
        question_lower = question.lower()
        
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix.lower()):
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
        
        # Validate length
        if len(question) < 10 or len(question) > 300:
            return ""
        
        return question
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on starting words"""
        question_lower = question.lower()
        
        if question_lower.startswith(('what', 'which', 'who', 'when', 'where')):
            return 'factual'
        elif question_lower.startswith(('how', 'in what way')):
            return 'analytical'
        elif question_lower.startswith(('why', 'what is the reason')):
            return 'causal'
        elif question_lower.startswith(('describe', 'explain', 'what are the characteristics')):
            return 'descriptive'
        elif question_lower.startswith(('compare', 'contrast', 'what is the difference')):
            return 'comparative'
        else:
            return 'general'
    
    def _generate_expected_answer(self, question: str, context_text: str) -> str:
        """Generate expected answer based on question and context"""
        # For now, use a portion of the context as expected answer
        # In the future, this could also use Gemini to generate more precise answers
        
        if len(context_text) <= 200:
            return context_text
        else:
            # Try to find the most relevant part of the context
            question_words = set(question.lower().split())
            sentences = re.split(r'[.!?]+', context_text)
            
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence
            
            if best_sentence:
                return best_sentence + "..."
            else:
                return context_text[:200] + "..."
    
    async def _generate_fallback_for_chunk(self, chunk: Dict, chunk_index: int) -> Dict:
        """Generate a simple fallback question for a chunk"""
        context_text = chunk.get("text", "").strip()
        
        fallback_templates = [
            "What information is provided in this section?",
            "What key points are mentioned here?",
            "What details are discussed in this part?",
            "What is the main focus of this content?",
            "What important aspects are covered?"
        ]
        
        question = random.choice(fallback_templates)
        expected_answer = context_text[:150] + "..." if len(context_text) > 150 else context_text
        
        return {
            "question": question,
            "expected_answer": expected_answer,
            "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
            "question_type": "fallback",
            "chunk_index": chunk_index
        }
    
    async def _generate_intelligent_fallbacks(self, chunks: List[Dict], needed: int, existing_questions: set) -> List[Dict]:
        """Generate intelligent fallback questions"""
        fallback_questions = []
        
        # More sophisticated fallback questions
        intelligent_fallbacks = [
            "What is the main topic discussed in this document?",
            "What key information is provided?",
            "What important details are mentioned?",
            "What concepts are explained?",
            "What facts or data are presented?",
            "What processes or methods are described?",
            "What conclusions can be drawn?",
            "What examples are given?",
            "What problems or solutions are discussed?",
            "What recommendations are made?"
        ]
        
        random.shuffle(intelligent_fallbacks)
        
        for i, fallback_q in enumerate(intelligent_fallbacks[:needed]):
            if fallback_q.lower() not in existing_questions:
                # Use different chunks for variety
                chunk_index = i % len(chunks)
                chunk = chunks[chunk_index]
                context_text = chunk.get("text", "")[:300]
                
                fallback_questions.append({
                    "question": fallback_q,
                    "expected_answer": f"Based on the document content: {context_text}...",
                    "context": context_text + "...",
                    "question_type": "intelligent_fallback",
                    "chunk_index": chunk_index
                })
        
        return fallback_questions
    
    async def _generate_fallback_questions(self, max_questions: int) -> List[Dict]:
        """Generate complete fallback questions when everything else fails"""
        selected_questions = random.sample(
            self.fallback_questions, 
            min(max_questions, len(self.fallback_questions))
        )
        
        questions_data = []
        for i, question in enumerate(selected_questions):
            questions_data.append({
                'question': question,
                'expected_answer': 'Based on document content',
                'context': 'General document content',
                'question_type': 'complete_fallback',
                'chunk_index': i
            })
        
        return questions_data
    
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
