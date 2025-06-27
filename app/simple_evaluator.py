import random
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self):
        """Initialize the evaluator with sentence transformer model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SimpleEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SimpleEvaluator: {e}")
            raise

    def generate_questions_from_document(self, document_text: str, max_questions: int = 8, seed: int = None) -> Dict[str, Any]:
        """
        Generate diverse questions from document text with guaranteed variety
        """
        try:
            # Set random seed for reproducibility if provided, otherwise use timestamp
            if seed is None:
                seed = int(time.time() * 1000) % 10000
            random.seed(seed)
            
            # Ensure minimum 5 questions
            if max_questions < 5:
                max_questions = 5
            
            # Split document into chunks
            chunks = self._split_into_chunks(document_text)
            if not chunks:
                return self._generate_fallback_questions(max_questions)
            
            # Shuffle chunks for variety
            random.shuffle(chunks)
            
            # Generate questions using multiple strategies
            questions_data = []
            
            # Strategy 1: Template-based questions (primary)
            template_questions = self._generate_template_questions(chunks, max_questions // 2)
            questions_data.extend(template_questions)
            
            # Strategy 2: Content-analysis questions (secondary)
            if len(questions_data) < max_questions:
                content_questions = self._generate_content_questions(chunks, max_questions - len(questions_data))
                questions_data.extend(content_questions)
            
            # Strategy 3: Analytical questions (tertiary)
            if len(questions_data) < max_questions:
                analytical_questions = self._generate_analytical_questions(chunks, max_questions - len(questions_data))
                questions_data.extend(analytical_questions)
            
            # Strategy 4: Fallback questions (guarantee minimum)
            if len(questions_data) < 5:
                fallback_needed = 5 - len(questions_data)
                fallback_questions = self._generate_varied_fallback_questions(fallback_needed)
                questions_data.extend(fallback_questions)
            
            # Shuffle final questions for variety
            random.shuffle(questions_data)
            
            # Limit to max_questions
            questions_data = questions_data[:max_questions]
            
            # Count question types
            question_types = {}
            for q_data in questions_data:
                q_type = q_data.get('question_type', 'general')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            logger.info(f"Generated {len(questions_data)} questions with seed {seed}")
            
            return {
                "questions": [q['question'] for q in questions_data],
                "questions_data": questions_data,
                "questions_generated": len(questions_data),
                "question_types": question_types,
                "seed_used": seed
            }
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._generate_fallback_questions(max_questions)

    def _split_into_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only include substantial chunks
                chunks.append(chunk.strip())
        
        return chunks

    def _generate_template_questions(self, chunks: List[str], target_count: int) -> List[Dict[str, Any]]:
        """Generate questions using diverse templates"""
        templates = {
            'factual': [
                "What specific information does the text provide about {}?",
                "What details are mentioned regarding {}?",
                "What facts are presented about {}?",
                "What specific aspects of {} are discussed?"
            ],
            'analytical': [
                "How does the text explain the concept of {}?",
                "What approach does the document take regarding {}?",
                "How is {} analyzed or described in the text?",
                "What methodology is used to discuss {}?"
            ],
            'descriptive': [
                "What characteristics of {} are highlighted?",
                "What features or properties of {} are described?",
                "What attributes of {} does the text emphasize?",
                "What qualities of {} are mentioned?"
            ],
            'comparative': [
                "What differences regarding {} are noted in the text?",
                "What similarities or contrasts about {} are discussed?",
                "How does {} compare to other concepts mentioned?",
                "What relationships involving {} are described?"
            ]
        }
        
        questions = []
        usable_chunks = [chunk for chunk in chunks if len(chunk) > 100]
        
        if not usable_chunks:
            return questions
        
        # Randomly select chunks and templates
        for i in range(min(target_count, len(usable_chunks))):
            chunk = usable_chunks[i]
            
            # Extract key terms from chunk
            key_terms = self._extract_key_terms(chunk)
            if not key_terms:
                continue
            
            # Randomly select question type and template
            q_type = random.choice(list(templates.keys()))
            template = random.choice(templates[q_type])
            key_term = random.choice(key_terms)
            
            question = template.format(key_term)
            
            questions.append({
                'question': question,
                'question_type': q_type,
                'source_chunk': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'expected_answer': chunk[:300] + "..." if len(chunk) > 300 else chunk
            })
        
        return questions

    def _generate_content_questions(self, chunks: List[str], target_count: int) -> List[Dict[str, Any]]:
        """Generate questions based on content analysis"""
        questions = []
        
        for i, chunk in enumerate(chunks[:target_count]):
            if len(chunk) < 100:
                continue
            
            # Generate contextual questions
            if "process" in chunk.lower() or "method" in chunk.lower():
                question = f"What process or method is described in the document?"
                q_type = "process"
            elif "result" in chunk.lower() or "outcome" in chunk.lower():
                question = f"What results or outcomes are mentioned?"
                q_type = "result"
            elif "problem" in chunk.lower() or "issue" in chunk.lower():
                question = f"What problems or issues are identified?"
                q_type = "problem"
            else:
                question = f"What main points are covered in this section of the document?"
                q_type = "summary"
            
            questions.append({
                'question': question,
                'question_type': q_type,
                'source_chunk': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'expected_answer': chunk[:300] + "..." if len(chunk) > 300 else chunk
            })
        
        return questions

    def _generate_analytical_questions(self, chunks: List[str], target_count: int) -> List[Dict[str, Any]]:
        """Generate analytical questions for deeper understanding"""
        analytical_templates = [
            "What is the significance of the information presented?",
            "How do the concepts discussed relate to each other?",
            "What implications can be drawn from the content?",
            "What patterns or trends are evident in the text?",
            "What conclusions can be reached based on the information?",
            "What evidence supports the main arguments presented?"
        ]
        
        questions = []
        
        for i in range(min(target_count, len(chunks))):
            if i < len(analytical_templates):
                template = analytical_templates[i]
            else:
                template = random.choice(analytical_templates)
            
            chunk = chunks[i] if i < len(chunks) else random.choice(chunks)
            
            questions.append({
                'question': template,
                'question_type': 'analytical',
                'source_chunk': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'expected_answer': chunk[:300] + "..." if len(chunk) > 300 else chunk
            })
        
        return questions

    def _generate_varied_fallback_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate varied fallback questions when content analysis fails"""
        fallback_questions = [
            {"question": "What is the main topic discussed in the document?", "type": "topic"},
            {"question": "What key information is presented in the text?", "type": "information"},
            {"question": "What important details are mentioned?", "type": "details"},
            {"question": "What concepts are explained in the document?", "type": "concepts"},
            {"question": "What specific points are covered?", "type": "points"},
            {"question": "What subject matter does the text address?", "type": "subject"},
            {"question": "What themes are explored in the content?", "type": "themes"},
            {"question": "What aspects are highlighted in the document?", "type": "aspects"},
            {"question": "What elements are discussed in the text?", "type": "elements"},
            {"question": "What components are described in the document?", "type": "components"},
            {"question": "What features are mentioned in the content?", "type": "features"},
            {"question": "What characteristics are outlined in the text?", "type": "characteristics"}
        ]
        
        # Randomly select and shuffle
        selected = random.sample(fallback_questions, min(count, len(fallback_questions)))
        
        questions = []
        for item in selected:
            questions.append({
                'question': item["question"],
                'question_type': item["type"],
                'source_chunk': "General document content",
                'expected_answer': "Based on the overall document content"
            })
        
        return questions

    def _generate_fallback_questions(self, max_questions: int) -> Dict[str, Any]:
        """Generate fallback questions when document processing fails"""
        fallback_data = self._generate_varied_fallback_questions(max_questions)
        
        question_types = {}
        for q_data in fallback_data:
            q_type = q_data.get('question_type', 'general')
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return {
            "questions": [q['question'] for q in fallback_data],
            "questions_data": fallback_data,
            "questions_generated": len(fallback_data),
            "question_types": question_types,
            "seed_used": int(time.time() * 1000) % 10000
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for question generation"""
        # Simple keyword extraction
        words = text.split()
        
        # Filter for meaningful terms (longer than 3 characters, not common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'a', 'an'}
        
        key_terms = []
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}').lower()
            if len(clean_word) > 3 and clean_word not in common_words and clean_word.isalpha():
                key_terms.append(clean_word)
        
        # Return unique terms, limited to reasonable number
        unique_terms = list(set(key_terms))
        return unique_terms[:10]  # Limit to 10 key terms

    def calculate_f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score between predicted and expected answers"""
        try:
            if not predicted or not expected:
                return 0.0
            
            # Tokenize and normalize
            pred_tokens = set(predicted.lower().split())
            exp_tokens = set(expected.lower().split())
            
            if not exp_tokens:
                return 0.0
            
            # Calculate precision and recall
            common_tokens = pred_tokens.intersection(exp_tokens)
            
            if not common_tokens:
                return 0.0
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(exp_tokens) if exp_tokens else 0
            
            # Calculate F1 score
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        except Exception as e:
            logger.error(f"Error calculating F1 score: {e}")
            return 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Get embeddings
            embeddings = self.model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def contains_answer(self, predicted: str, expected: str, threshold: float = 0.3) -> bool:
        """Check if predicted answer contains relevant information from expected answer"""
        try:
            if not predicted or not expected:
                return False
            
            # Use semantic similarity as a measure
            similarity = self.calculate_semantic_similarity(predicted, expected)
            return similarity >= threshold
            
        except Exception as e:
            logger.error(f"Error checking answer containment: {e}")
            return False

    def evaluate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a list of predictions and return comprehensive results"""
        try:
            if not predictions:
                return {
                    "average_f1": 0.0,
                    "average_semantic": 0.0,
                    "accuracy_rate": 0.0,
                    "total_questions": 0,
                    "predictions": [],
                    "f1_scores": [],
                    "semantic_scores": []
                }
            
            f1_scores = []
            semantic_scores = []
            contains_answers = []
            
            # Process each prediction
            for pred in predictions:
                predicted = pred.get('predicted', '')
                expected = pred.get('expected', '')
                
                # Calculate metrics
                f1 = self.calculate_f1_score(predicted, expected)
                semantic = self.calculate_semantic_similarity(predicted, expected)
                contains = self.contains_answer(predicted, expected)
                
                f1_scores.append(f1)
                semantic_scores.append(semantic)
                contains_answers.append(contains)
                
                # Update prediction with scores
                pred.update({
                    'f1_score': f1,
                    'semantic_score': semantic,
                    'contains_answer': contains
                })
            
            # Calculate averages
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0
            accuracy_rate = np.mean(contains_answers) if contains_answers else 0.0
            
            # Count question types
            question_types = {}
            for pred in predictions:
                q_type = pred.get('question_type', 'general')
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            # Calculate response time statistics
            response_times = [pred.get('response_time', 0) for pred in predictions]
            avg_response_time = np.mean(response_times) if response_times else 0.0
            
            return {
                "average_f1": float(avg_f1),
                "average_semantic": float(avg_semantic),
                "accuracy_rate": float(accuracy_rate),
                "total_questions": len(predictions),
                "average_response_time": float(avg_response_time),
                "predictions": predictions,
                "f1_scores": f1_scores,
                "semantic_scores": semantic_scores,
                "question_types": question_types
            }
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {
                "average_f1": 0.0,
                "average_semantic": 0.0,
                "accuracy_rate": 0.0,
                "total_questions": 0,
                "predictions": [],
                "f1_scores": [],
                "semantic_scores": [],
                "error": str(e)
            }
