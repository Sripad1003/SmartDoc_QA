import random
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, rag_system, document_processor):
        """Initialize the evaluator with RAG system and document processor"""
        self.rag_system = rag_system
        self.document_processor = document_processor
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enhanced question templates with more variety
        self.question_templates = {
            'factual': [
                "What specific information does the text provide about {}?",
                "What details are mentioned regarding {}?",
                "What facts are stated about {}?",
                "What key information is given about {}?",
                "What specific details does the document mention about {}?",
                "What concrete information is provided about {}?",
                "What particular facts are highlighted about {}?",
                "What explicit details are shared about {}?"
            ],
            'analytical': [
                "How does the text explain the concept of {}?",
                "What approach does the document take regarding {}?",
                "How is {} described or characterized in the text?",
                "What perspective does the text offer on {}?",
                "How does the document analyze {}?",
                "What interpretation is given for {}?",
                "How does the text break down the topic of {}?",
                "What analytical framework is used for {}?"
            ],
            'descriptive': [
                "What characteristics of {} are described?",
                "What features of {} does the text highlight?",
                "What attributes of {} are mentioned?",
                "What properties of {} are discussed?",
                "What aspects of {} are covered in the text?",
                "What qualities of {} are emphasized?",
                "What elements of {} are detailed?",
                "What components of {} are described?"
            ],
            'comparative': [
                "What differences regarding {} are mentioned?",
                "What similarities about {} are discussed?",
                "How does {} compare to other concepts in the text?",
                "What contrasts are drawn about {}?",
                "What comparisons are made regarding {}?",
                "How is {} differentiated from other topics?",
                "What parallels are drawn with {}?",
                "What distinctions are made about {}?"
            ]
        }
        
        # Fallback questions with more variety
        self.fallback_questions = [
            "What is the main topic discussed in this document?",
            "What are the key points mentioned in the text?",
            "What important information is provided in this document?",
            "What are the primary themes covered?",
            "What significant details are highlighted?",
            "What core concepts are explained?",
            "What essential information is shared?",
            "What fundamental ideas are presented?",
            "What crucial points are emphasized?",
            "What vital information is contained in the text?",
            "What central themes are explored?",
            "What principal ideas are discussed?"
        ]
        
        logger.info("SimpleEvaluator initialized with enhanced question generation")

    def generate_questions_from_document(self, doc_id: str, max_questions: int = 8, seed: int = None) -> Dict[str, Any]:
        """Generate diverse questions from a document with guaranteed variety"""
        try:
            # Set random seed for reproducibility while ensuring variety
            if seed is None:
                seed = int(time.time() * 1000) % 10000
            random.seed(seed)
            
            # Ensure minimum questions
            if max_questions < 5:
                max_questions = 5
            
            # Get document chunks
            doc_data = self.document_processor.get_document_data(doc_id)
            if not doc_data or not doc_data.get('chunks'):
                logger.warning(f"No chunks found for document {doc_id}")
                return self._generate_fallback_questions(max_questions)
            
            chunks = doc_data['chunks']
            usable_chunks = [chunk for chunk in chunks if len(chunk.get('content', '').strip()) > 50]
            
            if not usable_chunks:
                logger.warning(f"No usable chunks found for document {doc_id}")
                return self._generate_fallback_questions(max_questions)
            
            # Shuffle chunks for variety
            random.shuffle(usable_chunks)
            
            questions_data = []
            question_types = {'factual': 0, 'analytical': 0, 'descriptive': 0, 'comparative': 0}
            
            # Strategy 1: Template-based questions with diverse types
            template_questions = self._generate_template_questions(usable_chunks, max_questions // 2)
            questions_data.extend(template_questions)
            
            # Strategy 2: Content-analysis questions
            content_questions = self._generate_content_questions(usable_chunks, max_questions // 3)
            questions_data.extend(content_questions)
            
            # Strategy 3: Analytical questions
            analytical_questions = self._generate_analytical_questions(usable_chunks, max_questions // 4)
            questions_data.extend(analytical_questions)
            
            # Remove duplicates while preserving order
            seen_questions = set()
            unique_questions = []
            for q_data in questions_data:
                question_lower = q_data['question'].lower().strip()
                if question_lower not in seen_questions:
                    seen_questions.add(question_lower)
                    unique_questions.append(q_data)
                    question_types[q_data['question_type']] += 1
            
            # Ensure minimum questions with fallbacks
            while len(unique_questions) < max_questions:
                fallback_q = random.choice(self.fallback_questions)
                if fallback_q.lower() not in seen_questions:
                    unique_questions.append({
                        'question': fallback_q,
                        'question_type': 'general',
                        'expected_answer': 'Based on document content',
                        'source_chunk': 'general'
                    })
                    question_types['general'] = question_types.get('general', 0) + 1
                    seen_questions.add(fallback_q.lower())
            
            # Shuffle final questions for variety
            random.shuffle(unique_questions)
            
            # Limit to requested number
            final_questions = unique_questions[:max_questions]
            
            # Update question type counts
            final_question_types = {}
            for q_data in final_questions:
                q_type = q_data['question_type']
                final_question_types[q_type] = final_question_types.get(q_type, 0) + 1
            
            result = {
                'questions': [q['question'] for q in final_questions],
                'questions_data': final_questions,
                'questions_generated': len(final_questions),
                'question_types': final_question_types,
                'seed_used': seed
            }
            
            logger.info(f"Generated {len(final_questions)} questions for document {doc_id} with seed {seed}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating questions for document {doc_id}: {str(e)}")
            return self._generate_fallback_questions(max_questions)

    def _generate_template_questions(self, chunks: List[Dict], target_count: int) -> List[Dict]:
        """Generate questions using diverse templates"""
        questions = []
        question_types = list(self.question_templates.keys())
        
        for i in range(target_count):
            if i >= len(chunks):
                break
                
            chunk = chunks[i]
            content = chunk.get('content', '')
            
            # Extract key terms for question generation
            key_terms = self._extract_key_terms(content)
            if not key_terms:
                continue
            
            # Rotate through question types for variety
            q_type = question_types[i % len(question_types)]
            templates = self.question_templates[q_type]
            template = random.choice(templates)
            key_term = random.choice(key_terms)
            
            question = template.format(key_term)
            
            questions.append({
                'question': question,
                'question_type': q_type,
                'expected_answer': content[:200] + "...",
                'source_chunk': chunk.get('chunk_id', f'chunk_{i}')
            })
        
        return questions

    def _generate_content_questions(self, chunks: List[Dict], target_count: int) -> List[Dict]:
        """Generate questions based on content analysis"""
        questions = []
        content_types = ['process', 'result', 'problem', 'summary']
        
        for i in range(min(target_count, len(chunks))):
            chunk = chunks[i]
            content = chunk.get('content', '')
            
            if len(content.strip()) < 30:
                continue
            
            content_type = content_types[i % len(content_types)]
            
            if 'process' in content_type:
                question = f"What process or method is described in this section?"
            elif 'result' in content_type:
                question = f"What outcomes or results are mentioned?"
            elif 'problem' in content_type:
                question = f"What challenges or issues are identified?"
            else:
                question = f"What is the main point of this section?"
            
            questions.append({
                'question': question,
                'question_type': content_type,
                'expected_answer': content[:200] + "...",
                'source_chunk': chunk.get('chunk_id', f'chunk_{i}')
            })
        
        return questions

    def _generate_analytical_questions(self, chunks: List[Dict], target_count: int) -> List[Dict]:
        """Generate analytical questions"""
        questions = []
        analytical_types = ['significance', 'implications', 'patterns', 'relationships']
        
        for i in range(min(target_count, len(chunks))):
            chunk = chunks[i]
            content = chunk.get('content', '')
            
            if len(content.strip()) < 50:
                continue
            
            anal_type = analytical_types[i % len(analytical_types)]
            
            if 'significance' in anal_type:
                question = "What is the significance of the information presented?"
            elif 'implications' in anal_type:
                question = "What are the implications of the points discussed?"
            elif 'patterns' in anal_type:
                question = "What patterns or trends can be identified?"
            else:
                question = "What relationships are established in the text?"
            
            questions.append({
                'question': question,
                'question_type': anal_type,
                'expected_answer': content[:200] + "...",
                'source_chunk': chunk.get('chunk_id', f'chunk_{i}')
            })
        
        return questions

    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content for question generation"""
        # Simple key term extraction
        words = content.split()
        
        # Filter for meaningful terms (longer than 3 characters, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
        
        key_terms = []
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}').lower()
            if len(clean_word) > 3 and clean_word not in common_words and clean_word.isalpha():
                key_terms.append(clean_word)
        
        # Return unique terms, limited to reasonable number
        unique_terms = list(set(key_terms))[:10]
        return unique_terms if unique_terms else ['topic', 'subject', 'content']

    def _generate_fallback_questions(self, max_questions: int) -> Dict[str, Any]:
        """Generate fallback questions when document processing fails"""
        selected_questions = random.sample(
            self.fallback_questions, 
            min(max_questions, len(self.fallback_questions))
        )
        
        questions_data = []
        for i, question in enumerate(selected_questions):
            questions_data.append({
                'question': question,
                'question_type': 'general',
                'expected_answer': 'Based on document content',
                'source_chunk': f'fallback_{i}'
            })
        
        return {
            'questions': selected_questions,
            'questions_data': questions_data,
            'questions_generated': len(selected_questions),
            'question_types': {'general': len(selected_questions)},
            'seed_used': int(time.time() * 1000) % 10000
        }

    def calculate_f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score between predicted and expected answers"""
        try:
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
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        except Exception as e:
            logger.error(f"Error calculating F1 score: {str(e)}")
            return 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            # Get embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def evaluate_document(self, doc_id: str) -> Dict[str, Any]:
        """Evaluate the RAG system performance on a document"""
        try:
            logger.info(f"Starting evaluation for document {doc_id}")
            
            # Generate questions
            questions_result = self.generate_questions_from_document(doc_id)
            questions_data = questions_result.get('questions_data', [])
            
            if not questions_data:
                logger.error(f"No questions generated for document {doc_id}")
                return {"error": "No questions generated"}
            
            predictions = []
            f1_scores = []
            semantic_scores = []
            response_times = []
            
            # Evaluate each question
            for q_data in questions_data:
                question = q_data['question']
                expected = q_data['expected_answer']
                
                # Get prediction from RAG system
                start_time = time.time()
                try:
                    result = self.rag_system.ask_question(question)
                    predicted = result.get('answer', '')
                    confidence = result.get('confidence', 0.0)
                except Exception as e:
                    logger.error(f"Error getting answer for question '{question}': {str(e)}")
                    predicted = "Error generating answer"
                    confidence = 0.0
                
                response_time = time.time() - start_time
                
                # Calculate metrics
                f1_score = self.calculate_f1_score(predicted, expected)
                semantic_score = self.calculate_semantic_similarity(predicted, expected)
                contains_answer = len(predicted.strip()) > 0 and "error" not in predicted.lower()
                
                prediction_data = {
                    'question': question,
                    'question_type': q_data.get('question_type', 'general'),
                    'expected': expected,
                    'predicted': predicted,
                    'f1_score': f1_score,
                    'semantic_score': semantic_score,
                    'contains_answer': contains_answer,
                    'response_time': response_time,
                    'confidence': confidence
                }
                
                predictions.append(prediction_data)
                f1_scores.append(f1_score)
                semantic_scores.append(semantic_score)
                response_times.append(response_time)
            
            # Calculate aggregate metrics
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0
            avg_response_time = np.mean(response_times) if response_times else 0.0
            accuracy_rate = sum(1 for p in predictions if p['contains_answer']) / len(predictions) if predictions else 0.0
            
            results = {
                'predictions': predictions,
                'average_f1': float(avg_f1),
                'average_semantic': float(avg_semantic),
                'average_response_time': float(avg_response_time),
                'accuracy_rate': float(accuracy_rate),
                'total_questions': len(predictions),
                'f1_scores': f1_scores,
                'semantic_scores': semantic_scores,
                'question_types': questions_result.get('question_types', {}),
                'evaluation_timestamp': time.time()
            }
            
            logger.info(f"Evaluation completed for document {doc_id}. Average F1: {avg_f1:.3f}, Average Semantic: {avg_semantic:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating document {doc_id}: {str(e)}")
            return {"error": str(e)}
