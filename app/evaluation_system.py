import asyncio
import json
import time
import statistics
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)

class EvaluationSystem:
    def __init__(self, rag_system, document_processor):
        self.rag_system = rag_system
        self.document_processor = document_processor
        self.evaluation_results = {}
        self.performance_metrics = {
            "response_times": [],
            "memory_usage": [],
            "accuracy_scores": []
        }
    
    async def generate_sample_data_from_document(self, doc_id: str, max_questions: int = 5) -> List[Dict]:
        """
        Generate sample questions and answers from the uploaded document context.
        This method extracts chunks from the document, generates questions using Gemini,
        and obtains answers using the RAG system.
        """
        sample_data = []
        try:
            # Get document chunks
            chunks = self.document_processor.get_document_chunks(doc_id)
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return sample_data
            
            # Limit chunks to max_questions for question generation
            selected_chunks = chunks[:max_questions]
            
            for chunk in selected_chunks:
                context_text = chunk.get("text", "")
                # Generate question from context using Gemini via rag_system
                prompt = f"Generate a question based on the following context:\n{context_text}\nQuestion:"
                question = await self.rag_system._generate_with_gemini(prompt)
                question = question.strip().rstrip("?") + "?"
                
                # Get answer from RAG system for the generated question
                answer_data = await self.rag_system.generate_answer(question)
                answer = answer_data.get("answer", "")
                
                sample_data.append({
                    "question": question,
                    "answers": [answer],
                    "context": context_text
                })
            
            logger.info(f"Generated {len(sample_data)} sample questions from document {doc_id}")
        except Exception as e:
            logger.error(f"Error generating sample data from document {doc_id}: {str(e)}")
        
        return sample_data
    
    async def evaluate_squad_format(self, squad_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate system using SQUAD format data"""
        logger.info(f"Starting SQUAD evaluation with {len(squad_data)} questions")
        
        results = {
            "total_questions": len(squad_data),
            "exact_matches": 0,
            "f1_scores": [],
            "response_times": [],
            "predictions": [],
            "errors": []
        }
        
        for i, item in enumerate(squad_data):
            try:
                question = item["question"]
                expected_answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
                
                # Measure response time
                start_time = time.time()
                
                # Get prediction from our system
                answer_data = await self.rag_system.generate_answer(question)
                predicted_answer = answer_data["answer"]
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                # Calculate metrics
                exact_match = self._calculate_exact_match(predicted_answer, expected_answers)
                f1_score = self._calculate_f1_score(predicted_answer, expected_answers)
                
                if exact_match:
                    results["exact_matches"] += 1
                
                results["f1_scores"].append(f1_score)
                results["predictions"].append({
                    "question": question,
                    "predicted": predicted_answer,
                    "expected": expected_answers,
                    "exact_match": exact_match,
                    "f1_score": f1_score,
                    "confidence": answer_data.get("confidence", 0),
                    "response_time": response_time
                })
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(squad_data)} questions")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                results["errors"].append({
                    "question_index": i,
                    "error": str(e),
                    "question": item.get("question", "Unknown")
                })
        
        # Calculate final metrics
        results["exact_match_score"] = results["exact_matches"] / results["total_questions"] if results["total_questions"] > 0 else 0
        results["average_f1"] = statistics.mean(results["f1_scores"]) if results["f1_scores"] else 0
        results["average_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        results["evaluation_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"SQUAD evaluation completed: EM={results['exact_match_score']:.3f}, F1={results['average_f1']:.3f}")
        return results
    
    def _calculate_exact_match(self, predicted: str, expected_answers: List[str]) -> bool:
        """Calculate exact match score"""
        predicted_normalized = self._normalize_answer(predicted)
        
        for expected in expected_answers:
            expected_normalized = self._normalize_answer(expected)
            if predicted_normalized == expected_normalized:
                return True
        return False
    
    def _calculate_f1_score(self, predicted: str, expected_answers: List[str]) -> float:
        """Calculate F1 score using token overlap"""
        predicted_tokens = self._tokenize(predicted)
        
        f1_scores = []
        for expected in expected_answers:
            expected_tokens = self._tokenize(expected)
            
            if not expected_tokens:
                f1_scores.append(1.0 if not predicted_tokens else 0.0)
                continue
            
            common_tokens = Counter(predicted_tokens) & Counter(expected_tokens)
            num_common = sum(common_tokens.values())
            
            if num_common == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_common / len(predicted_tokens) if predicted_tokens else 0
            recall = num_common / len(expected_tokens)
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = ''.join(char for char in answer if char not in string.punctuation)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        return answer.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for F1 calculation"""
        normalized = self._normalize_answer(text)
        return normalized.split()
    
    async def benchmark_performance(self, test_questions: List[str], iterations: int = 2) -> Dict[str, Any]:
        """Benchmark system performance"""
        logger.info(f"Starting performance benchmark with {len(test_questions)} questions, {iterations} iterations")
        
        results = {
            "response_times": [],
            "memory_usage": [],
            "error_rate": 0,
            "errors": []
        }
        
        total_requests = len(test_questions) * iterations
        successful_requests = 0
        
        for iteration in range(iterations):
            for i, question in enumerate(test_questions):
                try:
                    start_time = time.time()
                    
                    # Get memory usage if available
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    except ImportError:
                        memory_before = 0
                    
                    # Make request
                    answer_data = await self.rag_system.generate_answer(question)
                    
                    # Calculate metrics
                    response_time = time.time() - start_time
                    
                    try:
                        import psutil
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_used = memory_after - memory_before
                    except ImportError:
                        memory_used = 0
                    
                    results["response_times"].append(response_time)
                    results["memory_usage"].append(memory_used)
                    successful_requests += 1
                    
                except Exception as e:
                    logger.error(f"Benchmark error on question {i}, iteration {iteration}: {str(e)}")
                    results["errors"].append({
                        "question": question,
                        "iteration": iteration,
                        "error": str(e)
                    })
        
        # Calculate final metrics
        results["error_rate"] = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
        results["average_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        
        # Calculate P95 safely
        if len(results["response_times"]) >= 20:
            sorted_times = sorted(results["response_times"])
            p95_index = int(0.95 * len(sorted_times))
            results["p95_response_time"] = sorted_times[p95_index]
        else:
            results["p95_response_time"] = max(results["response_times"]) if results["response_times"] else 0
        
        results["average_memory_usage"] = statistics.mean(results["memory_usage"]) if results["memory_usage"] else 0
        results["throughput"] = successful_requests / sum(results["response_times"]) if results["response_times"] and sum(results["response_times"]) > 0 else 0
        results["benchmark_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Performance benchmark completed: Avg response time={results['average_response_time']:.3f}s, Error rate={results['error_rate']:.3f}")
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
# Q&A System Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
"""
        
        if "squad_results" in evaluation_results:
            squad = evaluation_results["squad_results"]
            report += f"""
### SQUAD 2.0 Evaluation
- **Questions Evaluated**: {squad.get('total_questions', 0)}
- **Exact Match Score**: {squad.get('exact_match_score', 0):.3f}
- **F1 Score**: {squad.get('average_f1', 0):.3f}
- **Average Response Time**: {squad.get('average_response_time', 0):.3f}s
"""
        
        if "performance_results" in evaluation_results:
            perf = evaluation_results["performance_results"]
            report += f"""
### Performance Benchmarks
- **Average Response Time**: {perf.get('average_response_time', 0):.3f}s
- **P95 Response Time**: {perf.get('p95_response_time', 0):.3f}s
- **Throughput**: {perf.get('throughput', 0):.2f} requests/second
- **Error Rate**: {perf.get('error_rate', 0):.3f}
- **Average Memory Usage**: {perf.get('average_memory_usage', 0):.2f} MB
"""
        
        report += """
## Recommendations

### Performance Optimization
- Monitor response times and optimize if > 5 seconds
- Consider caching for frequently asked questions
- Implement request queuing for high load scenarios

### Accuracy Improvement
- Fine-tune retrieval parameters based on F1 scores
- Improve document chunking strategy
- Consider domain-specific training data

### Production Readiness
- Implement comprehensive error handling
- Add monitoring and alerting
- Set up automated testing pipeline
"""
        
        return report

    async def load_squad_sample(self) -> List[Dict]:
        """Load a sample of SQUAD-format questions for testing"""
        sample_data = [
            {
                "question": "What are the main phases of the AI/ML exercise?",
                "answers": ["Phase 1: Document Processing Pipeline, Phase 2: RAG System Implementation, Phase 3: System Integration, Phase 4: Evaluation and Testing"],
                "context": "The exercise consists of multiple phases including document processing, RAG implementation, system integration, and evaluation."
            },
            {
                "question": "What is the recommended technology stack for backend development?",
                "answers": ["FastAPI for high-performance API development, Google Gemini Pro for text generation, Google Gemini Embedding Model for embeddings"],
                "context": "The recommended backend technologies include FastAPI, Google Gemini Pro, and various document processing libraries."
            },
            {
                "question": "What are the evaluation criteria for the system?",
                "answers": ["Technical Excellence (40%), Functionality (35%), Innovation (25%)"],
                "context": "The system is evaluated based on technical excellence, functionality, and innovation with specific weightings."
            },
            {
                "question": "What document formats should be supported?",
                "answers": ["PDF, DOCX, TXT, HTML, and Markdown files"],
                "context": "The system should support multiple document formats including PDF, Word documents, text files, HTML, and Markdown."
            },
            {
                "question": "What are the performance targets for the system?",
                "answers": ["Document processing: < 30 seconds per document, Query response time: < 5 seconds, Answer accuracy: > 80% user satisfaction"],
                "context": "Performance targets include fast document processing, quick query responses, and high accuracy rates."
            }
        ]
        
        return sample_data
    
    async def run_comprehensive_evaluation(self, doc_id: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation suite"""
        logger.info("Starting comprehensive evaluation suite")
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "system_info": {
                "documents_indexed": len(self.rag_system.chunk_embeddings),
                "cache_size": len(self.rag_system.query_cache)
            }
        }
        
        try:
            # Load sample data dynamically from document if doc_id provided
            if doc_id:
                squad_sample = await self.generate_sample_data_from_document(doc_id)
            else:
                squad_sample = await self.load_squad_sample()
            
            # Run SQUAD evaluation
            logger.info("Running SQUAD evaluation...")
            results["squad_results"] = await self.evaluate_squad_format(squad_sample)
            
            # Run performance benchmark
            logger.info("Running performance benchmark...")
            test_questions = [item["question"] for item in squad_sample]
            results["performance_results"] = await self.benchmark_performance(test_questions)
            
            logger.info("Comprehensive evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {str(e)}")
            results["error"] = str(e)
        
        return results
