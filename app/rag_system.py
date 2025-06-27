import asyncio,logging,re
import numpy as np
from typing import List, Dict, Any

# Gemini API
import google.generativeai as genai # type: ignore

from .config import Config

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        self.embedding_model = Config.EMBEDDING_MODEL
        self.generation_model = genai.GenerativeModel(
            Config.GENERATION_MODEL,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=Config.MAX_ANSWER_LENGTH,
                temperature=0.1,  # Lower temperature for more focused answers
                top_p=0.8,
                top_k=40
            )
        )
        
        # Vector storage
        self.chunk_embeddings = {}
        
        # Cache for performance
        self.query_cache = {}
        self.embedding_cache = {}
        
        logger.info("RAG System initialized with enhanced Gemini configuration")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini with better error handling"""
        try:
            if not Config.GEMINI_API_KEY:
                logger.warning("No Gemini API key configured, using fallback embeddings")
                return await self._fallback_embeddings(texts)
        
            embeddings = []
            batch_size = 5  # Process in smaller batches
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
            
                for text in batch:
                    # Check cache first
                    text_hash = hash(text)
                    if text_hash in self.embedding_cache:
                        embeddings.append(self.embedding_cache[text_hash])
                        continue
                
                    # Clean text before embedding
                    cleaned_text = self._clean_text_for_embedding(text)
                
                    try:
                        # Generate embedding
                        result = genai.embed_content(
                            model=self.embedding_model,
                            content=cleaned_text,
                            task_type="retrieval_document"
                        )
                        embedding = result['embedding']
                    
                        # Cache the embedding
                        self.embedding_cache[text_hash] = embedding
                        embeddings.append(embedding)
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for text, using fallback: {e}")
                        fallback_embedding = await self._fallback_embeddings([text])
                        embeddings.extend(fallback_embedding)
                
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
            
            # Longer delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.5)
        
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
    
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fallback to TF-IDF
            return await self._fallback_embeddings(texts)
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text specifically for embedding generation"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might confuse embeddings
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', ' ', text)
        
        # Limit length for embedding (Gemini has token limits)
        if len(text) > 8000:  # Conservative limit
            text = text[:8000] + "..."
        
        return text.strip()
    
    async def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Enhanced fallback embedding generation"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
            from sklearn.decomposition import TruncatedSVD# type: ignore
            
            # Use TF-IDF with better parameters
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams, uigrams
                min_df=1, #freq
                max_df=0.90
            )
            
            vectors = vectorizer.fit_transform(texts)
            
            # Reduce dimensionality to match typical embedding size
            svd = TruncatedSVD(n_components=384)
            reduced_vectors = svd.fit_transform(vectors)
            
            embeddings = []
            for i in range(reduced_vectors.shape[0]):
                embedding = reduced_vectors[i].tolist()
                embeddings.append(embedding)
            
            logger.warning("Using enhanced TF-IDF fallback embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Fallback embedding generation failed: {str(e)}")
            # Last resort: random embeddings
            return [[0.1] * 384 for _ in texts]
    
    async def index_documents(self, chunks: List[Dict[str, Any]]):
        """Index document chunks with embeddings"""
        try:
            if not chunks:
                return
            
            logger.info(f"Indexing {len(chunks)} chunks...")
            
            texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.generate_embeddings(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = chunk["chunk_id"]
                self.chunk_embeddings[chunk_id] = {
                    "embedding": embedding,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    async def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks with improved ranking"""
        try:
            if not self.chunk_embeddings:
                logger.warning("No documents indexed yet")
                return []
            
            if top_k is None:
                top_k = Config.MAX_RETRIEVAL_CHUNKS
            
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Calculate similarities with multiple metrics
            similarities = []
            query_words = set(query.lower().split())
            
            for chunk_id, chunk_data in self.chunk_embeddings.items():
                # Cosine similarity
                cosine_sim = self._cosine_similarity(query_embedding, chunk_data["embedding"])
                
                # Keyword overlap bonus
                chunk_words = set(chunk_data["text"].lower().split())
                keyword_overlap = len(query_words.intersection(chunk_words)) / len(query_words)
                
                # Combined score
                combined_score = cosine_sim * 0.8 + keyword_overlap * 0.2
                
                similarities.append({
                    "chunk_id": chunk_id,
                    "similarity": combined_score,
                    "cosine_similarity": cosine_sim,
                    "keyword_overlap": keyword_overlap,
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"]
                })
            
            # Sort by combined score and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Filter by minimum confidence threshold
            filtered_similarities = [
                s for s in similarities 
                if s["similarity"] >= Config.MIN_CONFIDENCE_THRESHOLD
            ]
            
            return filtered_similarities[:top_k]
        
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def generate_answer(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive answer using enhanced RAG"""
        try:
            # Check cache first
            cache_key = f"{question}_{len(conversation_history or [])}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # Retrieve relevant chunks
            relevant_chunks = await self.retrieve_relevant_chunks(question)
            
            if not relevant_chunks:
                return {
                    "answer": "I don't have enough information in the uploaded documents to answer this question. Please upload relevant documents first, or try rephrasing your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieved_chunks": 0,
                    "reasoning": "No relevant content found in indexed documents"
                }
            
            # Prepare enhanced context
            context = self._prepare_enhanced_context(relevant_chunks, question)
            
            # Prepare conversation context
            conversation_context = self._prepare_conversation_context(conversation_history)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(question, context, conversation_context)
            
            # Generate response using Gemini
            try:
                response = await self._generate_with_gemini(prompt)
                answer = self._post_process_answer(response)
            except Exception as e:
                logger.error(f"Gemini generation failed: {str(e)}")
                answer = self._generate_extractive_answer(question, relevant_chunks)
            
            # Calculate enhanced confidence score
            confidence = self._calculate_enhanced_confidence(relevant_chunks, question, answer)
            
            # Prepare detailed sources
            sources = self._prepare_detailed_sources(relevant_chunks)
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "retrieved_chunks": len(relevant_chunks),
                "reasoning": f"Based on {len(relevant_chunks)} relevant document sections"
            }
            
            # Cache the result
            self.query_cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try rephrasing your question or check if the documents are properly uploaded.",
                "sources": [],
                "confidence": 0.0,
                "retrieved_chunks": 0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def _prepare_enhanced_context(self, chunks: List[Dict], question: str) -> str:
        """Prepare enhanced context with better organization"""
        if not chunks:
            return ""
        
        # Sort chunks by relevance and organize
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            chunk_text = chunk["text"].strip()
            
            # Add source information and chunk content
            context_part = f"[Source {i}: {source}]\n{chunk_text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _prepare_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Prepare conversation context"""
        if not conversation_history:
            return ""
        
        context_parts = []
        recent_history = conversation_history[-3:]  # Last 3 interactions
        
        for interaction in recent_history:
            context_parts.append(f"Previous Q: {interaction['question']}")
            context_parts.append(f"Previous A: {interaction['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, question: str, context: str, conversation_context: str) -> str:
        """Create enhanced prompt for better answer generation"""
        prompt = f"""You are an expert document analysis assistant. Your task is to provide comprehensive, accurate answers based on the provided document context.

        CONVERSATION HISTORY:
        {conversation_context}

        DOCUMENT CONTEXT:
        {context}

        QUESTION: {question}

        INSTRUCTIONS:
        1. Provide a detailed, comprehensive answer based PRIMARILY on the document context
        2. If the context contains relevant information, elaborate and explain thoroughly
        3. Structure your answer with clear sections if the topic is complex
        4. Include specific details, examples, and explanations from the documents
        5. If information is incomplete, clearly state what is missing
        6. If the answer requires information not in the context, clearly state this limitation
        7. Use bullet points or numbered lists when appropriate for clarity
        8. Aim for a thorough response (200-800 words depending on complexity)
        9. Cite specific sections when referencing document content

        ANSWER:"""
        
        return prompt
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.generation_model.generate_content(prompt)
                
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the generated answer"""
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any unwanted prefixes
        prefixes_to_remove = ["Answer:", "Response:", "Based on the context:"]
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper formatting
        answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)  # Remove excessive line breaks
        
        return answer
    
    def _generate_extractive_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate extractive answer as fallback"""
        try:
            question_words = set(question.lower().split())
            
            # Find most relevant sentences
            relevant_sentences = []
            
            for chunk in chunks[:3]:  # Use top 3 chunks
                sentences = re.split(r'[.!?]+', chunk["text"])
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 20:  # Skip very short sentences
                        continue
                    
                    sentence_words = set(sentence.lower().split())
                    overlap = len(question_words.intersection(sentence_words))
                    
                    if overlap > 0:
                        relevant_sentences.append((sentence, overlap, chunk["metadata"].get("source", "Unknown")))
            
            # Sort by relevance and combine
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                answer_parts = []
                used_sources = set()
                
                for sentence, _, source in relevant_sentences[:5]:
                    if source not in used_sources:
                        answer_parts.append(f"From {source}: {sentence}")
                        used_sources.add(source)
                    else:
                        answer_parts.append(sentence)
                
                return ". ".join(answer_parts) + "."
            
            # Final fallback
            return f"Based on the available documents, I found relevant information but cannot provide a specific answer to '{question}'. Please try rephrasing your question or provide more specific documents."
        
        except Exception as e:
            logger.error(f"Error generating extractive answer: {str(e)}")
            return "I encountered an error while processing the documents. Please try again."
    
    def _calculate_enhanced_confidence(self, chunks: List[Dict], question: str, answer: str) -> float:
        """Calculate enhanced confidence score"""
        if not chunks:
            return 0.0
        
        # Base confidence from similarity scores
        similarities = [chunk["similarity"] for chunk in chunks[:3]]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Answer quality indicators
        answer_length_score = min(len(answer.split()) / 50, 1.0)  # Longer answers get higher score
        
        # Question-answer relevance (simple keyword matching)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        relevance_score = len(question_words.intersection(answer_words)) / len(question_words)
        
        # Combined confidence
        confidence = (avg_similarity * 0.5 + answer_length_score * 0.2 + relevance_score * 0.3)
        
        return round(min(confidence, 1.0), 2)
    
    def _prepare_detailed_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Prepare detailed source information"""
        sources = []
        
        for i, chunk in enumerate(chunks[:5], 1):  # Top 5 sources
            source_info = {
                "source": chunk["metadata"].get("source", "Unknown"),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
                "similarity": round(chunk["similarity"], 3),
                "cosine_similarity": round(chunk.get("cosine_similarity", 0), 3),
                "keyword_overlap": round(chunk.get("keyword_overlap", 0), 3),
                "preview": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"]
            }
            sources.append(source_info)
        
        return sources
