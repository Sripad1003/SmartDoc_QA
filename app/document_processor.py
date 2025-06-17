import asyncio
import hashlib
import uuid
from typing import List, Dict, Any
from datetime import datetime
import logging
import io

# Document processing libraries
import PyPDF2
import docx
from bs4 import BeautifulSoup
import markdown
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.processed_documents = {}
        self.chunks_db = {}
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK data: {e}")
                # Continue without NLTK sentence tokenization
    
    async def process_document(self, content: bytes, filename: str, content_type: str) -> str:
        """Process document and return document ID"""
        doc_id = str(uuid.uuid4())
        
        try:
            # Extract text based on file type
            text = await self._extract_text(content, filename, content_type)
            
            # Apply intelligent chunking
            chunks = self._intelligent_chunking(text, filename)
            
            # Store document metadata
            self.processed_documents[doc_id] = {
                "filename": filename,
                "content_type": content_type,
                "processed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "text_length": len(text)
            }
            
            # Store chunks
            self.chunks_db[doc_id] = chunks
            
            logger.info(f"Processed document {filename} into {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    async def _extract_text(self, content: bytes, filename: str, content_type: str) -> str:
        """Extract text from different file formats"""
        try:
            if content_type == "application/pdf" or filename.endswith('.pdf'):
                return self._extract_pdf_text(content)
            elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or filename.endswith('.docx'):
                return self._extract_docx_text(content)
            elif content_type == "text/html" or filename.endswith('.html'):
                return self._extract_html_text(content)
            elif filename.endswith('.md'):
                return self._extract_markdown_text(content)
            else:  # Default to plain text
                return content.decode('utf-8', errors='ignore')
        
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            return content.decode('utf-8', errors='ignore')
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def _extract_html_text(self, content: bytes) -> str:
        """Extract text from HTML"""
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()
    
    def _extract_markdown_text(self, content: bytes) -> str:
        """Extract text from Markdown"""
        md_text = content.decode('utf-8', errors='ignore')
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _intelligent_chunking(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Apply intelligent chunking strategies"""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        # Semantic chunking based on paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_size = 1000  # Target chunk size
        overlap_size = 200  # Overlap between chunks
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph exceeds chunk size, create a new chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunk_id = f"{filename}_chunk_{len(chunks)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": len(chunks),
                        "char_count": len(current_chunk),
                        "word_count": len(current_chunk.split())
                    }
                })
                
                # Start new chunk with overlap
                sentences = self._safe_sentence_tokenize(current_chunk)
                overlap_text = " ".join(sentences[-2:]) if len(sentences) > 2 else current_chunk[-overlap_size:]
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunk_id = f"{filename}_chunk_{len(chunks)}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": len(chunks),
                    "char_count": len(current_chunk),
                    "word_count": len(current_chunk.split())
                }
            })
        
        return chunks
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get chunks for a document"""
        return self.chunks_db.get(doc_id, [])
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks from all documents"""
        all_chunks = []
        for chunks in self.chunks_db.values():
            all_chunks.extend(chunks)
        return all_chunks

    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get document information"""
        return self.processed_documents.get(doc_id, {})

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        if not self.processed_documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "extraction_methods": [],
                "average_chunks_per_doc": 0
            }
        
        total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents.values())
        extraction_methods = list(set(doc.get("extraction_method", "unknown") for doc in self.processed_documents.values()))
        
        return {
            "total_documents": len(self.processed_documents),
            "total_chunks": total_chunks,
            "extraction_methods": extraction_methods,
            "average_chunks_per_doc": total_chunks / len(self.processed_documents) if self.processed_documents else 0,
            "total_text_length": sum(doc.get("text_length", 0) for doc in self.processed_documents.values())
        }

    def add_manual_content(self, filename: str, content: str) -> str:
        """Add manual content directly (for sample content)"""
        doc_id = str(uuid.uuid4())
        
        try:
            # Apply intelligent chunking
            chunks = self._intelligent_chunking(content, filename)
            
            # Store document metadata
            self.processed_documents[doc_id] = {
                "filename": filename,
                "content_type": "text/plain",
                "processed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "text_length": len(content),
                "extraction_method": "manual"
            }
            
            # Store chunks
            self.chunks_db[doc_id] = chunks
            
            logger.info(f"Added manual content {filename} with {len(chunks)} chunks")
            return doc_id
        
        except Exception as e:
            logger.error(f"Error adding manual content {filename}: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;$$$$\[\]\"\'\/]', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _safe_sentence_tokenize(self, text: str) -> List[str]:
        """Safe sentence tokenization with fallback"""
        try:
            return sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
