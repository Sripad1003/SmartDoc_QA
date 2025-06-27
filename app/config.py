import os

class Config:
    """Configuration settings for the Q&A system"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")    
    # Gemini API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyDv5zVdk7kaPnHrcSUiyZi1lkZggVgm7ZA")
    
    # Model Configuration
    EMBEDDING_MODEL = "models/embedding-001"
    GENERATION_MODEL = "gemini-1.5-flash"  # Updated to better model
    
    # System Configuration - Increased for better answers
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1500"))  # Increased
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))     # Increased
    MAX_RETRIEVAL_CHUNKS = int(os.getenv("MAX_RETRIEVAL_CHUNKS", "8"))  # Increased
    RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "60"))  # Increased
    
    # Answer Generation Settings
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "2000"))  # New setting
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.4"))
    
    # Cache Configuration
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Security
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10")) * 1024 * 1024  # 50MB
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            print("⚠️  Warning: GEMINI_API_KEY environment variable not set")
            print("   Please set your Gemini API key in the .env file")
            return False
        return True