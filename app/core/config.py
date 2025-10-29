"""
Configuration management for Graph-Vector-RAG Application
Loads settings from environment variables using python-dotenv
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env file if it exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✓ Loaded environment from: {ENV_FILE}")
else:
    print(f"⚠️  No .env file found at: {ENV_FILE}")
    print("   Using default values and environment variables")


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""
    
    # =============================================================================
    # API KEYS & AUTHENTICATION
    # =============================================================================
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key for embeddings and Gemini"
    )

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    APP_NAME: str = Field(default="Graph-Vector-RAG", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    APP_ROOT_PATH: str = Field(default="/rag", description="API root path")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    
    # =============================================================================
    # PDF PROCESSING SETTINGS
    # =============================================================================
    PDF_DATA_DIR: str = Field(default="data", description="Directory containing PDF files")
    PDF_CHUNK_SIZE: int = Field(default=1000, description="Size of text chunks in characters")
    PDF_CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks in characters")
    PDF_CLEAN_TEXT: bool = Field(default=True, description="Enable text cleaning")
    PDF_ADD_START_INDEX: bool = Field(default=True, description="Add start index to chunk metadata")
    
    # =============================================================================
    # EMBEDDINGS SETTINGS
    # =============================================================================
    EMBEDDING_MODEL: str = Field(
        default="models/embedding-001",
        description="Google embedding model name"
    )
    EMBEDDING_DIMENSION: int = Field(default=768, description="Dimension of embedding vectors")
    EMBEDDING_BATCH_SIZE: int = Field(default=100, description="Batch size for embedding generation")
    
    # =============================================================================
    # VECTOR STORE SETTINGS
    # =============================================================================
    VECTOR_STORE_PATH: str = Field(default="vectorstore_db", description="Path to save/load vector store")
    VECTOR_STORE_TYPE: str = Field(default="FAISS", description="Vector store type")
    VECTOR_SEARCH_K: int = Field(default=4, description="Number of top results to retrieve")
    VECTOR_SEARCH_TYPE: str = Field(default="similarity", description="Search type")
    
    # =============================================================================
    # LLM SETTINGS
    # =============================================================================
    LLM_MODEL: str = Field(default="gemini-pro", description="Gemini model for chat/generation")
    LLM_TEMPERATURE: float = Field(default=0.3, description="Temperature for LLM responses")
    LLM_MAX_TOKENS: int = Field(default=2048, description="Maximum tokens in LLM response")
    LLM_TOP_P: float = Field(default=0.95, description="Top-p sampling parameter")
    
    # =============================================================================
    # RAG SETTINGS
    # =============================================================================
    RAG_CHAIN_TYPE: str = Field(default="stuff", description="Chain type for RAG")
    RAG_RETURN_SOURCES: bool = Field(default=True, description="Return source documents with answer")
    RAG_VERBOSE: bool = Field(default=False, description="Verbose mode for debugging")
    
    # =============================================================================
    # KNOWLEDGE GRAPH SETTINGS
    # =============================================================================
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="", description="Neo4j password")
    GRAPH_DB_TYPE: str = Field(default="neo4j", description="Graph database type")
    ENABLE_KNOWLEDGE_GRAPH: bool = Field(default=False, description="Enable knowledge graph extraction")
    
    # =============================================================================
    # LOGGING SETTINGS
    # =============================================================================
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_TO_CONSOLE: bool = Field(default=True, description="Enable console logging")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Create global settings instance
settings = get_settings()
