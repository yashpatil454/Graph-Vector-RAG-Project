"""
Embeddings Manager Service for Knowledge Graph and Vector RAG Project
Handles embedding generation, vector storage, and retrieval operations
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pickle

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Service for managing embeddings and vector storage.
    Handles embedding generation, FAISS vector store operations, and retrieval.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        embedding_dimension: Optional[int] = None
    ):
        """
        Initialize Embeddings Manager.
        Uses config values from settings if not provided.

        Args:
            embedding_model: Google embedding model name (default from config)
            api_key: Google API key (default from config/env)
            vector_store_path: Path to save/load vector store (default from config)
            embedding_dimension: Dimension of embedding vectors (default from config)
        """
        # Use config values as defaults
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.vector_store_path = Path(settings.VECTOR_STORE_PATH)
        self.embedding_dimension = settings.EMBEDDING_DIMENSION

        # Initialize embeddings model
        self.embeddings_model = self._initialize_embeddings_model()
        
        # Vector store (initialized when created/loaded)
        self.vectorstore: Optional[FAISS] = None
        
        # Chunk mapping for direct lookup (chunk_id -> Document)
        self.chunk_mapping: Dict[str, Document] = {}
        
        logger.info(
            f"EmbeddingsManager initialized with model: {self.embedding_model_name}"
        )

    def _initialize_embeddings_model(self) -> Embeddings:
        """Initialize the embeddings model."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        logger.info(f"Embeddings model '{self.embedding_model_name}' initialized")
        return embedding_model

    def create_embeddings(
        self,
        chunks: List[Document],
        save: bool = True
    ) -> FAISS:
        """
        Create embeddings for document chunks and store in FAISS vector store.

        Args:
            chunks: List of document chunks to embed
            save: Whether to save vector store to disk
            batch_size: Number of chunks to process at once

        Returns:
            FAISS vector store
        """
        try:
            logger.info(f"Creating embeddings for {len(chunks)} chunks...")
            
            # Create FAISS vector store from documents
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings_model
            )
            
            # # Build chunk mapping for direct lookup
            # self._build_chunk_mapping(chunks)
            
            logger.info(
                f"Created vector store with {self.vectorstore.index.ntotal} vectors"
            )
            
            # Save if requested
            if save:
                self.save_vector_store()
            
            return self.vectorstore

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def add_embeddings(
        self,
        new_chunks: List[Document],
        save: bool = True
    ) -> FAISS:
        """
        Add new embeddings to existing vector store.

        Args:
            new_chunks: New document chunks to add
            save: Whether to save updated vector store

        Returns:
            Updated FAISS vector store
        """
        try:
            if self.vectorstore is None:
                logger.warning("No existing vector store. Creating new one.")
                return self.create_embeddings(new_chunks, save)
            
            logger.info(f"Adding {len(new_chunks)} new chunks to vector store...")
            
            # Add documents to existing store
            self.vectorstore.add_documents(new_chunks)
            
            # Update chunk mapping
            self._build_chunk_mapping(new_chunks)
            
            logger.info(
                f"✓ Added chunks. Total vectors: {self.vectorstore.index.ntotal}"
            )
            
            if save:
                self.save_vector_store()
            
            return self.vectorstore

        except Exception as e:
            logger.error(f"Error adding embeddings: {str(e)}")
            raise

    def save_vector_store(self, path: Optional[str] = None) -> None:
        """
        Save vector store and chunk mapping to disk.

        Args:
            path: Optional custom path (uses default if not provided)
        """
        try:
            save_path = Path(path) if path else self.vector_store_path
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vectorstore.save_local(str(save_path))
            
            # Save chunk mapping separately
            mapping_path = save_path / "chunk_mapping.pkl"
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.chunk_mapping, f)
            
            logger.info(f"✓ Vector store saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self, path: Optional[str] = None) -> FAISS:
        """
        Load vector store and chunk mapping from disk.

        Args:
            path: Optional custom path (uses default if not provided)

        Returns:
            Loaded FAISS vector store
        """
        try:
            load_path = Path(path) if path else self.vector_store_path
            
            if not load_path.exists():
                raise FileNotFoundError(f"Vector store not found at {load_path}")
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(load_path),
                self.embeddings_model,
                allow_dangerous_deserialization=True
            )
            
            # Load chunk mapping
            mapping_path = load_path / "chunk_mapping.pkl"
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    self.chunk_mapping = pickle.load(f)
            else:
                logger.warning("Chunk mapping file not found. Direct lookup unavailable.")
                self.chunk_mapping = {}
            
            logger.info(
                f"✓ Vector store loaded from {load_path} "
                f"({self.vectorstore.index.ntotal} vectors)"
            )
            
            return self.vectorstore

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    # def_build_chunk_mapping(self, chunks: List[Document]) -> None:
    #     """Build mapping from chunk_id to Document for direct lookup."""
    #     for chunk in chunks:
    #         chunk_id = chunk.metadata.get('chunk_id')
    #         if chunk_id:
    #             self.chunk_mapping[chunk_id] = chunk

    def search_by_similarity(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant document chunks
        """
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Create or load first.")
            
            if score_threshold:
                # Search with score threshold
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                filtered_results = [
                    doc for doc, score in results 
                    if score >= score_threshold
                ]
                return filtered_results
            else:
                # Regular similarity search
                return self.vectorstore.similarity_search(query, k=k)

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def search_by_similarity_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized.")
            
            return self.vectorstore.similarity_search_with_score(query, k=k)

        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise

    def search_by_chunk_id(self, chunk_id: str) -> Optional[Document]:
        """
        Retrieve chunk directly by its ID (no similarity search).

        Args:
            chunk_id: Chunk identifier (e.g., 'chunk_00005')

        Returns:
            Document chunk or None if not found
        """
        chunk = self.chunk_mapping.get(chunk_id)
        
        if chunk:
            logger.info(f"Found chunk: {chunk_id}")
        else:
            logger.warning(f"Chunk not found: {chunk_id}")
        
        return chunk

    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search chunks by metadata criteria.

        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            k: Maximum number of results (None = all matches)

        Returns:
            List of matching document chunks
        """
        matching_chunks = []
        
        for chunk_id, chunk in self.chunk_mapping.items():
            # Check if all filter criteria match
            match = all(
                chunk.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            
            if match:
                matching_chunks.append(chunk)
                if k and len(matching_chunks) >= k:
                    break
        
        logger.info(
            f"Found {len(matching_chunks)} chunks matching metadata filter"
        )
        
        return matching_chunks

    def search_by_source_file(
        self,
        source_file: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Get all chunks from a specific source file.

        Args:
            source_file: Source file name or path
            k: Maximum number of results

        Returns:
            List of chunks from the source file
        """
        return self.search_by_metadata({"source": source_file}, k)

    def search_by_page(
        self,
        source_file: str,
        page_number: int
    ) -> List[Document]:
        """
        Get all chunks from a specific page.

        Args:
            source_file: Source file name or path
            page_number: Page number (0-indexed)

        Returns:
            List of chunks from the specified page
        """
        return self.search_by_metadata({
            "source": source_file,
            "page": page_number
        })

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Get a LangChain retriever for use in chains.

        Args:
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters

        Returns:
            LangChain retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        search_kwargs = search_kwargs or {"k": 4}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        # Get unique sources
        sources = set(
            chunk.metadata.get('source', 'unknown')
            for chunk in self.chunk_mapping.values()
        )
        
        # Get page count per source
        source_pages = {}
        for chunk in self.chunk_mapping.values():
            source = chunk.metadata.get('source', 'unknown')
            page = chunk.metadata.get('page', 0)
            if source not in source_pages:
                source_pages[source] = set()
            source_pages[source].add(page)
        
        return {
            "status": "initialized",
            "total_vectors": self.vectorstore.index.ntotal,
            "total_chunks": len(self.chunk_mapping),
            "total_sources": len(sources),
            "sources": list(sources),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "vector_store_path": str(self.vector_store_path),
            "source_page_counts": {
                source: len(pages) 
                for source, pages in source_pages.items()
            }
        }

    def delete_vector_store(self, path: Optional[str] = None) -> None:
        """
        Delete vector store from disk.

        Args:
            path: Optional custom path
        """
        import shutil
        
        delete_path = Path(path) if path else self.vector_store_path
        
        if delete_path.exists():
            shutil.rmtree(delete_path)
            logger.info(f"✓ Deleted vector store at {delete_path}")
        else:
            logger.warning(f"Vector store not found at {delete_path}")
        
        # Reset in-memory store
        self.vectorstore = None
        self.chunk_mapping = {}


# Singleton instance
_embeddings_manager_instance: Optional[EmbeddingsManager] = None


def get_embeddings_manager(
    embedding_model: str = "models/embedding-001",
    api_key: Optional[str] = None,
    vector_store_path: str = "vectorstore_db"
) -> EmbeddingsManager:
    """
    Get or create singleton EmbeddingsManager instance.

    Args:
        embedding_model: Google embedding model name
        api_key: Google API key
        vector_store_path: Path to vector store

    Returns:
        EmbeddingsManager instance
    """
    global _embeddings_manager_instance
    
    if _embeddings_manager_instance is None:
        _embeddings_manager_instance = EmbeddingsManager(
            embedding_model=embedding_model,
            api_key=api_key,
            vector_store_path=vector_store_path
        )
    
    return _embeddings_manager_instance
