"""Vector Store Service
Provides functionality to build, persist, and query a FAISS vector index
from processed PDF document chunks.

Features:
- Supports Google Gemini embeddings (requires GOOGLE_API_KEY) via LangChain integration.
- Builds FAISS index from LangChain `Document` objects.
- Similarity search (with and without scores) and MMR search.
- Persistence (save/load local index directory).
- Optional embedding caching via CacheBackedEmbeddings for both providers.

Notes:
- Set the environment variable GOOGLE_API_KEY before using this service
    depending on the chosen provider.
- Call `save()` to persist, then later instantiate and call `load()` to restore.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Any

from app.core.logger import SingletonLogger

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

from langchain_classic.embeddings import CacheBackedEmbeddings  
from langchain_classic.storage import LocalFileStore 

logger = SingletonLogger().get_logger()

class VectorStoreService:
    """Service wrapping FAISS vector store operations.

    Parameters:
        embedding_provider: "gemini"
        persist_dir: Directory to persist FAISS index artifacts.
        use_cache: If True and supported, wrap embeddings with file-cache.
        gemini_model: Model name for Gemini embeddings (default models/gemini-embedding-001).
    """

    def __init__(
        self,
        embedding_provider: str = "openai",
        persist_dir: str = "vector_store",
        use_cache: bool = True,
        gemini_model: str = "models/gemini-embedding-001",
    ):
        self.embedding_provider = embedding_provider.lower()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.gemini_model = gemini_model

        self._embeddings: Embeddings = self._init_embeddings()
        self._vector_store: Optional[FAISS] = None

        logger.info(
            f"VectorStoreService initialized provider={self.embedding_provider} "
            f"persist_dir={self.persist_dir} cache={self.use_cache}"
        )

    def _init_embeddings(self) -> Embeddings:
        """Initialize embeddings based on provider selection."""
        provider = self.embedding_provider

        if provider == "gemini":
            # Gemini embedding expects GOOGLE_API_KEY in environment
            google_api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
            if not google_api_key:
                raise EnvironmentError("GOOGLE_API_KEY missing from environment for Gemini embeddings.")
            # The class reads from environment variable; ensure it's set for downstream libs
            os.environ.setdefault("GOOGLE_API_KEY", google_api_key)
            base = GoogleGenerativeAIEmbeddings(model=self.gemini_model)
            namespace = getattr(base, "model", "gemini")
        else:
            raise ValueError(
                f"Unsupported embedding_provider '{provider}'. Only 'gemini' is supported."
            )

        if self.use_cache:
            cache_dir = self.persist_dir / "cache"
            store = LocalFileStore(str(cache_dir))
            cached = CacheBackedEmbeddings.from_bytes_store(base, store, namespace=namespace)
            logger.info(f"Using CacheBackedEmbeddings for provider '{provider}'.")
            return cached
        return base

    # ------------------------------------------------------------------
    # Build / Add
    # ------------------------------------------------------------------
    def from_documents(self, documents: List[Document]) -> None:
        """Create a new FAISS index from documents (replaces existing)."""
        if not documents:
            raise ValueError("No documents provided to build vector store.")

        import faiss  # Local import to avoid unnecessary dependency issues early
        embedding_dim = len(self._embeddings.embed_query("dimension probe"))
        index = faiss.IndexFlatL2(embedding_dim)

        self._vector_store = FAISS(
            embedding_function=self._embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self._vector_store.add_documents(documents)
        logger.info(f"FAISS index built with {len(documents)} documents (chunks).")

    def add_documents(self, documents: List[Document]) -> int:
        """Add additional documents to existing index."""
        if not documents:
            return 0
        if self._vector_store is None:
            self.from_documents(documents)
            return len(documents)
        self._vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store.")
        return len(documents)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        self._ensure_store()
        return self._vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        self._ensure_store()
        return self._vector_store.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Document]:
        self._ensure_store()
        return self._vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        self._ensure_store()
        self._vector_store.save_local(str(self.persist_dir))
        logger.info(f"Vector store persisted to {self.persist_dir}.")

    def load(self) -> bool:
        """Load existing persisted FAISS index. Returns True if successful."""
        if not self.persist_dir.exists():
            return False
        try:
            self._vector_store = FAISS.load_local(
                str(self.persist_dir),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,  # Controlled local usage
            )
            logger.info(
                f"Loaded FAISS index from {self.persist_dir} (size unknown until queried)."
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_store(self) -> None:
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized. Call from_documents() or load() first.")

    def is_initialized(self) -> bool:
        return self._vector_store is not None

    def count(self) -> int:
        self._ensure_store()
        # Access private index_to_docstore_id size
        return len(self._vector_store.index_to_docstore_id)


# Convenience factory with load attempt
_def_instance: Optional[VectorStoreService] = None


def get_vector_store_service(
    embedding_provider: str = "gemini",
    persist_dir: str = "vector_store",
    use_cache: bool = True,
    gemini_model: str = "models/gemini-embedding-001",
    auto_load: bool = True,
) -> VectorStoreService:
    """Get or create a singleton VectorStoreService instance.

    If auto_load=True, will attempt to load an existing index from persist_dir.
    """
    global _def_instance
    if _def_instance is None:
        svc = VectorStoreService(
            embedding_provider=embedding_provider,
            persist_dir=persist_dir,
            use_cache=use_cache,
            gemini_model=gemini_model,
        )
        if auto_load:
            svc.load()
        _def_instance = svc
    return _def_instance
