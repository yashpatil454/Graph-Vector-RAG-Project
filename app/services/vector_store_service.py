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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents in token-size batches (each under 20,000 tokens)."""
        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small"
        )

        max_tokens_per_batch = 20_000
        batch, batch_tokens, total_tokens, total_documents = [], 0, 0, len(documents)
        batches_added = 0

        for doc in documents:
            n_tokens = splitter._length_function(doc.page_content)

            # If adding this doc would exceed batch limit, flush batch first
            if batch and (batch_tokens + n_tokens > max_tokens_per_batch):
                self._vector_store.add_documents(batch)
                batches_added += 1
                logger.info(f"Committed batch {batches_added} (Approx. {batch_tokens} tokens, {len(batch)} docs).")
                batch, batch_tokens = [], 0  # reset after adding batch

            # Add current doc to batch
            batch.append(doc)
            batch_tokens += n_tokens
            total_tokens += n_tokens

        # Add remaining batch if any
        if batch:
            self._vector_store.add_documents(batch)
            batches_added += 1
            logger.info(f"Committed final batch {batches_added} (Approx. {batch_tokens} tokens, {len(batch)} docs).")

        logger.info(f"Total: {total_documents} documents processed in {batches_added} batches (Approx. {total_tokens} tokens total).")

        # Save vector store
        try:
            self.save()
        except Exception as e:
            logger.error(f"Failed to save vector store after adding documents: {e}")

        return total_documents

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

    def load_or_create_vector_store(self):
        """Load existing persisted FAISS index if present; otherwise create a new empty index.
        """
        # Attempt load if persistence directory exists and appears to contain FAISS artifacts.
        if self.persist_dir.exists():
            try:
                self._vector_store = FAISS.load_local(
                    str(self.persist_dir),
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True,  # Controlled local usage
                )
                logger.info(
                    f"Loaded FAISS index from {self.persist_dir}."
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load vector store from '{self.persist_dir}': {e}. Will create new index.")

        # Create a new empty FAISS index using a dimension probe.
        try:
            embedding_dim = len(self._embeddings.embed_query("dimension probe"))
        except Exception as e:
            logger.error(f"Failed to probe embedding dimension: {e}")
            raise

        try:
            index = faiss.IndexFlatL2(embedding_dim)
            self._vector_store = FAISS(
                embedding_function=self._embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            logger.info(
                f"Created new empty FAISS index (dimension={embedding_dim}) at {self.persist_dir}."
            )
        except Exception as e:
            logger.error(f"Failed to create new FAISS index: {e}")
            raise
        return

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
    gemini_model: str = "models/gemini-embedding-001"
) -> VectorStoreService:
    """Get or create a singleton VectorStoreService instance.
    """
    global _def_instance
    if _def_instance is None:
        svc = VectorStoreService(
            embedding_provider=embedding_provider,
            persist_dir=persist_dir,
            use_cache=use_cache,
            gemini_model=gemini_model,
        )
        svc.load_or_create_vector_store()
        _def_instance = svc
    return _def_instance
