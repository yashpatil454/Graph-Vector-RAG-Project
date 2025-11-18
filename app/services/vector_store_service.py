"""
Async Vector Store Service
Provides async-safe operations to build, persist, and query a FAISS vector index
from processed PDF document chunks.
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Callable

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
    """
    Async-safe service wrapping FAISS vector store operations.

    Parameters:
        embedding_provider: "gemini"
        persist_dir: Directory to persist FAISS files
        use_cache: Enable embedding cache
        gemini_model: Gemini embedding model
    """

    def __init__(
        self,
        embedding_provider: str = "gemini",
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
            f"Async VectorStoreService initialized provider={self.embedding_provider} "
            f"persist_dir={self.persist_dir} cache={self.use_cache}"
        )

    # ------------------------------------------------------------------
    # Async Helper
    # ------------------------------------------------------------------
    async def _run_blocking(self, func, *args, **kwargs):
        """Run blocking CPU-bound FAISS/embedding operations off the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    # ------------------------------------------------------------------
    # Embeddings Init
    # ------------------------------------------------------------------
    def _init_embeddings(self) -> Embeddings:
        provider = self.embedding_provider

        if provider == "gemini":
            google_api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
            if not google_api_key:
                raise EnvironmentError("GOOGLE_API_KEY missing for Gemini embeddings.")

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
            logger.info("Using CacheBackedEmbeddings for Gemini.")
            return cached

        return base

    # ------------------------------------------------------------------
    # Add Documents (Async)
    # ------------------------------------------------------------------
    async def add_documents(self, documents: List[Document]) -> int:
        """Asynchronously add documents in token-size batches."""
        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small"
        )
        logger.info(f"Adding {len(documents)} documents to vector store...")
        max_tokens_per_batch = 28000
        total_documents = len(documents)
        logger.info(
            f"VectorStore add_documents start | total_docs={total_documents} max_tokens_per_batch={max_tokens_per_batch}"
        )
        batch: List[Document] = []
        batch_tokens = 0
        total_tokens = 0
        batches_added = 0
        tokens_per_doc: List[int] = []

        for idx, doc in enumerate(documents, start=1):
            n_tokens = splitter._length_function(doc.page_content)
            tokens_per_doc.append(n_tokens)
            logger.info(
                f"Doc {idx}/{total_documents} | first_5='{doc.page_content[:5].replace('\n',' ')}' | tokens={n_tokens}"
            )

            # Flush current batch if adding this doc would exceed threshold
            if batch and (batch_tokens + n_tokens > max_tokens_per_batch):
                batches_added += 1
                logger.info(
                    f"Flushing batch {batches_added} | docs_in_batch={len(batch)} batch_tokens={batch_tokens} cumulative_tokens={total_tokens}"
                )
                await self._run_blocking(self._vector_store.add_documents, batch)
                batch = []
                batch_tokens = 0
                await asyncio.sleep(62.0)

            batch.append(doc)
            batch_tokens += n_tokens
            total_tokens += n_tokens

        # After loop, flush remaining batch
        if batch:
            batches_added += 1
            logger.info(
                f"Flushing final batch {batches_added} | docs_in_batch={len(batch)} batch_tokens={batch_tokens} cumulative_tokens={total_tokens}"
            )
            await self._run_blocking(self._vector_store.add_documents, batch)

        # Stats
        min_tokens = min(tokens_per_doc) if tokens_per_doc else 0
        max_tokens = max(tokens_per_doc) if tokens_per_doc else 0
        avg_tokens_doc = (total_tokens / total_documents) if total_documents else 0
        avg_tokens_batch = (total_tokens / batches_added) if batches_added else 0

        logger.info(
            "VectorStore add_documents complete | "
            f"documents={total_documents} batches={batches_added} total_tokens={total_tokens} "
            f"min_tokens_doc={min_tokens} max_tokens_doc={max_tokens} avg_tokens_doc={avg_tokens_doc:.2f} "
            f"avg_tokens_batch={avg_tokens_batch:.2f}"
        )

        await self.save()
        return total_documents

    # ------------------------------------------------------------------
    # Search (Async)
    # ------------------------------------------------------------------
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        self._ensure_store()
        return await self._run_blocking(self._vector_store.similarity_search, query, k)

    async def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        self._ensure_store()
        return await self._run_blocking(
            self._vector_store.similarity_search_with_score, query, k
        )

    async def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Document]:
        self._ensure_store()
        return await self._run_blocking(
            self._vector_store.max_marginal_relevance_search,
            query,
            k,
            fetch_k,
            lambda_mult,
        )

    # ------------------------------------------------------------------
    # Persistence (Async)
    # ------------------------------------------------------------------
    async def save(self):
        self._ensure_store()
        await self._run_blocking(self._vector_store.save_local, str(self.persist_dir))
        logger.info(f"Vector store saved to {self.persist_dir}")

    async def load_or_create_vector_store(self):
        """Async load existing FAISS index or create new empty one."""
        if self.persist_dir.exists():
            try:
                loaded = await self._run_blocking(
                FAISS.load_local,
                str(self.persist_dir),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
                self._vector_store = loaded
                logger.info(f"Loaded FAISS index from {self.persist_dir}")
                return
            except Exception as e:
                logger.warning(f"Load failed, creating new index. Reason: {e}")

        # Create fresh index
        embedding_dim = len(self._embeddings.embed_query("dimension probe"))
        index = faiss.IndexFlatL2(embedding_dim)

        self._vector_store = FAISS(
            embedding_function=self._embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        logger.info(f"Created new FAISS index (dim={embedding_dim}) at {self.persist_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_store(self):
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized. Call load_or_create_vector_store().")

    def is_initialized(self) -> bool:
        return self._vector_store is not None

    async def count(self) -> int:
        self._ensure_store()
        return len(self._vector_store.index_to_docstore_id)


# ----------------------------------------------------------------------
# Async Singleton Factory
# ----------------------------------------------------------------------
_def_instance: Optional[VectorStoreService] = None

async def get_vector_store_service(
    embedding_provider: str = "gemini",
    persist_dir: str = "vector_store",
    use_cache: bool = True,
    gemini_model: str = "models/gemini-embedding-001",
) -> VectorStoreService:
    global _def_instance
    if _def_instance is None:
        svc = VectorStoreService(
            embedding_provider=embedding_provider,
            persist_dir=persist_dir,
            use_cache=use_cache,
            gemini_model=gemini_model,
        )
        await svc.load_or_create_vector_store()
        _def_instance = svc
    return _def_instance