"""Unit tests for VectorStoreService.

Gemini embeddings, FAISS operations and disk I/O are all mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from app.services.vector_store_service import VectorStoreService


def _make_service(tmp_path, mock_embeddings):
    """Build a VectorStoreService with embeddings already injected, bypassing __init__."""
    svc = VectorStoreService.__new__(VectorStoreService)
    svc.embedding_provider = "gemini"
    svc.persist_dir = tmp_path
    svc.use_cache = False
    svc.gemini_model = "models/gemini-embedding-001"
    svc._embeddings = mock_embeddings
    svc._vector_store = None
    return svc


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 768
    emb.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
    return emb


@pytest.fixture
def mock_faiss_store():
    store = MagicMock()
    store.similarity_search.return_value = []
    store.similarity_search_with_score.return_value = []
    store.add_documents.return_value = None
    store.save_local.return_value = None
    store.index_to_docstore_id = {}
    return store


# ---------------------------------------------------------------------------
# is_initialized / _ensure_store
# ---------------------------------------------------------------------------

class TestIsInitialized:
    def test_false_before_store_set(self, tmp_path, mock_embeddings):
        svc = _make_service(tmp_path, mock_embeddings)
        assert svc.is_initialized() is False

    def test_true_after_store_set(self, tmp_path, mock_embeddings, mock_faiss_store):
        svc = _make_service(tmp_path, mock_embeddings)
        svc._vector_store = mock_faiss_store
        assert svc.is_initialized() is True

    def test_ensure_store_raises_when_not_initialized(self, tmp_path, mock_embeddings):
        svc = _make_service(tmp_path, mock_embeddings)
        with pytest.raises(RuntimeError, match="not initialized"):
            svc._ensure_store()


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    @pytest.mark.asyncio
    async def test_returns_documents(self, tmp_path, mock_embeddings, mock_faiss_store, sample_documents):
        svc = _make_service(tmp_path, mock_embeddings)
        svc._vector_store = mock_faiss_store

        async def fake_run_blocking(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        svc._run_blocking = fake_run_blocking
        mock_faiss_store.similarity_search.return_value = sample_documents

        results = await svc.similarity_search("AI in medicine", k=2)
        assert results == sample_documents
        mock_faiss_store.similarity_search.assert_called_once_with("AI in medicine", 2)

    @pytest.mark.asyncio
    async def test_raises_when_store_not_set(self, tmp_path, mock_embeddings):
        svc = _make_service(tmp_path, mock_embeddings)
        with pytest.raises(RuntimeError):
            await svc.similarity_search("query")


# ---------------------------------------------------------------------------
# similarity_search_with_score
# ---------------------------------------------------------------------------

class TestSimilaritySearchWithScore:
    @pytest.mark.asyncio
    async def test_returns_tuples_with_scores(self, tmp_path, mock_embeddings, mock_faiss_store, sample_documents):
        svc = _make_service(tmp_path, mock_embeddings)
        svc._vector_store = mock_faiss_store
        expected = [(sample_documents[0], 0.92)]
        mock_faiss_store.similarity_search_with_score.return_value = expected

        async def fake_run_blocking(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        svc._run_blocking = fake_run_blocking
        results = await svc.similarity_search_with_score("AI", k=1)
        assert results == expected


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    @pytest.mark.asyncio
    async def test_count_returns_length_of_index(self, tmp_path, mock_embeddings, mock_faiss_store):
        svc = _make_service(tmp_path, mock_embeddings)
        mock_faiss_store.index_to_docstore_id = {0: "a", 1: "b"}
        svc._vector_store = mock_faiss_store
        count = await svc.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_raises_when_not_initialized(self, tmp_path, mock_embeddings):
        svc = _make_service(tmp_path, mock_embeddings)
        with pytest.raises(RuntimeError):
            await svc.count()


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------

class TestAddDocuments:
    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_input(self, tmp_path, mock_embeddings, mock_faiss_store):
        svc = _make_service(tmp_path, mock_embeddings)
        svc._vector_store = mock_faiss_store

        async def fake_run_blocking(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        svc._run_blocking = fake_run_blocking
        result = await svc.add_documents([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_total_document_count(self, tmp_path, mock_embeddings, mock_faiss_store, sample_documents):
        svc = _make_service(tmp_path, mock_embeddings)
        svc._vector_store = mock_faiss_store

        async def fake_run_blocking(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        svc._run_blocking = fake_run_blocking
        result = await svc.add_documents(sample_documents)
        assert result == len(sample_documents)


# ---------------------------------------------------------------------------
# load_or_create_vector_store
# ---------------------------------------------------------------------------

class TestLoadOrCreate:
    @pytest.mark.asyncio
    async def test_creates_new_store_when_dir_missing(self, tmp_path, mock_embeddings):
        svc = _make_service(tmp_path / "new_dir", mock_embeddings)

        with patch("app.services.vector_store_service.faiss") as mock_faiss, \
             patch("app.services.vector_store_service.FAISS") as mock_faiss_cls, \
             patch("app.services.vector_store_service.InMemoryDocstore"):
            mock_faiss.IndexFlatL2.return_value = MagicMock()
            mock_faiss_cls.return_value = MagicMock()
            await svc.load_or_create_vector_store()

        assert svc._vector_store is not None

    @pytest.mark.asyncio
    async def test_loads_existing_store_when_dir_exists(self, tmp_path, mock_embeddings):
        tmp_path.mkdir(exist_ok=True)
        svc = _make_service(tmp_path, mock_embeddings)
        mock_loaded = MagicMock()

        async def fake_run_blocking(fn, *args, **kwargs):
            return mock_loaded

        svc._run_blocking = fake_run_blocking
        await svc.load_or_create_vector_store()
        assert svc._vector_store is mock_loaded


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_vector_store_service_returns_same_instance(tmp_path):
    import app.services.vector_store_service as mod
    mod._def_instance = None  # reset singleton

    with patch.object(VectorStoreService, "__init__", return_value=None), \
         patch.object(VectorStoreService, "load_or_create_vector_store", new=AsyncMock()):
        svc_a = await mod.get_vector_store_service(persist_dir=str(tmp_path))
        svc_b = await mod.get_vector_store_service(persist_dir=str(tmp_path))

    assert svc_a is svc_b
    mod._def_instance = None  # clean up
