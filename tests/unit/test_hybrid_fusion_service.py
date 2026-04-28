"""Unit tests for HybridFusionService.

VectorStoreService and KnowledgeGraphService are fully mocked.
No Gemini API, FAISS, or Neo4j calls are made.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document

from app.models.request_models import FusionRequest
from app.services.hybrid_fusion_service import HybridFusionService


def _make_doc(content: str, source: str = "doc.pdf", page: int = 1) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


def _make_service(vector_hits=None, kg_entity_names=None, graph_rows=None):
    """Build HybridFusionService with both sub-services mocked."""
    if vector_hits is None:
        vector_hits = [
            (_make_doc("AI raises ethical challenges in medicine.", "paper.pdf", 2), float(0.88)),
        ]
    if kg_entity_names is None:
        kg_entity_names = ["AI", "Ethics", "Medicine", "Physicians"]
    if graph_rows is None:
        graph_rows = [
            {"subject": "AI", "predicate": "raises", "object": "Ethics"},
            {"subject": "Physicians", "predicate": "rely on", "object": "AI"},
        ]

    vector_svc = AsyncMock()
    vector_svc.similarity_search_with_score.return_value = vector_hits

    kg_svc = AsyncMock()
    # First call: _list_existing_entities; second call: graph triples query
    kg_svc.query.side_effect = [
        [{"name": n} for n in kg_entity_names],
        graph_rows,
    ]

    svc = HybridFusionService()
    svc._vector_service = vector_svc
    svc._kg_service = kg_svc
    return svc


# ---------------------------------------------------------------------------
# fuse — response structure
# ---------------------------------------------------------------------------

class TestFuseResponseStructure:
    @pytest.mark.asyncio
    async def test_returns_all_expected_keys(self):
        svc = _make_service()
        req = FusionRequest(query="AI ethics", k=2)
        result = await svc.fuse(req)
        for key in ("query", "k", "vector_hits_count", "vector_hits",
                    "graph_results_count", "graph_results", "context"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_query_echoed_in_result(self):
        svc = _make_service()
        req = FusionRequest(query="physician trust")
        result = await svc.fuse(req)
        assert result["query"] == "physician trust"

    @pytest.mark.asyncio
    async def test_vector_hits_count_matches_list(self):
        svc = _make_service()
        result = await svc.fuse(FusionRequest(query="test"))
        assert result["vector_hits_count"] == len(result["vector_hits"])

    @pytest.mark.asyncio
    async def test_graph_results_count_matches_list(self):
        svc = _make_service()
        result = await svc.fuse(FusionRequest(query="test"))
        assert result["graph_results_count"] == len(result["graph_results"])

    @pytest.mark.asyncio
    async def test_scores_are_python_floats(self):
        import numpy as np
        hits = [(_make_doc("text"), np.float32(0.75))]
        svc = _make_service(vector_hits=hits)
        result = await svc.fuse(FusionRequest(query="test"))
        score = result["vector_hits"][0]["score"]
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# fuse — score threshold filtering
# ---------------------------------------------------------------------------

class TestScoreThreshold:
    @pytest.mark.asyncio
    async def test_filters_hits_below_threshold(self):
        hits = [
            (_make_doc("high relevance"), float(0.90)),
            (_make_doc("low relevance"), float(0.30)),
        ]
        svc = _make_service(vector_hits=hits)
        result = await svc.fuse(FusionRequest(query="test", score_threshold=0.80))
        assert result["vector_hits_count"] == 1
        assert result["vector_hits"][0]["content"] == "high relevance"

    @pytest.mark.asyncio
    async def test_zero_threshold_keeps_all_hits(self):
        hits = [
            (_make_doc("doc1"), float(0.10)),
            (_make_doc("doc2"), float(0.20)),
        ]
        svc = _make_service(vector_hits=hits)
        result = await svc.fuse(FusionRequest(query="test", score_threshold=0.0))
        assert result["vector_hits_count"] == 2

    @pytest.mark.asyncio
    async def test_threshold_above_all_scores_returns_empty_hits(self):
        hits = [(_make_doc("text"), float(0.50))]
        svc = _make_service(vector_hits=hits)
        result = await svc.fuse(FusionRequest(query="test", score_threshold=0.99))
        assert result["vector_hits_count"] == 0


# ---------------------------------------------------------------------------
# fuse — graph deduplication
# ---------------------------------------------------------------------------

class TestGraphDeduplication:
    @pytest.mark.asyncio
    async def test_duplicate_triples_removed(self):
        graph_rows = [
            {"subject": "AI", "predicate": "helps", "object": "doctors"},
            {"subject": "AI", "predicate": "helps", "object": "doctors"},  # duplicate
        ]
        svc = _make_service(graph_rows=graph_rows)
        result = await svc.fuse(FusionRequest(query="test"))
        assert result["graph_results_count"] == 1

    @pytest.mark.asyncio
    async def test_different_triples_all_kept(self):
        graph_rows = [
            {"subject": "AI", "predicate": "helps", "object": "doctors"},
            {"subject": "AI", "predicate": "raises", "object": "Ethics"},
        ]
        svc = _make_service(graph_rows=graph_rows)
        result = await svc.fuse(FusionRequest(query="test"))
        assert result["graph_results_count"] == 2


# ---------------------------------------------------------------------------
# fuse — custom Cypher
# ---------------------------------------------------------------------------

class TestCustomCypher:
    @pytest.mark.asyncio
    async def test_custom_cypher_bypasses_auto_generation(self):
        svc = _make_service()
        custom = "MATCH (e:Entity) RETURN e.name AS subject, 'is' AS predicate, 'entity' AS object"
        req = FusionRequest(query="test", cypher=custom)
        await svc.fuse(req)
        # Second kg_svc.query call uses the custom cypher (no $names param)
        call_args = svc._kg_service.query.call_args_list[1]
        assert call_args[0][0] == custom


# ---------------------------------------------------------------------------
# fuse — parallel execution
# ---------------------------------------------------------------------------

class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_vector_and_kg_entity_calls_both_made(self):
        svc = _make_service()
        await svc.fuse(FusionRequest(query="test"))
        svc._vector_service.similarity_search_with_score.assert_called_once()
        # kg_svc.query called at least once for entity listing
        assert svc._kg_service.query.call_count >= 1


# ---------------------------------------------------------------------------
# _extract_entity_candidates
# ---------------------------------------------------------------------------

class TestExtractEntityCandidates:
    def test_returns_capitalised_tokens(self):
        svc = HybridFusionService()
        hits = [{"content": "AI and Machine Learning improve Physicians outcomes."}]
        candidates = svc._extract_entity_candidates(hits)
        assert "Machine" in candidates or "Physicians" in candidates

    def test_deduplicates_tokens(self):
        svc = HybridFusionService()
        # Use a token >2 chars so it matches the regex [A-Z][A-Za-z0-9_-]{2,}
        hits = [{"content": "Artificial is great. Artificial is powerful."}]
        candidates = svc._extract_entity_candidates(hits)
        assert candidates.count("Artificial") == 1

    def test_respects_max_entities(self):
        svc = HybridFusionService()
        # Generate a string with many capitalised tokens
        content = " ".join(f"Token{i}" for i in range(100))
        hits = [{"content": content}]
        candidates = svc._extract_entity_candidates(hits, max_entities=10)
        assert len(candidates) <= 10

    def test_empty_hits_returns_empty(self):
        svc = HybridFusionService()
        candidates = svc._extract_entity_candidates([])
        assert candidates == []


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_includes_query_in_context(self):
        svc = HybridFusionService()
        req = FusionRequest(query="AI ethics")
        context = svc._build_context(req, [], [])
        assert "AI ethics" in context

    def test_includes_source_citation(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test", include_scores=True)
        hits = [{"content": "some text", "metadata": {"source": "paper.pdf", "page": 7}, "score": 0.9}]
        context = svc._build_context(req, hits, [])
        assert "paper.pdf" in context
        assert "page 7" in context

    def test_includes_score_when_requested(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test", include_scores=True)
        hits = [{"content": "text", "metadata": {"source": "f.pdf", "page": 1}, "score": 0.85}]
        context = svc._build_context(req, hits, [])
        assert "0.85" in context

    def test_omits_score_when_not_requested(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test", include_scores=False)
        hits = [{"content": "text", "metadata": {"source": "f.pdf", "page": 1}, "score": 0.85}]
        context = svc._build_context(req, hits, [])
        assert "score=0.85" not in context

    def test_includes_graph_triples(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test")
        graph = [{"subject": "AI", "predicate": "helps", "object": "doctors"}]
        context = svc._build_context(req, [], graph)
        assert "AI -[helps]-> doctors" in context

    def test_no_graph_results_shows_placeholder(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test")
        context = svc._build_context(req, [], [])
        assert "no graph relationships found" in context

    def test_includes_instructions_block(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test")
        context = svc._build_context(req, [], [])
        assert "Instructions" in context

    def test_snippet_max_800_chars(self):
        svc = HybridFusionService()
        req = FusionRequest(query="test")
        long_content = "A" * 2000
        hits = [{"content": long_content, "metadata": {"source": "f.pdf", "page": 1}, "score": 0.9}]
        context = svc._build_context(req, hits, [])
        # The snippet in context should be at most 800 chars of the content
        assert "A" * 801 not in context


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_hybrid_fusion_service_returns_same_instance():
    import app.services.hybrid_fusion_service as mod
    mod._fusion_instance = None  # reset
    a = await mod.get_hybrid_fusion_service()
    b = await mod.get_hybrid_fusion_service()
    assert a is b
    mod._fusion_instance = None  # clean up
