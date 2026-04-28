"""Unit tests for KnowledgeGraphService.

Neo4j driver and Gemini LLM are fully mocked — no real DB or API calls.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from app.models.request_models import Triple


def _make_service(tmp_path):
    """Instantiate KnowledgeGraphService bypassing __init__ to avoid real Neo4j/LLM setup."""
    from app.services.knowledge_graph_service import KnowledgeGraphService
    svc = KnowledgeGraphService.__new__(KnowledgeGraphService)
    svc.data_dir = Path(tmp_path)

    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '[{"subject": "AI", "predicate": "is used in", "object": "medicine"}]'
    mock_llm.invoke.return_value = mock_response
    svc._llm = mock_llm

    # Mock Neo4j driver
    session_mock = AsyncMock()
    result_mock = AsyncMock()
    result_mock.single.return_value = {"relationships_created": 1}
    result_mock.__aiter__ = AsyncMock(return_value=iter([]))
    session_mock.run.return_value = result_mock
    driver_mock = MagicMock()
    driver_mock.session.return_value.__aenter__ = AsyncMock(return_value=session_mock)
    driver_mock.session.return_value.__aexit__ = AsyncMock(return_value=False)
    svc._driver = driver_mock
    svc._session_mock = session_mock  # expose for assertions
    svc._result_mock = result_mock

    return svc


# ---------------------------------------------------------------------------
# extract_triples
# ---------------------------------------------------------------------------

class TestExtractTriples:
    @pytest.mark.asyncio
    async def test_writes_triples_to_jsonl(self, tmp_path, sample_documents):
        svc = _make_service(tmp_path)
        count = await svc.extract_triples(sample_documents[:1])
        assert count == 1
        output = tmp_path / "processed_triples" / "triples.jsonl"
        assert output.exists()
        line = json.loads(output.read_text().strip())
        assert line["subject"] == "AI"
        assert line["predicate"] == "is used in"
        assert line["object"] == "medicine"

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_input(self, tmp_path):
        svc = _make_service(tmp_path)
        count = await svc.extract_triples([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_non_json_response(self, tmp_path, sample_documents):
        svc = _make_service(tmp_path)
        svc._llm.invoke.return_value.content = "This is not JSON"
        count = await svc.extract_triples(sample_documents[:1])
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_incomplete_triples(self, tmp_path, sample_documents):
        svc = _make_service(tmp_path)
        # Missing "object" key
        svc._llm.invoke.return_value.content = '[{"subject": "AI", "predicate": "helps"}]'
        count = await svc.extract_triples(sample_documents[:1])
        assert count == 0

    @pytest.mark.asyncio
    async def test_appends_to_existing_file(self, tmp_path, sample_documents):
        svc = _make_service(tmp_path)
        output_dir = tmp_path / "processed_triples"
        output_dir.mkdir()
        existing = {"subject": "X", "predicate": "Y", "object": "Z", "provenance": {}}
        (output_dir / "triples.jsonl").write_text(json.dumps(existing) + "\n")

        await svc.extract_triples(sample_documents[:1])
        lines = (output_dir / "triples.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# load_persisted_triples
# ---------------------------------------------------------------------------

class TestLoadPersistedTriples:
    def test_loads_valid_triples(self, tmp_path):
        svc = _make_service(tmp_path)
        triple_dir = tmp_path / "processed_triples"
        triple_dir.mkdir()
        line = json.dumps({"subject": "AI", "predicate": "helps", "object": "doctors", "provenance": {}})
        (triple_dir / "triples.jsonl").write_text(line + "\n")

        triples = svc.load_persisted_triples()
        assert len(triples) == 1
        assert triples[0].subject == "AI"
        assert triples[0].predicate == "helps"
        assert triples[0].object == "doctors"

    def test_returns_empty_when_file_missing(self, tmp_path):
        svc = _make_service(tmp_path)
        triples = svc.load_persisted_triples()
        assert triples == []

    def test_skips_malformed_lines(self, tmp_path):
        svc = _make_service(tmp_path)
        triple_dir = tmp_path / "processed_triples"
        triple_dir.mkdir()
        good = json.dumps({"subject": "A", "predicate": "B", "object": "C", "provenance": {}})
        (triple_dir / "triples.jsonl").write_text("not json\n" + good + "\n")

        triples = svc.load_persisted_triples()
        assert len(triples) == 1

    def test_respects_limit(self, tmp_path):
        svc = _make_service(tmp_path)
        triple_dir = tmp_path / "processed_triples"
        triple_dir.mkdir()
        lines = [
            json.dumps({"subject": f"S{i}", "predicate": "P", "object": "O", "provenance": {}})
            for i in range(10)
        ]
        (triple_dir / "triples.jsonl").write_text("\n".join(lines) + "\n")

        triples = svc.load_persisted_triples(limit=3)
        assert len(triples) == 3

    def test_skips_triples_with_empty_fields(self, tmp_path):
        svc = _make_service(tmp_path)
        triple_dir = tmp_path / "processed_triples"
        triple_dir.mkdir()
        bad = json.dumps({"subject": "", "predicate": "P", "object": "O", "provenance": {}})
        (triple_dir / "triples.jsonl").write_text(bad + "\n")

        triples = svc.load_persisted_triples()
        assert triples == []


# ---------------------------------------------------------------------------
# ingest_triples
# ---------------------------------------------------------------------------

class TestIngestTriples:
    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_input(self, tmp_path):
        svc = _make_service(tmp_path)
        count = await svc.ingest_triples([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_deduplicates_before_ingestion(self, tmp_path):
        svc = _make_service(tmp_path)
        triples = [
            Triple("AI", "helps", "doctors"),
            Triple("AI", "helps", "doctors"),  # exact duplicate
            Triple("AI", "helps", "patients"),
        ]
        count = await svc.ingest_triples(triples)
        assert count == 2  # only 2 unique

    @pytest.mark.asyncio
    async def test_calls_neo4j_session_run(self, tmp_path):
        svc = _make_service(tmp_path)
        triples = [Triple("A", "rel", "B")]
        await svc.ingest_triples(triples)
        svc._session_mock.run.assert_called_once()


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, tmp_path):
        svc = _make_service(tmp_path)

        mock_record = MagicMock()
        mock_record.keys.return_value = ["name"]
        mock_record.get.return_value = "AI"

        # Use a plain async generator function (not bound method) to avoid signature issues
        async def fake_run(*args, **kwargs):
            class FakeResult:
                async def __aiter__(self):
                    yield mock_record
            return FakeResult()

        svc._session_mock.run = fake_run

        results = await svc.query("MATCH (e:Entity) RETURN e.name AS name")
        assert isinstance(results, list)
        assert results[0]["name"] == "AI"

    @pytest.mark.asyncio
    async def test_passes_params_to_session(self, tmp_path):
        svc = _make_service(tmp_path)
        received_args = {}

        async def fake_run(*args, **kwargs):
            received_args["args"] = args
            received_args["kwargs"] = kwargs
            class FakeResult:
                async def __aiter__(self):
                    return
                    yield  # empty async generator
            return FakeResult()

        svc._session_mock.run = fake_run
        await svc.query("MATCH (e) RETURN e", params={"limit": 10})
        assert "MATCH (e) RETURN e" in received_args["args"]
        assert received_args["kwargs"].get("limit") == 10
