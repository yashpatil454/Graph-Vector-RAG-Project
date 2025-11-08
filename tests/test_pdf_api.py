import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_process_all_pdfs():
    resp = client.post("/data_processor/process")
    assert resp.status_code == 200
    data = resp.json()
    assert "chunks" in data
    if data["total_chunks"] > 0:
        # If there are chunks, ensure serialization shape
        assert isinstance(data["chunks"], list)
        if data["chunks"]:
            sample = data["chunks"][0]
            assert set(sample.keys()) == {"chunk_id", "content", "char_count", "metadata"}

