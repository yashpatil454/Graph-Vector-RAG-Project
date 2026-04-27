from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

from app.services.knowledge_graph_service import get_knowledge_graph_service, KnowledgeGraphService
from app.services.data_processor import get_pdf_processor
from app.models.request_models import (
    KnowledgeGraphExtractResponse,
    KnowledgeGraphIngestResponse,
    KnowledgeGraphQueryResponse,
)

router = APIRouter(prefix="/knowledge_graph", tags=["knowledge_graph"])

async def get_service() -> KnowledgeGraphService:
    return await get_knowledge_graph_service()


@router.post("/extract", response_model=KnowledgeGraphExtractResponse)
async def extract_triples(
    service: KnowledgeGraphService = Depends(get_service),
):
    """Load persisted PDF chunks and extract (subject, predicate, object) triples
    using Gemini 2.5 Flash. Triples are streamed to data/processed_triples/triples.jsonl.
    """
    try:
        processor = get_pdf_processor()
        documents = processor.load_persisted_chunks()
        triples_written = await service.extract_triples(documents)
        return KnowledgeGraphExtractResponse(
            total_documents=len(documents),
            triples_written=triples_written,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=KnowledgeGraphIngestResponse)
async def ingest_triples(
    service: KnowledgeGraphService = Depends(get_service),
):
    """Load persisted triples from data/processed_triples/triples.jsonl and
    ingest them into Neo4j using MERGE semantics (no duplicates).
    """
    try:
        triples = service.load_persisted_triples()
        ingested = await service.ingest_triples(triples)
        return KnowledgeGraphIngestResponse(
            loaded_triples=len(triples),
            ingested_triples=ingested,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query", response_model=KnowledgeGraphQueryResponse)
async def query_knowledge_graph(
    cypher: str = Query(..., description="Cypher query to execute"),
    service: KnowledgeGraphService = Depends(get_service),
):
    """Execute an arbitrary Cypher query against the Neo4j knowledge graph."""
    try:
        results = await service.query(cypher)
        return KnowledgeGraphQueryResponse(query=cypher, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
