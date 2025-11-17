from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

from app.services.knowledge_graph_service import get_knowledge_graph_service, KnowledgeGraphService
from app.models.request_models import KnowledgeGraphBuildResponse, KnowledgeGraphQueryResponse

router = APIRouter(prefix="/knowledge_graph", tags=["knowledge_graph"])

async def get_service() -> KnowledgeGraphService:
    return await get_knowledge_graph_service()

@router.post("/build", response_model=KnowledgeGraphBuildResponse)
async def build_knowledge_graph(service: KnowledgeGraphService = Depends(get_service)):
    try:
        summary = await service.build_graph()
        return KnowledgeGraphBuildResponse(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query", response_model=KnowledgeGraphQueryResponse)
async def query_knowledge_graph(
    cypher: str = Query(..., description="Cypher query to execute"),
    service: KnowledgeGraphService = Depends(get_service)
):
    try:
        results = await service.query(cypher)
        return KnowledgeGraphQueryResponse(query=cypher, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
