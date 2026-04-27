"""Hybrid Fusion Service
Combines vector similarity search results with Neo4j knowledge graph query results
into a unified context for downstream LLM consumption.
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from app.core.logger import SingletonLogger
from app.services.vector_store_service import get_vector_store_service, VectorStoreService
from app.services.knowledge_graph_service import get_knowledge_graph_service, KnowledgeGraphService
from app.models.request_models import FusionRequest

logger = SingletonLogger().get_logger()

_fusion_instance: Optional["HybridFusionService"] = None

class HybridFusionService:
    def __init__(self) -> None:
        logger.info("HybridFusionService initialized")
        self._kg_service: Optional[KnowledgeGraphService] = None
        self._vector_service: Optional[VectorStoreService] = None

    @classmethod
    async def get_instance(cls) -> "HybridFusionService":
        global _fusion_instance
        if _fusion_instance is None:
            _fusion_instance = HybridFusionService()
        return _fusion_instance

    async def fuse(self, req: FusionRequest) -> Dict[str, Any]:
        # Acquire dependencies (cache locally for optional close)
        if self._vector_service is None:
            self._vector_service = await get_vector_store_service()
        if self._kg_service is None:
            self._kg_service = await get_knowledge_graph_service()
        vector_service = self._vector_service
        kg_service = self._kg_service

        # Vector similarity search with scores
        hits_with_scores = await vector_service.similarity_search_with_score(req.query, k=req.k)
        vector_hits: List[Dict[str, Any]] = []
        for doc, score in hits_with_scores:
            vector_hits.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        logger.info(f"HybridFusion: Retrieved {len(vector_hits)} vector hits for query '{req.query}'")
        logger.info(f"HybridFusion: Retrieved {vector_hits}")
        # Extract entity candidate names from vector hits (heuristic)
        raw_candidates = self._extract_entity_candidates(vector_hits)
        # Intersect with existing KG entity names for precision
        kg_entities = await self._list_existing_entities(kg_service, limit=500)
        logger.info(f"HybridFusion: Existing Retrieved {kg_entities}")
        # entity_candidates = [c for c in raw_candidates if c in kg_entities]
        entity_candidates = kg_entities
        logger.info(f"HybridFusion: ieved {entity_candidates}")
        logger.info(
            f"HybridFusion: Derived {len(raw_candidates)} raw candidates; {len(entity_candidates)} matched existing KG entities"
        )
        if not entity_candidates:
            logger.warning("HybridFusion: No matched KG entities; graph query may return 0 rows")

        # Build Cypher if not provided
        if req.cypher:
            cypher = req.cypher
            params = {}
            logger.info("HybridFusion: Using user-supplied Cypher query")
        else:
            cypher = (
                "WITH $names AS names "
                "UNWIND names AS n "
                "MATCH (e:Entity {name: n})-[r:RELATION]->(o:Entity) "
                "RETURN e.name AS subject, r.predicate AS predicate, o.name AS object LIMIT 50"
            )
            params = {"names": entity_candidates}
            logger.info("HybridFusion: Using auto-generated Cypher from entity candidates")
        graph_results = await kg_service.query(cypher, params)
        logger.info(f"HybridFusion: Retrieved {len(graph_results)} graph rows")

        # Build fused context string
        context = self._build_context(req, vector_hits, graph_results)
        logger.info("HybridFusion: Context assembled for LLM")

        return {
            "query": req.query,
            "k": req.k,
            "vector_hits_count": len(vector_hits),
            "vector_hits": vector_hits,
            "graph_results_count": len(graph_results),
            "graph_results": graph_results,
            "context": context,
        }

    def _extract_entity_candidates(self, vector_hits: List[Dict[str, Any]], max_entities: int = 30) -> List[str]:
        text_concat = " \n ".join(hit["content"][:3000] for hit in vector_hits)  # limit size
        # Heuristic: words starting with uppercase, length>2, strip punctuation
        raw_tokens = re.findall(r"[A-Z][A-Za-z0-9_-]{2,}", text_concat)
        # Deduplicate preserving order
        seen = set()
        entities = []
        for tok in raw_tokens:
            if tok not in seen:
                seen.add(tok)
                entities.append(tok)
            if len(entities) >= max_entities:
                break
        return entities

    async def _list_existing_entities(self, kg_service: KnowledgeGraphService, limit: int = 1000) -> List[str]:
        cypher = "MATCH (e:Entity) RETURN e.name AS name LIMIT $limit"
        rows = await kg_service.query(cypher, {"limit": limit})
        names = [row.get("name") for row in rows if row.get("name")]
        return names

    def _build_context(self, req: FusionRequest, vector_hits: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append(f"User Query: {req.query}\n")
        lines.append("=== Vector Similarity Hits ===")
        for i, hit in enumerate(vector_hits, start=1):
            snippet = hit["content"][:400].replace("\n", " ")
            score_part = f" (score={hit['score']:.4f})" if req.include_scores and 'score' in hit else ""
            lines.append(f"[{i}] {snippet}{score_part}")
        lines.append("\n=== Knowledge Graph Relationships ===")
        for row in graph_results[:50]:
            subj = row.get("subject") or row.get("s") or row.get("s.name")
            pred = row.get("predicate") or row.get("r.predicate")
            obj = row.get("object") or row.get("o") or row.get("o.name")
            if subj and pred and obj:
                lines.append(f"{subj} -[{pred}]-> {obj}")
        lines.append("\n=== Guidance ===")
        lines.append("Use the entities and relationships above along with semantic snippets to answer the user query accurately.")
        return "\n".join(lines)

async def get_hybrid_fusion_service() -> HybridFusionService:
    return await HybridFusionService.get_instance()

async def close_hybrid_fusion_service():
    if _fusion_instance and _fusion_instance._kg_service:
        await _fusion_instance._kg_service.close()
        logger.info("HybridFusionService closed underlying KG service")
