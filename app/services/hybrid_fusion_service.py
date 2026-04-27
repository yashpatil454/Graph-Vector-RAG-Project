"""Hybrid Fusion Service
Combines vector similarity search results with Neo4j knowledge graph query results
into a unified context for downstream LLM consumption.

Improvements over v1:
 - Vector search + KG entity fetch run in parallel via asyncio.gather
 - Entity candidates intersected with KG entities (query-focused graph retrieval)
 - Optional 2-hop graph traversal for richer relational context
 - Score threshold filter to drop low-quality vector hits
 - Full source citations (PDF file + page) in context
 - Longer snippets (800 chars) for better LLM comprehension
 - Deduplication of graph triples before context assembly
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple

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
        # Acquire service singletons
        if self._vector_service is None:
            self._vector_service = await get_vector_store_service()
        if self._kg_service is None:
            self._kg_service = await get_knowledge_graph_service()
        vector_service = self._vector_service
        kg_service = self._kg_service

        # --- Step 1: Run vector search + KG entity fetch in parallel ---
        hits_with_scores, kg_entity_names = await asyncio.gather(
            vector_service.similarity_search_with_score(req.query, k=req.k),
            self._list_existing_entities(kg_service),
        )

        # --- Step 2: Build vector hits, apply score threshold filter ---
        # Cast score to Python float — FAISS returns numpy.float32 which Pydantic cannot serialize
        all_vector_hits: List[Dict[str, Any]] = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": float(score)}
            for doc, score in hits_with_scores
        ]
        vector_hits = (
            [h for h in all_vector_hits if h["score"] >= req.score_threshold]
            if req.score_threshold > 0.0
            else all_vector_hits
        )
        logger.info(
            f"HybridFusion: {len(all_vector_hits)} raw hits → {len(vector_hits)} after "
            f"score_threshold={req.score_threshold} for query '{req.query}'"
        )

        # --- Step 3: Entity intersection — query-focused graph retrieval ---
        raw_candidates = self._extract_entity_candidates(vector_hits)
        kg_entity_set = set(kg_entity_names)
        entity_candidates = [c for c in raw_candidates if c in kg_entity_set]
        # Fallback: if intersection is empty, use top-30 KG entities
        if not entity_candidates:
            entity_candidates = kg_entity_names[:30]
            logger.warning(
                f"HybridFusion: No intersection between {len(raw_candidates)} raw candidates "
                f"and KG entities; falling back to top-30 KG entities"
            )
        logger.info(
            f"HybridFusion: {len(raw_candidates)} raw candidates → "
            f"{len(entity_candidates)} matched KG entities"
        )

        # --- Step 4: Build and run Cypher query ---
        if req.cypher:
            cypher = req.cypher
            params: Dict[str, Any] = {}
            logger.info("HybridFusion: Using user-supplied Cypher query")
        elif req.two_hop:
            cypher = (
                "WITH $names AS names "
                "UNWIND names AS n "
                "MATCH (e:Entity {name: n})-[r1:RELATION]->(m:Entity)-[r2:RELATION]->(o:Entity) "
                "RETURN e.name AS subject, "
                "       r1.predicate + ' → ' + r2.predicate AS predicate, "
                "       o.name AS object "
                "LIMIT 80"
            )
            params = {"names": entity_candidates}
            logger.info("HybridFusion: Using 2-hop auto-generated Cypher")
        else:
            cypher = (
                "WITH $names AS names "
                "UNWIND names AS n "
                "MATCH (e:Entity {name: n})-[r:RELATION]->(o:Entity) "
                "RETURN e.name AS subject, r.predicate AS predicate, o.name AS object "
                "LIMIT 50"
            )
            params = {"names": entity_candidates}
            logger.info("HybridFusion: Using 1-hop auto-generated Cypher")

        graph_results = await kg_service.query(cypher, params)

        # Deduplicate graph triples
        seen_triples: set = set()
        deduped_graph: List[Dict[str, Any]] = []
        for row in graph_results:
            key = (row.get("subject"), row.get("predicate"), row.get("object"))
            if None not in key and key not in seen_triples:
                seen_triples.add(key)
                deduped_graph.append(row)
        logger.info(
            f"HybridFusion: {len(graph_results)} graph rows → {len(deduped_graph)} after deduplication"
        )

        # --- Step 5: Assemble fused context ---
        context = self._build_context(req, vector_hits, deduped_graph)
        logger.info("HybridFusion: Context assembled for LLM")

        return {
            "query": req.query,
            "k": req.k,
            "vector_hits_count": len(vector_hits),
            "vector_hits": vector_hits,
            "graph_results_count": len(deduped_graph),
            "graph_results": deduped_graph,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_entity_candidates(
        self, vector_hits: List[Dict[str, Any]], max_entities: int = 50
    ) -> List[str]:
        """Extract capitalised tokens from vector hit text as entity candidates."""
        text_concat = " \n ".join(hit["content"][:3000] for hit in vector_hits)
        raw_tokens = re.findall(r"[A-Z][A-Za-z0-9_-]{2,}", text_concat)
        seen: set = set()
        entities: List[str] = []
        for tok in raw_tokens:
            if tok not in seen:
                seen.add(tok)
                entities.append(tok)
            if len(entities) >= max_entities:
                break
        return entities

    async def _list_existing_entities(
        self, kg_service: KnowledgeGraphService, limit: int = 1000
    ) -> List[str]:
        """Fetch all entity names currently stored in Neo4j."""
        cypher = "MATCH (e:Entity) RETURN e.name AS name LIMIT $limit"
        rows = await kg_service.query(cypher, {"limit": limit})
        return [row.get("name") for row in rows if row.get("name")]

    def _build_context(
        self,
        req: FusionRequest,
        vector_hits: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> str:
        """Assemble a structured context string ready for LLM injection."""
        lines: List[str] = []
        lines.append(f"User Query: {req.query}\n")

        # Vector hits — longer snippets with source citations
        lines.append("=== Relevant Document Passages ===")
        for i, hit in enumerate(vector_hits, start=1):
            snippet = hit["content"][:800].replace("\n", " ")
            score_part = f" | score={hit['score']:.4f}" if req.include_scores and "score" in hit else ""
            source = hit["metadata"].get("source", "unknown")
            page = hit["metadata"].get("page", "?")
            # Shorten path to just filename for readability
            source_name = source.split("\\")[-1].split("/")[-1]
            lines.append(f"[{i}] (file: {source_name}, page {page}{score_part})")
            lines.append(f"    {snippet}")

        # Knowledge graph triples
        lines.append("\n=== Knowledge Graph Relationships ===")
        if graph_results:
            for row in graph_results:
                subj = row.get("subject") or row.get("s") or row.get("s.name")
                pred = row.get("predicate") or row.get("r.predicate")
                obj = row.get("object") or row.get("o") or row.get("o.name")
                if subj and pred and obj:
                    lines.append(f"  {subj} -[{pred}]-> {obj}")
        else:
            lines.append("  (no graph relationships found for this query)")

        # Guidance block for the LLM
        lines.append("\n=== Instructions ===")
        lines.append(
            "Answer the user query using the document passages and knowledge graph relationships above. "
            "Cite the source file and page number when referencing a passage. "
            "If the answer cannot be determined from the provided context, say so explicitly."
        )
        return "\n".join(lines)


async def get_hybrid_fusion_service() -> HybridFusionService:
    return await HybridFusionService.get_instance()


async def close_hybrid_fusion_service():
    if _fusion_instance and _fusion_instance._kg_service:
        await _fusion_instance._kg_service.close()
        logger.info("HybridFusionService closed underlying KG service")
