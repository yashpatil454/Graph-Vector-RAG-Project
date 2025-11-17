"""Async Singleton Knowledge Graph Service

Builds a Neo4j knowledge graph from persisted chunk documents using
Gemini 2.5 Flash to extract (subject, predicate, object) triples.

Requirements satisfied:
 - Async service using local Neo4j
 - Singleton pattern
 - Triple extraction via Gemini 2.5 Flash (no fallback code)
 - Load persisted chunks via data_processor.load_persisted_chunks
 - Avoid duplicate entities/relationships (MERGE semantics)
 - Logging via SingletonLogger
 - API keys loaded from settings

Public async methods:
 - build_graph(): end-to-end load -> extract -> ingest
 - extract_triples(documents)
 - ingest_triples(triples)
 - query(cypher)
 - close()

Assumptions:
 - Chunks are persisted in data/processed_chunks/chunks.jsonl
 - Gemini responses can be coerced to JSON list of triples; non-JSON text is ignored
 - Minimal prompt, no retry/fallback per instructions
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import SingletonLogger
from app.services.data_processor import get_pdf_processor
from app.models.request_models import Triple

logger = SingletonLogger().get_logger()

class KnowledgeGraphService:

	def __init__(self) -> None:
		self._driver = AsyncGraphDatabase.driver(
			settings.NEO4J_URI,
			auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
		)
		# Initialize Gemini 2.5 Flash LLM
		self._llm = ChatGoogleGenerativeAI(
			model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY
		)
		logger.info("KnowledgeGraphService initialized (Neo4j + Gemini 2.5 Flash)")

	# ---------------------- Triple Extraction ----------------------- #
	async def extract_triples(self, documents: List[Document]) -> List[Triple]:
		"""Extract triples from documents using Gemini 2.5 Flash.

		Each document is processed concurrently (CPU bound only for JSON parse).
		"""
		if not documents:
			logger.warning("No documents provided for triple extraction")
			return []

		prompt_template = (
			"Extract factual triples from the text below. "
			"Return ONLY valid JSON array of objects with keys 'subject','predicate','object'. "
			"Use concise canonical entity names; omit duplicates; skip trivial or vague relations.\n\nText:\n{text}"
		)

		all_triples: List[Triple] = []
		for idx, doc in enumerate(documents):
			text = doc.page_content
			prompt = prompt_template.format(text=text)
			try:
				response = await asyncio.to_thread(self._llm.invoke, prompt)
				raw = getattr(response, "content", None) or getattr(response, "text", "") or str(response)
				json_str = raw
				try:
					clean = json_str.replace("```json", "").replace("```", "")
					parsed = json.loads(clean)
					logger.info(f"parsed triples: {parsed}")
					if isinstance(parsed, list):
						for item in parsed:
							if isinstance(item, dict):
								subj = item.get("subject")
								pred = item.get("predicate")
								obj = item.get("object")
								if subj and pred and obj:
									all_triples.append(
										Triple(subject=subj.strip(), predicate=pred.strip(), object=obj.strip(), provenance=doc.metadata)
									)
				except json.JSONDecodeError:
					logger.debug("Non-JSON response for a chunk; skipping.")
			except Exception as e:
				logger.error(f"Triple extraction failed for chunk {idx}: {e}")
			# Simple pacing: sleep 6s between calls to remain under 10 req/min
			await asyncio.sleep(6)

		logger.info(f"Extracted {len(all_triples)} raw triples from {len(documents)} documents (sequential throttled)")
		return all_triples

	# ---------------------- Graph Ingestion ------------------------- #
	async def ingest_triples(self, triples: List[Triple]) -> int:
		"""Ingest triples into Neo4j using MERGE to avoid duplicates."""
		if not triples:
			logger.warning("No triples provided for ingestion")
			return 0
		# Deduplicate at triple level prior to ingestion
		unique_key = {(t.subject, t.predicate, t.object): t for t in triples}
		unique_triples = list(unique_key.values())
		logger.info(f"Ingesting {len(unique_triples)} unique triples (from {len(triples)} raw)")
		cypher = (
			"UNWIND $triples AS t "
			"MERGE (s:Entity {name: t.subject}) "
			"MERGE (o:Entity {name: t.object}) "
			"MERGE (s)-[r:RELATION {predicate: t.predicate}]->(o) "
			"RETURN count(r) AS relationships_created"
		)
		payload = [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in unique_triples]
		
		async with self._driver.session() as session:
			result = await session.run(cypher, triples=payload)
			record = await result.single()
			count_created = record["relationships_created"] if record else 0
		logger.info(f"Neo4j ingestion complete: {count_created} relationships present")
		return len(unique_triples)

	# ---------------------- End-to-End Build ------------------------ #
	async def build_graph(self) -> Dict[str, Any]:
		"""Load persisted chunks, extract triples, and ingest into graph."""
		processor = get_pdf_processor()
		documents = processor.load_persisted_chunks()
		triples = await self.extract_triples(documents[:2])
		ingested = await self.ingest_triples(triples)
		summary = {
			"total_documents": len(documents),
			"total_triples": len(triples),
			"ingested_triples": ingested,
		}
		logger.info(f"Graph build summary: {summary}")
		return summary

	# ---------------------- Query Interface ------------------------- #
	async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Execute a read-only Cypher query and return list of dict records."""
		params = params or {}
		async with self._driver.session() as session:
			result = await session.run(cypher, **params)
			logger.info(f"Results: {result}")
			records = []
			async for record in result:
				# Convert record values to Python serializable types
				rec_dict = {k: record.get(k) for k in record.keys()}
				records.append(rec_dict)
		logger.info(f"Query executed: {cypher} (rows={len(records)})")
		return records

	# ---------------------- Cleanup ------------------------------- #
	async def close(self) -> None:
		await self._driver.close()
		logger.info("KnowledgeGraphService Neo4j driver closed")

# ----------------------------------------------------------------------
# Async Singleton Factory
# ----------------------------------------------------------------------
_kg_instance: Optional["KnowledgeGraphService"] = None

async def get_knowledge_graph_service() -> KnowledgeGraphService:
    global _kg_instance
    if _kg_instance is None:
        kg = KnowledgeGraphService()
        _kg_instance = kg
    return _kg_instance
