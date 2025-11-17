from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict


class PDFProcessRequest(BaseModel):
	split: bool = Field(default=True, description="Whether to split pages into chunks")
	parallel: bool = Field(default=True, description="Use parallel processing")
	glob_pattern: str = Field(default="**/*.pdf", description="Glob pattern for PDFs")
	max_workers: Optional[int] = Field(default=None, description="Override worker count")
	persist: bool = Field(default=True, description="Persist chunks to disk")
	persist_dir: Optional[str] = Field(default=None, description="Directory to store persisted chunks")
	persist_format: str = Field(default="jsonl", description="Persistence format")

class PDFProcessResponse(BaseModel):
	total_files: int
	total_pages: int
	total_chunks: int
	processed_files: List[str]
	processed_at: str
	persisted: Optional[bool] = Field(default=None, description="Whether chunks were persisted")
	persist_dir: Optional[str] = Field(default=None, description="Directory where chunks were persisted")
	persist_error: Optional[str] = Field(default=None, description="Error message if persistence failed")

class VectorStoreRequest(BaseModel):
	embedding_provider: str = Field(default="gemini", description="Embedding provider to use")
	persist_dir: str = Field(default="vector_store", description="Directory to store persisted chunks")
	use_cache: bool = Field(default=True, description="Whether to use caching")
	gemini_model: str = Field(default="models/gemini-embedding-001", description="Gemini model to use")
	
class VectorStoreResponse(BaseModel):
    len_documents: int
	
@dataclass
class Entity:
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    canonical: Optional[str] = None


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    provenance: Optional[Dict] = None

class KnowledgeGraphBuildResponse(BaseModel):
	total_documents: int = Field(description="Number of documents processed for triple extraction")
	total_triples: int = Field(description="Total triples extracted before deduplication")
	ingested_triples: int = Field(description="Number of unique triples ingested into the graph")

class KnowledgeGraphQueryResponse(BaseModel):
	query: str = Field(description="Cypher query executed")
	results: List[Dict[str, Any]] = Field(description="Raw result records returned from Neo4j")

class FusionRequest(BaseModel):
	query: str = Field(description="Natural language user query driving similarity search")
	k: int = Field(default=4, description="Top-k vector hits to retrieve")
	cypher: Optional[str] = Field(default=None, description="Optional explicit Cypher query; if omitted auto-generated from vector hits")
	include_scores: bool = Field(default=True, description="Include similarity scores in output context")

class FusionResponse(BaseModel):
	query: str
	k: int
	vector_hits_count: int
	vector_hits: List[Dict[str, Any]]
	graph_results_count: int
	graph_results: List[Dict[str, Any]]
	context: str = Field(description="Fused context prepared for LLM consumption")
    