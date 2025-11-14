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
    