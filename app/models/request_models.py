from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PDFProcessRequest(BaseModel):
	split: bool = Field(default=True, description="Whether to split pages into chunks")
	parallel: bool = Field(default=True, description="Use parallel processing")
	glob_pattern: str = Field(default="**/*.pdf", description="Glob pattern for PDFs")
	max_workers: Optional[int] = Field(default=None, description="Override worker count")

class PDFProcessResponse(BaseModel):
	total_files: int
	total_pages: int
	total_chunks: int
	processed_files: List[str]
	processed_at: str

