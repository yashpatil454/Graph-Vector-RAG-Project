"""
PDF Processing Service for Knowledge Graph and Vector RAG
Handles PDF loading, text extraction, chunking, and metadata extraction
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Service for processing PDF documents for RAG applications.
    Handles loading, chunking, and metadata extraction.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_start_index: Optional[bool] = None,
        clean_text: Optional[bool] = None
    ):
        """
        Initialize PDF Processor.

        Args:
            data_dir: Directory containing PDF files (default from config)
            chunk_size: Size of text chunks in characters (default from config)
            chunk_overlap: Overlap between chunks in characters (default from config)
            add_start_index: Whether to track index in original document (default from config)
            clean_text: Whether to clean text (default from config)
        """
        # Use config values as defaults
        self.data_dir = Path(settings.PDF_DATA_DIR)
        self.chunk_size = settings.PDF_CHUNK_SIZE
        self.chunk_overlap = settings.PDF_CHUNK_OVERLAP
        self.add_start_index = settings.PDF_ADD_START_INDEX
        self.clean_text = settings.PDF_CLEAN_TEXT

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(
            f"PDFProcessor initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, clean_text={self.clean_text}"
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing whitespace.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not self.clean_text:
            return text

        # Replace non-breaking spaces (\xa0) with regular spaces
        text = text.replace('\xa0', ' ')
        
        # Replace multiple newlines with double newline (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace single newlines that are likely line breaks within sentences
        # Keep newlines that separate paragraphs (double newlines)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final cleanup: remove any remaining multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()


    def load_all_pdfs(self, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """Load all PDF files from the data directory (sequential).

        Prefer using `load_all_pdfs_parallel` for large batches.
        """
        try:
            loader = DirectoryLoader(
                str(self.data_dir),
                glob=glob_pattern,
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()

            if self.clean_text:
                for doc in documents:
                    doc.page_content = self._clean_text(doc.page_content)

            logger.info(f"Loaded {len(documents)} total pages from {self.data_dir}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDFs from directory: {str(e)}")
            raise

    @staticmethod
    def _clean_text_worker(text: str) -> str:
        """Static version of text cleaning for parallel workers."""
        # Replace non-breaking spaces (\xa0) with regular spaces
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def _load_single_pdf_worker(file_path: str, clean_text: bool) -> List[Document]:
        """Worker function to load and optionally clean a single PDF (per process)."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if clean_text:
            for doc in documents:
                doc.page_content = PDFProcessor._clean_text_worker(doc.page_content)
        return documents

    def load_all_pdfs_parallel(
        self,
        glob_pattern: str = "**/*.pdf",
        max_workers: Optional[int] = None
    ) -> List[Document]:
        """Load all PDF files using multiple processes.
        Args:
            glob_pattern: Glob pattern for PDF discovery.
            max_workers: Override number of processes (defaults to CPU count or number of files).
        Returns:
            List[Document]: All page Documents across all PDFs.
        """
        # Collect file paths matching glob
        pdf_files = [str(p) for p in Path(self.data_dir).rglob("*.pdf") if p.is_file()]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return []

        if max_workers is None:
            cpu_cnt = os.cpu_count() or 1
            max_workers = min(len(pdf_files), cpu_cnt)

        logger.info(
            f"Starting parallel load of {len(pdf_files)} PDFs with {max_workers} workers"
        )

        all_documents: List[Document] = []
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        PDFProcessor._load_single_pdf_worker, file_path, self.clean_text
                    ): file_path for file_path in pdf_files
                }
                for future in as_completed(future_map):
                    file_path = future_map[future]
                    try:
                        docs = future.result()
                        all_documents.extend(docs)
                        logger.debug(
                            f"Loaded {len(docs)} pages from {Path(file_path).name} (parallel)"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
        except Exception as e:
            logger.error(f"Parallel PDF loading encountered an error: {e}")
            raise

        logger.info(
            f"Parallel loading complete: {len(all_documents)} total pages from {len(pdf_files)} PDFs"
        )
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of Document objects to split

        Returns:
            List of split Document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)

            logger.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise

    # ------------------------ Persistence Helpers ------------------------ #

    def save_chunks(
        self,
        chunks: List[Document],
        output_dir: str,
        format: str = "jsonl"
    ) -> None:
        """Persist chunk documents to disk.

        Args:
            chunks: List of chunk Documents
            output_dir: Directory where chunks will be stored
            format: Persistence format (only 'jsonl' supported currently)
        """
        if format.lower() != "jsonl":
            raise ValueError("Only 'jsonl' format is currently supported.")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Persisting {len(chunks)} chunks to {output_dir}")
        target_path = Path(output_dir) / "chunks.jsonl"
        tmp_path = target_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for d in chunks:
                record = {"page_content": d.page_content, "metadata": d.metadata}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(target_path)
        logger.debug(f"Wrote chunks file: {target_path}")

        logger.info("Chunk persistence complete.")

    def load_persisted_chunks(
        self,
        input_dir: str = "data/processed_chunks",
        format: str = "jsonl"
    ) -> List[Document]:
        """Load persisted chunk documents from disk.

        Args:
            input_dir: Directory containing persisted chunk files
            format: Persistence format used (only 'jsonl')

        Returns:
            List[Document]: Reconstructed chunk documents
        """
        if format.lower() != "jsonl":
            raise ValueError("Only 'jsonl' format is currently supported.")

        path_obj = Path(input_dir)
        if not path_obj.exists():
            raise FileNotFoundError(f"Persistence directory not found: {input_dir}")

        target = path_obj / "chunks.jsonl"
        if not target.exists():
            logger.warning(f"Chunks file missing in {input_dir}")
            return []
        documents: List[Document] = []
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    documents.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata", {})))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line in {target.name}: {e}")
        logger.info(f"Loaded {len(documents)} persisted chunks from {target}")
        return documents

    def process_all_pdfs(
        self,
        split: bool = True,
        glob_pattern: str = "**/*.pdf",
        parallel: bool = True,
        max_workers: Optional[int] = None,
        persist: bool = True,
        persist_dir: Optional[str] = None,
        persist_format: str = "jsonl"
    ) -> Dict[str, Any]:
        """
        Process all PDF files in the data directory.

        Args:
            split: Whether to split documents into chunks
            glob_pattern: Pattern to match PDF files
            parallel: Use parallel loading
            max_workers: Number of processes for parallel mode
            persist: Whether to persist chunks to disk
            persist_dir: Target directory for persistence (created if missing)
            persist_format: Currently only 'jsonl' supported

        Returns:
            Dictionary containing all processed documents and metadata
        """
        try:
            # Load all PDFs (parallel or sequential)
            if parallel:
                documents = self.load_all_pdfs_parallel(glob_pattern=glob_pattern, max_workers=max_workers)
            else:
                documents = self.load_all_pdfs(glob_pattern)

            if not documents:
                logger.warning(f"No PDF files found in {self.data_dir}")
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "chunks": [],
                    "processed_at": datetime.now().isoformat()
                }

            # Split if requested
            chunks = self.split_documents(documents) if split else documents

            # Get list of processed files
            processed_files = list(set(doc.metadata.get("source", "") for doc in documents))

            result = {
                "total_files": len(processed_files),
                "total_pages": len(documents),
                "total_chunks": len(chunks),
                "processed_files": processed_files,
                "chunks": chunks,
                "processed_at": datetime.now().isoformat(),
                "persisted": False,
                "persist_dir": None,
                "persist_error": None,
            }

            # Persist if requested
            if persist:
                try:
                    self.save_chunks(
                        chunks,
                        output_dir=persist_dir or os.path.join(str(self.data_dir), "processed_chunks"),
                        format=persist_format
                    )
                    result["persisted"] = True
                    result["persist_dir"] = persist_dir or os.path.join(str(self.data_dir), "processed_chunks")
                except Exception as e:
                    logger.error(f"Failed to persist chunks: {e}")
                    result["persisted"] = False
                    result["persist_error"] = str(e)

            mode = "parallel" if parallel else "sequential"
            logger.info(
                f"Successfully processed {len(processed_files)} PDFs ({mode}): "
                f"{len(documents)} pages → {len(chunks)} chunks"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing all PDFs: {str(e)}")
            raise

# Singleton instance for easy access
_processor_instance: Optional[PDFProcessor] = None


def get_pdf_processor(
    data_dir: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    clean_text: Optional[bool] = None
) -> PDFProcessor:
    """Get or create a singleton PDFProcessor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = PDFProcessor(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_text=clean_text
        )
    return _processor_instance
