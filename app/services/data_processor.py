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

    def process_all_pdfs(
        self,
        split: bool = True,
        glob_pattern: str = "**/*.pdf",
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all PDF files in the data directory.

        Args:
            split: Whether to split documents into chunks
            glob_pattern: Pattern to match PDF files

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
                "processed_at": datetime.now().isoformat()
            }

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
    """
    Get or create a singleton PDFProcessor instance.

    Args:
        data_dir: Directory containing PDF files (default from config)
        chunk_size: Size of text chunks (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        clean_text: Whether to clean text (default from config)

    Returns:
        PDFProcessor instance
    """
    global _processor_instance

    if _processor_instance is None:
        _processor_instance = PDFProcessor(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_text=clean_text
        )

    return _processor_instance
