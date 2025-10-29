"""
PDF Processing Service for Knowledge Graph and Vector RAG
Handles PDF loading, text extraction, chunking, and metadata extraction
"""

import os
import re
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

    # def load_single_pdf(self, file_path: str) -> List[Document]:
    #     """
    #     Load a single PDF file and return document objects.

    #     Args:
    #         file_path: Path to PDF file

    #     Returns:
    #         List of Document objects (one per page)
    #     """
    #     try:
    #         loader = PyPDFLoader(file_path)
    #         documents = loader.load()

    #         # Clean text if enabled
    #         if self.clean_text:
    #             for doc in documents:
    #                 doc.page_content = self._clean_text(doc.page_content)

    #         logger.info(
    #             f"Loaded {len(documents)} pages from {Path(file_path).name}"
    #         )

    #         return documents

    #     except Exception as e:
    #         logger.error(f"Error loading PDF {file_path}: {str(e)}")
    #         raise

    def load_all_pdfs(self, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """
        Load all PDF files from the data directory.

        Args:
            glob_pattern: Pattern to match PDF files

        Returns:
            List of all Document objects from all PDFs
        """
        try:
            loader = DirectoryLoader(
                str(self.data_dir),
                glob=glob_pattern,
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()

            # Clean text if enabled
            if self.clean_text:
                for doc in documents:
                    doc.page_content = self._clean_text(doc.page_content)

            logger.info(
                f"Loaded {len(documents)} total pages from {self.data_dir}"
            )

            return documents

        except Exception as e:
            logger.error(f"Error loading PDFs from directory: {str(e)}")
            raise

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

    # def process_pdf(
    #     self,
    #     file_path: str,
    #     split: bool = True
    # ) -> Dict[str, Any]:
    #     """
    #     Process a single PDF file: load and optionally split.

    #     Args:
    #         file_path: Path to PDF file
    #         split: Whether to split into chunks

    #     Returns:
    #         Dictionary containing documents/chunks and metadata
    #     """
    #     try:
    #         # Load PDF
    #         documents = self.load_single_pdf(file_path)

    #         # Split if requested
    #         chunks = self.split_documents(documents) if split else documents

    #         # Extract metadata
    #         metadata = self._extract_metadata(file_path, documents, chunks)

    #         result = {
    #             "file_path": file_path,
    #             "file_name": Path(file_path).name,
    #             "total_pages": len(documents),
    #             "total_chunks": len(chunks),
    #             "documents": documents if not split else None,
    #             "chunks": chunks,
    #             "metadata": metadata,
    #             "processed_at": datetime.now().isoformat()
    #         }

    #         logger.info(
    #             f"Successfully processed {Path(file_path).name}: "
    #             f"{len(documents)} pages → {len(chunks)} chunks"
    #         )

    #         return result

    #     except Exception as e:
    #         logger.error(f"Error processing PDF: {str(e)}")
    #         raise

    def process_all_pdfs(
        self,
        split: bool = True,
        glob_pattern: str = "**/*.pdf"
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
            # Load all PDFs
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

            logger.info(
                f"Successfully processed {len(processed_files)} PDFs: "
                f"{len(documents)} pages → {len(chunks)} chunks"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing all PDFs: {str(e)}")
            raise

    # def _extract_metadata(
    #     self,
    #     file_path: str,
    #     documents: List[Document],
    #     chunks: List[Document]
    # ) -> Dict[str, Any]:
    #     """
    #     Extract metadata from processed documents.

    #     Args:
    #         file_path: Path to PDF file
    #         documents: Original documents (pages)
    #         chunks: Split chunks

    #     Returns:
    #         Dictionary of metadata
    #     """
    #     file_path_obj = Path(file_path)

    #     # Calculate basic statistics
    #     total_chars = sum(len(doc.page_content) for doc in documents)
    #     avg_chars_per_page = total_chars / len(documents) if documents else 0
    #     avg_chars_per_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0

    #     metadata = {
    #         "file_name": file_path_obj.name,
    #         "file_size_bytes": file_path_obj.stat().st_size if file_path_obj.exists() else 0,
    #         "total_pages": len(documents),
    #         "total_chunks": len(chunks),
    #         "total_characters": total_chars,
    #         "avg_chars_per_page": round(avg_chars_per_page, 2),
    #         "avg_chars_per_chunk": round(avg_chars_per_chunk, 2),
    #         "chunk_size": self.chunk_size,
    #         "chunk_overlap": self.chunk_overlap,
    #     }

    #     return metadata

    # def get_chunk_by_index(
    #     self,
    #     chunks: List[Document],
    #     index: int
    # ) -> Optional[Document]:
    #     """
    #     Get a specific chunk by index.

    #     Args:
    #         chunks: List of document chunks
    #         index: Index of chunk to retrieve

    #     Returns:
    #         Document chunk or None if index out of range
    #     """
    #     if 0 <= index < len(chunks):
    #         return chunks[index]
    #     else:
    #         logger.warning(f"Chunk index {index} out of range (0-{len(chunks)-1})")
    #         return None

    # def extract_text_content(self, documents: List[Document]) -> str:
    #     """
    #     Extract all text content from documents.

    #     Args:
    #         documents: List of Document objects

    #     Returns:
    #         Combined text content
    #     """
    #     return "\n\n".join(doc.page_content for doc in documents)

    # def get_chunks_with_metadata(
    #     self,
    #     chunks: List[Document]
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Get chunks with their metadata in a structured format.

    #     Args:
    #         chunks: List of document chunks

    #     Returns:
    #         List of dictionaries with chunk content and metadata
    #     """
    #     result = []
    #     for i, chunk in enumerate(chunks):
    #         result.append({
    #             "chunk_id": i,
    #             "content": chunk.page_content,
    #             "metadata": chunk.metadata,
    #             "char_count": len(chunk.page_content)
    #         })

    #     return result


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
