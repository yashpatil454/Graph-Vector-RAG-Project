"""Unit tests for PDFProcessor service.

All file I/O and PDF loading is mocked — no real PDFs required.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.data_processor import PDFProcessor, get_pdf_processor


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def setup_method(self):
        self.processor = PDFProcessor()

    def test_removes_non_breaking_spaces(self):
        result = self.processor._clean_text("Hello\xa0World")
        assert "\xa0" not in result
        assert "Hello World" in result

    def test_collapses_multiple_newlines(self):
        result = self.processor._clean_text("line1\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_replaces_inline_newlines_with_space(self):
        result = self.processor._clean_text("word1\nword2")
        assert "word1 word2" in result

    def test_collapses_multiple_spaces(self):
        result = self.processor._clean_text("hello   world")
        assert "  " not in result

    def test_removes_space_before_punctuation(self):
        result = self.processor._clean_text("Hello , World .")
        assert "Hello," in result
        assert "World." in result

    def test_strips_leading_trailing_whitespace(self):
        result = self.processor._clean_text("  hello  ")
        assert result == "hello"

    def test_passthrough_when_clean_text_disabled(self):
        processor = PDFProcessor()
        processor.clean_text = False
        raw = "Hello   world\n\n\n"
        result = processor._clean_text(raw)
        assert result == raw


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------

class TestSplitDocuments:
    def setup_method(self):
        self.processor = PDFProcessor()

    def test_returns_list_of_documents(self, sample_documents):
        chunks = self.processor.split_documents(sample_documents)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_short_docs_produce_at_least_one_chunk_each(self, sample_documents):
        chunks = self.processor.split_documents(sample_documents)
        assert len(chunks) >= len(sample_documents)

    def test_empty_input_returns_empty(self):
        assert self.processor.split_documents([]) == []

    def test_metadata_preserved_in_chunks(self, sample_documents):
        chunks = self.processor.split_documents(sample_documents)
        sources = {c.metadata.get("source") for c in chunks}
        assert "doc1.pdf" in sources or "doc2.pdf" in sources


# ---------------------------------------------------------------------------
# save_chunks / load_persisted_chunks
# ---------------------------------------------------------------------------

class TestPersistence:
    def setup_method(self):
        self.processor = PDFProcessor()

    def test_roundtrip_jsonl(self, tmp_path, sample_documents):
        self.processor.save_chunks(sample_documents, output_dir=str(tmp_path), format="jsonl")
        loaded = self.processor.load_persisted_chunks(input_dir=str(tmp_path), format="jsonl")
        assert len(loaded) == len(sample_documents)
        contents = {d.page_content for d in loaded}
        assert "AI is transforming medicine and clinical decision-making." in contents

    def test_save_creates_file(self, tmp_path, sample_documents):
        self.processor.save_chunks(sample_documents, output_dir=str(tmp_path), format="jsonl")
        files = list(Path(tmp_path).glob("*.jsonl"))
        assert len(files) == 1

    def test_load_missing_dir_raises(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            self.processor.load_persisted_chunks(input_dir=missing)

    def test_loaded_metadata_preserved(self, tmp_path, sample_documents):
        self.processor.save_chunks(sample_documents, output_dir=str(tmp_path))
        loaded = self.processor.load_persisted_chunks(input_dir=str(tmp_path))
        sources = {d.metadata.get("source") for d in loaded}
        assert "doc1.pdf" in sources


# ---------------------------------------------------------------------------
# load_all_pdfs
# ---------------------------------------------------------------------------

class TestLoadAllPDFs:
    @patch("app.services.data_processor.DirectoryLoader")
    def test_calls_directory_loader(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        processor = PDFProcessor()
        result = processor.load_all_pdfs()
        assert result == []
        mock_loader.load.assert_called_once()

    @patch("app.services.data_processor.DirectoryLoader")
    def test_returns_documents_from_loader(self, mock_loader_cls, sample_documents):
        mock_loader = MagicMock()
        mock_loader.load.return_value = sample_documents
        mock_loader_cls.return_value = mock_loader

        processor = PDFProcessor()
        result = processor.load_all_pdfs()
        assert len(result) == len(sample_documents)


# ---------------------------------------------------------------------------
# process_all_pdfs
# ---------------------------------------------------------------------------

class TestProcessAllPDFs:
    def test_returns_expected_keys(self, sample_documents):
        processor = PDFProcessor()
        with patch.object(processor, "load_all_pdfs_parallel", return_value=sample_documents):
            result = processor.process_all_pdfs(split=True, parallel=True, persist=False)
        assert "total_files" in result
        assert "total_pages" in result
        assert "total_chunks" in result
        assert "processed_at" in result

    def test_total_pages_matches_input(self, sample_documents):
        processor = PDFProcessor()
        with patch.object(processor, "load_all_pdfs_parallel", return_value=sample_documents):
            result = processor.process_all_pdfs(split=True, parallel=True, persist=False)
        assert result["total_pages"] == len(sample_documents)

    def test_persist_false_does_not_write(self, tmp_path, sample_documents):
        processor = PDFProcessor(data_dir=str(tmp_path))
        with patch.object(processor, "load_all_pdfs_parallel", return_value=sample_documents):
            result = processor.process_all_pdfs(persist=False)
        # persist=False sets persisted to False (not True), and no file is written
        assert result.get("persisted") is not True
        assert not list(tmp_path.glob("**/*.jsonl"))

    def test_persist_true_writes_file(self, tmp_path, sample_documents):
        processor = PDFProcessor(data_dir=str(tmp_path))
        with patch.object(processor, "load_all_pdfs_parallel", return_value=sample_documents):
            result = processor.process_all_pdfs(persist=True)
        assert result.get("persisted") is True


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def test_get_pdf_processor_returns_same_instance():
    # Reset singleton for isolation
    import app.services.data_processor as mod
    mod._processor_instance = None
    a = get_pdf_processor()
    b = get_pdf_processor()
    assert a is b
