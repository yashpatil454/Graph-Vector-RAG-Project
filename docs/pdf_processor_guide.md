# PDF Processor Service Documentation

## Overview
The PDF Processor Service is a comprehensive solution for processing PDF documents in a Knowledge Graph and Vector RAG (Retrieval-Augmented Generation) application. It leverages LangChain's document loaders and text splitters to efficiently extract, chunk, and prepare PDF content for downstream tasks.

## Features

- ✅ **PDF Loading**: Load single or multiple PDF files
- ✅ **Text Extraction**: Extract text content from PDFs page by page
- ✅ **Intelligent Chunking**: Split documents using RecursiveCharacterTextSplitter
- ✅ **Metadata Tracking**: Preserve source, page numbers, and chunk positions
- ✅ **Configurable**: Customize chunk size, overlap, and processing options
- ✅ **Batch Processing**: Process entire directories of PDFs
- ✅ **Statistics**: Extract detailed metadata and statistics

## Installation

Required dependencies (already in `requirements.txt`):

```bash
pip install langchain langchain-community langchain-text-splitters pypdf
```

## Quick Start

### Basic Usage

```python
from app.services.data_processor import PDFProcessor

# Initialize processor
processor = PDFProcessor(
    data_dir="data",
    chunk_size=1000,
    chunk_overlap=200
)

# Process a single PDF
result = processor.process_pdf("data/document.pdf", split=True)

print(f"Pages: {result['total_pages']}")
print(f"Chunks: {result['total_chunks']}")
```

### Process All PDFs

```python
# Process all PDFs in the data directory
result = processor.process_all_pdfs(split=True)

# Access the chunks
chunks = result['chunks']

# Each chunk is a LangChain Document with:
# - page_content: The text content
# - metadata: Source file, page number, etc.
```

### Using the Singleton Instance

```python
from app.services.data_processor import get_pdf_processor

# Get or create processor instance
processor = get_pdf_processor(
    data_dir="data",
    chunk_size=1000,
    chunk_overlap=200
)

result = processor.process_all_pdfs()
```

## API Reference

### PDFProcessor Class

#### Initialization

```python
PDFProcessor(
    data_dir: str = "data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True
)
```

**Parameters:**
- `data_dir`: Directory containing PDF files
- `chunk_size`: Maximum size of text chunks in characters (default: 1000)
- `chunk_overlap`: Number of overlapping characters between chunks (default: 200)
- `add_start_index`: Track the starting character index in original document (default: True)

#### Methods

##### `load_single_pdf(file_path: str) -> List[Document]`
Load a single PDF file and return one Document per page.

**Returns:** List of LangChain Document objects

##### `load_all_pdfs(glob_pattern: str = "**/*.pdf") -> List[Document]`
Load all PDF files matching the pattern from the data directory.

**Parameters:**
- `glob_pattern`: Pattern to match PDF files (default: "**/*.pdf")

**Returns:** List of all Document objects from all PDFs

##### `split_documents(documents: List[Document]) -> List[Document]`
Split documents into smaller chunks using RecursiveCharacterTextSplitter.

**Returns:** List of split Document chunks

##### `process_pdf(file_path: str, split: bool = True) -> Dict[str, Any]`
Process a single PDF file with loading and optional splitting.

**Returns:** Dictionary containing:
- `file_path`: Path to the PDF file
- `file_name`: Name of the file
- `total_pages`: Number of pages
- `total_chunks`: Number of chunks created
- `chunks`: List of document chunks
- `metadata`: Processing metadata
- `processed_at`: ISO timestamp

##### `process_all_pdfs(split: bool = True, glob_pattern: str = "**/*.pdf") -> Dict[str, Any]`
Process all PDF files in the data directory.

**Returns:** Dictionary containing:
- `total_files`: Number of files processed
- `total_pages`: Total number of pages
- `total_chunks`: Total number of chunks
- `processed_files`: List of processed file paths
- `chunks`: All document chunks
- `processed_at`: ISO timestamp

##### `get_chunks_with_metadata(chunks: List[Document]) -> List[Dict[str, Any]]`
Get chunks in a structured format with metadata.

**Returns:** List of dictionaries with:
- `chunk_id`: Sequential chunk identifier
- `content`: Text content
- `metadata`: Original metadata (source, page, etc.)
- `char_count`: Number of characters

##### `extract_text_content(documents: List[Document]) -> str`
Extract and concatenate all text content from documents.

**Returns:** Combined text as a single string

##### `get_chunk_by_index(chunks: List[Document], index: int) -> Optional[Document]`
Retrieve a specific chunk by its index.

**Returns:** Document chunk or None if out of range

## Document Structure

Each processed document chunk is a LangChain `Document` object with:

```python
{
    "page_content": "The actual text content of the chunk...",
    "metadata": {
        "source": "data/document.pdf",
        "page": 5,
        "start_index": 1250  # Character position in original doc
    }
}
```

## Configuration Options

### Chunk Size Guidelines

- **Small chunks (300-500 chars)**: Better for precise retrieval, more chunks
- **Medium chunks (1000-1500 chars)**: Balanced approach (recommended)
- **Large chunks (2000-3000 chars)**: More context, fewer chunks

### Chunk Overlap

- **Purpose**: Ensures important information at chunk boundaries isn't lost
- **Recommended**: 10-20% of chunk_size
- **Example**: chunk_size=1000, chunk_overlap=200

## Integration Examples

### For Vector Embeddings

```python
from app.services.data_processor import get_pdf_processor

# Process PDFs
processor = get_pdf_processor()
result = processor.process_all_pdfs(split=True)
chunks = result['chunks']

# Now use with your embedding model
# Example: embeddings = embed_model.embed_documents([c.page_content for c in chunks])
```

### For Knowledge Graph Extraction

```python
# Use larger chunks for better entity relationship context
processor = PDFProcessor(chunk_size=2000, chunk_overlap=400)
result = processor.process_all_pdfs(split=True)

# Extract entities and relationships from each chunk
for chunk in result['chunks']:
    text = chunk.page_content
    metadata = chunk.metadata
    # Process with NER, relation extraction, etc.
```

### For FastAPI Endpoint

```python
from fastapi import APIRouter, HTTPException
from app.services.data_processor import get_pdf_processor

router = APIRouter()

@router.post("/process-pdfs")
async def process_pdfs():
    try:
        processor = get_pdf_processor()
        result = processor.process_all_pdfs(split=True)
        
        return {
            "status": "success",
            "total_files": result['total_files'],
            "total_chunks": result['total_chunks'],
            "processed_at": result['processed_at']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Example Output

```python
{
    "total_files": 2,
    "total_pages": 45,
    "total_chunks": 156,
    "processed_files": [
        "data/document1.pdf",
        "data/document2.pdf"
    ],
    "chunks": [
        Document(
            page_content="Introduction to AI...",
            metadata={"source": "data/document1.pdf", "page": 0}
        ),
        # ... more chunks
    ],
    "processed_at": "2025-10-29T10:30:00.123456"
}
```

## Performance Tips

1. **Batch Processing**: Process all PDFs at once for better efficiency
2. **Adjust Chunk Size**: Larger chunks = fewer API calls, but less precise retrieval
3. **Use Caching**: Store processed chunks to avoid re-processing
4. **Parallel Processing**: For large document sets, consider concurrent processing

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install langchain langchain-community langchain-text-splitters pypdf
```

**PDF Not Found:**
- Verify the file path is correct
- Ensure the PDF is in the `data/` directory
- Check file permissions

**Memory Issues with Large PDFs:**
- Process PDFs individually instead of batch
- Reduce chunk_size to create fewer chunks in memory
- Use streaming for very large files

## Next Steps

After processing PDFs, you can:

1. **Generate Embeddings**: Use OpenAI, Google, or other embedding models
2. **Store in Vector DB**: FAISS, Pinecone, Chroma, etc.
3. **Build Knowledge Graph**: Extract entities and relationships
4. **Create RAG Pipeline**: Combine with retrieval and generation

## Demo Script

Run the demo to see all features in action:

```bash
python demo_pdf_processor.py
```

## License

Part of the Graph-Vector-RAG-Project
