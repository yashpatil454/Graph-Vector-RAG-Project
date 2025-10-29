# PDF Processor - Quick Reference

## Import
```python
from app.services.data_processor import PDFProcessor, get_pdf_processor
```

## Initialize
```python
# Create new instance
processor = PDFProcessor(
    data_dir="data",
    chunk_size=1000,      # characters per chunk
    chunk_overlap=200,    # overlap between chunks
    add_start_index=True, # track position in original doc
    clean_text=True       # remove \n, \xa0, normalize whitespace (recommended)
)

# Or use singleton
processor = get_pdf_processor(clean_text=True)
```

## Process PDFs

### Single File
```python
result = processor.process_pdf("data/document.pdf", split=True)
```

### All Files
```python
result = processor.process_all_pdfs(split=True)
chunks = result['chunks']  # List[Document]
```

## Access Data

### Get Chunks
```python
for chunk in chunks:
    text = chunk.page_content
    source = chunk.metadata['source']
    page = chunk.metadata['page']
```

### Structured Format
```python
structured = processor.get_chunks_with_metadata(chunks)
for item in structured:
    print(item['chunk_id'])
    print(item['content'])
    print(item['metadata'])
```

### Full Text
```python
docs = processor.load_single_pdf("data/doc.pdf")
full_text = processor.extract_text_content(docs)
```

## Result Structure
```python
{
    'total_files': 2,
    'total_pages': 45,
    'total_chunks': 156,
    'processed_files': ['data/doc1.pdf', 'data/doc2.pdf'],
    'chunks': [Document(...), Document(...), ...],
    'processed_at': '2025-10-29T10:30:00'
}
```

## Common Patterns

### For Vector Embeddings
```python
processor = get_pdf_processor(chunk_size=1000)
result = processor.process_all_pdfs()
texts = [chunk.page_content for chunk in result['chunks']]
# Pass texts to embedding model
```

### For Knowledge Graph
```python
processor = PDFProcessor(chunk_size=2000)  # Larger chunks
result = processor.process_all_pdfs()
for chunk in result['chunks']:
    # Extract entities and relationships
    pass
```

### FastAPI Endpoint
```python
@router.post("/process")
async def process():
    processor = get_pdf_processor()
    result = processor.process_all_pdfs()
    return {"chunks": result['total_chunks']}
```

## Tips
- Use larger chunks (1500-2000) for better context
- Increase overlap (300-400) to avoid losing information
- Process PDFs once, cache results
- Use `split=False` if you need full pages
- **Enable `clean_text=True` (default)** to remove `\n`, `\xa0`, and normalize whitespace
- Disable cleaning (`clean_text=False`) only if you need exact raw text

## Text Cleaning (New Feature!)
Text cleaning is **enabled by default** and provides:
- Removes `\n` (newline) characters that break words
- Converts `\xa0` (non-breaking spaces) to regular spaces
- Normalizes multiple spaces to single space
- Preserves paragraph breaks (double newlines)
- Cleans up spacing around punctuation

```python
# With cleaning (default, recommended)
processor = PDFProcessor(clean_text=True)
# Output: "Generative AI and foundation models in medical image"

# Without cleaning (raw text)
processor = PDFProcessor(clean_text=False)
# Output: "Generative AI and\xa0foundation models in\xa0medical image"
```
