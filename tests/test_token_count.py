from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys
from pathlib import Path
# Ensure project root is on sys.path BEFORE importing app.* modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from app.services.data_processor import get_pdf_processor

processor = get_pdf_processor()
loaded = processor.load_persisted_chunks("data/processed_chunks")
# Example: your document list
docs = loaded  # List[Document]

# Use a text splitter to simulate token-sized chunks
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="text-embedding-3-small"
)

# split_docs = splitter.split_documents(docs)

# Optional: count tokens per chunk
for d in docs:
    n_tokens = splitter._length_function(d.page_content)
    print(f"Chunk: {d.page_content[:50]}... | Tokens: {n_tokens}")

print(len(docs))