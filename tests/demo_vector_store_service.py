"""Demo script for VectorStoreService.
Run: python tests/demo_vector_store_service.py

Requires OpenAI embeddings (OPENAI_API_KEY must be set).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path BEFORE importing app.* modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.data_processor import get_pdf_processor
from app.services.vector_store_service import get_vector_store_service, VectorStoreService
from langchain_core.documents import Document

def _fallback_docs() -> list[Document]:
    samples = [
        "Clinical trial results indicate improved patient-reported outcomes in phase II.",
        "Adverse events were managed through dose titration and supportive therapy.",
        "Pharmacokinetic analysis shows rapid absorption and moderate half-life.",
        "Regulatory submission requires consolidated safety data and labeling strategy.",
    ]
    return [Document(page_content=s, metadata={"source": "sample", "line": i}) for i, s in enumerate(samples)]


def build_or_load_index(service: VectorStoreService, docs: list[Document]):
    if service.is_initialized():
        # Already loaded from disk
        return
    # if not docs:
    #     docs = _fallback_docs()
    service.from_documents(docs)
    service.save()


def main():
    processor = get_pdf_processor()
    result = processor.process_all_pdfs(split=True, parallel=True)
    chunks = result.get("chunks", [])
    print(f"Loaded {len(chunks)} chunks from PDFs.")

    # Switch to Gemini embeddings (requires GOOGLE_API_KEY in environment)
    try:
        vs_service = get_vector_store_service(embedding_provider="gemini", auto_load=True)
    except EnvironmentError as e:
        print(f"[WARN] {e}. Falling back to sample in-memory docs with mock build.")
        vs_service = get_vector_store_service(embedding_provider="openai", auto_load=True)

    build_or_load_index(vs_service, chunks[:10])  # Use a few chunks for demo

    print(f"Vector store initialized with {vs_service.count()} documents using provider='{vs_service.embedding_provider}'.")

    queries = [
        "patient outcomes",
        "adverse events",
        "pharmacokinetic",
        "regulatory submission",
    ]

    for q in queries:
        print("\n=== Query:", q)
        for doc, score in vs_service.similarity_search_with_score(q, k=2):
            snippet = doc.page_content[:160].replace("\n", " ")
            print(f"Score={score:.4f} | {snippet}...")

    # MMR demo
    mmr_docs = vs_service.max_marginal_relevance_search("clinical", k=3, fetch_k=10)
    print("\nMMR Diversified Results:")
    for d in mmr_docs:
        print("-", d.page_content[:120].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
