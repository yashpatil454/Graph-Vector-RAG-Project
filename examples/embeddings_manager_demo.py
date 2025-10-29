"""
Demo: Using Embeddings Manager Service
Shows all features of the embeddings manager for RAG applications
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_processor import get_pdf_processor
from app.services.embeddings_manager import EmbeddingsManager, get_embeddings_manager


def demo_create_embeddings():
    """Demo 1: Create embeddings from PDF chunks"""
    print("\n" + "="*70)
    print("DEMO 1: Creating Embeddings from PDF Chunks")
    print("="*70)
    
    # Step 1: Process PDFs to get chunks
    print("\n📄 Step 1: Processing PDFs...")
    processor = get_pdf_processor(
        data_dir="data",
        chunk_size=1000,
        chunk_overlap=200,
        clean_text=True
    )
    
    result = processor.process_all_pdfs(split=True)
    chunks = result['chunks']
    
    print(f"✓ Processed {result['total_files']} files")
    print(f"✓ Created {result['total_chunks']} chunks")
    
    # Step 2: Create embeddings
    print("\n🔢 Step 2: Creating embeddings...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GOOGLE_API_KEY not set!")
        print("Set it with: $env:GOOGLE_API_KEY='your-key-here'")
        print("\nSkipping embedding creation...")
        return None
    
    embeddings_manager = EmbeddingsManager(
        embedding_model="models/embedding-001",
        api_key=api_key,
        vector_store_path="vectorstore_db"
    )
    
    # Create and save embeddings
    vectorstore = embeddings_manager.create_embeddings(
        chunks=chunks,
        save=True
    )
    
    print(f"✓ Created vector store with {vectorstore.index.ntotal} vectors")
    print(f"✓ Saved to: {embeddings_manager.vector_store_path}")
    
    return embeddings_manager


def demo_load_embeddings():
    """Demo 2: Load existing embeddings"""
    print("\n" + "="*70)
    print("DEMO 2: Loading Existing Embeddings")
    print("="*70)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping (no API key)")
        return None
    
    embeddings_manager = EmbeddingsManager(
        api_key=api_key,
        vector_store_path="vectorstore_db"
    )
    
    try:
        embeddings_manager.load_vector_store()
        print(f"✓ Loaded vector store with {embeddings_manager.vectorstore.index.ntotal} vectors")
        return embeddings_manager
    except FileNotFoundError:
        print("❌ Vector store not found. Create one first with demo_create_embeddings()")
        return None


def demo_similarity_search(embeddings_manager):
    """Demo 3: Search by semantic similarity"""
    print("\n" + "="*70)
    print("DEMO 3: Semantic Similarity Search")
    print("="*70)
    
    if embeddings_manager is None or embeddings_manager.vectorstore is None:
        print("⚠️  Skipping (vector store not available)")
        return
    
    queries = [
        "What is generative AI in medical imaging?",
        "Explain foundation models",
        "What are diffusion models?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Search with scores
        results = embeddings_manager.search_by_similarity_with_scores(
            query=query,
            k=3
        )
        
        print(f"\n📚 Top 3 Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n   Result {i} (Score: {score:.4f}):")
            print(f"   • Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"   • Source: {Path(doc.metadata.get('source', 'Unknown')).name}")
            print(f"   • Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   • Preview: {doc.page_content[:150]}...")
        print("\n" + "-"*70)


def demo_search_by_chunk_id(embeddings_manager):
    """Demo 4: Direct chunk retrieval by ID"""
    print("\n" + "="*70)
    print("DEMO 4: Direct Chunk Retrieval by ID")
    print("="*70)
    
    if embeddings_manager is None:
        print("⚠️  Skipping (embeddings manager not available)")
        return
    
    chunk_ids = ["chunk_00000", "chunk_00010", "chunk_00050"]
    
    for chunk_id in chunk_ids:
        print(f"\n🔍 Looking up: {chunk_id}")
        
        chunk = embeddings_manager.search_by_chunk_id(chunk_id)
        
        if chunk:
            print(f"✓ Found!")
            print(f"   • Source: {Path(chunk.metadata.get('source', 'Unknown')).name}")
            print(f"   • Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"   • Index: {chunk.metadata.get('chunk_index', 'N/A')}")
            print(f"   • Content: {chunk.page_content[:200]}...")
        else:
            print(f"❌ Not found")


def demo_search_by_metadata(embeddings_manager):
    """Demo 5: Search by metadata filters"""
    print("\n" + "="*70)
    print("DEMO 5: Search by Metadata Filters")
    print("="*70)
    
    if embeddings_manager is None:
        print("⚠️  Skipping (embeddings manager not available)")
        return
    
    # Example 1: Get chunks from specific page
    print("\n📄 Example 1: Get chunks from page 0")
    results = embeddings_manager.search_by_metadata(
        metadata_filter={"page": 0},
        k=5
    )
    
    print(f"Found {len(results)} chunks from page 0")
    for i, chunk in enumerate(results[:3], 1):
        print(f"   {i}. Chunk {chunk.metadata.get('chunk_id')} - {chunk.page_content[:100]}...")
    
    # Example 2: Get all chunks from a specific source
    if embeddings_manager.chunk_mapping:
        first_chunk = next(iter(embeddings_manager.chunk_mapping.values()))
        source = first_chunk.metadata.get('source')
        
        print(f"\n📁 Example 2: Get chunks from source: {Path(source).name}")
        results = embeddings_manager.search_by_source_file(source, k=5)
        
        print(f"Found {len(results)} chunks from this source (showing first 3)")
        for i, chunk in enumerate(results[:3], 1):
            print(f"   {i}. Page {chunk.metadata.get('page')} - {chunk.page_content[:100]}...")


def demo_get_statistics(embeddings_manager):
    """Demo 6: Get vector store statistics"""
    print("\n" + "="*70)
    print("DEMO 6: Vector Store Statistics")
    print("="*70)
    
    if embeddings_manager is None:
        print("⚠️  Skipping (embeddings manager not available)")
        return
    
    stats = embeddings_manager.get_statistics()
    
    print(f"\n📊 Statistics:")
    print(f"   • Status: {stats.get('status')}")
    print(f"   • Total Vectors: {stats.get('total_vectors', 0)}")
    print(f"   • Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"   • Total Sources: {stats.get('total_sources', 0)}")
    print(f"   • Embedding Model: {stats.get('embedding_model')}")
    print(f"   • Embedding Dimension: {stats.get('embedding_dimension')}")
    print(f"   • Storage Path: {stats.get('vector_store_path')}")
    
    print(f"\n📁 Sources:")
    for source in stats.get('sources', []):
        print(f"   • {Path(source).name}")
    
    print(f"\n📄 Pages per Source:")
    for source, page_count in stats.get('source_page_counts', {}).items():
        print(f"   • {Path(source).name}: {page_count} pages")


def demo_retriever_for_rag(embeddings_manager):
    """Demo 7: Get retriever for RAG chains"""
    print("\n" + "="*70)
    print("DEMO 7: Creating Retriever for RAG")
    print("="*70)
    
    if embeddings_manager is None or embeddings_manager.vectorstore is None:
        print("⚠️  Skipping (vector store not available)")
        return None
    
    # Create retriever
    retriever = embeddings_manager.get_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    print(f"✓ Created retriever")
    print(f"   • Search Type: similarity")
    print(f"   • K (results): 4")
    
    # Test retriever
    test_query = "What is generative AI?"
    print(f"\n🔍 Testing retriever with query: '{test_query}'")
    
    docs = retriever.get_relevant_documents(test_query)
    
    print(f"\n📚 Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n   Document {i}:")
        print(f"   • Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"   • Source: {Path(doc.metadata.get('source', 'Unknown')).name}")
        print(f"   • Page: {doc.metadata.get('page')}")
        print(f"   • Preview: {doc.page_content[:150]}...")
    
    return retriever


def demo_add_new_chunks(embeddings_manager):
    """Demo 8: Add new chunks to existing vector store"""
    print("\n" + "="*70)
    print("DEMO 8: Adding New Chunks to Existing Store")
    print("="*70)
    
    if embeddings_manager is None or embeddings_manager.vectorstore is None:
        print("⚠️  Skipping (vector store not available)")
        return
    
    from langchain_core.documents import Document
    
    # Create sample new chunks
    new_chunks = [
        Document(
            page_content="This is a new test chunk about AI in healthcare.",
            metadata={"chunk_id": "chunk_99999", "source": "test.pdf", "page": 0}
        ),
        Document(
            page_content="Another test chunk about machine learning applications.",
            metadata={"chunk_id": "chunk_99998", "source": "test.pdf", "page": 0}
        )
    ]
    
    print(f"\n📝 Adding {len(new_chunks)} new chunks...")
    
    initial_count = embeddings_manager.vectorstore.index.ntotal
    embeddings_manager.add_embeddings(new_chunks, save=False)
    final_count = embeddings_manager.vectorstore.index.ntotal
    
    print(f"✓ Added successfully")
    print(f"   • Before: {initial_count} vectors")
    print(f"   • After: {final_count} vectors")
    print(f"   • Added: {final_count - initial_count} vectors")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print(" "*20 + "EMBEDDINGS MANAGER SERVICE - DEMO")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        # Full demo with API key
        print("\n✓ API key found. Running full demo...\n")
        
        # Create embeddings
        embeddings_manager = demo_create_embeddings()
        
        if embeddings_manager:
            # Run all other demos
            demo_similarity_search(embeddings_manager)
            demo_search_by_chunk_id(embeddings_manager)
            demo_search_by_metadata(embeddings_manager)
            demo_get_statistics(embeddings_manager)
            demo_retriever_for_rag(embeddings_manager)
            demo_add_new_chunks(embeddings_manager)
    else:
        # Limited demo without API key
        print("\n⚠️  No API key found. Running limited demo...\n")
        print("To run full demo, set: $env:GOOGLE_API_KEY='your-key-here'\n")
        
        # Try to load existing embeddings
        embeddings_manager = demo_load_embeddings()
        
        if embeddings_manager:
            demo_search_by_chunk_id(embeddings_manager)
            demo_search_by_metadata(embeddings_manager)
            demo_get_statistics(embeddings_manager)
    
    print("\n" + "="*80)
    print("✅ Demo Completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
