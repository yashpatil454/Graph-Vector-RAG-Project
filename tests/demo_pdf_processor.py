"""
Example usage of the PDF Processor Service
Demonstrates how to process PDFs for Knowledge Graph and Vector RAG
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_processor import PDFProcessor, get_pdf_processor


def example_single_pdf():
    """Example: Process a single PDF file"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Processing a Single PDF File")
    print("="*70)
    
    # Initialize processor
    processor = PDFProcessor(data_dir="data", chunk_size=1000, chunk_overlap=200)
    
    # Process the first PDF in data folder
    pdf_path = "data/Generative_AI_and_foundation_models_in_medical_ima.pdf"
    
    try:
        result = processor.process_pdf(pdf_path, split=True)
        
        print(f"\n✓ File: {result['file_name']}")
        print(f"✓ Total Pages: {result['total_pages']}")
        print(f"✓ Total Chunks: {result['total_chunks']}")
        print(f"\n📊 Metadata:")
        for key, value in result['metadata'].items():
            print(f"   • {key}: {value}")
        print(result)
        # Show first chunk
        if result['chunks']:
            first_chunk = result['chunks'][0]
            print(f"\n📄 First Chunk Preview:")
            print(f"   Content: {first_chunk.page_content[:200]}...")
            print(f"   Metadata: {first_chunk.metadata}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def example_all_pdfs():
    # """Example: Process all PDFs in the data folder"""
    # print("\n" + "="*70)
    # print("EXAMPLE 2: Processing All PDFs in Data Folder")
    # print("="*70)
    
    # Use singleton instance
    processor = get_pdf_processor(data_dir="data", chunk_size=1000, chunk_overlap=200)
    
    try:
        result = processor.process_all_pdfs(split=True)
        print(result['chunks'][0])
        
        # print(f"\n✓ Total Files Processed: {result['total_files']}")
        # print(f"✓ Total Pages: {result['total_pages']}")
        # print(f"✓ Total Chunks: {result['total_chunks']}")
        
        # print(f"\n📁 Processed Files:")
        # for file_path in result['processed_files']:
        #     print(f"   • {Path(file_path).name}")
        
        # # Show chunk statistics
        # if result['chunks']:
        #     chunk_lengths = [len(chunk.page_content) for chunk in result['chunks']]
        #     print(f"\n📊 Chunk Statistics:")
        #     print(f"   • Min chunk size: {min(chunk_lengths)} chars")
        #     print(f"   • Max chunk size: {max(chunk_lengths)} chars")
        #     print(f"   • Avg chunk size: {sum(chunk_lengths) / len(chunk_lengths):.2f} chars")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def example_chunks_with_metadata():
    """Example: Get structured chunks with metadata"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Extracting Chunks with Metadata")
    print("="*70)
    
    processor = get_pdf_processor()
    
    try:
        # Process all PDFs
        result = processor.process_all_pdfs(split=True)
        
        # Get structured chunks
        structured_chunks = processor.get_chunks_with_metadata(result['chunks'][:5])
        
        print(f"\n📦 Showing first 5 chunks with metadata:\n")
        for chunk_data in structured_chunks:
            print(f"Chunk ID: {chunk_data['chunk_id']}")
            print(f"Character Count: {chunk_data['char_count']}")
            print(f"Source: {chunk_data['metadata'].get('source', 'Unknown')}")
            print(f"Page: {chunk_data['metadata'].get('page', 'N/A')}")
            print(f"Content Preview: {chunk_data['content'][:150]}...")
            print("-" * 70)
            
        return structured_chunks
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def example_custom_chunking():
    """Example: Process with custom chunk size"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Chunking Configuration")
    print("="*70)
    
    # Create processor with smaller chunks for more granular retrieval
    processor = PDFProcessor(
        data_dir="data",
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,  # Less overlap
        add_start_index=True
    )
    
    try:
        result = processor.process_all_pdfs(split=True)
        
        print(f"\n✓ Configuration:")
        print(f"   • Chunk Size: 500 characters")
        print(f"   • Chunk Overlap: 100 characters")
        
        print(f"\n✓ Results:")
        print(f"   • Total Chunks: {result['total_chunks']}")
        print(f"   • Chunks per Page: {result['total_chunks'] / result['total_pages']:.2f}")
        
        # Show a chunk with start_index
        if result['chunks'] and 'start_index' in result['chunks'][0].metadata:
            sample_chunk = result['chunks'][10]
            print(f"\n📄 Sample Chunk with Start Index:")
            print(f"   Start Index: {sample_chunk.metadata.get('start_index', 'N/A')}")
            print(f"   Content: {sample_chunk.page_content[:200]}...")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def example_extract_full_text():
    """Example: Extract full text content"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Extracting Full Text Content")
    print("="*70)
    
    processor = get_pdf_processor()
    
    try:
        # Load first PDF
        pdf_path = "data/Generative_AI_and_foundation_models_in_medical_ima.pdf"
        documents = processor.load_single_pdf(pdf_path)
        
        # Extract full text
        full_text = processor.extract_text_content(documents)
        
        print(f"\n✓ Total Characters: {len(full_text)}")
        print(f"✓ Total Words (approx): {len(full_text.split())}")
        print(f"\n📝 Text Preview (first 500 chars):")
        print(full_text[:500])
        print("...")
        
        return full_text
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    # """Run all examples"""
    # print("\n" + "="*70)
    # print("PDF PROCESSOR SERVICE - DEMO")
    # print("Knowledge Graph and Vector RAG Application")
    # print("="*70)
    
    # Run examples
    # example_single_pdf()
    example_all_pdfs()
    # example_chunks_with_metadata()
    # example_custom_chunking()
    # example_extract_full_text()
    
    # print("\n" + "="*70)
    # print("✅ All examples completed!")
    # print("="*70 + "\n")


if __name__ == "__main__":
    main()
