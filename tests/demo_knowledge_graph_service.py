"""Simple manual test harness for KnowledgeGraphService (no pytest).
Run directly: python tests/demo_knowledge_graph_service.py
"""
import asyncio
from pathlib import Path
import sys

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.services.data_processor import get_pdf_processor

async def main():
    service = await get_knowledge_graph_service()
    # processor = get_pdf_processor()

    # print("Loading persisted chunk documents ...")
    # documents = processor.load_persisted_chunks()
    # print(f"Loaded {len(documents)} documents/chunks")

    # if not documents:
    #     print("No chunks found. First run the PDF processing endpoint to create them.")
    #     return

    # print("Extracting triples from first 2 chunks for preview ...")
    # preview_docs = documents[:1]
    # triples = await service.extract_triples(preview_docs)
    # print(f"Extracted {len(triples)} triples (preview): {triples}")
    # for t in triples[:10]:
    #     print(f"  ({t.subject}) -[{t.predicate}]-> ({t.object})")

    # print("Building full graph (all chunks) ...")
    # summary = await service.build_graph()
    # print("Graph build summary:", summary)

    print("Querying sample relationships ...")
    sample_results = await service.query("MATCH (s:Entity)-[r:RELATION]->(o:Entity) RETURN s.name AS subject, r.predicate AS predicate, o.name AS object")
    for row in sample_results:
        print(row)

    await service.close()

if __name__ == "__main__":
    asyncio.run(main())
