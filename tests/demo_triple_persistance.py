"""Standalone demo to test streaming triple persistence to data/processed_triples/triples.jsonl.

This does NOT call the real Gemini model; instead it mocks responses to focus on file writing logic.

Run:
	python tests/demo_triple_persistance.py
"""
import json
import asyncio
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Dict

@dataclass
class Triple:
	subject: str
	predicate: str
	object: str
	provenance: Dict

class MockDocument:
	def __init__(self, page_content: str, metadata: Dict):
		self.page_content = page_content
		self.metadata = metadata

documents: List[MockDocument] = [
	MockDocument(
		"DrugA treats DiseaseX and may cause Nausea as a side effect.",
		{"source": "mock_doc_1.pdf"}
	),
	MockDocument(
		"DrugB interacts with DrugA and is metabolized in the Liver.",
		{"source": "mock_doc_2.pdf"}
	),
]

MOCK_JSON_OUTPUTS = [
	[
		{"subject": "DrugA", "predicate": "TREATS", "object": "DiseaseX"},
		{"subject": "DrugA", "predicate": "HAS_SIDE_EFFECT", "object": "Nausea"},
	],
	[
		{"subject": "DrugB", "predicate": "INTERACTS_WITH", "object": "DrugA"},
		{"subject": "DrugB", "predicate": "METABOLIZED_IN", "object": "Liver"},
	],
]

prompt_template = (
	"Extract factual triples from the text below. "
	"Return ONLY valid JSON array of objects with keys 'subject','predicate','object'. "
	"Use concise canonical entity names; omit duplicates; skip trivial or vague relations.\n\nText:\n{text}"
)

async def main():
	output_dir = Path("data") / "processed_triples"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "triples.jsonl"
	if output_path.exists():
		output_path.unlink()  # start fresh
	print(f"Opening triples output file for streaming writes: {output_path}")
	all_triples: List[Triple] = []
	written_count = 0
	with open(output_path, "a", encoding="utf-8") as f:
		for idx, doc in enumerate(documents):
			prompt = prompt_template.format(text=doc.page_content)
			parsed = MOCK_JSON_OUTPUTS[idx]
			print(f"Parsed triples (doc {idx}): {parsed}")
			for item in parsed:
				subj = item.get("subject")
				pred = item.get("predicate")
				obj = item.get("object")
				if subj and pred and obj:
					triple_obj = Triple(subject=subj.strip(), predicate=pred.strip(), object=obj.strip(), provenance=doc.metadata)
					all_triples.append(triple_obj)
					f.write(json.dumps(asdict(triple_obj), ensure_ascii=False) + "\n")
					written_count += 1
			await asyncio.sleep(0.01)
	print(f"Closed triples output file. Total streamed triples written: {written_count}")
	print(f"Extracted {len(all_triples)} raw triples from {len(documents)} documents (mocked)")
	lines = output_path.read_text(encoding="utf-8").strip().splitlines()
	print(f"File line count: {len(lines)}")
	for i, line in enumerate(lines[:10], start=1):
		print(f"Line {i}: {line}")

if __name__ == "__main__":
	asyncio.run(main())