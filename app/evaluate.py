import json
from pathlib import Path

from .ingest import ParsedDocument, chunk_text
from .retrieval import LocalRetriever


def main() -> None:
    data = json.loads((Path(__file__).parents[1] / "evaluation" / "questions.json").read_text())
    retriever = LocalRetriever()
    for item in data["corpus"]:
        retriever.add(chunk_text(ParsedDocument(item["name"], item["text"]), size=90, overlap=15))
    passed = 0
    for item in data["questions"]:
        hits = retriever.search(item["question"], 3)
        ok = bool(hits and any(keyword in hits[0].text.lower() for keyword in item["expected_keywords"]))
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}: {item['question']}")
    print(f"retrieval_hit_rate={passed}/{len(data['questions'])} ({passed / len(data['questions']):.0%})")


if __name__ == "__main__":
    main()
