import json
import sys
import time
from pathlib import Path

from .ingest import ParsedDocument, chunk_text
from .retrieval import InMemoryVectorStore, LocalRetriever, tokens
from .main import has_sufficient_evidence


class EvaluationEmbedding:
    """Offline deterministic fixture, not a SentenceTransformer embedding."""

    vocabulary = ("return", "unused", "packaging", "refund", "payment", "shipping", "express", "domestic", "tracking", "keys", "registry", "address", "gift", "weekend", "cash")
    synonyms = {"returns": "return", "refunded": "refund", "refunds": "refund", "sent": "refund", "back": "refund", "delivery": "shipping", "arrive": "shipping", "quickly": "shipping", "credentials": "keys", "provider": "keys", "persisted": "registry", "update": "address", "cards": "gift", "when": "return", "are": "return"}

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = []
        for text in texts:
            normalized = [self.synonyms.get(token, token) for token in tokens(text)]
            output.append([float(normalized.count(word)) for word in self.vocabulary])
        return output


def supported_match(item: dict[str, object], hits: list[object], mode: str) -> bool:
    return bool(has_sufficient_evidence(item["question"], hits, mode) and any(keyword in hits[0].text.lower() for keyword in item["expected_keywords"]))


def main() -> None:
    data = json.loads((Path(__file__).parents[1] / "evaluation" / "questions.json").read_text())
    embedding = EvaluationEmbedding()
    retriever = LocalRetriever(embedding_provider=embedding, vector_store=InMemoryVectorStore(embedding.dimension))
    for item in data["corpus"]:
        retriever.add(chunk_text(ParsedDocument(item["name"], item["text"]), size=35, overlap=5))
    failures = []
    for mode in ("lexical", "semantic", "hybrid"):
        supported_hits = rejected = 0
        latencies = []
        for item in data["questions"]:
            started = time.perf_counter()
            hits, _, _ = retriever.search(item["question"], 3, mode)
            latencies.append((time.perf_counter() - started) * 1000)
            if item["supported"] and supported_match(item, hits, mode):
                supported_hits += 1
            if not item["supported"] and not has_sufficient_evidence(item["question"], hits, mode):
                rejected += 1
        supported = sum(item["supported"] for item in data["questions"])
        unsupported = sum(not item["supported"] for item in data["questions"])
        print(f"mode={mode} retrieval_hit_rate={supported_hits}/{supported} ({supported_hits / supported:.0%}) unsupported_rejection={rejected}/{unsupported} ({rejected / unsupported:.0%}) avg_latency_ms={sum(latencies) / len(data['questions']):.3f}")
        if rejected != unsupported:
            failures.append(mode)
        if supported_hits != supported:
            failures.append(f"{mode} supported evidence/answer coverage")
    if failures:
        print(f"ERROR: unsupported questions were not rejected for: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
