import json
import time
from pathlib import Path

from .ingest import ParsedDocument, chunk_text
from .retrieval import InMemoryVectorStore, LocalRetriever


class EvaluationEmbedding:
    """Offline deterministic semantic fixture; never downloads a model."""

    vocabulary = ("return", "unused", "packaging", "refund", "shipping", "express", "domestic", "tracking", "keys", "upload", "address", "gift", "weekend")
    synonyms = {"returns": "return", "refunded": "refund", "refunds": "refund", "delivery": "shipping", "credentials": "keys", "provider": "keys", "uploads": "upload", "persisted": "upload", "update": "address", "cards": "gift"}

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = []
        for text in texts:
            normalized = [self.synonyms.get(token, token) for token in text.lower().replace("?", "").split()]
            output.append([float(normalized.count(word)) for word in self.vocabulary])
        return output


def hit(item: dict[str, object], hits: list[object]) -> bool:
    if not item["supported"]:
        return not hits
    return bool(hits and any(keyword in hits[0].text.lower() for keyword in item["expected_keywords"]))


def main() -> None:
    data = json.loads((Path(__file__).parents[1] / "evaluation" / "questions.json").read_text())
    embedding = EvaluationEmbedding()
    retriever = LocalRetriever(embedding_provider=embedding, vector_store=InMemoryVectorStore(embedding.dimension))
    for item in data["corpus"]:
        retriever.add(chunk_text(ParsedDocument(item["name"], item["text"]), size=35, overlap=5))
    for mode in ("lexical", "semantic", "hybrid"):
        supported_hits = rejected = 0
        latencies = []
        for item in data["questions"]:
            started = time.perf_counter()
            hits, _, _ = retriever.search(item["question"], 3, mode)
            latencies.append((time.perf_counter() - started) * 1000)
            if item["supported"]:
                supported_hits += hit(item, hits)
            if not item["supported"] and not hits:
                rejected += 1
        supported = sum(item["supported"] for item in data["questions"])
        unsupported = sum(not item["supported"] for item in data["questions"])
        print(f"mode={mode} retrieval_hit_rate={supported_hits}/{supported} ({supported_hits / supported:.0%}) unsupported_rejection={rejected}/{unsupported} ({rejected / unsupported:.0%}) avg_latency_ms={sum(latencies) / len(data['questions']):.3f}")


if __name__ == "__main__":
    main()
