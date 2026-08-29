from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN = re.compile(r"[a-z0-9]{2,}")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document: str
    text: str
    score: float


class LocalRetriever:
    def __init__(self) -> None:
        self.chunks: list[dict[str, str]] = []

    def add(self, chunks: list[dict[str, str]]) -> None:
        self.chunks.extend(chunks)

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        query = Counter(tokens(question))
        if not query:
            return []
        docs = [set(tokens(c["text"])) for c in self.chunks]
        df = Counter(token for doc in docs for token in doc)
        scored: list[Hit] = []
        for chunk, doc_tokens in zip(self.chunks, docs):
            overlap = sum((1 + math.log((len(docs) + 1) / (df[t] + 1))) * query[t] for t in query if t in doc_tokens)
            score = overlap / max(1.0, sum(query.values()))
            if score > 0:
                scored.append(Hit(chunk["chunk_id"], chunk["document"], chunk["text"], round(score, 4)))
        return sorted(scored, key=lambda hit: (-hit.score, hit.chunk_id))[:top_k]
