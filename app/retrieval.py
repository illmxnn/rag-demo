from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9]{2,}")
STOPWORDS = {"about", "are", "can", "does", "for", "from", "how", "is", "made", "of", "the", "to", "what", "when", "where"}


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document: str
    text: str
    score: float


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    @property
    def dimension(self) -> int: ...

    def upsert(self, chunks: list[dict[str, str]], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], limit: int) -> list[Hit]: ...

    def delete_document(self, name: str) -> int: ...

    def reset(self) -> None: ...


class EmbeddingUnavailable(RuntimeError):
    pass


class SentenceTransformerProvider:
    """Lazy local model loader: importing/running lexical mode never downloads a model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise EmbeddingUnavailable("local embedding model is unavailable; use lexical mode or install semantic extras") from exc

    @property
    def dimension(self) -> int:
        self._load()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]


class InMemoryVectorStore:
    """Small deterministic implementation used by tests; production uses Qdrant."""

    def __init__(self, dimension: int) -> None:
        self._dimension, self.items = dimension, {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _validate(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

    def upsert(self, chunks: list[dict[str, str]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        for chunk, vector in zip(chunks, vectors):
            self._validate(vector)
            self.items[chunk["chunk_id"]] = (chunk, vector)

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        self._validate(vector)
        scored = []
        for chunk, candidate in self.items.values():
            score = sum(a * b for a, b in zip(vector, candidate))
            scored.append(Hit(chunk["chunk_id"], chunk["document"], chunk["text"], round(score, 4)))
        return sorted((hit for hit in scored if hit.score > 0), key=lambda hit: (-hit.score, hit.chunk_id))[:limit]

    def delete_document(self, name: str) -> int:
        ids = [key for key, (chunk, _) in self.items.items() if chunk["document"] == name]
        for key in ids:
            del self.items[key]
        return len(ids)

    def reset(self) -> None:
        self.items.clear()


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, dimension: int) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise EmbeddingUnavailable("qdrant-client is unavailable; install semantic extras") from exc
        self._models, self._dimension = models, dimension
        self.client = QdrantClient(url=url, timeout=5)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
        else:
            stored_dimension = self.client.get_collection(collection).config.params.vectors.size
            if stored_dimension != dimension:
                raise ValueError(f"vector dimension mismatch: collection has {stored_dimension}, embedding provider has {dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _validate(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

    def upsert(self, chunks: list[dict[str, str]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        points = []
        for chunk, vector in zip(chunks, vectors):
            self._validate(vector)
            point_id = int(hashlib.sha256(chunk["chunk_id"].encode()).hexdigest()[:15], 16)
            points.append(self._models.PointStruct(id=point_id, vector=vector, payload=chunk))
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        self._validate(vector)
        points = self.client.search(self.collection, query_vector=vector, limit=limit)
        return [Hit(str(p.payload["chunk_id"]), str(p.payload["document"]), str(p.payload["text"]), round(float(p.score), 4)) for p in points if float(p.score) > 0]

    def delete_document(self, name: str) -> int:
        points, _ = self.client.scroll(self.collection, scroll_filter=self._models.Filter(must=[self._models.FieldCondition(key="document", match=self._models.MatchValue(value=name))]), with_payload=False, with_vectors=False, limit=10_000)
        if points:
            self.client.delete(self.collection, points_selector=self._models.PointIdsList(points=[p.id for p in points]), wait=True)
        return len(points)

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=self._models.VectorParams(size=self.dimension, distance=self._models.Distance.COSINE))


class LocalRetriever:
    def __init__(self, state_path: str | Path | None = None, embedding_provider: EmbeddingProvider | None = None, vector_store: VectorStore | None = None) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.chunks: list[dict[str, str]] = []
        self.documents: dict[str, str] = {}
        self.embedding_provider, self.vector_store = embedding_provider, vector_store
        if self.state_path and self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.chunks, self.documents = saved.get("chunks", []), saved.get("documents", {})
            if self.embedding_provider and self.vector_store and self.chunks:
                self._sync_vectors()

    def _save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"chunks": self.chunks, "documents": self.documents}), encoding="utf-8")

    def _sync_vectors(self) -> None:
        if self.embedding_provider and self.vector_store and self.chunks:
            self.vector_store.upsert(self.chunks, self.embedding_provider.embed([chunk["text"] for chunk in self.chunks]))

    def add(self, chunks: list[dict[str, str]], checksum: str | None = None) -> None:
        if not chunks:
            raise ValueError("document produced no chunks")
        name = chunks[0]["document"]
        if name in self.documents:
            raise FileExistsError(f"document already indexed: {name}")
        if any(chunk["document"] != name for chunk in chunks):
            raise ValueError("chunks must belong to one document")
        checksum = checksum or hashlib.sha256("\n".join(c["text"] for c in chunks).encode()).hexdigest()
        self.chunks.extend(chunks)
        self.documents[name] = checksum
        try:
            if self.embedding_provider and self.vector_store:
                vectors = self.embedding_provider.embed([chunk["text"] for chunk in chunks])
                self.vector_store.upsert(chunks, vectors)
        except Exception:
            self.chunks = self.chunks[:-len(chunks)]
            del self.documents[name]
            raise
        self._save()

    def delete_document(self, name: str) -> int:
        count = sum(chunk["document"] == name for chunk in self.chunks)
        if not count:
            return 0
        self.chunks = [chunk for chunk in self.chunks if chunk["document"] != name]
        del self.documents[name]
        if self.vector_store:
            self.vector_store.delete_document(name)
        self._save()
        return count

    def reset(self) -> int:
        count = len(self.chunks)
        self.chunks, self.documents = [], {}
        if self.vector_store:
            self.vector_store.reset()
        self._save()
        return count

    def lexical_search(self, question: str, top_k: int = 3) -> list[Hit]:
        query = Counter(tokens(question))
        if not query:
            return []
        docs = [set(tokens(c["text"])) for c in self.chunks]
        df = Counter(token for doc in docs for token in doc)
        scored = [Hit(chunk["chunk_id"], chunk["document"], chunk["text"], round(sum((1 + math.log((len(docs) + 1) / (df[token] + 1))) * query[token] for token in query if token in doc_tokens) / max(1.0, sum(query.values())), 4)) for chunk, doc_tokens in zip(self.chunks, docs)]
        return sorted((hit for hit in scored if hit.score > 0), key=lambda hit: (-hit.score, hit.chunk_id))[:top_k]

    def semantic_search(self, question: str, top_k: int = 3) -> list[Hit]:
        if not self.embedding_provider or not self.vector_store:
            raise EmbeddingUnavailable("semantic retrieval is not configured")
        return self.vector_store.search(self.embedding_provider.embed([question])[0], top_k)

    def search(self, question: str, top_k: int = 3, mode: str = "lexical") -> tuple[list[Hit], str, str | None]:
        if mode == "lexical":
            return self.lexical_search(question, top_k), "lexical", None
        try:
            semantic = self.semantic_search(question, top_k)
            if mode == "semantic":
                return semantic, "semantic", None
            lexical = self.lexical_search(question, top_k)
            return reciprocal_rank_fusion(lexical, semantic, limit=top_k), "hybrid", None
        except (EmbeddingUnavailable, ValueError, OSError) as exc:
            return self.lexical_search(question, top_k), "lexical", str(exc)


def reciprocal_rank_fusion(*rankings: list[Hit], limit: int = 3, k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    source: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + rank)
            source[hit.chunk_id] = hit
    return [Hit(source[key].chunk_id, source[key].document, source[key].text, round(score, 6)) for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]
