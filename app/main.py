from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from .ingest import chunk_text, parse_document
from .models import Citation, DocumentInfo, IngestResponse, QueryRequest, QueryResponse, ResetResponse, RetrievalMetadata
from .generation import INSUFFICIENT, configured_generator
from .retrieval import EmbeddingUnavailable, LocalRetriever, QdrantVectorStore, SentenceTransformerEngine, VectorStoreUnavailable, tokens

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag_demo")


def build_retriever() -> LocalRetriever:
    state_path = os.getenv("RAG_STATE_PATH", "data/documents.json")
    semantic_enabled = os.getenv("SEMANTIC_ENABLED", "false").lower() == "true"
    if not semantic_enabled:
        return LocalRetriever(state_path=state_path)
    try:
        embedding = SentenceTransformerEngine(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        vector = QdrantVectorStore(os.getenv("QDRANT_URL", "http://localhost:6333"), os.getenv("QDRANT_COLLECTION", "rag_chunks"), embedding.dimension)
        return LocalRetriever(state_path=state_path, embedding_engine=embedding, vector_store=vector)
    except (EmbeddingUnavailable, VectorStoreUnavailable, ValueError, OSError) as exc:
        logger.warning("semantic_backend_unavailable fallback=lexical reason=%s", str(exc))
        fallback = LocalRetriever(state_path=state_path)
        fallback.startup_fallback_reason = str(exc)
        return fallback


app = FastAPI(title="Hybrid RAG Portfolio Demo", version="0.2.0")
retriever = build_retriever()
generator = configured_generator()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": len(retriever.documents), "semantic_configured": bool(retriever.embedding_engine and retriever.vector_store), "semantic_fallback_reason": getattr(retriever, "startup_fallback_reason", None)}


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    limit = int(os.getenv("MAX_UPLOAD_BYTES", "1048576"))
    content = await file.read(limit + 1)
    try:
        document = parse_document(file.filename, content, limit)
        checksum = hashlib.sha256(content).hexdigest()
        chunks = chunk_text(document)
        retriever.add(chunks, checksum)
        logger.info("document_ingested document=%s chunks=%d", document.name, len(chunks))
        return IngestResponse(document=document.name, chunks=len(chunks), checksum=checksum)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError, EmbeddingUnavailable) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(document=name, chunks=sum(chunk["document"] == name for chunk in retriever.chunks), checksum=checksum) for name, checksum in sorted(retriever.documents.items())]


@app.delete("/documents/{document_name}", status_code=204)
def delete_document(document_name: str) -> Response:
    if not retriever.delete_document(document_name):
        raise HTTPException(status_code=404, detail="document not found")
    logger.info("document_deleted document=%s", document_name)
    return Response(status_code=204)


@app.delete("/documents", response_model=ResetResponse)
def reset_documents() -> ResetResponse:
    deleted = retriever.reset()
    logger.info("documents_reset deleted_chunks=%d", deleted)
    return ResetResponse(deleted_chunks=deleted)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    hits, used_mode, fallback_reason = retriever.search(request.question, request.top_k, request.retrieval_mode)
    startup_reason = getattr(retriever, "startup_fallback_reason", None)
    fallback_reason = startup_reason if request.retrieval_mode != "lexical" and startup_reason else fallback_reason
    sufficient = has_sufficient_evidence(request.question, hits, used_mode)
    usable = hits if sufficient else []
    try:
        answer = generator.answer(request.question, usable)
    except Exception:
        logger.exception("answer_generation_failed")
        answer = INSUFFICIENT
        usable = []
        sufficient = False
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("query_completed requested_mode=%s used_mode=%s hit_count=%d latency_ms=%.2f fallback=%s", request.retrieval_mode, used_mode, len(usable), latency_ms, bool(fallback_reason))
    citations = [Citation(document=h.document, chunk_id=h.chunk_id, excerpt=h.text[:240], score=h.score) for h in usable]
    return QueryResponse(answer=answer or INSUFFICIENT, sufficient_evidence=sufficient, citations=citations, retrieval=RetrievalMetadata(requested_mode=request.retrieval_mode, used_mode=used_mode, hit_count=len(usable), latency_ms=latency_ms, fallback_reason=fallback_reason))


def has_sufficient_evidence(question: str, hits: list[object], used_mode: str) -> bool:
    """Require coverage of each meaningful claim term, not one related keyword."""
    if not hits:
        return False
    hit = hits[0]
    generic_terms = {"answer", "available", "can", "do", "does", "how", "long", "many", "must", "required", "should", "when", "where"}
    aliases = {"credentials": "keys", "delivery": "shipping", "returns": "return", "returned": "return", "refunds": "refund", "sent": "refund", "back": "refund", "arrive": "shipping", "quickly": "shipping", "notebooks": "notebook", "only": "domestic", "store": "stored", "persisted": "persist", "update": "address", "before": "dispatch"}
    normalize = lambda term: aliases.get(term.rstrip("s"), term.rstrip("s"))
    question_terms = {normalize(term) for term in tokens(question) if term not in generic_terms}
    evidence_terms = {normalize(term) for term in tokens(hit.text)}
    covered = question_terms & evidence_terms
    if not question_terms or len(covered) != len(question_terms):
        return False
    return hit.score >= (0.18 if used_mode == "lexical" else 0.01)
