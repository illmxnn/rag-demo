from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from .ingest import chunk_text, parse_document
from .models import Citation, IngestResponse, QueryRequest, QueryResponse
from .providers import INSUFFICIENT, configured_provider
from .retrieval import LocalRetriever

app = FastAPI(title="Small Document QA", version="0.1.0")
retriever = LocalRetriever()
provider = configured_provider()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "retrieval": "local-token-similarity"}


@app.post("/documents", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    limit = int(os.getenv("MAX_UPLOAD_BYTES", "1048576"))
    content = await file.read(limit + 1)
    try:
        document = parse_document(file.filename, content, limit)
        chunks = chunk_text(document)
        retriever.add(chunks)
        return IngestResponse(document=document.name, chunks=len(chunks))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    hits = retriever.search(request.question, request.top_k)
    sufficient = bool(hits and hits[0].score >= 0.18)
    usable = hits if sufficient else []
    answer = provider.answer(request.question, usable)
    citations = [Citation(document=h.document, chunk_id=h.chunk_id, excerpt=h.text[:240], score=h.score) for h in usable]
    return QueryResponse(answer=answer or INSUFFICIENT, sufficient_evidence=sufficient, citations=citations)
