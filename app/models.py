from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    chunk_id: str
    excerpt: str
    score: float


class RetrievalMetadata(BaseModel):
    requested_mode: Literal["lexical", "semantic", "hybrid"]
    used_mode: Literal["lexical", "semantic", "hybrid"]
    hit_count: int
    latency_ms: float
    fallback_reason: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    sufficient_evidence: bool
    citations: list[Citation]
    retrieval: RetrievalMetadata


class IngestResponse(BaseModel):
    document: str
    chunks: int
    checksum: str


class DocumentInfo(BaseModel):
    document: str
    chunks: int
    checksum: str


class ResetResponse(BaseModel):
    deleted_chunks: int
