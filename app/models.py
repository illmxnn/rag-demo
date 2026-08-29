from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    chunk_id: str
    excerpt: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    sufficient_evidence: bool
    citations: list[Citation]


class IngestResponse(BaseModel):
    document: str
    chunks: int
