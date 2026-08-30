# Hybrid RAG Portfolio Demo

An honest local document-QA demonstration: bounded ingestion, lexical retrieval, optional local sentence-transformer embeddings, persistent Qdrant vectors, hybrid reciprocal-rank fusion (RRF), citations, and conservative unsupported-question handling. It is a learning/portfolio project, **not production-ready**.

## Architecture

```text
                  +-> persistent JSON document registry
upload -> parse -> chunks -> lexical IDF index ----+ 
                  +-> local embeddings -> Qdrant --+-> RRF -> evidence gate -> answer + citations
query -> lexical | semantic | hybrid ----------------^                         \-> retrieval metadata/logs
```

`lexical` has no model dependency. `semantic` uses `sentence-transformers/all-MiniLM-L6-v2` locally and Qdrant. `hybrid` fuses lexical and semantic rankings using RRF (`1 / (60 + rank)`). If the optional embedding/vector backend is unavailable, semantic/hybrid requests explicitly fall back to lexical and return `fallback_reason` in response metadata.

## Setup

Requires Python 3.11+.

### Lexical-only (offline, no model download)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,documents]"
uvicorn app.main:app --reload
```

### Semantic/vector mode

```bash
python -m pip install -e ".[dev,documents,semantic]"
docker compose up -d qdrant
# The first semantic query downloads the compact public model into the local model cache.
# PowerShell:
$env:SEMANTIC_ENABLED = "true"
$env:QDRANT_URL = "http://localhost:6333"
uvicorn app.main:app --reload
```

Or run the complete stack with `SEMANTIC_ENABLED=true docker compose up --build`. Compose persists document metadata in `rag-data` and vectors in `qdrant-data` volumes.

## API

```bash
curl -F "file=@evaluation/fixtures/product-handbook.txt" http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"question":"What is the return window?","retrieval_mode":"hybrid"}'
curl -X DELETE http://127.0.0.1:8000/documents/product-handbook.txt
curl -X DELETE http://127.0.0.1:8000/documents
```

`POST /documents` rejects duplicate filenames (409). `GET /documents`, `DELETE /documents/{name}`, and `DELETE /documents` provide local corpus management. Query metadata records requested/used mode, hit count, latency, and any safe fallback. Logs contain only mode, counts, latency and error class—not keys, question content, or document text.

## Evaluation and verification

The synthetic corpus has 16 questions (13 supported, 3 unsupported). It injects a **deterministic in-memory embedding and vector-store fixture**; it is not a SentenceTransformer/Qdrant benchmark and never downloads a model. Evaluation exits nonzero unless every declared unsupported question is rejected by the deterministic evidence gate.

```bash
python -m compileall -q app tests
pytest -q
python -m app.evaluate
docker compose config
```

Measured on this checkout (offline deterministic evaluation):

| Mode | Retrieval hit rate | Unsupported rejection | Average latency |
|---|---:|---:|---:|
| lexical | 11/13 (85%) | 3/3 (100%) | host-dependent |
| semantic | 10/13 (77%) | 3/3 (100%) | host-dependent |
| hybrid | 11/13 (85%) | 3/3 (100%) | host-dependent |

The hit-rate values were measured locally with the offline deterministic fixture; rerun it because latency is host-dependent. The evaluation is a regression fixture, not a benchmark. It measures whether expected synthetic evidence appears first and whether the deterministic evidence gate rejects declared unsupported prompts.

## Optional answer provider

Local extraction is the default. An OpenAI-compatible provider is opt-in with `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`; it has a bounded timeout and one retry by default. A provider failure returns the conservative insufficient-evidence response. Keys remain environment-only and are never logged.

## Correct interview claims and limitations

You can accurately say this project implements local semantic embeddings, a persistent Qdrant vector index, lexical fallback, RRF hybrid ranking, offline-injected tests, basic document lifecycle controls, explicit unsupported handling, and an optional guarded answer-provider boundary.

Do **not** claim production readiness, data isolation, authentication/authorization, malware scanning, OCR, tenant controls, observability infrastructure, benchmark-grade quality, or that semantic mode works without first downloading the model and running Qdrant. The JSON metadata registry and Qdrant operations are intentionally small-demo scope; cross-process ingest locking, robust recovery, reranking, access control, and operational deployment are future work.
