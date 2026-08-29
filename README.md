# Small Document QA

A standalone, public-safe portfolio demonstration of a document question-answering pipeline. It ingests synthetic text documents, creates bounded chunks, retrieves evidence with a deterministic local token-similarity index, and returns citations. An optional OpenAI-compatible adapter can generate an answer, but the default path is fully local and the test suite uses a deterministic fake provider.

This is a learning/demo project, not a production service and not evidence of external model usage.

## Architecture

```text
upload -> validation/parsing -> normalized document -> chunks -> local retriever
                                                        \-> optional answer provider
query  -> retrieval -> evidence threshold -> answer + source citations
```

- **FastAPI + Pydantic** expose `/health`, `/documents`, and `/query`.
- **Parsers** support TXT by default. PDF (`pypdf`) and DOCX (`python-docx`) are optional dependencies and are rejected with a clear error when unavailable.
- **Retrieval** uses normalized token overlap with IDF weighting. It has no network or credentials requirement.
- **Providers** implement a tiny protocol. `LocalExtractiveProvider` is deterministic; `OpenAICompatibleProvider` is opt-in through environment variables only.
- **Safety** enforces extensions, byte limits, filename/path normalization, prompt-injection-resistant behavior (retrieved text is evidence, never instructions), and an explicit insufficient-evidence response.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,documents]"
uvicorn app.main:app --reload
```

OpenAPI is available at http://127.0.0.1:8000/docs.

The minimal install is `python -m pip install -e ".[dev]"`. Add `[documents]` for PDF/DOCX parsing. No provider key is needed for tests or local extraction.

## API examples

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@evaluation/fixtures/product-handbook.txt" http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is the return window?\"}"
```

The query response contains `answer`, `sufficient_evidence`, and `citations` with document name, chunk id, and a short excerpt. Questions below the evidence threshold return a conservative “insufficient evidence” answer rather than guessing.

## Docker

```bash
docker compose up --build
```

The container stores uploaded documents only in its ephemeral `/tmp/rag-demo-data` volume. Do not mount sensitive material into it.

## Evaluation

`evaluation/questions.json` contains a small synthetic corpus and expected evidence keywords. Run:

```bash
pytest -q
python -m app.evaluate
```

The evaluation reports measured retrieval hit rate for the included questions. It is intentionally small and is not a benchmark or a quality claim about real-world data.

## Configuration

Copy `.env.example` to `.env` if desired. `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` enable the optional OpenAI-compatible provider. The API key is read only from the environment and is never logged or stored. If any required setting is absent, the service remains local and deterministic.

## Limitations and security notes

- In-memory storage is used; restarts remove the index.
- Token overlap is intentionally simple and does not provide semantic understanding, reranking, access control, OCR, malware scanning, authentication, or tenant isolation.
- PDF/DOCX parsing depends on optional libraries and should be treated as untrusted input handling, not a complete file security boundary.
- Uploads are size-limited, extension allowlisted, decoded safely, and never interpreted as filesystem paths. Filenames are metadata only.
- Retrieved content can contain prompt-injection text. Providers receive it in a clearly delimited evidence section, and the application never treats retrieved text as executable instructions.
- Synthetic fixtures contain no private, customer, or real candidate data.

## Exact verification commands

```bash
python -m compileall -q app tests
pytest -q
python -m app.evaluate
docker compose config
```
