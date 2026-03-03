# Screenplay PDF Parser

FastAPI service that parses screenplay PDFs and optionally stores them in **Postgres with pgvector** for RAG (retrieval-augmented generation).

## Endpoints

- **POST /parse** — Parse PDF, return structured JSON (no DB).
- **POST /ingest** — Parse PDF, chunk by scene, store in Postgres. Optionally compute and store embeddings when `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set (Gemini gemini-embedding-001, 1536 dims).
- **POST /ask** — Ask a natural-language question about stored screenplays. Uses Gemini with function calling to query the DB (list screenplays, get scenes, semantic search over chunks). Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Database (Postgres + pgvector)

- **ORM:** SQLAlchemy 2.x (async with asyncpg).
- **Tables:** `screenplays`, `scenes`, `rag_chunks` (with vector column for embeddings).
- **Screenplay ID:** UUID v7 (time-ordered). If you previously used integer IDs, drop tables `screenplays`, `scenes`, and `rag_chunks` and restart the app to recreate with UUID primary keys.
- **Chunking:** Scene-level by default; each chunk = one scene (heading + action + dialogue).

Set the connection URL (defaults to local dev):

```bash
export DATABASE_URL="postgresql+asyncpg://dev:dev@localhost:5432/dev"
```

Start Postgres with pgvector (e.g. from repo root after `./scripts/start.sh` for the db stack). The `umt/postgres` image in `infra/db` already includes pgvector; the app runs `CREATE EXTENSION IF NOT EXISTS vector` on startup.

## Embeddings and Q&A (Gemini)

- Set **GOOGLE_API_KEY** or **GEMINI_API_KEY** for:
  - **Ingest:** `/ingest` computes embeddings with Gemini (gemini-embedding-001, 1536 dims) and stores them in `rag_chunks.embedding`.
  - **Ask:** `/ask` uses Gemini with function calling to answer questions by querying the DB (list screenplays, get scenes, semantic search over chunks).
- For similarity search in code, use `store.search_chunks(session, query_embedding, ...)`.

## OpenTelemetry (Aspire dashboard)

The service is instrumented with OpenTelemetry **traces** (FastAPI requests) and **logs** (Python `logging`). To send both to the [Aspire dashboard](https://learn.microsoft.com/en-us/dotnet/aspire/fundamentals/observability/telemetry) (e.g. `infra/telemetry/aspire.stack.dev.yaml`), set:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

- **Traces** appear in the Traces view (each request is a span).
- **Logs** appear in the Logs view; use `logging.info()`, `logger.debug()`, etc. and they will be exported to OTLP.

Optional: `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` to override per-signal. If no endpoint is set, telemetry is no-op.

## Running

Copy env template and set variables (e.g. `GOOGLE_API_KEY` or `GEMINI_API_KEY` for embeddings/Q&A, `OTEL_EXPORTER_OTLP_ENDPOINT` for the dashboard):

```bash
cp .env.example .env
# Edit .env as needed
```

Then:

```bash
cd apps/screenplay-parser
pip install -e .
uvicorn main:app --reload --host 0.0.0.0
```
