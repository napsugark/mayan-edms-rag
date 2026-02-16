# Mayan EDMS + Hybrid RAG System

An AI-assisted document search system built on top of [Mayan EDMS](https://www.mayan-edms.com/), a free open-source document management system. A Retrieval-Augmented Generation (RAG) backend indexes documents stored in Mayan, enriches them with LLM-extracted metadata and summaries, and provides natural-language question answering directly from within the Mayan UI.

> **Status:** This is a learning / portfolio project. It works end-to-end but has known limitations (see [Limitations & Assessment](#limitations--assessment) below). The RAG pipeline is specialized for **Romanian official documents** (invoices, contracts, receipts) and will need adaptation for other document types or languages.

---

## What This Project Does

1. **Mayan EDMS** manages documents — upload, OCR, tagging, permissions, workflows.
2. When a document is uploaded or updated, a custom **`rag_integration`** Django app inside Mayan sends the document content to the RAG server via a Celery task.
3. The **Hybrid RAG server** (FastAPI + Haystack) processes the document through a multi-stage indexing pipeline:
   - Document type detection (invoice, contract, receipt, offer, report)
   - LLM-powered metadata extraction (company, client, date, amount, invoice number, entities, topics)
   - Semantic chunking
   - Boilerplate filtering (legal disclaimers, payment terms, regulatory text)
   - Per-chunk summarization
   - Hybrid embedding (dense + sparse) stored in Qdrant
4. Users search from the Mayan UI — an **"AI assisted"** search button sends the query to the RAG server, which retrieves relevant chunks, reranks them, and generates a natural-language answer citing source documents.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Mayan EDMS                       │
│  (Django app with rag_integration custom module)    │
│                                                     │
│  Upload ──► Celery task ──► POST /index ────────┐   │
│  Search ──► AI button   ──► POST /query ─────┐  │   │
└──────────────────────────────────────────────┼──┼───┘
                                               │  │
┌──────────────────────────────────────────────┼──┼───┐
│              Hybrid RAG Server (FastAPI)      │  │   │
│                                               ▼  ▼   │
│  Indexing Pipeline:                                   │
│    Doc type detection → Metadata extraction           │
│    → Semantic chunking → Boilerplate filter           │
│    → Summarization → Dense + Sparse embeddings        │
│    → Store in Qdrant                                  │
│                                                       │
│  Query Pipeline:                                      │
│    Query metadata extraction → Hybrid retrieval       │
│    → Permission filtering → Cross-encoder reranking   │
│    → Prompt building → LLM → Answer                   │
└───────────────────┬───────────────┬───────────────────┘
                    │               │
              ┌─────▼─────┐  ┌──────▼──────┐
              │   Qdrant   │  │ Azure OpenAI│
              │  (vectors) │  │  or Ollama  │
              └────────────┘  └─────────────┘
```

All services run in Docker containers on a single bridge network.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document management | Mayan EDMS 4.10 (Django) |
| RAG framework | [Haystack 2.x](https://haystack.deepset.ai/) (deepset) |
| API server | FastAPI + Uvicorn |
| Dense embeddings | Snowflake/snowflake-arctic-embed-l-v2.0 (1024-dim) |
| Sparse embeddings | Qdrant/bm25 |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual) |
| Vector database | Qdrant |
| LLM | Azure OpenAI (gpt-4o-mini) or Ollama (local, e.g. qwen2.5:3b) |
| Task queue | Celery + RabbitMQ (Mayan side) |
| Observability | Langfuse (self-hosted, with ClickHouse + MinIO) |
| Database | PostgreSQL 15 (Mayan), PostgreSQL 17 (Langfuse) |
| Container orchestration | Docker Compose |

---

## Romanian Document Specialization

This system was designed and tested primarily with **Romanian official documents**: invoices (facturi), contracts (contracte), receipts (chitanțe), and utility bills. Several components reflect this:

- **Document type detector** — regex patterns for Romanian document types (`factură`, `chitanță`, `contract`, `ofertă`, `raport`)
- **Boilerplate filter** — removes common Romanian legal/regulatory boilerplate (ANRE references, payment terms like "plata se va efectua", penalty clauses, `condiții generale`, etc.)
- **Metadata extraction prompt** — tuned for Romanian invoice fields (CUI, `nr. factură`, RON amounts, Romanian date formats like `18.01.2025`)
- **Evaluation dataset** — test queries are in Romanian
- **Multilingual models** — the reranker (mmarco-mMiniLMv2) and embeddings (Arctic) support Romanian, but were not specifically fine-tuned for it

The system will work with documents in other languages (English, German, etc.), but the boilerplate filter and document type detection are most effective with Romanian content. Adapting to another language would require updating the regex patterns and prompt templates in the `prompts/` directory.

---

## Services & Ports

All containers are managed by a single `docker-compose.yml` at the project root.

| Service | Container | Port | Description |
|---|---|---|---|
| Mayan EDMS | `mayan-app-1` | 80 | Document management UI |
| RAG API | `rag` | 8001 | FastAPI server (Swagger at `/docs`) |
| Qdrant | `qdrant` | 6333 / 6334 | Vector database (REST / gRPC) |
| Ollama | `ollama` | 11434 | Local LLM (optional, profile: `ollama`) |
| Langfuse | `rag-langfuse-web` | 3111 | Observability UI |
| Mayan PostgreSQL | `mayan-postgresql-1` | — | Mayan database |
| RabbitMQ | `mayan-rabbitmq-1` | — | Celery broker |
| Mayan Redis | `mayan-redis-1` | — | Mayan cache & locks |
| Langfuse PostgreSQL | `rag-langfuse-pg` | — | Langfuse database |
| Langfuse Redis | `rag-langfuse-redis` | — | Langfuse cache |
| Langfuse ClickHouse | `rag-langfuse-ch` | — | Langfuse analytics |
| Langfuse MinIO | `rag-langfuse-minio` | 9190 | Langfuse S3 storage |
| Langfuse Worker | `rag-langfuse-worker` | — | Langfuse background processing |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2
- At least 8 GB RAM (16 GB recommended if using Ollama)
- An Azure OpenAI key **or** enough RAM/GPU for a local Ollama model

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd mayan-edms-rag

# 2. Create your .env from the example
cp .env.example .env
# Edit .env — fill in AZURE_OPENAI_API_KEY, MAYAN_API_TOKEN, etc.

# 3. Start everything
docker compose up -d --build

# 4. Wait for services to become healthy (RAG takes ~3-5 min on first boot
#    to download embedding models)
docker compose ps

# 5. Access the apps
#    Mayan EDMS:  http://localhost
#    RAG API:     http://localhost:8001/docs
#    Langfuse:    http://localhost:3111
#    Qdrant:      http://localhost:6333/dashboard
```

### First-time setup

1. Log into Mayan EDMS at http://localhost (default: `admin` / `admin`)
2. Generate an API token: *User menu → API tokens → Create*
3. Paste the token into `.env` as `MAYAN_API_TOKEN`
4. Restart the RAG container: `docker compose restart rag`
5. Upload some documents in Mayan — the `rag_integration` app will automatically send them to the RAG server for indexing

### Using Ollama instead of Azure OpenAI

```bash
# In .env, change:
MODEL_TO_USE=OLLAMA
# Make sure COMPOSE_PROFILES includes "ollama"
COMPOSE_PROFILES=all_in_one,postgresql,rabbitmq,redis,ollama

docker compose up -d
```

---

## Project Structure

```
mayan-edms-rag/
├── docker-compose.yml          # Unified compose — all services
├── .env                        # Secrets & configuration (gitignored)
├── .env.example                # Template with placeholder values
├── Makefile                    # Convenience commands (make up, make down, etc.)
│
├── hybrid_rag_qa/              # RAG server (FastAPI + Haystack)
│   ├── docker/
│   │   ├── Dockerfile          # Multi-stage Python build
│   │   ├── docker-compose.yml  # Standalone compose (reference only)
│   │   └── ollama-entrypoint.sh
│   ├── src/
│   │   ├── app.py              # Main application facade
│   │   ├── config.py           # Pydantic BaseSettings configuration
│   │   ├── cache.py            # Embedding, retrieval & response caches
│   │   ├── resilience.py       # Circuit breakers, rate limiters, retries
│   │   ├── langfuse_tracker.py # Langfuse/OpenTelemetry tracing
│   │   ├── api/
│   │   │   └── server.py       # FastAPI endpoints (/index, /query, /health)
│   │   ├── components/         # Custom Haystack components
│   │   │   ├── boilerplate_filter.py      # Romanian boilerplate removal
│   │   │   ├── document_type_detector.py  # Invoice/contract/receipt detection
│   │   │   ├── metadata_enricher.py       # LLM-based entity & field extraction
│   │   │   ├── query_metadata_extractor.py
│   │   │   ├── semantic_chunker.py
│   │   │   └── summarizer.py
│   │   ├── integrations/
│   │   │   └── mayan_client.py # Mayan EDMS REST API client
│   │   └── pipelines/
│   │       ├── indexing.py     # Document indexing pipeline
│   │       └── retrieval.py    # Query + generation pipeline
│   ├── prompts/                # LLM prompt templates (01-08)
│   ├── evaluation/             # Automated evaluation framework
│   ├── scripts/                # CLI utilities
│   └── tests/                  # Pytest test suite
│
└── mayan-edms-new/             # Mayan EDMS (with rag_integration app)
    ├── docker/
    │   └── docker-compose.yml  # Standalone compose (reference only)
    └── mayan/
        ├── apps/
        │   └── rag_integration/  # Custom Django app for AI search
        │       ├── apps.py       # App config, signal registration
        │       ├── handlers.py   # Post-save signal → Celery task
        │       ├── tasks.py      # Celery tasks (sync, bulk sync, resync)
        │       ├── models.py     # RAGDocumentVersionSync tracking model
        │       ├── views.py      # AI search results view
        │       ├── services.py   # HTTP calls to RAG server
        │       ├── templates/    # UI templates (search box, results page)
        │       └── ...
        └── settings/
            └── base.py           # Adds rag_integration to INSTALLED_APPS
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/index` | Index a document (multipart: file + metadata) |
| `POST` | `/query` | Natural-language search with permission filtering |
| `GET` | `/health` | Health check (Qdrant status, document count) |
| `GET` | `/status` | Detailed configuration status |
| `GET` | `/docs` | Swagger UI |

---

## Hybrid Search

The retrieval pipeline combines two complementary methods:

- **Sparse embeddings** (Qdrant/bm25) — keyword/term matching; good for exact names, invoice numbers, specific phrases
- **Dense embeddings** (Snowflake Arctic Embed L v2.0, 1024-dim) — semantic similarity; good for conceptual queries like "show me all contracts from last year"

Retrieved results are fused and re-scored by a **cross-encoder reranker** (mmarco-mMiniLMv2-L12-H384-v1, multilingual) to produce the final ranking.

---

## Observability

All RAG operations (indexing, retrieval, generation) are traced via **Langfuse** (self-hosted). Access the dashboard at http://localhost:3111 to see:

- Trace latency and token usage per query
- Full prompt/completion pairs
- Evaluation scores from automated test runs
- Cost tracking (when using Azure OpenAI)

---

## Limitations & Assessment

This is a working prototype / learning project. Here is a look at where it stands:

### What works well
- End-to-end flow: upload a document in Mayan → automatic indexing → AI search from the Mayan UI
- Hybrid search (sparse + dense) with cross-encoder reranking gives meaningfully better results than either method alone
- Metadata extraction is useful for structured filtering (by company, date, amount)
- The Langfuse integration provides real visibility into what the LLM is doing
- Docker Compose makes the full stack reproducible with a single command

### What needs improvement
- **Retrieval accuracy is not production-grade.** The system retrieves relevant documents for straightforward queries, but struggles with:
  - Ambiguous queries that require reasoning across multiple documents
  - Negation queries ("which invoices are NOT from company X")
  - Temporal reasoning ("invoices from the last 3 months")
- **Boilerplate filter is regex-based** and tuned specifically for Romanian invoices/utility bills. It will miss boilerplate in other document types or incorrectly filter content that looks like boilerplate but isn't.
- **Document type detection is pattern-matching only.** A document that doesn't contain expected Romanian keywords (e.g., `factură`, `contract`) will be classified as `other`.
- **No fine-tuning.** The embedding models and reranker are used off-the-shelf. Fine-tuning on a Romanian document corpus would likely improve retrieval quality significantly.
- **Evaluation dataset is small** (~20 test cases) and covers only Romanian queries. A larger, more diverse evaluation suite would give better confidence in accuracy metrics.
- **Permission filtering is basic** — based on user IDs passed by Mayan, not deeply integrated with Mayan's full ACL system.
- **The `rag_integration` Django app modifies upstream Mayan code.** It's mounted into the container via Docker volumes, which is fragile and will break if Mayan changes its internal paths between versions.
- **No chunking strategy optimization.** Semantic chunking is used with hardcoded parameters. Different document types might benefit from different chunk sizes or strategies.
- **LLM-dependent steps (metadata extraction, summarization) are slow** and add significant latency to indexing. With Azure OpenAI this is rate-limited; with Ollama on CPU it can be very slow.

### Potential improvements
- Fine-tune the embedding model on a Romanian document corpus
- Train a small classifier for document type detection instead of regex
- Implement query expansion and hypothetical document embeddings (HyDE)
- Add a feedback loop where users can rate answers to improve retrieval over time
- Move the boilerplate filter to an LLM-based approach for better generalization
- Implement incremental re-indexing (currently re-indexes the full document on any change)
- Add support for table extraction from PDFs (currently treats tables as plain text)

---

## Configuration

All RAG settings are defined in `hybrid_rag_qa/src/config.py` as a Pydantic `BaseSettings` class. Every field can be overridden via environment variables or the root `.env` file. See the `.env.example` for the most commonly changed values.

LLM prompt templates are stored in `hybrid_rag_qa/prompts/` as numbered text files (01-08) and can be edited without code changes.

---

## Development

### Running tests

```bash
cd hybrid_rag_qa
poetry install
poetry run pytest tests/
```

### Running evaluation

```bash
cd hybrid_rag_qa
poetry run python evaluation/run_evaluation.py
```

Results are saved to `evaluation/results/` and tracked in Langfuse.

### Local development (without Docker)

```bash
cd hybrid_rag_qa
cp .env.local.example .env
poetry install
poetry run api-server
```

Point the env vars at locally published Docker ports (Qdrant on 6333, Mayan on 80, etc.).

---

## License

- **Mayan EDMS** is licensed under the [Apache License 2.0](https://www.mayan-edms.com/)
- **This project** (the RAG server and rag_integration app) is a personal learning project
