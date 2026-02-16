# Hybrid RAG Server — Developer Reference

> For the full project overview, architecture, quick start, and deployment instructions, see the [root README](../README.md).

This document covers **RAG-server-specific** details for developers working in `hybrid_rag_qa/`.

---

## Local Development (without Docker)

Run the FastAPI server on the host while Qdrant, Langfuse, Mayan, and Ollama remain in Docker. Useful for rapid iteration with auto-reload.

```bash
pip install poetry
poetry install
cp .env.local.example .env
# Edit .env — point at published Docker ports:
#   QDRANT_ENDPOINT=http://localhost:6333
#   MAYAN_URL=http://localhost:8000
#   MODEL_TO_USE=AZURE_OPENAI (or OLLAMA with OLLAMA_URL=http://localhost:11434)

poetry run api-server          # starts FastAPI with auto-reload
poetry run pytest tests/       # run tests
```

---

## Connecting to Mayan EDMS

The RAG server calls Mayan's REST API to fetch OCR content for image files.

**From a container** (default): all services join the same Docker network, so Mayan is at `http://app:8000`.

**From the host** (local dev): use `http://localhost:8000` if Mayan publishes that port.

### Authentication

1. **API token** (preferred) — set `MAYAN_API_TOKEN` in `.env`. Generate one via Mayan UI: *Account → API tokens*.
2. **Username / password** — set `MAYAN_USERNAME` + `MAYAN_PASSWORD`; the client exchanges them for a token automatically.

---

## Environment Variables

Copy `.env.example` (Docker) or `.env.local.example` (host) to `.env`. Key variables:

| Variable | Default (Docker) | Description |
|---|---|---|
| `QDRANT_ENDPOINT` | `http://qdrant:6333` | Qdrant URL |
| `MODEL_TO_USE` | `AZURE_OPENAI` | `OLLAMA` or `AZURE_OPENAI` |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model name |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o-mini` | Azure deployment name |
| `MAYAN_URL` | `http://app:8000` | Mayan EDMS API base URL |
| `MAYAN_API_TOKEN` | — | Mayan API token |
| `WEBHOOK_SECRET` | — | Shared secret for webhook verification |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-local-dev-public` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | `sk-lf-local-dev-secret` | Langfuse secret key |
| `LANGFUSE_HOST` | `http://langfuse-web:3000` | Langfuse URL |

Full list in `src/config.py` (Pydantic `BaseSettings` — all values can be overridden via env vars or `.env`).

---

## CLI Scripts

| Script | Poetry shortcut | Description |
|---|---|---|
| `scripts/run_api_server.py` | `poetry run api-server` | Start the FastAPI server (dev mode) |
| `scripts/create_payload_indexes.py` | — | Create Qdrant payload indexes for metadata filtering |
| `scripts/recreate_collection.py` | — | Drop and recreate the Qdrant collection |
| `evaluation/run_evaluation.py` | — | Run the automated evaluation suite |

> `scripts/index_documents.py` and `scripts/query_app.py` exist for legacy/debug use but require a local `data/documents_ro/` folder.

---

## LLM Prompt Templates

Stored in `prompts/` as numbered text files — editable without code changes:

| File | Purpose |
|---|---|
| `01_boilerplate_detection.txt` | Detect boilerplate sections |
| `02_metadata_extraction.txt` | Extract structured metadata |
| `03_summarization.txt` | Generate chunk summaries |
| `04_query_extraction.txt` | Extract filters from user queries |
| `05_rag_system.txt` | RAG system prompt |
| `06_rag_user.txt` | RAG user prompt (default) |
| `07_rag_user_concise.txt` | RAG user prompt (concise variant, currently active) |
| `08_rag_user_structured.txt` | RAG user prompt (structured variant) |

---

## Configuration Categories

All settings live in `src/config.py`. Key categories:

- **LLM selection** — `MODEL_TO_USE`, Ollama settings, Azure OpenAI settings
- **Embedding models** — dense (Snowflake Arctic), sparse (Qdrant/bm25)
- **Document processing** — chunk size/overlap, semantic chunking, boilerplate filtering, document-type detection
- **Retrieval** — `TOP_K`, reranker model, hybrid search weights
- **Metadata enrichment** — fields to extract (company, client, date, amount, entities)
- **Summarization** — which section types get summaries, max length, style
- **Resilience** — circuit breaker thresholds, rate limits, retry config
- **Caching** — embedding / retrieval / response cache sizes and TTLs
- **Langfuse** — keys and host for observability

---

## Performance Tips

- **GPU acceleration** — set `EMBEDDING_DEVICE=cuda` in `.env` if you have a GPU
- **Batch sizes** — increase `INDEXING_BATCH_SIZE` / `QUERY_BATCH_SIZE` for throughput
- **Ollama** — use quantized models (e.g. `llama3.1:8b-q4_0`) for faster inference
- **Caching** — embedding and retrieval caches are enabled by default; tune TTLs in config

---

## Troubleshooting

### Qdrant connection
```bash
curl http://localhost:6333/collections                        # from host
docker exec rag curl http://qdrant:6333/collections           # from container
```

### Ollama connection
```bash
curl http://localhost:11434/api/tags
```

### Mayan connectivity
```bash
docker exec rag curl http://app:8000/api/v4/
```

If you get connection errors, verify:
- The Docker network exists (`docker network ls`)
- The Mayan web service is reachable on that network
- Your `MAYAN_API_TOKEN` is valid

### Out of memory
- Reduce `INDEXING_BATCH_SIZE` in `.env`
- Use a smaller embedding model
- Process documents in smaller batches
