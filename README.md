# RAG API

A small, Dockerized Retrieval-Augmented Generation service. Upload documents, ask questions, and get answers grounded in those documents, with the source passages returned alongside every answer.

- **FastAPI** with Pydantic validation and auto-generated Swagger docs at `/docs`
- **Local LLM stack** via [Ollama](https://ollama.com): `nomic-embed-text` for embeddings, `llama3.2` for generation
- **Pluggable vector store**: in-memory exact cosine search by default, or an external vector database over HTTP (built for a C++ HNSW VectorDB)
- **Hallucination guardrails**: a retrieval-score threshold, a context-only system prompt with few-shot examples, and detection of model refusals
- **Evaluation harness** with answerable and unanswerable questions
- **pytest** suite using fake embedder/LLM (no Ollama needed) and a **GitHub Actions** workflow

## Architecture

```
            POST /documents            POST /query
                  |                        |
                  v                        v
 text --> chunk (overlap) --> embed    question --> embed
                                |                      |
                                v                      v
                         +--------------+   top-k   +---------+
                         | Vector store | <-------- | search  |
                         +--------------+           +----+----+
                                                         |
                          best score < MIN_SCORE? -------+--> fallback answer (LLM not called)
                                                         |
                                          context + few-shot prompt --> Ollama LLM --> answer + sources
```

## Quick start (Docker)

```bash
docker compose up --build
```

The first run downloads the two Ollama models (a few GB), so give it a few minutes. When it is up:

```bash
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"doc_id":"faq","text":"Our refund window is 30 days from purchase."}'

curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"How long is the refund window?"}'
```

Interactive docs: <http://localhost:8000/docs>

## Run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ollama pull nomic-embed-text && ollama pull llama3.2   # Ollama must be running
uvicorn ragapi.main:app --reload
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness plus document and chunk counts |
| POST | `/documents` | Ingest `{doc_id, text}`; re-ingesting a `doc_id` replaces it |
| POST | `/documents/upload` | Ingest a UTF-8 text file (multipart, max 2 MB) |
| DELETE | `/documents/{doc_id}` | Remove a document (in-memory store only) |
| POST | `/query` | Ask `{question, top_k?}`; returns `answer`, `answered`, `sources` |

`answered: false` means the guardrail declined, either because retrieval was weak or because the model said the context did not contain the answer.

## Configuration

All settings are environment variables (see `ragapi/config.py`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `EMBED_MODEL` / `LLM_MODEL` | `nomic-embed-text` / `llama3.2` | Models |
| `STORE_BACKEND` | `memory` | `memory` or `vectordb` |
| `VECTORDB_URL` | `http://localhost:8080` | External vector DB base URL |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters per chunk / overlap |
| `TOP_K` | `3` | Chunks passed to the LLM |
| `MIN_SCORE` | `0.5` | Minimum cosine similarity of the best chunk to answer |

`MIN_SCORE` depends on your embedding model and data. Tune it with the eval script: too high and answerable questions are refused, too low and off-topic questions slip through.

## Using your own vector database

Set `STORE_BACKEND=vectordb` and `VECTORDB_URL`. `RemoteVectorDBStore` in `ragapi/vectorstore.py` assumes:

- `POST /insert` with `{"id", "vector", "metadata": {"doc_id", "text"}}`
- `POST /search` with `{"vector", "k"}`, returning `{"results": [{"id", "score", "metadata": {...}}]}` where `score` is cosine similarity

If your server uses different paths or field names, change the class attributes and the two request/response mappings. They are the only place that knows about the wire format.

## Tests

```bash
pytest
```

Chunking, vector store and RAG logic tests use a deterministic fake embedder and a scripted LLM, so they run in under a second with no network. API tests use FastAPI's `TestClient` and need `httpx` (included in `requirements-dev.txt`).

## Evaluation

With the stack running:

```bash
python eval/run_eval.py --base-url http://localhost:8000 --output eval/results.json
```

`eval/eval_set.json` ships 16 questions over 4 short policy documents: 14 answerable (pass if the answer contains an expected phrase) and 2 unanswerable (pass if the guardrail declines). It is a smoke test for regressions when you change chunking, prompts or `MIN_SCORE`, not a benchmark. Replace the documents and questions with your own domain for meaningful numbers.

Record your own results here after running it:

| Config | Passed | Notes |
| --- | --- | --- |
| _default_ | _x/16_ | _run it and fill this in_ |

## Project layout

```
ragapi/          application code
  main.py        FastAPI app and routes
  rag.py         ingest / retrieve / guardrail / generate
  vectorstore.py in-memory store and remote VectorDB adapter
  ollama_client.py  embeddings + chat over HTTP
  chunking.py    overlapping chunker
tests/           pytest suite
eval/            evaluation set and runner
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```
