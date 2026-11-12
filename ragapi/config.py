"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    # "memory" keeps vectors in-process; "vectordb" talks to an external
    # vector database over HTTP (see ragapi/vectorstore.py).
    store_backend: str = "memory"
    vectordb_url: str = "http://localhost:8080"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120  # characters
    top_k: int = 3
    # Guardrail: if the best retrieved chunk scores below this cosine
    # similarity, the API refuses to answer instead of guessing.
    min_score: float = 0.5
    request_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_url=_get("OLLAMA_URL", cls.ollama_url),
            embed_model=_get("EMBED_MODEL", cls.embed_model),
            llm_model=_get("LLM_MODEL", cls.llm_model),
            store_backend=_get("STORE_BACKEND", cls.store_backend),
            vectordb_url=_get("VECTORDB_URL", cls.vectordb_url),
            chunk_size=int(_get("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(_get("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            top_k=int(_get("TOP_K", str(cls.top_k))),
            min_score=float(_get("MIN_SCORE", str(cls.min_score))),
            request_timeout=float(_get("REQUEST_TIMEOUT", str(cls.request_timeout))),
        )
