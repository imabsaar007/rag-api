"""Vector store backends.

``InMemoryStore`` is the default and needs nothing but NumPy.
``RemoteVectorDBStore`` is an adapter for an external vector database that
exposes a small REST API (for example the C++ HNSW VectorDB project). Its
endpoint paths and JSON field names are class attributes so they can be
adjusted to match the real server without touching the rest of the code.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import requests


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity, higher is better


@dataclass
class _Record:
    chunk_id: str
    doc_id: str
    text: str
    vector: np.ndarray = field(repr=False)


class VectorStore(Protocol):
    def add(
        self, doc_id: str, chunks: list[str], vectors: list[list[float]]
    ) -> list[str]: ...

    def search(self, vector: list[float], k: int) -> list[Hit]: ...

    def delete_doc(self, doc_id: str) -> int: ...

    def stats(self) -> dict: ...


def _normalise(vec: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


class InMemoryStore:
    """Exact (brute-force) cosine search over normalised vectors."""

    def __init__(self) -> None:
        self._records: list[_Record] = []
        self._lock = threading.Lock()

    def add(
        self, doc_id: str, chunks: list[str], vectors: list[list[float]]
    ) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            # re-ingesting a doc replaces its old chunks
            self._records = [r for r in self._records if r.doc_id != doc_id]
            ids = []
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{doc_id}#{i}"
                self._records.append(_Record(chunk_id, doc_id, text, _normalise(vec)))
                ids.append(chunk_id)
            return ids

    def search(self, vector: list[float], k: int) -> list[Hit]:
        with self._lock:
            if not self._records:
                return []
            matrix = np.vstack([r.vector for r in self._records])
            records = list(self._records)
        scores = matrix @ _normalise(vector)
        order = np.argsort(-scores)[:k]
        return [
            Hit(records[i].chunk_id, records[i].doc_id, records[i].text, float(scores[i]))
            for i in order
        ]

    def delete_doc(self, doc_id: str) -> int:
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.doc_id != doc_id]
            return before - len(self._records)

    def stats(self) -> dict:
        with self._lock:
            docs = {r.doc_id for r in self._records}
            return {"documents": len(docs), "chunks": len(self._records)}


class RemoteVectorDBStore:
    """HTTP adapter for an external vector database.

    Assumed contract (edit the attributes below if your server differs):

    * ``POST {base}/insert``  body ``{"id", "vector", "metadata": {"doc_id", "text"}}``
    * ``POST {base}/search``  body ``{"vector", "k"}`` returning
      ``{"results": [{"id", "score", "metadata": {"doc_id", "text"}}]}``
      where ``score`` is a cosine similarity (higher is better).
    """

    insert_path = "/insert"
    search_path = "/search"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = requests.Session()
        self._doc_chunks: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(
        self, doc_id: str, chunks: list[str], vectors: list[list[float]]
    ) -> list[str]:
        ids = []
        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{doc_id}#{i}"
            resp = self._http.post(
                f"{self.base_url}{self.insert_path}",
                json={
                    "id": chunk_id,
                    "vector": list(map(float, vec)),
                    "metadata": {"doc_id": doc_id, "text": text},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            ids.append(chunk_id)
        with self._lock:
            self._doc_chunks[doc_id] = len(ids)
        return ids

    def search(self, vector: list[float], k: int) -> list[Hit]:
        resp = self._http.post(
            f"{self.base_url}{self.search_path}",
            json={"vector": list(map(float, vector)), "k": k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hits = []
        for item in resp.json().get("results", []):
            meta = item.get("metadata", {})
            hits.append(
                Hit(
                    chunk_id=str(item.get("id", "")),
                    doc_id=str(meta.get("doc_id", "")),
                    text=str(meta.get("text", "")),
                    score=float(item.get("score", 0.0)),
                )
            )
        return hits

    def delete_doc(self, doc_id: str) -> int:
        raise NotImplementedError(
            "Deleting documents is not supported by the remote VectorDB adapter"
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "documents": len(self._doc_chunks),
                "chunks": sum(self._doc_chunks.values()),
            }


def build_store(backend: str, vectordb_url: str) -> VectorStore:
    if backend == "memory":
        return InMemoryStore()
    if backend == "vectordb":
        return RemoteVectorDBStore(vectordb_url)
    raise ValueError(f"Unknown STORE_BACKEND: {backend!r} (use 'memory' or 'vectordb')")
