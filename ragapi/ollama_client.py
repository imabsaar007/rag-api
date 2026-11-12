"""Thin client for the Ollama HTTP API (embeddings and chat generation)."""

from __future__ import annotations

from typing import Protocol

import requests


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Implements both the Embedder and LLM protocols against an Ollama server."""

    def __init__(
        self,
        base_url: str,
        embed_model: str,
        llm_model: str,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.timeout = timeout
        self._http = session or requests.Session()

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._http.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request to {path} failed: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": text})
        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise OllamaError("Ollama returned no embedding")
        return embeddings[0]

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self._post(
            "/api/chat",
            {
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise OllamaError("Unexpected response from Ollama chat API") from exc
