"""Shared test doubles: a deterministic embedder and a scripted LLM."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragapi.rag import RAGService  # noqa: E402
from ragapi.vectorstore import InMemoryStore  # noqa: E402

DIM = 256


class FakeEmbedder:
    """Hashed bag-of-words embedding: texts sharing words get similar vectors."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        vec = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        return vec.tolist()


class FakeLLM:
    """Records prompts and returns a canned answer."""

    def __init__(self, answer: str = "The refund window is 30 days [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.answer


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(embedder, llm) -> RAGService:
    return RAGService(
        store=InMemoryStore(),
        embedder=embedder,
        llm=llm,
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        min_score=0.3,
    )
