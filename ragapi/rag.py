"""RAG orchestration: ingest, retrieve, guardrail, generate."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_text
from .ollama_client import Embedder, LLM
from .vectorstore import Hit, VectorStore

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the passages do not contain the answer, reply exactly: "
    "I don't have enough information to answer that. "
    "Keep answers short and factual, and cite passage numbers like [1]."
)

FEW_SHOT = (
    "Example\n"
    "Context:\n[1] The warehouse opens at 9 am and closes at 6 pm on weekdays.\n"
    "Question: When does the warehouse close?\n"
    "Answer: It closes at 6 pm on weekdays [1].\n\n"
    "Example\n"
    "Context:\n[1] The warehouse opens at 9 am and closes at 6 pm on weekdays.\n"
    "Question: Who is the warehouse manager?\n"
    "Answer: I don't have enough information to answer that.\n"
)

FALLBACK_ANSWER = (
    "I don't have enough information in the ingested documents to answer that."
)


@dataclass
class QueryResult:
    answer: str
    answered: bool
    sources: list[Hit]


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 3,
        min_score: float = 0.5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score

    def ingest(self, doc_id: str, text: str) -> int:
        """Chunk, embed and store a document. Returns the number of chunks."""
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("document has no text to ingest")
        vectors = [self.embedder.embed(c) for c in chunks]
        self.store.add(doc_id, chunks, vectors)
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        k = top_k or self.top_k
        hits = self.store.search(self.embedder.embed(question), k)

        # Guardrail: nothing retrieved, or the best match is too weak.
        if not hits or hits[0].score < self.min_score:
            return QueryResult(FALLBACK_ANSWER, answered=False, sources=hits)

        context = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
        prompt = f"{FEW_SHOT}\nNow answer.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        # Guardrail: the model itself declined, so report that honestly.
        declined = answer.lower().startswith("i don't have enough information")
        return QueryResult(answer, answered=not declined, sources=hits)
