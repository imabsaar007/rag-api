"""Pydantic request/response models (these also drive the Swagger docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=200, examples=["handbook"])
    text: str = Field(..., min_length=1, examples=["Our refund window is 30 days."])


class DocumentOut(BaseModel):
    doc_id: str
    chunks: int


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, examples=["How long is the refund window?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    answered: bool = Field(
        description="False when the guardrail declined to answer (weak retrieval or model refusal)."
    )
    sources: list[SourceOut]


class HealthOut(BaseModel):
    status: str
    documents: int
    chunks: int
