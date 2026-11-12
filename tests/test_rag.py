import pytest

from ragapi.rag import FALLBACK_ANSWER, RAGService
from ragapi.vectorstore import InMemoryStore

REFUND = "Our refund window is 30 days from the date of purchase for all items."
SHIPPING = "Standard shipping takes five business days within the country."


def test_ingest_returns_chunk_count(service, embedder):
    n = service.ingest("policy", REFUND + " " + SHIPPING * 5)
    assert n >= 1
    assert embedder.calls == n


def test_ingest_rejects_blank_text(service):
    with pytest.raises(ValueError):
        service.ingest("empty", "   ")


def test_relevant_question_is_answered_with_sources(service, llm):
    service.ingest("refunds", REFUND)
    service.ingest("shipping", SHIPPING)
    result = service.query("How long is the refund window?")
    assert result.answered
    assert result.answer == llm.answer
    assert result.sources[0].doc_id == "refunds"
    assert "refund window" in llm.prompts[0]


def test_prompt_contains_guardrail_instructions_and_few_shot(service, llm):
    service.ingest("refunds", REFUND)
    service.query("refund window days")
    assert "ONLY" in llm.systems[0]
    assert "Example" in llm.prompts[0]


def test_weak_retrieval_triggers_fallback_without_calling_llm(embedder, llm):
    svc = RAGService(InMemoryStore(), embedder, llm, min_score=0.9)
    svc.ingest("refunds", REFUND)
    result = svc.query("quantum chromodynamics lagrangian")
    assert not result.answered
    assert result.answer == FALLBACK_ANSWER
    assert llm.prompts == []


def test_empty_store_triggers_fallback(service, llm):
    result = service.query("anything at all")
    assert not result.answered
    assert result.sources == []
    assert llm.prompts == []


def test_model_refusal_is_reported_as_not_answered(service, llm):
    llm.answer = "I don't have enough information to answer that."
    service.ingest("refunds", REFUND)
    result = service.query("refund window days")
    assert not result.answered


def test_blank_question_raises(service):
    with pytest.raises(ValueError):
        service.query("   ")


def test_top_k_override(service):
    for i in range(5):
        service.ingest(f"doc{i}", f"refund policy variant number {i} applies here")
    assert len(service.query("refund policy", top_k=2).sources) == 2
