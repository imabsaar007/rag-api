"""API tests. Skipped automatically if FastAPI / httpx are not installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ragapi.main import create_app  # noqa: E402


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health_empty(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "documents": 0, "chunks": 0}


def test_ingest_then_query(client):
    r = client.post(
        "/documents", json={"doc_id": "faq", "text": "Our refund window is 30 days."}
    )
    assert r.status_code == 201
    assert r.json()["doc_id"] == "faq"

    r = client.post("/query", json={"question": "What is the refund window?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is True
    assert body["sources"][0]["doc_id"] == "faq"
    assert set(body["sources"][0]) == {"doc_id", "chunk_id", "score", "text"}


def test_query_with_nothing_ingested_is_not_answered(client):
    body = client.post("/query", json={"question": "hello?"}).json()
    assert body["answered"] is False
    assert body["sources"] == []


def test_validation_errors(client):
    assert client.post("/documents", json={"doc_id": "x"}).status_code == 422
    assert client.post("/documents", json={"doc_id": "x", "text": "   "}).status_code == 422
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "q", "top_k": 0}).status_code == 422


def test_upload_text_file(client):
    files = {"file": ("notes.txt", b"Our refund window is 30 days.", "text/plain")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 201
    assert r.json()["doc_id"] == "notes.txt"


def test_upload_rejects_binary(client):
    files = {"file": ("x.bin", b"\xff\xfe\x00\x01", "application/octet-stream")}
    assert client.post("/documents/upload", files=files).status_code == 415


def test_delete_document(client):
    client.post("/documents", json={"doc_id": "faq", "text": "Refunds take 30 days."})
    assert client.delete("/documents/faq").status_code == 200
    assert client.delete("/documents/faq").status_code == 404
