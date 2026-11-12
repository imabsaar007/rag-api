"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__
from .config import Settings
from .ollama_client import OllamaClient, OllamaError
from .rag import RAGService
from .schemas import DocumentIn, DocumentOut, HealthOut, QueryIn, QueryOut, SourceOut
from .vectorstore import build_store

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB of plain text is plenty for a demo


def build_service(settings: Settings) -> RAGService:
    ollama = OllamaClient(
        settings.ollama_url,
        settings.embed_model,
        settings.llm_model,
        timeout=settings.request_timeout,
    )
    return RAGService(
        store=build_store(settings.store_backend, settings.vectordb_url),
        embedder=ollama,
        llm=ollama,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )


def create_app(service: RAGService | None = None) -> FastAPI:
    """App factory. Tests pass in a service wired with fake embedder/LLM."""
    app = FastAPI(
        title="RAG API",
        version=__version__,
        description="Ingest documents and ask questions answered from them (RAG).",
    )
    rag = service or build_service(Settings.from_env())

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return HealthOut(status="ok", **rag.store.stats())

    @app.post("/documents", response_model=DocumentOut, status_code=201, tags=["documents"])
    def add_document(doc: DocumentIn) -> DocumentOut:
        try:
            chunks = rag.ingest(doc.doc_id, doc.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return DocumentOut(doc_id=doc.doc_id, chunks=chunks)

    @app.post(
        "/documents/upload", response_model=DocumentOut, status_code=201, tags=["documents"]
    )
    async def upload_document(
        file: UploadFile = File(...), doc_id: str | None = Form(default=None)
    ) -> DocumentOut:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large (max 2 MB)")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=415, detail="only UTF-8 text files are supported") from exc
        name = doc_id or file.filename or "upload"
        try:
            chunks = rag.ingest(name, text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return DocumentOut(doc_id=name, chunks=chunks)

    @app.delete("/documents/{doc_id}", tags=["documents"])
    def delete_document(doc_id: str) -> dict:
        try:
            removed = rag.store.delete_doc(doc_id)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if removed == 0:
            raise HTTPException(status_code=404, detail="document not found")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post("/query", response_model=QueryOut, tags=["query"])
    def query(body: QueryIn) -> QueryOut:
        try:
            result = rag.query(body.question, body.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryOut(
            answer=result.answer,
            answered=result.answered,
            sources=[
                SourceOut(doc_id=h.doc_id, chunk_id=h.chunk_id, score=round(h.score, 4), text=h.text)
                for h in result.sources
            ],
        )

    return app


def _default_app() -> FastAPI:
    return create_app()


# `uvicorn ragapi.main:app` serves this; created lazily so importing the module
# in tests does not require a running Ollama.
def __getattr__(name: str):
    if name == "app":
        return _default_app()
    raise AttributeError(name)
