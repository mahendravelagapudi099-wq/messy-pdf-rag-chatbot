"""High-level RAG facade.

:class:`RAGEngine` wires the pipeline together behind one object so the Streamlit
UI (and tests, and the benchmark) never touch individual modules. Application
logic stays here; ``app.py`` stays presentational.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .chunking import PageAwareChunker
from .config import Settings, get_settings
from .citations import build_citations
from .generation import generate_answer
from .ingestion import ingest_bytes, ingest_file
from .ingestion.ocr import probe as probe_ocr
from .models import Answer, Chunk, IngestionReport, RetrievedChunk
from .retrieval import RetrievalResult, retrieve
from .vector_store import DocumentInfo, VectorStore

log = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Environment capability report, shown on the settings page."""

    llm_configured: bool
    llm_detail: str
    ocr_detail: str
    embedding_detail: str
    doc_count: int
    chunk_count: int

    @property
    def ready(self) -> bool:
        return self.chunk_count > 0


class RAGEngine:
    """Facade over ingestion, retrieval and generation."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = VectorStore(self.settings, persist_dir=persist_dir)
        self._chunker = PageAwareChunker(
            self.settings.chunk_size, self.settings.chunk_overlap
        )

    # -- configuration ---------------------------------------------------
    def with_settings(self, settings: Settings) -> "RAGEngine":
        """Return an engine bound to new settings (chunking re-created)."""
        engine = RAGEngine(settings, persist_dir=self.store.persist_dir)
        engine.store = self.store  # reuse the loaded collection
        return engine

    @property
    def chunker(self) -> PageAwareChunker:
        return self._chunker

    # -- ingestion -------------------------------------------------------
    def add_upload(
        self, filename: str, data: bytes, *, force: bool = False,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> IngestionReport:
        return ingest_bytes(
            filename, data, self.store, self._chunker, self.settings,
            progress=progress, force=force,
        )

    def add_file(self, path: Path | str, *, force: bool = False, progress: Optional[Callable[[str, float], None]] = None) -> IngestionReport:
        return ingest_file(Path(path), self.store, self._chunker, self.settings, force=force, progress=progress)

    def add_directory(self, directory: Path | str, *, force: bool = False) -> list[IngestionReport]:
        return [
            self.add_file(p, force=force)
            for p in sorted(Path(directory).glob("*.pdf"))
            if p.is_file()
        ]

    def remove_document(self, doc_id: str) -> int:
        return self.store.delete_document(doc_id)

    def clear_index(self) -> None:
        self.store.clear()

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    # -- query -----------------------------------------------------------
    def search(
        self, question: str, *, doc_ids: Optional[Sequence[str]] = None
    ) -> RetrievalResult:
        s = self.settings
        return retrieve(
            self.store,
            question,
            top_k=s.top_k,
            similarity_threshold=s.similarity_threshold,
            use_reranker=s.use_reranker,
            reranker_model=s.reranker_model,
            reranker_candidates=s.reranker_candidates,
            doc_ids=doc_ids,
        )

    def ask(
        self,
        question: str,
        *,
        history: Optional[list[dict[str, str]]] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> Answer:
        """Retrieve then answer, attaching citations. Never raises."""
        try:
            retrieval = self.search(question, doc_ids=doc_ids)
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            return Answer(question, "", error=f"Retrieval failed: {exc}")

        answer = generate_answer(question, retrieval.chunks, self.settings, history=history)
        answer.citations = build_citations(answer.retrieved, answer.text)
        answer.latency_s = answer.latency_s + retrieval.latency_s
        return answer

    def page_excerpts(self, doc_id: str, page: int) -> list[Chunk]:
        """All stored chunks from one page, for the citation viewer."""
        return self.store.get_page_text(doc_id, page)

    # -- diagnostics -----------------------------------------------------
    def status(self) -> SystemStatus:
        s = self.settings
        try:
            ocr = probe_ocr(s.ocr_engine)
            ocr_detail = str(ocr)
        except Exception as exc:
            ocr_detail = f"probe failed: {exc}"

        try:
            emb = self.store.provider.name
        except Exception as exc:
            emb = f"unavailable: {exc}"

        llm_detail = (
            f"{s.llm_model} via {s.llm_base_url}" if s.llm_configured
            else "not configured - retrieval-only mode (set LLM_API_KEY)"
        )
        return SystemStatus(
            llm_configured=s.llm_configured,
            llm_detail=llm_detail,
            ocr_detail=ocr_detail,
            embedding_detail=emb,
            doc_count=len(self.store.list_documents()),
            chunk_count=self.store.count(),
        )