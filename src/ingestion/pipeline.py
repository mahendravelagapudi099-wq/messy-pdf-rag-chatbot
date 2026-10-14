"""Ingestion pipeline: validate -> extract -> OCR -> chunk -> embed -> store.

This module owns the end-to-end flow and is the only place that coordinates the
individual ingestion stages. It returns an :class:`IngestionReport` describing
precisely what happened, which the UI renders verbatim.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..config import Settings, get_settings
from ..models import Chunk, ExtractedDocument, IngestionReport
from .extractor import extract_pdf
from .validator import safe_filename, validate_upload

log = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Minimal interface the pipeline needs from a store (keeps this testable)."""

    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...
    def has_document(self, doc_id: str) -> bool: ...


class Chunker(Protocol):
    def chunk(self, doc: ExtractedDocument) -> list[Chunk]: ...


ProgressFn = Callable[[str, float], None]


def ingest_bytes(
    filename: str,
    data: bytes,
    store: VectorStore,
    chunker: Chunker,
    settings: Optional[Settings] = None,
    *,
    progress: Optional[ProgressFn] = None,
    force: bool = False,
) -> IngestionReport:
    """Validate, extract, chunk and index a single uploaded PDF.

    Ingestion is idempotent: re-uploading identical bytes is reported as
    ``skipped`` unless ``force=True`` re-indexes them.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    safe = safe_filename(filename)

    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    report("Validating file", 0.05)
    validation = validate_upload(filename, data, settings.max_upload_bytes)
    if not validation.ok:
        return IngestionReport(
            filename=safe, doc_id="", status="failed", message=validation.message,
            elapsed_s=time.perf_counter() - started,
        )

    report("Checking index", 0.12)
    from ..models import doc_hash

    doc_id = doc_hash(data)
    if not force and store.has_document(doc_id):
        return IngestionReport(
            filename=safe,
            doc_id=doc_id,
            status="skipped",
            message="Already indexed (identical file). Use force re-index to replace it.",
            elapsed_s=time.perf_counter() - started,
        )

    report("Extracting text and tables", 0.3)
    extracted = extract_pdf(
        data,
        safe,
        ocr_engine=settings.ocr_engine,
        ocr_languages=settings.ocr_languages,
        ocr_min_chars_per_page=settings.ocr_min_chars_per_page,
    )

    if extracted.errors:
        return IngestionReport(
            filename=safe,
            doc_id=doc_id,
            status="failed",
            message=extracted.errors[0],
            page_count=extracted.page_count,
            warnings=extracted.warnings,
            errors=extracted.errors,
            elapsed_s=time.perf_counter() - started,
        )

    if not extracted.is_usable:
        return IngestionReport(
            filename=safe,
            doc_id=doc_id,
            status="failed",
            message=(
                "No usable text was extracted (document appears to be fully scanned "
                "and OCR could not recover any content)."
            ),
            page_count=extracted.page_count,
            warnings=extracted.warnings,
            elapsed_s=time.perf_counter() - started,
        )

    report("Creating chunks", 0.6)
    chunks = chunker.chunk(extracted)
    if not chunks:
        return IngestionReport(
            filename=safe, doc_id=doc_id, status="failed",
            message="Extracted text produced no chunks (try a smaller chunk size).",
            page_count=extracted.page_count, warnings=extracted.warnings,
            elapsed_s=time.perf_counter() - started,
        )

    report("Embedding and storing", 0.8)
    # Replace any prior vectors so a re-index cannot leave stale rows behind.
    store.delete_document(doc_id)
    store.upsert_chunks(chunks)

    report("Done", 1.0)
    return IngestionReport(
        filename=safe,
        doc_id=doc_id,
        status="indexed",
        message=f"Indexed {len(chunks)} chunks from {extracted.page_count} pages.",
        page_count=extracted.page_count,
        chunk_count=len(chunks),
        ocr_pages=extracted.ocr_pages,
        warnings=extracted.warnings,
        elapsed_s=time.perf_counter() - started,
    )


def ingest_file(
    path: Path,
    store: VectorStore,
    chunker: Chunker,
    settings: Optional[Settings] = None,
    *,
    force: bool = False,
    progress: Optional[ProgressFn] = None,
) -> IngestionReport:
    """Ingest a PDF already on disk (used by the benchmark and test harnesses)."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return IngestionReport(
            filename=Path(path).name, doc_id="", status="failed", message=str(exc)
        )
    return ingest_bytes(Path(path).name, data, store, chunker, settings, force=force, progress=progress)