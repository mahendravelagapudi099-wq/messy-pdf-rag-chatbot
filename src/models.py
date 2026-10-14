"""Domain models shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


@dataclass
class Table:
    """A table extracted from a single page."""

    page: int
    rows: list[list[str]]
    bbox: Optional[tuple[float, float, float, float]] = None

    def to_markdown(self) -> str:
        """Render the table as a Markdown table.

        Ragged rows are padded so that the result is always a valid table, which
        matters because real-world PDFs frequently have ragged grids.
        """
        if not self.rows:
            return ""
        width = max(len(r) for r in self.rows)
        norm = [[(c or "").strip().replace("\n", " ") for c in r] + [""] * (width - len(r)) for r in self.rows]
        norm[0] = [c or f"col{i + 1}" for i, c in enumerate(norm[0])]
        lines = ["| " + " | ".join(norm[0]) + " |", "|" + "|".join([" --- "] * width) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
        return "\n".join(lines)


@dataclass
class Page:
    """Extracted content for one PDF page."""

    page_number: int  # 1-based, matches what the user sees
    text: str = ""
    tables: list[Table] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    used_ocr: bool = False
    char_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Page text followed by rendered tables."""
        parts: list[str] = []
        for h in self.headings:
            parts.append(f"## {h}")
        if self.text.strip():
            parts.append(self.text.strip())
        for tbl in self.tables:
            md = tbl.to_markdown()
            if md:
                parts.append(f"[Table on page {self.page_number}]\n{md}")
        return "\n\n".join(parts)

    def is_effectively_empty(self, threshold: int) -> bool:
        """True when the page has too little *meaningful* text to trust."""
        # A page of digits/dashes (a scanned cover, a blank form) is not useful.
        alnum = sum(ch.isalnum() for ch in self.text)
        return alnum < threshold


@dataclass
class ExtractedDocument:
    """Full extraction result for one PDF."""

    filename: str
    doc_id: str
    pages: list[Page]
    page_count: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def is_usable(self) -> bool:
        return self.total_chars > 0


@dataclass
class IngestionReport:
    """Outcome of ingesting one file, surfaced directly in the UI."""

    filename: str
    doc_id: str
    status: str = "pending"  # pending | indexed | skipped | failed
    message: str = ""
    page_count: int = 0
    chunk_count: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in {"indexed", "skipped"}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """A page-anchored passage with full provenance."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    heading: Optional[str] = None
    token_count: int = 0
    used_ocr: bool = False
    chunk_index: int = 0

    def metadata(self) -> dict[str, Any]:
        """Chroma-safe metadata (primitives only)."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "heading": self.heading or "",
            "used_ocr": self.used_ocr,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }

    @classmethod
    def from_storage(
        cls, chunk_id: str, text: str, meta: dict[str, Any]
    ) -> "Chunk":
        return cls(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            filename=str(meta.get("filename", "")),
            page=int(meta.get("page", 0) or 0),
            text=text,
            heading=meta.get("heading") or None,
            used_ocr=bool(meta.get("used_ocr", False)),
            chunk_index=int(meta.get("chunk_index", 0) or 0),
            token_count=int(meta.get("token_count", 0) or 0),
        )

    def label(self) -> str:
        base = f"{self.filename} p.{self.page}"
        return f"{base} — {self.heading}" if self.heading else base


# --------------------------------------------------------------------------
# Retrieval & answers
# --------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk returned by search, with its scores."""

    chunk: Chunk
    similarity: float  # 1.0 = identical, 0.0 = orthogonal
    rerank_score: Optional[float] = None

    def label(self) -> str:
        return self.chunk.label()


@dataclass
class Citation:
    """A claim in the answer mapped to the evidence that supports it."""

    marker: str  # "[1]"
    doc_id: str
    filename: str
    page: int
    chunk_id: str
    excerpt: str
    similarity: float
    grounded: bool = True

    def render(self) -> str:
        return f"[{self.filename}, p.{self.page}]"


@dataclass
class Answer:
    """A generated (or abstained) answer with full provenance."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[str] = None
    #: Set when the system answered, but not by full generation -- e.g. no LLM
    #: configured. Deliberately distinct from ``abstained``: degraded output is
    #: still an answer, and conflating the two would corrupt abstention metrics.
    degraded_reason: Optional[str] = None
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")
_BULLET = re.compile(r"^\s*(?:[-•●▪*o]\s+|\d+[.)]\s+)", re.MULTILINE)


def content_hash(text: str) -> str:
    """Stable hash of normalised text, used for de-duplication."""
    norm = _WS.sub(" ", text).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def doc_hash(data: bytes) -> str:
    """Stable hash of raw file bytes, used for idempotent ingestion."""
    return hashlib.sha256(data).hexdigest()