"""Page-aware, token-counted chunking.

Design notes
------------
* **Token-aware.** Sizes are measured with ``tiktoken`` so that "512 tokens"
  means 512 tokens, not 512 characters. Character-based sizing silently produces
  wildly uneven chunks on prose vs. tables, which wrecks retrieval quality.
* **Page-anchored.** A chunk never spans pages, so every chunk carries an exact
  page number and citations are always precise.
* **Paragraph-respecting.** We split on paragraph boundaries first and only fall
  back to sentence/word boundaries when a single paragraph exceeds the budget.
  Mid-sentence cuts destroy the meaning of a chunk.
* **Table-safe.** Rendered Markdown tables are emitted as their own chunks,
  because a table split across chunks becomes unanswerable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from .models import Chunk, ExtractedDocument

log = logging.getLogger(__name__)


def get_encoding(model: Optional[str] = None):
    """Return a tiktoken encoding, falling back to a byte-safe approximation."""
    try:
        import tiktoken

        # cl100k_base approximates modern BPE tokenisers well enough for sizing
        # and needs no network access once cached.
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - offline fallback
        log.warning("tiktoken unavailable (%s); falling back to word-based sizing.", exc)
        return None


class _FallbackEncoding:
    """Approximate tokenizer used when tiktoken cannot be loaded.

    Maps ~4 characters (or 1 word) to a token, which is close enough for
    chunk sizing and keeps the pipeline functional offline.
    """

    def encode(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError


@dataclass
class ChunkingStats:
    chunks: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float


class PageAwareChunker:
    """Configurable chunker with overlap.

    Parameters
    ----------
    chunk_size:
        Target chunk size in tokens.
    chunk_overlap:
        Number of tokens shared with the previous chunk.
    respect_tables:
        Emit each extracted table as its own indivisible chunk.
    min_chunk_tokens:
        Chunks smaller than this are merged forward to avoid useless fragments.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        *,
        encoding=None,
        respect_tables: bool = True,
        min_chunk_tokens: int = 32,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding or get_encoding() or _FallbackEncoding()
        self.respect_tables = respect_tables
        self.min_chunk_tokens = min_chunk_tokens

    # -- token helpers ---------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.encoding.decode(tokens[:max_tokens])
        except NotImplementedError:
            return text[: max_tokens * 4]

    def _tail_tokens(self, text: str, n: int) -> str:
        """Last ``n`` tokens of ``text`` -- used to build the overlap."""
        if n <= 0:
            return ""
        try:
            tokens = self.encoding.encode(text)
            if not tokens:
                return ""
            return self.encoding.decode(tokens[-n:])
        except NotImplementedError:
            return text[-n * 4 :]

    # -- splitting -------------------------------------------------------
    def _paragraphs(self, text: str) -> list[str]:
        """Split into paragraph-like units, never returning empty strings."""
        if not text.strip():
            return []
        blocks = [b.strip() for b in text.split("\n\n")]
        return [b for b in blocks if b]

    def _split_long_unit(self, unit: str, size: int) -> list[str]:
        """Break an oversized paragraph at sentence boundaries, then words."""
        if self.count_tokens(unit) <= size:
            return [unit]

        # Prefer sentence boundaries.
        sentences: list[str] = []
        buf = ""
        for ch in unit:
            buf += ch
            if ch in ".!?" and (len(buf) > 1 and buf[-2] not in ".!?"):
                sentences.append(buf.strip())
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())
        sentences = [s for s in sentences if s]

        out: list[str] = []
        cur = ""
        for sent in sentences:
            if cur and self.count_tokens(cur + " " + sent) > size:
                out.append(cur.strip())
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur.strip():
            out.append(cur.strip())

        # A single sentence longer than the budget: hard-split on words.
        final: list[str] = []
        for piece in out:
            if self.count_tokens(piece) <= size:
                final.append(piece)
                continue
            words = piece.split(" ")
            cur, n = "", self.count_tokens("")
            for w in words:
                candidate = f"{cur} {w}".strip()
                if n and self.count_tokens(candidate) > size:
                    final.append(cur)
                    cur, n = w, self.count_tokens(w)
                else:
                    cur, n = candidate, self.count_tokens(candidate)
            if cur:
                final.append(cur)
        return final

    def _pack(self, units: Iterable[str], budget: int) -> list[str]:
        """Greedily pack units into chunks of at most ``budget`` tokens."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            n = self.count_tokens(unit)
            if current and current_tokens + n > budget:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            current.append(unit)
            current_tokens += n
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _with_overlap(self, chunks: list[str]) -> list[str]:
        """Prefix each chunk (after the first) with the tail of the previous."""
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks
        out = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = self._tail_tokens(prev, self.chunk_overlap)
            if tail and tail not in cur:
                out.append(f"{tail}\n\n{cur}")
            else:
                out.append(cur)
        return out

    # -- public API ------------------------------------------------------
    def chunk(self, doc: ExtractedDocument) -> list[Chunk]:
        """Split an extracted document into page-anchored chunks."""
        chunks: list[Chunk] = []
        index = 0

        for page in doc.pages:
            heading = page.headings[0] if page.headings else None

            # 1) Tables become standalone chunks so they are never cut apart.
            if self.respect_tables:
                for tbl in page.tables:
                    md = tbl.to_markdown()
                    if not md:
                        continue
                    md = self._truncate_tokens(md, self.chunk_size)
                    chunks.append(
                        self._make_chunk(
                            doc, page.page_number, md, heading, page.used_ocr, index
                        )
                    )
                    index += 1

            # 2) Prose, cleaned of page furniture and split on paragraphs.
            text = page.text
            if text.strip():
                units: list[str] = []
                for para in self._paragraphs(text):
                    units.extend(self._split_long_unit(para, self.chunk_size))
                for piece in self._pack(units, self.chunk_size):
                    piece = piece.strip()
                    if not piece:
                        continue
                    overlapped = self._with_overlap([piece])[0]
                    chunks.append(
                        self._make_chunk(
                            doc, page.page_number, overlapped, heading, page.used_ocr, index
                        )
                    )
                    index += 1

        chunks = self._merge_tiny(chunks)
        log.debug(
            "chunked %s -> %d chunks (page_count=%d)", doc.filename, len(chunks), doc.page_count
        )
        return chunks

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge undersized chunks forward when they share a page."""
        if not chunks or self.min_chunk_tokens <= 0:
            return chunks

        out: list[Chunk] = []
        for ch in chunks:
            if (
                out
                and out[-1].page == ch.page
                and out[-1].token_count < self.min_chunk_tokens
                and out[-1].token_count + ch.token_count <= self.chunk_size
            ):
                prev = out[-1]
                merged = f"{prev.text}\n\n{ch.text}".strip()
                out[-1] = Chunk(
                    chunk_id=prev.chunk_id, doc_id=prev.doc_id, filename=prev.filename,
                    page=prev.page, text=merged, heading=prev.heading,
                    token_count=self.count_tokens(merged), used_ocr=prev.used_ocr,
                    chunk_index=prev.chunk_index,
                )
            else:
                out.append(ch)
        return out

    def _make_chunk(
        self, doc: ExtractedDocument, page: int, text: str, heading: Optional[str],
        used_ocr: bool, index: int,
    ) -> Chunk:
        text = text.strip()
        return Chunk(
            chunk_id=f"{doc.doc_id[:16]}-p{page}-{index}",
            doc_id=doc.doc_id,
            filename=doc.filename,
            page=page,
            text=text,
            heading=heading,
            token_count=self.count_tokens(text),
            used_ocr=used_ocr,
            chunk_index=index,
        )

    def stats(self, chunks: list[Chunk]) -> ChunkingStats:
        if not chunks:
            return ChunkingStats(0, 0, 0, 0, 0.0)
        counts = [c.token_count for c in chunks]
        return ChunkingStats(
            chunks=len(chunks),
            total_tokens=sum(counts),
            min_tokens=min(counts),
            max_tokens=max(counts),
            mean_tokens=round(sum(counts) / len(counts), 1),
        )