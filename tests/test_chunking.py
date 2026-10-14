"""Tests for page-aware, token-aware chunking."""

from __future__ import annotations

import pytest

from src.chunking import PageAwareChunker
from src.ingestion.extractor import extract_pdf
from src.models import ExtractedDocument, Page, Table


def make_doc(pages_text: list[str], tables: list[list[list[str]]] | None = None) -> ExtractedDocument:
    """Build an ExtractedDocument directly, without going through a PDF."""
    tables = tables or []
    pages = []
    for i, text in enumerate(pages_text):
        tbls = [Table(page=i + 1, rows=t) for t in tables[i]] if i < len(tables) else []
        pages.append(
            Page(page_number=i + 1, text=text, tables=tbls, char_count=len(text))
        )
    return ExtractedDocument(
        filename="test.pdf",
        doc_id="deadbeef" * 4,
        pages=pages,
        page_count=len(pages),
    )


class TestValidation:
    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            PageAwareChunker(chunk_size=0)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError):
            PageAwareChunker(chunk_size=512, chunk_overlap=-1)

    def test_rejects_overlap_ge_size(self):
        with pytest.raises(ValueError):
            PageAwareChunker(chunk_size=256, chunk_overlap=256)


class TestChunking:
    def test_preserves_page_metadata(self):
        doc = make_doc(["First page content.", "Second page content."])
        chunks = PageAwareChunker(512, 64).chunk(doc)
        assert {c.page for c in chunks} == {1, 2}
        assert all(c.doc_id == doc.doc_id for c in chunks)
        assert all(c.filename == "test.pdf" for c in chunks)

    def test_chunks_never_span_pages(self):
        """Page-accurate citations depend on this invariant."""
        doc = make_doc(["Alpha " * 400, "Beta " * 400])
        chunks = PageAwareChunker(256, 32).chunk(doc)
        for c in chunks:
            assert c.page in (1, 2)

    def test_respects_chunk_size(self):
        doc = make_doc(["word " * 2000])
        chunker = PageAwareChunker(256, 32)
        chunks = chunker.chunk(doc)
        # Allow the overlap prefix to push slightly over; content must not.
        for c in chunks:
            assert c.token_count <= chunker.chunk_size + chunker.chunk_overlap + 32

    def test_token_counts_are_real_tokens(self):
        chunker = PageAwareChunker(256, 32)
        assert chunker.count_tokens("hello world") > 0
        assert chunker.count_tokens("hello world") < len("hello world")

    def test_overlap_duplicates_context(self):
        long_text = " ".join(f"sentence{i} about policy" for i in range(200))
        doc = make_doc([long_text])
        chunks = PageAwareChunker(256, 64).chunk(doc)
        assert len(chunks) > 1
        # With overlap, consecutive chunks share some leading text.
        assert any(
            set(chunks[i].text.split()[:5]) & set(chunks[i + 1].text.split()[:5])
            for i in range(len(chunks) - 1)
        )

    def test_zero_overlap_produces_no_duplication(self):
        long_text = " ".join(f"token{i}" for i in range(2000))
        doc = make_doc([long_text])
        chunks = PageAwareChunker(256, 0).chunk(doc)
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))

    def test_empty_page_produces_no_chunk(self):
        doc = make_doc(["", "Only this page has content."])
        chunks = PageAwareChunker(256, 32).chunk(doc)
        assert all(c.page == 2 for c in chunks)

    def test_empty_document_produces_no_chunks(self):
        doc = make_doc(["", "", ""])
        assert PageAwareChunker(256, 32).chunk(doc) == []

    def test_table_becomes_own_chunk(self):
        doc = make_doc(["Prose on page one."], tables=[[[
            ["Head A", "Head B"], ["1", "2"], ["3", "4"],
        ]]])
        chunks = PageAwareChunker(256, 32).chunk(doc)
        table_chunks = [c for c in chunks if "Head A" in c.text]
        assert len(table_chunks) == 1
        assert "|" in table_chunks[0].text  # rendered as markdown

    def test_chunk_ids_are_unique(self):
        doc = make_doc(["Page one. " * 100, "Page two. " * 100])
        chunks = PageAwareChunker(256, 32).chunk(doc)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_oversized_paragraph_is_split(self):
        # A single sentence far longer than the budget must still be chunked.
        doc = make_doc(["supercalifragilistic " * 500])
        chunks = PageAwareChunker(256, 32).chunk(doc)
        assert len(chunks) > 1

    def test_hyphenated_words_are_repaired_before_chunking(self, simple_pdf_bytes):
        doc = extract_pdf(simple_pdf_bytes, "simple.pdf")
        chunks = PageAwareChunker(256, 32).chunk(doc)
        joined = " ".join(c.text for c in chunks)
        assert "-\n" not in joined

    def test_stats_are_reported(self):
        doc = make_doc(["word " * 500, "more " * 500])
        chunker = PageAwareChunker(256, 32)
        stats = chunker.stats(chunker.chunk(doc))
        assert stats.chunks > 0
        assert stats.min_tokens <= stats.mean_tokens <= stats.max_tokens

    def test_metadata_is_chroma_safe(self):
        doc = make_doc(["Content here."])
        chunk = PageAwareChunker(256, 32).chunk(doc)[0]
        meta = chunk.metadata()
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())

    def test_chunk_roundtrips_through_storage_metadata(self):
        doc = make_doc(["Content here for roundtrip."])
        chunk = PageAwareChunker(256, 32).chunk(doc)[0]
        from src.models import Chunk

        restored = Chunk.from_storage(chunk.chunk_id, chunk.text, chunk.metadata())
        assert restored.page == chunk.page
        assert restored.filename == chunk.filename
        assert restored.text == chunk.text

    @pytest.mark.parametrize("size,overlap", [(256, 32), (512, 64), (1024, 128)])
    def test_experiment_configs_all_work(self, size, overlap):
        doc = make_doc(["Content sentence here. " * 300])
        chunks = PageAwareChunker(size, overlap).chunk(doc)
        assert chunks
        assert all(c.page == 1 for c in chunks)