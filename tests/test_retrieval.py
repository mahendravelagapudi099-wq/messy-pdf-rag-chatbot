"""Tests for the vector store, retrieval and citations."""

from __future__ import annotations

import pytest

from src.chunking import PageAwareChunker
from src.citations import (
    build_citations,
    content_tokens,
    excerpt,
    overlap_score,
    split_claims,
    verify_citations,
)
from src.ingestion.extractor import extract_pdf
from src.models import Chunk, RetrievedChunk
from src.rag import RAGEngine
from src.retrieval import retrieve
from src.vector_store import VectorStore


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, demo_corpus):
    """An engine with the demo corpus ingested."""
    from dataclasses import replace

    from src.config import get_settings

    base = replace(
        get_settings(),
        chroma_persist_dir=tmp_path_factory.mktemp("vs") / "db",
        chroma_collection="test_vs",
        similarity_threshold=0.0,
    )
    engine = RAGEngine(base)
    for pdf in sorted(demo_corpus.glob("*.pdf")):
        engine.add_file(pdf, force=True)
    return engine


class TestVectorStore:
    def test_upsert_and_count(self, indexed):
        assert indexed.store.count() > 0

    def test_has_document(self, indexed):
        docs = indexed.store.list_documents()
        assert docs
        assert indexed.store.has_document(docs[0].doc_id)

    def test_missing_document_is_false(self, indexed):
        assert not indexed.store.has_document("nonexistent-doc-id")

    def test_delete_document_is_idempotent(self, indexed, demo_corpus):
        """Deleting twice must not error, and the doc must be gone after the first."""
        from src.models import doc_hash

        docs = indexed.store.list_documents()
        doc_id = docs[0].doc_id

        first = indexed.store.delete_document(doc_id)
        second = indexed.store.delete_document(doc_id)
        assert first > 0
        assert second == 0
        assert not indexed.store.has_document(doc_id)

        # Restore so later tests see a complete index.
        source = next(
            p for p in sorted(demo_corpus.glob("*.pdf"))
            if doc_hash(p.read_bytes()) == doc_id
        )
        indexed.add_file(source, force=True)
        assert indexed.store.has_document(doc_id)

    def test_list_documents_reports_pages(self, indexed):
        for d in indexed.store.list_documents():
            assert d.page_count >= 1
            assert d.chunk_count >= 1

    def test_get_page_text_roundtrip(self, indexed):
        doc = next(d for d in indexed.store.list_documents() if d.page_count > 1)
        chunks = indexed.store.get_page_text(doc.doc_id, 1)
        assert chunks
        assert all(c.page == 1 for c in chunks)
        assert all(c.text.strip() for c in chunks)

    def test_empty_query_returns_nothing(self, indexed):
        assert indexed.store.query("") == []

    def test_query_on_empty_store_is_safe(self, settings, tmp_path):
        from dataclasses import replace

        s = replace(settings, chroma_persist_dir=tmp_path / "empty")
        store = VectorStore(s)
        assert store.query("anything") == []


class TestRetrieval:
    def test_finds_relevant_document(self, indexed):
        result = indexed.search("What is the vehicle retention rate requirement?")
        assert result.chunks
        assert any("pmgsy" in c.chunk.filename for c in result.chunks)

    def test_returns_at_most_top_k(self, indexed):
        result = retrieve(indexed.store, "compliance threshold for transactions", top_k=2)
        assert len(result.chunks) <= 2

    def test_threshold_filters_weak_matches(self, indexed):
        from dataclasses import replace

        loose = retrieve(indexed.store, "repo rate monetary policy", top_k=5,
                         similarity_threshold=0.0)
        strict = retrieve(indexed.store, "repo rate monetary policy", top_k=5,
                          similarity_threshold=0.99)
        assert len(strict.chunks) < len(loose.chunks)
        assert strict.below_threshold > 0

    def test_empty_query_is_safe(self, indexed):
        result = retrieve(indexed.store, "", top_k=5)
        assert result.is_empty
        assert not result.chunks

    def test_result_reports_latency(self, indexed):
        result = indexed.search("What is the repo rate threshold?")
        assert result.latency_s > 0
        assert result.top_similarity > 0

    def test_metadata_survives_the_roundtrip(self, indexed):
        result = indexed.search("funding pattern for bridges")
        assert result.chunks
        for rc in result.chunks:
            assert rc.chunk.filename
            assert rc.chunk.page >= 1
            assert rc.chunk.doc_id


class TestAsk:
    def test_answers_with_citations(self, indexed):
        answer = indexed.ask("What is the vehicle retention rate required after three years?")
        assert answer.ok, answer.error
        assert not answer.abstained
        assert "80" in answer.text
        assert answer.citations
        assert all(c.filename.endswith(".pdf") for c in answer.citations)

    def test_citation_pages_are_valid(self, indexed):
        answer = indexed.ask("How many credits does ME501 carry?")
        docs = {d.doc_id: d for d in indexed.store.list_documents()}
        for c in answer.citations:
            assert c.page >= 1
            assert c.page <= docs[c.doc_id].page_count

    def test_abstains_when_threshold_filters_everything(self, indexed):
        """With a real threshold, an off-topic question yields no evidence at all."""
        from dataclasses import replace

        strict = indexed.with_settings(replace(indexed.settings, similarity_threshold=0.35))
        answer = strict.ask("What is the airspeed velocity of an unladen swallow?")
        assert answer.ok, answer.error
        assert answer.abstained
        assert answer.abstention_reason == "no_evidence_retrieved"

    def test_low_threshold_surfaces_weak_matches_instead(self, indexed):
        """With threshold 0 the same question returns weak passages.

        This documents the real trade-off: the threshold is what turns a weak
        match into an abstention, so lowering it makes the system talk more and
        abstain less.
        """
        result = indexed.search("What is the airspeed velocity of an unladen swallow?")
        assert result.chunks
        assert all(rc.similarity < 0.35 for rc in result.chunks)

    def test_never_raises(self, indexed):
        answer = indexed.ask("")
        assert answer.error or answer.abstained


class TestCitations:
    def test_content_tokens_drop_stopwords(self):
        assert "the" not in content_tokens("The quick brown fox")
        assert "quick" in content_tokens("The quick brown fox")

    def test_overlap_score_high_for_supporting_text(self):
        claim = "The vehicle retention rate must be 80 per cent after three years"
        ev = "The vehicle retention rate shall be not less than 80 per cent after three years"
        assert overlap_score(claim, ev) > 0.8

    def test_overlap_score_low_for_unrelated_text(self):
        assert overlap_score("repo rate monetary policy", "syllabus thermodynamics units") < 0.2

    def test_overlap_score_zero_on_empty(self):
        assert overlap_score("", "evidence") == 0.0
        assert overlap_score("claim", "") == 0.0

    def test_excerpt_truncates_on_word_boundary(self):
        long_text = "word " * 200
        out = excerpt(long_text, limit=100)
        assert len(out) <= 110
        assert out.endswith("...")

    def test_excerpt_leaves_short_text_alone(self):
        assert excerpt("short", limit=100) == "short"

    def test_build_citations_keeps_only_used_markers(self):
        chunks = [
            RetrievedChunk(chunk=Chunk("c1", "d", "a.pdf", 1, "text one"), similarity=0.9),
            RetrievedChunk(chunk=Chunk("c2", "d", "b.pdf", 2, "text two"), similarity=0.5),
        ]
        cites = build_citations(chunks, "An answer citing only [2].")
        assert [c.marker for c in cites] == ["[2]"]
        assert cites[0].filename == "b.pdf"

    def test_split_claims_maps_markers(self):
        claims = split_claims("The rate is 80 per cent [1]. Bridges cost 90 per cent [2].")
        assert 0 in claims
        assert 1 in claims
        assert "80" in claims[0]
        assert "90" in claims[1]

    def test_citation_grounding_is_flagged(self):
        chunk = Chunk("c1", "d", "a.pdf", 1, "The vehicle retention rate must be 80 per cent")
        rc = RetrievedChunk(chunk=chunk, similarity=0.9)
        cites = build_citations([rc], "The vehicle retention rate must be 80 per cent [1].")
        assert cites[0].grounded

    def test_ungrounded_citation_is_flagged_not_hidden(self):
        chunk = Chunk("c1", "d", "a.pdf", 1, "Thermodynamic system and boundary conditions")
        rc = RetrievedChunk(chunk=chunk, similarity=0.9)
        cites = build_citations([rc], "The repo rate was cut to four per cent [1].")
        assert cites and not cites[0].grounded

    def test_verify_citations_summary(self):
        cites = build_citations(
            [RetrievedChunk(chunk=Chunk("c1", "d", "a.pdf", 1, "x"), similarity=0.9)],
            "text [1]",
        )
        summary = verify_citations(cites, "text [1]")
        assert summary["total"] == 1
        assert summary["grounded"] + summary["ungrounded"] == 1

    def test_verify_citations_empty(self):
        assert verify_citations([], "")["total"] == 0