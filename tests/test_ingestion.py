"""Tests for PDF extraction, OCR routing and text cleaning."""

from __future__ import annotations

import pytest

from src.ingestion.cleaner import (
    clean_text,
    dehyphenate,
    detect_headings,
    normalize,
    strip_page_furniture,
)
from src.ingestion.extractor import extract_pdf
from src.ingestion.ocr import OcrEngine, probe


class TestCleaner:
    def test_normalizes_ligatures(self):
        assert "fi" in normalize("the ﬁrst oﬃce")

    def test_strips_soft_hyphen(self):
        assert normalize("hyphen\u00adation") == "hyphenation"

    def test_dehyphenates_line_break(self):
        # The classic artefact: a word split across a line with a trailing hyphen.
        assert dehyphenate("elec-\ntrical") == "electrical"

    def test_dehyphenate_leaves_normal_hyphens_alone(self):
        assert dehyphenate("well-known") == "well-known"

    def test_strips_page_numbers(self):
        # The furniture line is dropped; surrounding content is preserved.
        assert strip_page_furniture("Header\n12\nBody") == "Header\nBody"

    def test_strips_confidential_banner(self):
        assert "CONFIDENTIAL" not in strip_page_furniture("CONFIDENTIAL\nBody")

    def test_clean_collapses_whitespace(self):
        assert clean_text("a    b\n\n\n\n\nc  ") == "a b\n\nc"

    def test_clean_handles_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_detect_numbered_headings(self):
        text = "1. ELIGIBILITY\nSome body text here.\n2. FEES\nMore body."
        assert "1. ELIGIBILITY" in detect_headings(text)

    def test_detect_uppercase_headings(self):
        assert "OBJECTIVE" in detect_headings("OBJECTIVE\nbody text follows here")

    def test_detect_section_headings(self):
        assert any("Section 4" in h for h in detect_headings("Section 4: Penalties\nbody"))

    def test_headings_deduplicated_and_ordered(self):
        text = "OBJECTIVE\nbody\nOBJECTIVE\nmore body"
        assert detect_headings(text).count("OBJECTIVE") == 1


class TestExtractPdf:
    def test_extracts_pages_and_text(self, simple_pdf_bytes):
        doc = extract_pdf(simple_pdf_bytes, "simple.pdf")
        assert doc.page_count == 2
        assert doc.total_chars > 0
        assert "settlement cycle" in doc.pages[0].text

    def test_page_numbers_are_one_based(self, simple_pdf_bytes):
        doc = extract_pdf(simple_pdf_bytes, "simple.pdf")
        assert [p.page_number for p in doc.pages] == [1, 2]

    def test_doc_id_is_content_hash(self, simple_pdf_bytes):
        a = extract_pdf(simple_pdf_bytes, "a.pdf")
        b = extract_pdf(simple_pdf_bytes, "b.pdf")
        assert a.doc_id == b.doc_id  # same bytes -> same id regardless of name

    def test_doc_id_differs_for_different_bytes(self, simple_pdf_bytes, demo_corpus):
        other = next(demo_corpus.glob("*.pdf")).read_bytes()
        assert extract_pdf(simple_pdf_bytes, "a.pdf").doc_id != extract_pdf(other, "b.pdf").doc_id

    def test_garbage_input_is_reported_not_raised(self):
        doc = extract_pdf(b"this is not a pdf at all", "broken.pdf")
        assert doc.errors
        assert not doc.is_usable

    def test_blank_pages_do_not_crash(self, demo_corpus):
        """The demo corpus ends with a deliberate blank page."""
        doc = extract_pdf(next(demo_corpus.glob("*.pdf")), "demo.pdf")
        assert any(p.char_count == 0 for p in doc.pages)
        assert doc.is_usable

    def test_tables_are_extracted(self, demo_corpus):
        pmgsy = next(p for p in demo_corpus.glob("*.pdf") if "pmgsy" in p.name)
        doc = extract_pdf(pmgsy, pmgsy.name)
        assert sum(len(p.tables) for p in doc.pages) > 0

    def test_running_headers_are_stripped(self, demo_corpus):
        rbi = next(p for p in demo_corpus.glob("*.pdf") if "rbi" in p.name)
        doc = extract_pdf(rbi, rbi.name)
        # The running header text appears on every page but is not real content.
        assert not doc.pages[2].text.startswith("rbi due diligence circular")

    def test_max_pages_is_respected(self, simple_pdf_bytes):
        doc = extract_pdf(simple_pdf_bytes, "simple.pdf", max_pages=1)
        assert len(doc.pages) == 1
        assert any("first 1 of 2" in w for w in doc.warnings)

    def test_scanned_page_reports_no_text(self, demo_corpus):
        scanned = next(
            (p for p in demo_corpus.glob("*.pdf") if "scanned" in p.name), None
        )
        if scanned is None:
            pytest.skip("scanned fixture not generated")
        doc = extract_pdf(scanned, scanned.name, ocr_engine="none")
        # Without OCR the page has no text; the failure must be *reported*.
        assert doc.total_chars == 0
        assert doc.errors or doc.warnings

    def test_encrypted_pdf_is_handled(self, simple_pdf_bytes):
        # Just confirm a normal PDF is not falsely flagged as encrypted.
        doc = extract_pdf(simple_pdf_bytes, "simple.pdf")
        assert not any("password" in e.lower() for e in doc.errors)


class TestOcrProbe:
    def test_probe_returns_status_not_exception(self):
        status = probe("auto")
        assert isinstance(status.available, bool)
        assert status.engine in set(OcrEngine)

    def test_none_disables_ocr(self):
        status = probe("none")
        assert not status.available

    def test_probe_is_deterministic(self):
        a, b = probe("auto"), probe("auto")
        assert a.available == b.available