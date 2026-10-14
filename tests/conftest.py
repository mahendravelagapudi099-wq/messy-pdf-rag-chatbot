"""Shared pytest fixtures.

Synthetic PDFs are generated at test time rather than committed as binaries, so
the test corpus stays readable and regenerable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    """Settings pointed at a throwaway index so tests never touch user data."""
    return Settings(
        chroma_persist_dir=tmp_path_factory.mktemp("chroma") / "test_db",
        chroma_collection="test_collection",
    )


@pytest.fixture(scope="session")
def demo_corpus(tmp_path_factory) -> Path:
    """The synthetic demo corpus, generated once per session."""
    from scripts.make_demo_corpus import main as make_corpus

    out = tmp_path_factory.mktemp("corpus")
    make_corpus(["--out", str(out)])
    return out


@pytest.fixture(scope="session")
def eval_dataset_path() -> Path:
    path = ROOT / "data" / "eval_dataset.jsonl"
    assert path.exists(), f"missing dataset: {path}"
    return path


@pytest.fixture(scope="session")
def simple_pdf_bytes() -> bytes:
    """A two-page PDF with predictable, quotable content."""
    from reportlab.pdfgen import canvas

    from src.ingestion import extract_pdf

    buf = _BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Financial Services Notice")
    c.drawString(72, 700, "The settlement cycle is T+1 for all listed securities.")
    c.showPage()
    c.drawString(72, 720, "Risk Management")
    c.drawString(72, 700, "The capital adequacy ratio must not fall below 12 percent.")
    c.showPage()
    c.save()

    doc = extract_pdf(buf.getvalue(), "simple.pdf")
    assert doc.page_count == 2
    return buf.getvalue()


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A structurally valid PDF with no pages."""
    return b"%PDF-1.4\n%%EOF\n"


def _BytesIO():  # noqa: N802 - test helper
    import io

    return io.BytesIO()