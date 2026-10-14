"""PDF text and table extraction with page provenance.

PyMuPDF supplies both the text layer and (via its optional table finder) table
structure. Pages whose text layer is missing or effectively empty -- scanned
pages -- are routed through OCR.

Extraction never raises for page-level problems: failures are collected as
warnings so that one bad page cannot abort a whole document.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..models import ExtractedDocument, Page, Table, doc_hash
from . import ocr as ocr_mod
from .cleaner import clean_table_cell, clean_text, detect_headings

log = logging.getLogger(__name__)

# Rendering DPI used when rasterising a page for OCR. 200 is a good accuracy /
# speed trade-off for body text; 300 only helps tiny fonts.
OCR_DPI = 200


def _fitz():
    """Import PyMuPDF lazily with a helpful error message."""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore

            return pymupdf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install with: pip install PyMuPDF"
            ) from exc


def _extract_tables(page, max_tables: int = 8) -> list[Table]:
    """Best-effort table extraction.

    PyMuPDF's ``find_tables`` is the only table engine available without a heavy
    dependency, and it does fail on some real-world layouts. Any failure is
    swallowed -- losing a table is far better than losing the document.
    """
    pymupdf = _fitz()
    tables: list[Table] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        log.debug("table detection failed on page %s: %s", page.number, exc)
        return tables

    try:
        extracted = list(finder.tables)
    except Exception:
        extracted = []

    for tbl in extracted[:max_tables]:
        try:
            rows = [
                [clean_table_cell(c) for c in row]
                for row in tbl.extract()
            ]
            rows = [r for r in rows if any(c for c in r)]
            # Discard single-cell "tables" -- those are almost always layout boxes.
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                continue
            tables.append(Table(page=page.number + 1, rows=rows))
        except Exception as exc:
            log.debug("table parse failed on page %s: %s", page.number, exc)
            continue
    return tables


def _render_for_ocr(page) -> Optional[bytes]:
    """Rasterise a page to PNG bytes for OCR."""
    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        return pix.tobytes("png")
    except Exception as exc:
        log.debug("page rasterisation failed on page %s: %s", page.number, exc)
        return None


def _running_headers(pages: list[Page], min_ratio: float = 0.6) -> set[str]:
    """Find first lines that repeat across most pages -- running headers.

    Government circulars, syllabi and manuals almost always carry a header such as
    the document title or department on every page. Left in place, it is embedded
    into every chunk and dilutes the embedding signal, so we strip it once we know
    it repeats. A line appearing on only some pages is real content and is kept.
    """
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        first = page.text.strip().split("\n")[0].strip().lower() if page.text.strip() else ""
        if first and len(first) <= 120:
            counts[first] = counts.get(first, 0) + 1
    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold}


def _strip_headers(pages: list[Page]) -> None:
    """Remove detected running headers from each page's text (in place)."""
    headers = _running_headers(pages)
    if not headers:
        return
    for page in pages:
        if not page.text.strip():
            continue
        lines = page.text.split("\n")
        kept = [
            ln for i, ln in enumerate(lines)
            if not (i == 0 and ln.strip().lower() in headers)
        ]
        page.text = "\n".join(kept).strip()


def extract_pdf(
    source: str | Path | bytes,
    filename: str,
    *,
    ocr_engine: str = "auto",
    ocr_languages: str = "eng",
    ocr_min_chars_per_page: int = 40,
    extract_tables: bool = True,
    max_pages: Optional[int] = None,
) -> ExtractedDocument:
    """Extract a PDF into page-anchored content.

    ``source`` may be a path or raw bytes. Bytes are used for uploads so that we
    never write untrusted files to disk before validating them.

    Returns an :class:`ExtractedDocument`. Check ``.errors`` for failures and
    ``.warnings`` for degraded pages.
    """
    pymupdf = _fitz()

    # ---- open ------------------------------------------------------------
    doc = None
    errors: list[str] = []
    warnings: list[str] = []
    digest = ""

    try:
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            digest = doc_hash(data)
            doc = pymupdf.open(stream=data, filetype="pdf")
        else:
            path = Path(source)
            digest = doc_hash(path.read_bytes())
            doc = pymupdf.open(str(path))
    except Exception as exc:
        msg = f"Could not open PDF: {exc}"
        log.error("%s (%s)", msg, filename)
        return ExtractedDocument(
            filename=filename,
            doc_id=digest or "unknown",
            pages=[],
            page_count=0,
            errors=[msg],
        )

    if doc.is_encrypted and not doc.authenticate(""):
        msg = "PDF is password-protected; cannot extract content."
        errors.append(msg)
        return ExtractedDocument(filename, digest, [], 0, warnings=errors, errors=errors)

    try:
        page_count = doc.page_count
        if page_count == 0:
            errors.append("PDF contains no pages.")
            return ExtractedDocument(filename, digest, [], 0, warnings=[], errors=errors)

        # ---- OCR capability ---------------------------------------------
        ocr_status = ocr_mod.probe(ocr_engine)
        if not ocr_status.available:
            warnings.append(f"OCR unavailable: {ocr_status.detail}")
            log.info("OCR unavailable for %s: %s", filename, ocr_status.detail)

        pages: list[Page] = []
        ocr_pages: list[int] = []
        limit = page_count if max_pages is None else min(page_count, max_pages)

        for index in range(limit):
            page_no = index + 1
            page = None
            try:
                page = doc.load_page(index)
            except Exception as exc:
                warnings.append(f"Page {page_no}: could not be read ({exc}).")
                continue

            try:
                raw_text = page.get_text("text") or ""
            except Exception as exc:
                raw_text = ""
                warnings.append(f"Page {page_no}: text extraction failed ({exc}).")

            text = clean_text(raw_text)
            tables = _extract_tables(page) if extract_tables else []

            # Tables alone make a page non-empty even when the text layer is not.
            alnum = sum(ch.isalnum() for ch in text)
            needs_ocr = alnum < ocr_min_chars_per_page and not tables

            if needs_ocr and ocr_status.available:
                png = _render_for_ocr(page)
                if png:
                    ocr_text, err = ocr_mod.ocr_image_bytes(png, ocr_status, ocr_languages)
                    if err:
                        warnings.append(f"Page {page_no}: OCR failed ({err}).")
                    elif ocr_text.strip():
                        # OCR text is lower quality; keep it but mark the page.
                        text = ocr_text.strip()
                        ocr_pages.append(page_no)
                elif not warnings[-1:] or "rasterisation" not in warnings[-1]:
                    warnings.append(f"Page {page_no}: could not be rendered for OCR.")
            elif needs_ocr and not ocr_status.available:
                warnings.append(
                    f"Page {page_no}: no extractable text (likely scanned) and OCR is "
                    f"unavailable, so this page was skipped."
                )

            pages.append(
                Page(
                    page_number=page_no,
                    text=text,
                    tables=tables,
                    headings=detect_headings(text),
                    used_ocr=page_no in ocr_pages,
                    char_count=sum(ch.isalnum() for ch in text),
                )
            )

        if limit < page_count:
            warnings.append(f"Only the first {limit} of {page_count} pages were extracted.")

        # Strip repeating running headers only after every page is extracted, so
        # we can tell a header apart from genuine first-page content.
        _strip_headers(pages)
        for page in pages:
            page.char_count = sum(ch.isalnum() for ch in page.text)

        extracted = ExtractedDocument(
            filename=filename,
            doc_id=digest,
            pages=pages,
            page_count=page_count,
            warnings=warnings,
            errors=errors,
            ocr_pages=ocr_pages,
        )
        if extracted.total_chars == 0 and not errors:
            errors.append(
                "No extractable text found in any page. The document is probably "
                "scanned and OCR is unavailable."
            )
        return extracted
    finally:
        try:
            doc.close()
        except Exception:
            pass


def iter_documents(paths: Iterable[Path]) -> list[Path]:
    """List readable PDFs in a directory (or pass through an explicit list)."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(q for q in p.glob("*.pdf") if q.is_file()))
        elif p.is_file():
            out.append(p)
    return out