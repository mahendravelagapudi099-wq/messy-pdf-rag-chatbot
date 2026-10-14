"""Upload validation: type, size, readability and path safety."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# A real PDF starts with "%PDF-", not necessarily at offset 0 (some producers
# prepend junk), so we scan a small prefix rather than checking byte 0 only.
_PDF_MAGIC = b"%PDF-"
_PDF_HEADER_SCAN_BYTES = 1024

_UNSAFE = re.compile(r"[^A-Za-z0-9._\- ]+")


@dataclass
class ValidationResult:
    ok: bool
    filename: str
    message: str = ""
    size_bytes: int = 0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def safe_filename(name: str, fallback: str = "document.pdf") -> str:
    """Return a filesystem-safe basename.

    Strips directory components (both POSIX and Windows), rejects traversal
    sequences, and normalises unicode so that two visually identical names map
    to the same key in the store.
    """
    name = (name or "").strip()
    # Drop any directory component, whatever separator style was used.
    name = name.replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = _UNSAFE.sub("_", name).strip(". ")
    if not name:
        return fallback
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name[:180]


def looks_like_pdf(data: bytes) -> bool:
    """True when the PDF magic marker appears near the start of the file."""
    return _PDF_MAGIC in data[:_PDF_HEADER_SCAN_BYTES]


def validate_upload(
    filename: str,
    data: bytes,
    max_bytes: int,
    *,
    require_magic: bool = True,
) -> ValidationResult:
    """Validate an uploaded PDF.

    Checks, in order: non-empty, extension, size ceiling, then content sniffing.
    The content check is what actually matters -- renaming ``evil.exe`` to
    ``evil.pdf`` is caught by the magic-byte test, not the extension.
    """
    name = safe_filename(filename)

    if not data:
        return ValidationResult(False, name, "File is empty (0 bytes).")

    size = len(data)
    if size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return ValidationResult(
            False, name, f"File is {size / 1048576:.1f} MB; the limit is {limit_mb:.0f} MB.", size
        )

    if not (filename or "").strip().lower().endswith(".pdf"):
        # Check the *original* name: safe_filename() appends .pdf for display, so
        # testing the sanitised name here would silently accept a renamed file.
        return ValidationResult(False, name, "Only PDF files are accepted.", size)

    if require_magic and not looks_like_pdf(data):
        return ValidationResult(
            False,
            name,
            "File does not look like a PDF (missing %PDF- header). It may be corrupt "
            "or renamed from another format.",
            size,
        )

    return ValidationResult(True, name, "OK", size)


def validate_disk_file(path: Path, max_bytes: int) -> ValidationResult:
    """Validate a PDF that already exists on disk (used by the benchmark CLI)."""
    if not path.exists():
        return ValidationResult(False, path.name, "File does not exist.")
    if not path.is_file():
        return ValidationResult(False, path.name, "Path is not a file.")
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ValidationResult(False, path.name, f"Could not read file: {exc}")
    return validate_upload(path.name, data, max_bytes)