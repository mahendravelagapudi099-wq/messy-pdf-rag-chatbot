"""Tests for upload validation and path-safety handling."""

from __future__ import annotations

import pytest

from src.ingestion.validator import (
    looks_like_pdf,
    safe_filename,
    validate_disk_file,
    validate_upload,
)


class TestSafeFilename:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("report.pdf", "report.pdf"),
            ("../../etc/passwd", "passwd.pdf"),
            ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
            ("/absolute/path/doc.pdf", "doc.pdf"),
            ("C:\\Users\\me\\secret.pdf", "secret.pdf"),
            ("", "document.pdf"),
            ("no-extension", "no-extension.pdf"),
            ("spaces in name.pdf", "spaces in name.pdf"),
        ],
    )
    def test_strips_directories_and_traversal(self, raw, expected):
        assert safe_filename(raw) == expected

    def test_removes_control_and_shell_characters(self):
        assert ";" not in safe_filename("report;rm -rf.pdf")
        assert "|" not in safe_filename("re|port.pdf")

    def test_length_is_bounded(self):
        assert len(safe_filename("a" * 500 + ".pdf")) <= 180

    def test_unicode_is_normalised(self):
        # NFKD + ascii encode maps accented characters to plain equivalents.
        assert safe_filename("rapport-économique.pdf") == "rapport-economique.pdf"


class TestMagicBytes:
    def test_detects_real_pdf(self, simple_pdf_bytes):
        assert looks_like_pdf(simple_pdf_bytes)

    def test_rejects_renamed_executable(self):
        assert not looks_like_pdf(b"MZ\x90\x00\x03" + b"\x00" * 64)

    def test_tolerates_leading_junk(self):
        # Some producers emit junk before the %PDF- header.
        assert looks_like_pdf(b"\x00\x01\x02%PDF-1.7\nrest")

    def test_rejects_truncated_file(self):
        assert not looks_like_pdf(b"%PD")


class TestValidateUpload:
    def test_accepts_valid_pdf(self, simple_pdf_bytes):
        result = validate_upload("doc.pdf", simple_pdf_bytes, max_bytes=10_000_000)
        assert result.ok
        assert result.message == "OK"

    def test_rejects_empty_file(self):
        result = validate_upload("doc.pdf", b"", max_bytes=1000)
        assert not result.ok
        assert "empty" in result.message.lower()

    def test_rejects_oversized_file(self, simple_pdf_bytes):
        result = validate_upload("doc.pdf", simple_pdf_bytes, max_bytes=10)
        assert not result.ok
        assert "limit" in result.message.lower()

    def test_rejects_non_pdf_extension(self, simple_pdf_bytes):
        result = validate_upload("doc.txt", simple_pdf_bytes, max_bytes=10_000_000)
        assert not result.ok
        assert "PDF" in result.message

    def test_rejects_disguised_non_pdf(self):
        """A renamed .exe must be caught by content, not extension."""
        result = validate_upload("invoice.pdf", b"MZ\x90\x00" + b"\x00" * 128, max_bytes=10_000_000)
        assert not result.ok
        assert "%PDF" in result.message

    def test_size_mb_reported(self, simple_pdf_bytes):
        result = validate_upload("doc.pdf", simple_pdf_bytes, max_bytes=10_000_000)
        assert result.size_bytes == len(simple_pdf_bytes)
        assert result.size_mb > 0

    def test_validate_disk_file_missing(self, tmp_path):
        result = validate_disk_file(tmp_path / "nope.pdf", max_bytes=10_000_000)
        assert not result.ok
        assert "does not exist" in result.message

    def test_validate_disk_file_directory(self, tmp_path):
        result = validate_disk_file(tmp_path, max_bytes=10_000_000)
        assert not result.ok