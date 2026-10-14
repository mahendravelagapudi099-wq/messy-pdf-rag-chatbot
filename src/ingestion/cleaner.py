"""Text cleanup for text extracted from real-world PDFs.

The goals here are conservative: fix the artefacts that break embedding quality
(hyphenation across line breaks, ligatures, excessive whitespace, page furniture)
without destroying content that a user might legitimately ask about.
"""

from __future__ import annotations

import re

# Ligatures and typographic characters that PDF text layers emit.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
    "­": "",  # soft hyphen
    "ﬁ": "fi",
}
_LIG_RE = re.compile("|".join(map(re.escape, _LIGATURES)))

# "exam-\nple" -> "example" (hyphen at end of line).
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑‒–]\n(\w)")
# Non-breaking / ideographic spaces and stray control characters.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NBSP = "   "
_SPACES = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+\n")

# Page furniture that adds noise without adding information.
_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?[-–—]?\s*\d+\s*(?:[-–—]?|of\s+\d+)?\s*$", re.I)
_BANNER = re.compile(r"^\s*(?:confidential|restricted|page\s+\d+\s+of\s+\d+)\s*$", re.I)


def normalize(text: str) -> str:
    """Apply encoding-level fixes. Safe to run before any other step."""
    if not text:
        return ""
    text = _LIG_RE.sub(lambda m: _LIGATURES[m.group(0)], text)
    text = _CTRL.sub("", text)
    for sp in _NBSP:
        text = text.replace(sp, " ")
    return text


def dehyphenate(text: str) -> str:
    """Join words split across a line break by a hyphen."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def strip_page_furniture(text: str) -> str:
    """Drop standalone page numbers and confidentiality banners."""
    lines = [ln for ln in text.split("\n") if not (_PAGE_NUM.match(ln) or _BANNER.match(ln))]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Full cleanup pass used on page text before chunking."""
    if not text:
        return ""
    text = normalize(text)
    text = dehyphenate(text)
    text = strip_page_furniture(text)
    text = _SPACES.sub(" ", text)
    text = _TRAIL_WS.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def detect_headings(text: str) -> list[str]:
    """Heuristic heading detection.

    Government circulars, syllabi and academic regulations rarely carry explicit
    heading markup, so we use typography: short lines in title/upper case, or
    numbered section markers such as ``3. ELIGIBILITY`` / ``Section 4``.
    """
    headings: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not (3 <= len(line) <= 120):
            continue

        # "1. TITLE" / "1.2 Title" / "SECTION 5: Title" / "ARTICLE IV"
        if re.match(r"^(?:section|article|chapter|part|annex(?:ure)?)\s+[0-9IVX]+[.:)]?\s+\S", line, re.I):
            headings.append(line)
            continue
        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][\w &/()'-]{2,}$", line):
            headings.append(line)
            continue

        # Uppercase lines that look like titles rather than shouting body text.
        letters = [c for c in line if c.isalpha()]
        if letters and len(line) <= 90:
            upper_ratio = sum(c.isupper() for c in letters) / len(letters)
            if upper_ratio > 0.85 and line.isupper() and len(line.split()) <= 12:
                headings.append(line)

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for h in headings:
        key = h.lower()
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:20]


def clean_table_cell(value: object) -> str:
    """Normalise a single extracted table cell."""
    if value is None:
        return ""
    s = str(value)
    s = normalize(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = _SPACES.sub(" ", s)
    return s.strip()