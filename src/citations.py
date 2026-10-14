"""Citation construction and verification.

Two responsibilities:

1. **Attach** ``[1]``, ``[2]`` markers to the answer so each claim points at the
   evidence behind it.
2. **Verify** that a citation actually supports the sentence it is attached to.
   Verification is lexical (not an LLM judge) -- it computes content-word
   overlap between the claim and the cited chunk. That is reproducible, free, and
   catches the failure mode we actually care about: a citation pointing at an
   unrelated page.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .models import Citation, RetrievedChunk

log = logging.getLogger(__name__)

# Markers the model is asked to emit, e.g. "[2]".
_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

# Stopwords excluded from the overlap computation.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being do does did doing have has had having will would shall
    should can could may might must not no nor so such it its it's they them their there here
    what which who whom whose when where why how all any both each few more most other some
    only own same too very s t don now i you he she we me my your his her our""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Below this overlap score a citation is treated as unsupported.
DEFAULT_SUPPORT_THRESHOLD = 0.12


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


def content_tokens(text: str) -> set[str]:
    """Lowercase, de-punctuated content words, with stopwords removed."""
    words = _WORD_RE.findall(_normalise(text).lower())
    out: set[str] = set()
    for w in words:
        if w in _STOPWORDS or len(w) <= 1:
            continue
        # Keep the stem-ish prefix so "eligibility"/"eligible" can match.
        out.add(w)
    return out


def overlap_score(claim: str, evidence: str) -> float:
    """Fraction of the claim's content words present in the evidence (0..1).

    Recall-oriented on purpose: we ask how much of the claim is grounded, not how
    much of the evidence is unused. Long supporting paragraphs would otherwise be
    unfairly penalised.
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    ev_tokens = content_tokens(evidence)
    if not ev_tokens:
        return 0.0
    # Also allow simple plural/prefix matching.
    matched = 0
    for tok in claim_tokens:
        if tok in ev_tokens:
            matched += 1
        elif any(e.startswith(tok) or tok.startswith(e) for e in ev_tokens if len(e) > 3):
            matched += 1
    return matched / len(claim_tokens)


def excerpt(text: str, limit: int = 320) -> str:
    """Trim an excerpt to a readable length on a word boundary."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.strip() + " ..."


def build_citations(
    retrieved: Sequence[RetrievedChunk],
    answer_text: str = "",
    *,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    excerpt_limit: int = 320,
) -> list[Citation]:
    """Build numbered citations for retrieved chunks.

    When ``answer_text`` is supplied, only markers actually present in the answer
    are kept, so the citation list matches what the user can see. Chunks cited
    but poorly grounded are still listed, flagged ``grounded=False``, so the UI
    can warn rather than hide the problem.
    """
    used: set[int] | None = None
    if answer_text:
        used = {int(m) for m in _MARKER_RE.findall(answer_text)}

    citations: list[Citation] = []
    for i, rc in enumerate(retrieved, start=1):
        if used is not None and i not in used:
            continue
        score = similarity_of(rc)
        citations.append(
            Citation(
                marker=f"[{i}]",
                doc_id=rc.chunk.doc_id,
                filename=rc.chunk.filename,
                page=rc.chunk.page,
                chunk_id=rc.chunk.chunk_id,
                excerpt=excerpt(rc.chunk.text, excerpt_limit),
                similarity=score,
            )
        )

    # Flag citations whose marker-bearing claim is not well supported.
    if answer_text:
        claims = split_claims(answer_text)
        for cit in citations:
            idx = int(cit.marker.strip("[]")) - 1
            claim = claims.get(idx, "")
            if claim:
                best = max(
                    (overlap_score(claim, rc.chunk.text) for rc in retrieved), default=0.0
                )
                cit.grounded = best >= threshold
    return citations


def split_claims(answer_text: str) -> dict[int, str]:
    """Map marker index -> the sentence carrying that marker."""
    claims: dict[int, str] = {}
    # Split on sentence terminators, keeping trailing citations with the sentence.
    for sentence in re.split(r"(?<=[.!?])\s+", answer_text or ""):
        for marker in _MARKER_RE.findall(sentence):
            idx = int(marker) - 1
            cleaned = _MARKER_RE.sub("", sentence).strip()
            if cleaned and idx not in claims:
                claims[idx] = cleaned
    return claims


def similarity_of(rc: RetrievedChunk) -> float:
    """Score to display for a chunk: rerank score when present, else cosine."""
    if rc.rerank_score is not None:
        # Map an unbounded logit into a 0..1 range for display only.
        return 1.0 / (1.0 + pow(2.718281828, -rc.rerank_score))
    return rc.similarity


def verify_citations(
    citations: Iterable[Citation], answer_text: str
) -> dict[str, int]:
    """Summarise citation health for the UI and evaluation metrics."""
    cits = list(citations)
    if not cits:
        return {"total": 0, "grounded": 0, "ungrounded": 0}
    grounded = sum(1 for c in cits if c.grounded)
    return {
        "total": len(cits),
        "grounded": grounded,
        "ungrounded": len(cits) - grounded,
    }


def citation_to_line(c: Citation) -> str:
    return f"{c.marker} {c.filename}, page {c.page} (match {c.similarity:.2f})"