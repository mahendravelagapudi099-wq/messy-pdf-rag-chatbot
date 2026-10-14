"""Evaluation metrics.

Definitions (implemented literally as documented)
--------------------------------------------------

**Retrieval Hit Rate@k**
For each answerable question, the fraction whose *expected* source appears in the
top-k retrieved results. A hit counts when a retrieved chunk's page is within
``page_tolerance`` of any expected page of the expected document, or when the
retrieved text overlaps the reference answer enough to be the right passage.

**Mean Reciprocal Rank (MRR)**
Average of ``1 / rank_of_first_correct_result`` over answerable questions.
Questions with no correct result contribute 0. Only meaningful when page-level
gold data exists, which is why it is reported separately from hit rate.

**Answer correctness (token F1)**
F1 between the generated answer and the reference answer after normalisation
(stopword removal, lowercasing, punctuation stripping). Deterministic and free;
deliberately *not* an LLM judge.

**Required-fact recall**
Fraction of the ``required_facts`` keywords that appear in the answer. This is the
metric that actually catches a fluent-but-wrong answer, because it checks for the
specific values that must be present.

**Citation correctness**
Of the citations attached to an answer, the fraction that point at the expected
document *and* an expected page. Citations with no gold citation expectation are
excluded from the denominator.

**Unsupported-answer rate**
Among answerable questions where the model gave a non-abstaining answer, the
fraction whose required-fact recall is below ``support_threshold``. This is a
deterministic proxy for hallucination.

**Appropriate abstention rate**
Among questions marked ``should_abstain``, the fraction where the system *did*
abstain.

**Ingestion failure rate**
Fraction of documents in the corpus that could not be parsed into usable text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..models import Answer, RetrievedChunk

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being do does did doing have has had having will would shall
    should can could may might must not no nor so such it its they them their there here
    what which who whom whose when where why how all any both each few more most other some
    only own same too very i you he she we me my your his her our""".split()
)


def normalize_text(text: str) -> list[str]:
    """Lowercase content tokens, stopwords removed -- shared by all metrics."""
    if not text:
        return []
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [w for w in _WORD_RE.findall(norm) if w not in _STOPWORDS and len(w) > 1]


def token_f1(prediction: str, reference: str) -> float:
    """F1 of content-token multisets between prediction and reference."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return 1.0 if (not pred and not ref) else 0.0

    from collections import Counter

    p, r = Counter(pred), Counter(ref)
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def fact_recall(answer: str, required_facts: Sequence[str]) -> float:
    """Fraction of required facts present in the answer (substring + token aware)."""
    if not required_facts:
        return float("nan")  # undefined, not zero
    norm_answer = unicodedata.normalize("NFKD", answer or "").lower()
    answer_tokens = set(normalize_text(answer))
    hit = 0
    for fact in required_facts:
        f = (fact or "").strip()
        if not f:
            continue
        if f.lower() in norm_answer:
            hit += 1
            continue
        f_tokens = set(normalize_text(f))
        if f_tokens and f_tokens.issubset(answer_tokens):
            hit += 1
    return hit / len([f for f in required_facts if (f or "").strip()] or [1])


def retrieved_hit(
    retrieved: Sequence[RetrievedChunk],
    expected_doc: Optional[str],
    expected_pages: Sequence[int],
    *,
    reference: str = "",
    page_tolerance: int = 1,
    answer_min_overlap: float = 0.35,
) -> bool:
    """True when the expected source appears in the retrieved results.

    Two independent routes to a hit:
    * **Gold metadata** -- retrieved chunk is from ``expected_doc`` and its page is
      within ``page_tolerance`` of an expected page.
    * **Textual overlap** -- a retrieved chunk shares ``answer_min_overlap`` of its
      content tokens with the reference answer. This rescues datasets that lack
      page-level gold data, and guards against gold page numbers drifting when a
      document is re-typeset.
    """
    if not retrieved:
        return False

    if expected_doc:
        for rc in retrieved:
            if rc.chunk.filename != expected_doc:
                continue
            if not expected_pages:
                return True
            if any(abs(rc.chunk.page - p) <= page_tolerance for p in expected_pages):
                return True

    if reference:
        ref_tokens = set(normalize_text(reference))
        if ref_tokens:
            for rc in retrieved:
                overlap = len(ref_tokens & set(normalize_text(rc.chunk.text))) / len(ref_tokens)
                if overlap >= answer_min_overlap:
                    return True
    return False


def reciprocal_rank(
    retrieved: Sequence[RetrievedChunk],
    expected_doc: Optional[str],
    expected_pages: Sequence[int],
    *,
    reference: str = "",
    page_tolerance: int = 1,
    answer_min_overlap: float = 0.35,
) -> float:
    """``1/rank`` of the first correct result (0 when there is none)."""
    if not retrieved:
        return 0.0
    for i, rc in enumerate(retrieved, start=1):
        if retrieved_hit(
            [rc], expected_doc, expected_pages, reference=reference,
            page_tolerance=page_tolerance, answer_min_overlap=answer_min_overlap,
        ):
            return 1.0 / i
    return 0.0


def citation_correct(
    answer: Answer, expected_doc: Optional[str], expected_pages: Sequence[int],
    *, page_tolerance: int = 1,
) -> Optional[bool]:
    """Whether the answer's citations point at the supporting source and page.

    Returns ``None`` when the question has no citation expectation (excluded from
    the metric denominator).
    """
    if not expected_doc:
        return None
    if not answer.citations:
        return False
    return any(
        c.filename == expected_doc
        and (not expected_pages or any(abs(c.page - p) <= page_tolerance for p in expected_pages))
        for c in answer.citations
    )


def is_abstention(answer: Answer) -> bool:
    """Whether the system declined to answer."""
    return bool(answer.abstained)


@dataclass
class Metrics:
    """Aggregate metrics for one evaluation run."""

    n_questions: int = 0
    n_answerable: int = 0
    n_unanswerable: int = 0

    hit_rate_at_k: float = 0.0
    mrr: float = 0.0
    answer_f1: float = 0.0
    fact_recall: float = 0.0
    citation_correctness: float = 0.0
    unsupported_rate: float = 0.0
    abstention_rate: float = 0.0
    ingestion_failure_rate: float = 0.0

    mean_latency_s: float = 0.0
    answerable_answered: int = 0
    unanswerable_abstained: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_questions": self.n_questions,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            "mrr": round(self.mrr, 4),
            "answer_f1": round(self.answer_f1, 4),
            "fact_recall": round(self.fact_recall, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "unsupported_rate": round(self.unsupported_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "ingestion_failure_rate": round(self.ingestion_failure_rate, 4),
            "mean_latency_s": round(self.mean_latency_s, 4),
        }


@dataclass
class QuestionOutcome:
    """Per-question evaluation record, used for error analysis."""

    question_id: str
    question: str
    should_abstain: bool
    hit: Optional[bool] = None
    reciprocal_rank: float = 0.0
    answer_f1: Optional[float] = None
    fact_recall: Optional[float] = None
    citation_correct: Optional[bool] = None
    abstained: bool = False
    latency_s: float = 0.0
    retrieved_count: int = 0
    answer: str = ""
    notes: list[str] = field(default_factory=list)


def aggregate(
    outcomes: Sequence[QuestionOutcome],
    *,
    ingestion_total: int = 0,
    ingestion_failed: int = 0,
    support_threshold: float = 0.5,
) -> Metrics:
    """Reduce per-question outcomes to the documented aggregate metrics."""
    answerable = [o for o in outcomes if not o.should_abstain]
    unanswerable = [o for o in outcomes if o.should_abstain]

    m = Metrics(
        n_questions=len(outcomes),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )

    if answerable:
        hits = [o.hit for o in answerable if o.hit is not None]
        m.hit_rate_at_k = sum(hits) / len(hits) if hits else 0.0
        m.mrr = sum(o.reciprocal_rank for o in answerable) / len(answerable)

        f1s = [o.answer_f1 for o in answerable if o.answer_f1 is not None]
        m.answer_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        recalls = [o.fact_recall for o in answerable if o.fact_recall is not None]
        m.fact_recall = sum(recalls) / len(recalls) if recalls else 0.0

        cits = [o.citation_correct for o in answerable if o.citation_correct is not None]
        m.citation_correctness = sum(1 for c in cits if c) / len(cits) if cits else 0.0

        # Unsupported rate: answered but missing required facts.
        answered = [
            o for o in answerable
            if not o.abstained and o.fact_recall is not None
        ]
        m.answerable_answered = len(answered)
        m.unsupported_rate = (
            sum(1 for o in answered if o.fact_recall < support_threshold) / len(answered)
            if answered else 0.0
        )

    if unanswerable:
        abstained = sum(1 for o in unanswerable if o.abstained)
        m.unanswerable_abstained = abstained
        m.abstention_rate = abstained / len(unanswerable)

    if outcomes:
        m.mean_latency_s = sum(o.latency_s for o in outcomes) / len(outcomes)

    if ingestion_total:
        m.ingestion_failure_rate = ingestion_failed / ingestion_total

    return m


def by_category(outcomes: Iterable[QuestionOutcome]) -> dict[str, dict[str, float]]:
    """Break hit rate down by question category -- shows *where* retrieval fails."""
    buckets: dict[str, list[QuestionOutcome]] = {}
    for o in outcomes:
        cat = o.question_id.rsplit("-", 1)[0] if "-" in o.question_id else "other"
        buckets.setdefault(cat, []).append(o)

    out: dict[str, dict[str, float]] = {}
    for cat, group in sorted(buckets.items()):
        hits = [g.hit for g in group if g.hit is not None]
        recalls = [g.fact_recall for g in group if g.fact_recall is not None]
        out[cat] = {
            "n": len(group),
            "hit_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "fact_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        }
    return out