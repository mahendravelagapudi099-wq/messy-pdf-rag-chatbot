"""Evaluation dataset: JSONL schema, loading, validation and coverage checks.

One JSON object per line. Required fields are ``id``, ``question`` and
``should_abstain``; everything else enriches the metrics. Unknown keys are
preserved but ignored, so datasets can carry provenance (source URLs, notes).

Example
-------
.. code-block:: json

    {"id": "factual-001", "question": "What is the repo rate?",
     "reference_answer": "The repo rate was held at 6.50 percent.",
     "expected_source": "rbi_mps.pdf", "expected_pages": [3],
     "required_facts": ["6.50 percent"], "should_abstain": false,
     "category": "factual"}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

log = logging.getLogger(__name__)

VALID_CATEGORIES = (
    "factual",       # direct lookup of a stated fact
    "table",         # value lives in a table
    "multi_page",    # answer needs evidence from several pages
    "paraphrase",    # question wording differs from the document
    "scanned",       # answer lives on an OCR'd page
    "unanswerable",  # must be abstained from
)

TEMPLATE_DATASET = Path(__file__).resolve().parents[2] / "data" / "eval_dataset.jsonl"


@dataclass
class EvalQuestion:
    """A single evaluation case."""

    id: str
    question: str
    reference_answer: str = ""
    expected_source: Optional[str] = None
    expected_pages: list[int] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    should_abstain: bool = False
    category: str = "factual"
    notes: str = ""
    source_url: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalQuestion":
        pages = raw.get("expected_pages") or []
        if isinstance(pages, (int, str)):
            pages = [int(pages)]
        facts = raw.get("required_facts") or []
        if isinstance(facts, str):
            facts = [facts]

        return cls(
            id=str(raw.get("id") or raw.get("question_id") or "").strip(),
            question=str(raw.get("question") or "").strip(),
            reference_answer=str(raw.get("reference_answer") or raw.get("answer") or "").strip(),
            expected_source=(raw.get("expected_source") or raw.get("expected_document") or None),
            expected_pages=[int(p) for p in pages],
            required_facts=[str(f) for f in facts],
            should_abstain=bool(raw.get("should_abstain", False)),
            category=str(raw.get("category") or "factual"),
            notes=str(raw.get("notes") or ""),
            source_url=str(raw.get("source_url") or ""),
        )

    @property
    def is_answerable(self) -> bool:
        return not self.should_abstain


class DatasetError(ValueError):
    """Raised when a dataset file cannot be used as-is."""


def load_dataset(path: Path | str, *, strict: bool = False) -> list[EvalQuestion]:
    """Load and validate a JSONL evaluation dataset.

    ``strict=True`` raises on any malformed record; otherwise bad lines are logged
    and skipped, so a partially hand-edited dataset still runs.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")

    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{path.name}:{lineno}: invalid JSON ({exc.msg})"
            if strict:
                raise DatasetError(msg) from exc
            log.warning("skipping %s", msg)
            continue

        if not isinstance(raw, dict):
            msg = f"{path.name}:{lineno}: expected a JSON object"
            if strict:
                raise DatasetError(msg)
            log.warning("skipping %s", msg)
            continue

        q = EvalQuestion.from_dict(raw)
        if not q.id or not q.question:
            msg = f"{path.name}:{lineno}: missing 'id' or 'question'"
            if strict:
                raise DatasetError(msg)
            log.warning("skipping %s", msg)
            continue
        if q.id in seen_ids:
            if strict:
                raise DatasetError(f"{path.name}:{lineno}: duplicate id {q.id!r}")
            log.warning("skipping duplicate id %s at line %d", q.id, lineno)
            continue
        seen_ids.add(q.id)
        questions.append(q)

    if not questions:
        raise DatasetError(f"No usable questions in {path}")
    return questions


def dump_dataset(questions: Sequence[EvalQuestion], path: Path | str) -> Path:
    """Write a dataset to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(
                json.dumps(
                    {
                        "id": q.id,
                        "question": q.question,
                        "reference_answer": q.reference_answer,
                        "expected_source": q.expected_source,
                        "expected_pages": q.expected_pages,
                        "required_facts": q.required_facts,
                        "should_abstain": q.should_abstain,
                        "category": q.category,
                        "notes": q.notes,
                        "source_url": q.source_url,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


@dataclass
class CoverageReport:
    """Coverage statistics for a dataset -- guards against a lopsided test set."""

    total: int
    by_category: dict[str, int]
    answerable: int
    unanswerable: int
    with_gold_source: int
    with_gold_pages: int
    with_required_facts: int
    missing_reference: int

    @property
    def is_complete(self) -> bool:
        """True when every required category is represented."""
        return all(self.by_category.get(c, 0) > 0 for c in VALID_CATEGORIES)

    def summary(self) -> str:
        cats = ", ".join(f"{c}={self.by_category.get(c, 0)}" for c in VALID_CATEGORIES)
        return (
            f"{self.total} questions ({self.answerable} answerable / "
            f"{self.unanswerable} unanswerable)\n"
            f"  categories: {cats}\n"
            f"  with gold source: {self.with_gold_source} | "
            f"with gold pages: {self.with_gold_pages} | "
            f"with required facts: {self.with_required_facts}\n"
            f"  all required categories present: {'yes' if self.is_complete else 'NO'}"
        )


def coverage(questions: Sequence[EvalQuestion]) -> CoverageReport:
    by_cat: dict[str, int] = {}
    for q in questions:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    return CoverageReport(
        total=len(questions),
        by_category=by_cat,
        answerable=sum(1 for q in questions if q.is_answerable),
        unanswerable=sum(1 for q in questions if q.should_abstain),
        with_gold_source=sum(1 for q in questions if q.expected_source),
        with_gold_pages=sum(1 for q in questions if q.expected_pages),
        with_required_facts=sum(1 for q in questions if q.required_facts),
        missing_reference=sum(1 for q in questions if not q.reference_answer),
    )


def iter_categories(questions: Sequence[EvalQuestion]) -> Iterator[str]:
    for q in questions:
        yield q.category