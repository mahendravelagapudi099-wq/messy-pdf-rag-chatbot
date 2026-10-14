"""Benchmark runner: ingestion, retrieval, answering and scoring.

The runner is the single source of truth for how the project measures itself. It
supports two answer modes:

* ``extractive`` -- deterministic, no API key needed. Selects the evidence
  sentences most relevant to the question. This is what runs by default and in
  CI, so retrieval quality can be measured reproducibly on any machine.
* ``llm`` -- calls the configured OpenAI-compatible endpoint. Required for the
  real generation metrics (unsupported-answer rate, citation correctness).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..citations import build_citations
from ..config import Settings, get_settings
from ..chunking import PageAwareChunker
from ..generation import extractive_answer, generate_answer
from ..ingestion import ingest_file
from ..ingestion.pipeline import Chunker, VectorStore
from ..models import Answer, IngestionReport, RetrievedChunk
from ..retrieval import retrieve
from ..vector_store import VectorStore as ChromaVectorStore
from .dataset import EvalQuestion, coverage, load_dataset
from .metrics import (
    Metrics,
    QuestionOutcome,
    aggregate,
    citation_correct,
    fact_recall,
    reciprocal_rank,
    retrieved_hit,
    token_f1,
)

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class BenchmarkConfig:
    """Everything that defines a reproducible benchmark run."""

    dataset_path: Path
    corpus_dir: Path
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.0
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    answer_mode: str = "extractive"  # extractive | llm
    persist_dir: Optional[Path] = None
    collection_suffix: str = ""
    page_tolerance: int = 1
    label: str = ""

    def key(self) -> str:
        base = f"cs{self.chunk_size}_co{self.chunk_overlap}_k{self.top_k}"
        if self.use_reranker:
            base += "_rr"
        if self.collection_suffix:
            base += f"_{self.collection_suffix}"
        return base


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    metrics: Metrics
    outcomes: list[QuestionOutcome]
    ingestion: list[IngestionReport] = field(default_factory=list)
    dataset_coverage: str = ""
    elapsed_s: float = 0.0

    def row(self) -> dict[str, Any]:
        d = {
            "label": self.config.label or self.config.key(),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "top_k": self.config.top_k,
            "reranker": self.config.use_reranker,
            "answer_mode": self.config.answer_mode,
        }
        d.update(self.metrics.as_dict())
        return d


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------


def _answer(
    question: EvalQuestion,
    retrieved: Sequence[RetrievedChunk],
    settings: Settings,
    mode: str,
    history: Optional[list[dict[str, str]]] = None,
) -> Answer:
    """Produce an answer plus its citations, in the requested mode."""
    if mode == "llm":
        ans = generate_answer(question.question, retrieved, settings, history=history)
    else:
        result = extractive_answer(question.question, retrieved)
        ans = Answer(
            question=question.question,
            text=result.text,
            retrieved=list(retrieved),
            abstained=result.abstained,
            abstention_reason=result.reason,
        )
    ans.citations = build_citations(ans.retrieved, ans.text)
    return ans


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_benchmark(
    config: BenchmarkConfig,
    settings: Optional[Settings] = None,
    *,
    reuse_store: bool = False,
    progress: Optional[ProgressFn] = None,
) -> BenchmarkResult:
    """Ingest the corpus, answer every question, and score the run."""
    settings = settings or get_settings()
    settings = settings.with_overrides(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold,
        use_reranker=config.use_reranker,
        reranker_model=config.reranker_model,
    )

    started = time.perf_counter()
    questions = load_dataset(config.dataset_path)
    cov = coverage(questions)

    if progress:
        progress(f"Loaded {len(questions)} questions", 0.05)

    # -- fresh store per configuration so configs cannot contaminate each other
    persist_dir = config.persist_dir or (
        settings.chroma_persist_dir.parent / f"benchmark_{config.key()}"
    )
    collection = f"{settings.chroma_collection}{config.collection_suffix}"
    store: ChromaVectorStore
    if reuse_store:
        store = ChromaVectorStore(settings, persist_dir=persist_dir, collection=collection)
    else:
        import shutil

        if Path(persist_dir).exists():
            shutil.rmtree(persist_dir)
        store = ChromaVectorStore(settings, persist_dir=persist_dir, collection=collection)

    # -- ingest ----------------------------------------------------------
    pdfs = sorted(
        q for q in config.corpus_dir.glob("*.pdf") if q.is_file()
    )
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in corpus directory: {config.corpus_dir}\n"
            f"Add PDFs to data/corpus/ or pass --corpus."
        )

    chunker = PageAwareChunker(config.chunk_size, config.chunk_overlap)
    reports: list[IngestionReport] = []
    for i, pdf in enumerate(pdfs):
        if progress:
            progress(f"Ingesting {pdf.name} ({i + 1}/{len(pdfs)})", 0.1 + 0.4 * i / len(pdfs))
        reports.append(ingest_file(pdf, store, chunker, settings, force=True))

    failed = sum(1 for r in reports if r.status == "failed")
    if failed:
        log.warning("%d/%d documents failed ingestion", failed, len(reports))

    # -- question loop ---------------------------------------------------
    outcomes: list[QuestionOutcome] = []
    for i, q in enumerate(questions):
        if progress:
            progress(f"Question {i + 1}/{len(questions)}", 0.55 + 0.4 * i / len(questions))

        result = retrieve(
            store,
            q.question,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            use_reranker=config.use_reranker,
            reranker_model=config.reranker_model,
        )
        answer = _answer(q, result.chunks, settings, config.answer_mode)

        outcome = QuestionOutcome(
            question_id=q.id,
            question=q.question,
            should_abstain=q.should_abstain,
            abstained=answer.abstained,
            latency_s=result.latency_s,
            retrieved_count=len(result.chunks),
            answer=answer.text,
        )

        if q.is_answerable:
            outcome.hit = retrieved_hit(
                result.chunks, q.expected_source, q.expected_pages,
                reference=q.reference_answer, page_tolerance=config.page_tolerance,
            )
            outcome.reciprocal_rank = reciprocal_rank(
                result.chunks, q.expected_source, q.expected_pages,
                reference=q.reference_answer, page_tolerance=config.page_tolerance,
            )
            if q.reference_answer:
                outcome.answer_f1 = token_f1(answer.text, q.reference_answer)
            if q.required_facts:
                outcome.fact_recall = fact_recall(answer.text, q.required_facts)
            outcome.citation_correct = citation_correct(
                answer, q.expected_source, q.expected_pages,
                page_tolerance=config.page_tolerance,
            )
        outcomes.append(outcome)

    metrics = aggregate(outcomes, ingestion_total=len(reports), ingestion_failed=failed)

    return BenchmarkResult(
        config=config,
        metrics=metrics,
        outcomes=outcomes,
        ingestion=reports,
        dataset_coverage=cov.summary(),
        elapsed_s=time.perf_counter() - started,
    )