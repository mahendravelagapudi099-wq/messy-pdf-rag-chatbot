"""Retrieval: similarity search, thresholding and optional cross-encoder reranking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from .models import RetrievedChunk
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Retrieved passages plus timing and filtering diagnostics."""

    query: str
    chunks: list[RetrievedChunk]
    latency_s: float
    total_candidates: int
    below_threshold: int
    reranked: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_similarity(self) -> float:
        return self.chunks[0].similarity if self.chunks else 0.0


class Reranker:
    """Cross-encoder reranking, optional.

    A bi-encoder (what we embed with) scores query and document independently,
    which is fast but coarse. A cross-encoder reads both together and is far more
    accurate but costs one forward pass per candidate. We only rerank a small
    candidate pool, and only when explicitly enabled.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        # Cross-encoder logits are unbounded; sort by them, not by cosine.
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)


_reranker_cache: dict[str, Reranker] = {}


def get_reranker(model_name: str) -> Optional[Reranker]:
    """Load (and cache) a reranker, returning ``None`` if unavailable."""
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]
    try:
        r = Reranker(model_name)
    except Exception as exc:
        log.warning("Reranker unavailable (%s); using plain vector ranking.", exc)
        _reranker_cache[model_name] = None  # type: ignore[assignment]
        return None
    _reranker_cache[model_name] = r
    return r


def clear_reranker_cache() -> None:
    _reranker_cache.clear()


def retrieve(
    store: VectorStore,
    query: str,
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    use_reranker: bool = False,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_candidates: int = 20,
    doc_ids: Optional[Sequence[str]] = None,
) -> RetrievalResult:
    """Retrieve the passages most likely to answer ``query``.

    Over-fetch when reranking so the cross-encoder has candidates to reorder.
    Candidates below ``similarity_threshold`` are dropped -- this is what lets the
    downstream answer say "I don't know" instead of citing a loosely related page.
    """
    started = time.perf_counter()
    query = (query or "").strip()
    if not query:
        return RetrievalResult(query, [], 0.0, 0, 0)

    pool = max(top_k, reranker_candidates if use_reranker else top_k)
    candidates = store.query(query, n_results=pool, doc_ids=doc_ids)
    total = len(candidates)

    reranked = False
    if use_reranker and candidates:
        reranker = get_reranker(reranker_model)
        if reranker is not None:
            candidates = reranker.rerank(query, candidates)
            reranked = True

    if similarity_threshold > 0:
        kept = [c for c in candidates if c.similarity >= similarity_threshold]
        dropped = total - len(kept)
    else:
        kept, dropped = candidates, 0

    final = kept[:top_k]
    latency = time.perf_counter() - started
    return RetrievalResult(
        query=query,
        chunks=final,
        latency_s=latency,
        total_candidates=total,
        below_threshold=dropped,
        reranked=reranked,
    )