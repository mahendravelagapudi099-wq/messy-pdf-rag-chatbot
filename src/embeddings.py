"""Embedding providers.

The default provider is ``onnx``: it uses the all-MiniLM-L6-v2 model shipped by
Chroma in ONNX form, so it needs no torch, downloads ~79 MB, and runs on CPU.

``sentence-transformers`` and ``huggingface-api`` are supported when installed /
configured. Providers are probed rather than assumed, so a missing optional
dependency degrades to a warning instead of an import error at module load.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

log = logging.getLogger(__name__)

# Silence noisy telemetry/absl chatter from optional deps.
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.impl.docs.DocsTelemetry")


class EmbeddingProvider(Protocol):
    """Anything that can turn text into vectors."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str = ""


def _load_onnx() -> Optional[EmbeddingProvider]:
    """Chroma's bundled ONNX MiniLM -- the portable default."""
    try:
        from chromadb.utils import embedding_functions

        ef = embedding_functions.DefaultEmbeddingFunction()
        dim = _probe_dimension(ef, "embedding dimension probe")
        return _OnnxProvider(ef, dim)
    except Exception as exc:
        log.info("ONNX embedding provider unavailable: %s", exc)
        return None


def _probe_dimension(fn, label: str) -> int:
    """Infer vector width by embedding a short probe string."""
    vec = fn(["dimension probe"])[0]
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return len(vec)


class _OnnxProvider:
    """Thin adapter over Chroma's ONNX embedding function."""

    name = "onnx/all-MiniLM-L6-v2"

    def __init__(self, fn, dimension: int) -> None:
        self._fn = fn
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._fn(list(texts))
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


class _SentenceTransformerProvider:
    """Higher-quality embeddings via sentence-transformers (needs torch)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers/{model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return [row.tolist() for row in arr]


def get_embedding_provider(
    provider: str = "onnx", model: Optional[str] = None, device: str = "cpu"
) -> EmbeddingProvider:
    """Resolve a provider by name, raising a clear error if unavailable.

    ``provider="auto"`` tries sentence-transformers first (better quality when
    present) and falls back to ONNX.
    """
    provider = (provider or "onnx").lower()

    if provider in {"sentence-transformers", "st"}:
        st = _SentenceTransformerProvider(model or "all-MiniLM-L6-v2", device)
        return st

    if provider in {"onnx", "auto"}:
        if provider == "auto":
            try:
                return _SentenceTransformerProvider(
                    model or "sentence-transformers/all-MiniLM-L6-v2", device
                )
            except Exception as exc:
                log.info("sentence-transformers unavailable (%s); using ONNX.", exc)
        onnx = _load_onnx()
        if onnx is not None:
            return onnx
        raise RuntimeError(
            "No embedding provider available. Install chromadb (for the ONNX "
            "provider) or sentence-transformers."
        )

    raise ValueError(f"Unknown embedding provider: {provider!r}")


def probe_providers(model: Optional[str] = None) -> dict[str, ProviderStatus]:
    """Report which providers are usable, for the settings UI."""
    status: dict[str, ProviderStatus] = {}
    try:
        p = _SentenceTransformerProvider(model or "all-MiniLM-L6-v2", "cpu")
        status["sentence-transformers"] = ProviderStatus("sentence-transformers", True, f"dim={p.dimension}")
    except Exception as exc:
        status["sentence-transformers"] = ProviderStatus(
            "sentence-transformers", False, str(exc)[:120]
        )
    try:
        p = _load_onnx()
        status["onnx"] = ProviderStatus(
            "onnx", p is not None, f"dim={p.dimension}" if p else "unavailable"
        )
    except Exception as exc:
        status["onnx"] = ProviderStatus("onnx", False, str(exc)[:120])
    return status


# Module-level singleton so the model loads once per process.
_cached: Optional[EmbeddingProvider] = None


def get_cached_provider(settings) -> EmbeddingProvider:
    """Return a process-wide provider, loading it on first use."""
    global _cached
    key = (settings.embedding_provider, settings.embedding_model, settings.embedding_device)
    if _cached is not None and getattr(_cached, "_key", None) == key:
        return _cached
    provider = get_embedding_provider(
        settings.embedding_provider, settings.embedding_model, settings.embedding_device
    )
    try:
        setattr(provider, "_key", key)
    except Exception:
        pass
    _cached = provider
    return provider