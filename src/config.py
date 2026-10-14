"""Central configuration loaded from environment variables / .env.

All runtime settings are funnelled through :class:`Settings` so that the UI can
override them per-session without mutating global state.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # These are extremely chatty and add nothing for an end user. Chroma logs a
    # telemetry ERROR on every call when posthog's API signature has moved on,
    # so they must be pushed above ERROR (not just to WARNING) to go quiet.
    for noisy in (
        "httpx", "httpcore", "chromadb.telemetry", "chromadb.telemetry.product.posthog",
        "posthog", "urllib3", "matplotlib.font_manager",
    ):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration."""

    # --- LLM -------------------------------------------------------------
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 60.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    answer_mode: str = field(default_factory=lambda: _env("ANSWER_MODE", "auto"))

    # --- Embeddings ------------------------------------------------------
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "onnx")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "cpu"))

    # --- Vector store ----------------------------------------------------
    chroma_persist_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _env("CHROMA_PERSIST_DIR", "data/chroma_db")
    )
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "pdf_rag"))

    # --- Chunking --------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))

    # --- Retrieval -------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.35)
    )
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", False))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    reranker_candidates: int = field(
        default_factory=lambda: _env_int("RERANKER_CANDIDATES", 20)
    )

    # --- OCR -------------------------------------------------------------
    ocr_engine: str = field(default_factory=lambda: _env("OCR_ENGINE", "auto"))
    ocr_languages: str = field(default_factory=lambda: _env("OCR_LANGUAGES", "eng"))
    ocr_min_chars_per_page: int = field(
        default_factory=lambda: _env_int("OCR_MIN_CHARS_PER_PAGE", 40)
    )
    tesseract_cmd: str = field(default_factory=lambda: _env("TESSERACT_CMD"))

    # --- Uploads ---------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 50))

    # --- App -------------------------------------------------------------
    app_title: str = field(default_factory=lambda: _env("APP_TITLE", "PDF RAG Chatbot"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")

    # -- helpers ---------------------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """True when enough is set to call an OpenAI-compatible endpoint."""
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with non-None fields replaced (validates the result)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)  # type: ignore[arg-type]


def get_settings(reload: bool = False) -> Settings:
    """Load settings and initialise logging. If reload=True, re-reads .env."""
    if reload:
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    s = Settings()
    setup_logging(s.log_level)
    return s


def save_env_overrides(updates: dict[str, str]) -> None:
    """Persist key-value overrides into .env file while preserving existing lines."""
    env_path = PROJECT_ROOT / ".env"
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text("utf-8").splitlines()

    new_lines: list[str] = []
    seen: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            k = k.strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", "utf-8")
    load_dotenv(env_path, override=True)


# The chunking configurations compared by the benchmark command.
CHUNKING_EXPERIMENTS: tuple[dict[str, int], ...] = (
    {"chunk_size": 256, "chunk_overlap": 32},
    {"chunk_size": 512, "chunk_overlap": 64},
    {"chunk_size": 1024, "chunk_overlap": 128},
)