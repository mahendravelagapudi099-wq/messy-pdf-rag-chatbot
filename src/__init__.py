"""PDF RAG chatbot -- ingestion, retrieval, grounded generation and evaluation."""

from .config import Settings, get_settings, save_env_overrides

__all__ = ["Settings", "get_settings", "save_env_overrides"]
__version__ = "1.0.0"