"""PDF RAG chatbot -- ingestion, retrieval, grounded generation and evaluation."""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
__version__ = "1.0.0"