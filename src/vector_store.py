"""Persistent ChromaDB vector store with document-level lifecycle management.

We talk to Chroma directly rather than through a framework wrapper: the queries
needed here are simple, the metadata handling is fully under our control, and it
keeps the vector layer swappable and independently testable.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .embeddings import EmbeddingProvider, get_cached_provider
from .models import Chunk, RetrievedChunk

log = logging.getLogger(__name__)

# Batches larger than this cause Chroma to warn and slow down on some versions.
UPSERT_BATCH = 256


@dataclass
class DocumentInfo:
    """One indexed document, for the UI's document list."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    ocr_pages: int = 0

    @property
    def short_id(self) -> str:
        return self.doc_id[:12]


class VectorStore:
    """Persistent Chroma collection with idempotent document ingestion."""

    def __init__(
        self,
        settings,
        embedding_provider: Optional[EmbeddingProvider] = None,
        *,
        persist_dir: Optional[Path] = None,
        collection: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection or settings.chroma_collection
        self._provider = embedding_provider
        self._lock = threading.Lock()

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._make_client()
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "VectorStore ready at %s (collection=%s, count=%d)",
            self.persist_dir, self.collection_name, self.count(),
        )

    # -- construction ----------------------------------------------------
    def _make_client(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_cached_provider(self.settings)
        return self._provider

    # -- writes ----------------------------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Insert or update chunks. Returns the number written."""
        if not chunks:
            return 0

        ids, docs, metas = [], [], []
        for c in chunks:
            if not c.text.strip():
                continue  # never embed empty strings
            ids.append(c.chunk_id)
            docs.append(c.text)
            metas.append(c.metadata())

        if not ids:
            return 0

        vectors = self.provider.embed(docs)
        written = 0
        with self._lock:
            for start in range(0, len(ids), UPSERT_BATCH):
                sl = slice(start, start + UPSERT_BATCH)
                self._col.upsert(
                    ids=ids[sl],
                    documents=docs[sl],
                    metadatas=metas[sl],
                    embeddings=vectors[sl],
                )
                written += len(ids[sl])
        return written

    def delete_document(self, doc_id: str) -> int:
        """Remove every vector for a document. Returns rows deleted."""
        if not doc_id:
            return 0
        with self._lock:
            existing = self._col.get(where={"doc_id": doc_id}, include=[])
            if not existing or not existing.get("ids"):
                return 0
            self._col.delete(ids=existing["ids"])
            return len(existing["ids"])

    def clear(self) -> None:
        with self._lock:
            self._client.delete_collection(self.collection_name)
            self._col = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    # -- reads -----------------------------------------------------------
    def count(self) -> int:
        try:
            return int(self._col.count())
        except Exception as exc:
            log.warning("count() failed: %s", exc)
            return 0

    def has_document(self, doc_id: str) -> bool:
        """True if the document already has vectors (drives idempotency)."""
        if not doc_id:
            return False
        try:
            got = self._col.get(where={"doc_id": doc_id}, limit=1, include=[])
            return bool(got.get("ids"))
        except Exception as exc:
            log.warning("has_document failed: %s", exc)
            return False

    def list_documents(self) -> list[DocumentInfo]:
        """Every indexed document with page/chunk counts, for the UI."""
        try:
            data = self._col.get(include=["metadatas"])
        except Exception as exc:
            log.warning("list_documents failed: %s", exc)
            return []

        agg: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            if not meta:
                continue
            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue
            rec = agg.setdefault(
                doc_id,
                {"filename": str(meta.get("filename", doc_id)), "chunks": 0,
                 "pages": set(), "ocr": 0},
            )
            rec["chunks"] += 1
            page = meta.get("page")
            if isinstance(page, int):
                rec["pages"].add(page)
            if meta.get("used_ocr"):
                rec["ocr"] += 1

        out = [
            DocumentInfo(
                doc_id=doc_id,
                filename=rec["filename"],
                page_count=len(rec["pages"]),
                chunk_count=rec["chunks"],
                ocr_pages=rec["ocr"],
            )
            for doc_id, rec in agg.items()
        ]
        return sorted(out, key=lambda d: d.filename.lower())

    def get_page_text(self, doc_id: str, page: int) -> list[Chunk]:
        """All chunks from one page -- used to show the user cited excerpts."""
        try:
            data = self._col.get(where={"$and": [{"doc_id": doc_id}, {"page": page}]})
        except Exception as exc:
            log.warning("get_page_text failed: %s", exc)
            return []
        return _chunks_from_payload(data)

    def query(
        self,
        text: str,
        n_results: int = 5,
        *,
        where: Optional[dict] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> list[RetrievedChunk]:
        """Similarity search. Returns chunks ordered best-first.

        Chroma's cosine distance is ``1 - cosine_similarity``, so similarity is
        ``1 - distance`` and lands in ``[0, 1]`` for non-negative vectors.
        """
        if not text.strip() or self.count() == 0:
            return []

        n = max(1, min(n_results, self.count()))
        where_clause = where
        if doc_ids:
            doc_clause: dict[str, Any] = {"doc_id": {"$in": list(doc_ids)}}
            where_clause = {"$and": [doc_clause, where]} if where else doc_clause

        try:
            res = self._col.query(
                query_texts=[text],
                n_results=n,
                where=where_clause or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("Vector search failed: %s", exc)
            return []

        out: list[RetrievedChunk] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        for i, doc_text in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            dist = float(dists[i]) if i < len(dists) else 1.0
            chunk = Chunk(
                chunk_id=ids[i] if i < len(ids) else f"chunk-{i}",
                doc_id=str(meta.get("doc_id", "")),
                filename=str(meta.get("filename", "")),
                page=int(meta.get("page", 0) or 0),
                text=doc_text,
                heading=meta.get("heading") or None,
                used_ocr=bool(meta.get("used_ocr", False)),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
                token_count=int(meta.get("token_count", 0) or 0),
            )
            out.append(RetrievedChunk(chunk=chunk, similarity=max(0.0, 1.0 - dist)))
        return out

    def query_by_embedding(
        self, embedding: Sequence[float], n_results: int = 5
    ) -> list[RetrievedChunk]:
        """Search with a precomputed vector (used by the benchmark for reranking)."""
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[list(embedding)], n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[RetrievedChunk] = []
        for i, doc_text in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            dist = float(dists[i]) if i < len(dists) else 1.0
            chunk = Chunk.from_storage(f"c{i}", doc_text, meta)
            out.append(RetrievedChunk(chunk=chunk, similarity=max(0.0, 1.0 - dist)))
        return out


def _chunks_from_payload(data: dict) -> list[Chunk]:
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    out: list[Chunk] = []
    for i, doc_id in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) and metas[i] else {}
        out.append(Chunk.from_storage(doc_id, text, meta))
    return out