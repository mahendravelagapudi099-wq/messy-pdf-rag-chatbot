"""Streamlit entry point for the PDF RAG chatbot.

All application logic lives in ``src/``; this file only renders state and wires
up user interactions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.citations import overlap_score, similarity_of  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.evaluation import coverage, format_comparison_table, load_dataset  # noqa: E402
from src.models import Answer, RetrievedChunk  # noqa: E402
from src.rag import RAGEngine  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.WARNING)

PAGE_ICON = "📄"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("settings", None)
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("engine_key", None)
    st.session_state.setdefault("last_retrieval", None)
    st.session_state.setdefault("bench_results", None)
    st.session_state.setdefault("ingest_log", [])


@st.cache_resource(show_spinner=False)
def get_engine(persist_dir: str, settings_kwargs: tuple) -> RAGEngine:
    """Build (and cache) the engine. ``cache_resource`` keeps the model warm."""
    from dataclasses import replace

    base = get_settings()
    settings = replace(base, **dict(settings_kwargs))
    return RAGEngine(settings, persist_dir=Path(persist_dir))


def current_engine() -> RAGEngine:
    """Return an engine matching the current settings, rebuilding if needed."""
    settings = st.session_state["settings"]
    base = get_settings()
    from dataclasses import replace

    merged = replace(
        base,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
        use_reranker=settings.use_reranker,
        reranker_model=settings.reranker_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )
    key = (
        str(merged.chroma_persist_dir),
        merged.chunk_size,
        merged.chunk_overlap,
        merged.top_k,
        merged.similarity_threshold,
        merged.use_reranker,
        merged.embedding_provider,
        merged.embedding_model,
    )
    if st.session_state["engine"] is None or st.session_state["engine_key"] != key:
        st.session_state["engine"] = get_engine(
            str(merged.chroma_persist_dir),
            (
                ("chunk_size", merged.chunk_size),
                ("chunk_overlap", merged.chunk_overlap),
                ("top_k", merged.top_k),
                ("similarity_threshold", merged.similarity_threshold),
                ("use_reranker", merged.use_reranker),
                ("embedding_provider", merged.embedding_provider),
                ("embedding_model", merged.embedding_model),
            ),
        )
        st.session_state["engine_key"] = key
    return st.session_state["engine"]


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def sidebar() -> None:
    settings = st.session_state["settings"]
    base = get_settings()

    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Retrieval")
        top_k = st.slider(
            "Top-k passages", 1, 20, settings.top_k,
            help="How many chunks to retrieve and pass to the model.",
        )
        threshold = st.slider(
            "Similarity threshold", 0.0, 1.0, settings.similarity_threshold, 0.01,
            help=(
                "Passages scoring below this cosine similarity are discarded. "
                "Raising it makes the system abstain more often."
            ),
        )

        st.subheader("Chunking")
        chunk_size = st.number_input(
            "Chunk size (tokens)", 128, 2048, settings.chunk_size, step=64
        )
        overlap = st.number_input(
            "Chunk overlap (tokens)", 0, 512, settings.chunk_overlap, step=16
        )
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
            overlap = max(0, chunk_size // 4)

        st.subheader("Models")
        provider = st.selectbox(
            "Embedding provider",
            ["onnx", "sentence-transformers"],
            index=0 if settings.embedding_provider == "onnx" else 1,
            help=(
                "`onnx` needs no torch and downloads ~79 MB. "
                "`sentence-transformers` is more accurate but requires the optional extras."
            ),
        )
        use_reranker = st.checkbox(
            "Enable cross-encoder reranker",
            value=settings.use_reranker,
            help="Reorders the top candidates with a cross-encoder. Slower; needs extras.",
        )

        st.subheader("Uploads")
        max_mb = st.number_input("Max upload size (MB)", 1, 500, settings.max_upload_mb)

        st.divider()
        st.caption(f"LLM: {base.llm_model if base.llm_configured else 'not configured'}")
        st.caption(f"Index: {base.chroma_persist_dir}")

        if st.button("Apply settings", use_container_width=True, type="primary"):
            st.session_state["settings"] = base.with_overrides(
                top_k=int(top_k),
                similarity_threshold=float(threshold),
                chunk_size=int(chunk_size),
                chunk_overlap=int(overlap),
                embedding_provider=provider,
                use_reranker=use_reranker,
                max_upload_mb=int(max_mb),
            )
            st.rerun()

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_retrieval"] = None
            st.rerun()


def render_document_panel(engine: RAGEngine, max_mb: int) -> None:
    st.subheader("1. Add documents")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {max_mb} MB per file.",
    )

    if uploaded and st.button("Index uploaded PDFs", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Preparing...")
        for i, f in enumerate(uploaded):
            data = f.getvalue()
            report = engine.add_upload(
                f.name, data,
                progress=lambda msg, frac: bar.progress(frac, text=f"{f.name}: {msg}"),
            )
            st.session_state["ingest_log"].insert(0, report)
            icon = {"indexed": "✅", "skipped": "↩️", "failed": "❌"}.get(report.status, "•")
            if report.status == "indexed":
                st.success(
                    f"{icon} **{report.filename}** — {report.chunk_count} chunks, "
                    f"{report.page_count} pages"
                    + (f", OCR on {len(report.ocr_pages)} page(s)" if report.ocr_pages else "")
                )
            elif report.status == "skipped":
                st.info(f"{icon} **{report.filename}** — {report.message}")
            else:
                st.error(f"{icon} **{report.filename}** — {report.message}")
            for w in report.warnings[:2]:
                st.warning(f"   ↳ {w}")
        bar.empty()

    st.subheader("2. Indexed documents")
    docs = engine.list_documents()
    if not docs:
        st.info("No documents indexed yet. Upload a PDF to get started.")
        return

    for d in docs:
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{d.filename}**")
                meta = f"{d.page_count} pages · {d.chunk_count} chunks"
                if d.ocr_pages:
                    meta += f" · {d.ocr_pages} OCR"
                st.caption(meta)
            with col2:
                if st.button("Remove", key=f"rm_{d.doc_id}", use_container_width=True):
                    removed = engine.remove_document(d.doc_id)
                    st.toast(f"Removed {removed} chunks from {d.filename}")
                    st.rerun()


def render_citations(engine: RAGEngine, answer: Answer) -> None:
    if not answer.citations:
        return
    st.markdown("**Sources**")
    for cit in answer.citations:
        with st.expander(
            f"{cit.marker} {cit.filename} — page {cit.page}  ·  "
            f"match {cit.similarity:.2f}" + ("" if cit.grounded else "  ⚠️ weakly supported")
        ):
            st.caption(f"chunk `{cit.chunk_id}`")
            st.text(cit.excerpt)
            if engine is not None and cit.page:
                with st.expander("All text stored for this page"):
                    for ch in engine.page_excerpts(cit.doc_id, cit.page):
                        st.text(f"[{ch.chunk_id}] {ch.text[:400]}")


def render_retrieved(result) -> None:
    if result is None or not result.chunks:
        return
    with st.expander(f"Retrieved passages ({len(result.chunks)}) · {result.latency_s * 1000:.0f} ms"):
        if result.below_threshold:
            st.caption(
                f"{result.below_threshold} candidate(s) discarded as below the "
                f"similarity threshold."
            )
        for rc in result.chunks:
            score = similarity_of(rc)
            st.markdown(
                f"**{rc.chunk.filename} · p.{rc.chunk.page}** — "
                f"`{rc.chunk.text[:220].strip()}…`"
            )
            st.progress(min(score, 1.0))
            st.caption(f"similarity {rc.similarity:.3f}")
            if rc.rerank_score is not None:
                st.caption(f"rerank score {rc.rerank_score:.3f}")


def render_answer(answer: Answer) -> None:
    if answer.error:
        st.error(answer.error)
        return
    if answer.degraded_reason:
        st.info(
            f"**Degraded mode** (`{answer.degraded_reason}`) — showing retrieved "
            f"passages instead of a generated answer. Set `LLM_API_KEY` in `.env` to "
            f"enable full grounded generation."
        )
        st.markdown(answer.text)
    elif answer.abstained:
        st.warning(
            f"**Not answered from the documents.** {answer.text}\n\n"
            f"_(reason: `{answer.abstention_reason}`)_"
        )
    else:
        st.markdown(answer.text)


def chat_tab(engine: RAGEngine) -> None:
    status = engine.status()

    if status.chunk_count == 0:
        st.info("Index some documents first using the sidebar, then come back to chat.")
        return

    if not status.llm_configured:
        st.warning(
            "**Retrieval-only mode.** No `LLM_API_KEY` is set, so answers are built by "
            "extracting relevant sentences from the retrieved passages. Set the key in "
            "`.env` to enable full grounded generation."
        )

    st.subheader(f"Indexed: {status.doc_count} document(s), {status.chunk_count} chunks")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                for cit in msg.get("citations", []):
                    st.caption(
                        f"{cit['marker']} {cit['filename']}, p.{cit['page']} "
                        f"(match {cit['similarity']:.2f})"
                    )

    question = st.chat_input("Ask a question about the indexed documents")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and composing…"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1]
            ]
            retrieval = engine.search(question)
            answer = engine.ask(question, history=history)
            st.session_state["last_retrieval"] = retrieval

        render_answer(answer)
        render_retrieved(retrieval)
        render_citations(engine, answer)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer.text,
            "citations": [
                {"marker": c.marker, "filename": c.filename, "page": c.page,
                 "similarity": c.similarity}
                for c in answer.citations
            ],
        }
    )


def settings_tab(engine: RAGEngine) -> None:
    st.subheader("System status")
    status = engine.status()
    rows = [
        ("Documents indexed", str(status.doc_count)),
        ("Chunks stored", str(status.chunk_count)),
        ("Embedding provider", status.embedding_detail),
        ("Language model", status.llm_detail),
        ("OCR", status.ocr_detail),
        ("Vector store", f"{engine.store.persist_dir} (collection {engine.store.collection_name})"),
    ]
    for label, value in rows:
        st.markdown(f"- **{label}:** {value}")

    st.divider()
    st.subheader("Evaluation dataset")
    dataset_path = "data/eval_dataset.jsonl"
    if Path(dataset_path).exists():
        try:
            qs = load_dataset(dataset_path)
            st.code(coverage(qs).summary(), language="text")
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
    else:
        st.info("No dataset at `data/eval_dataset.jsonl`.")

    st.divider()
    st.subheader("Run a chunking benchmark")
    st.caption(
        "Compares 256/32, 512/64 and 1024/128 token chunking. Runs without an API key "
        "using the deterministic extractive answerer."
    )
    mode = st.selectbox("Answer mode", ["extractive", "llm"])
    use_rr = st.checkbox("Also test the reranker", value=False)

    if st.button("Run benchmark", type="primary"):
        if mode == "llm" and not get_settings().llm_configured:
            st.error("`--mode llm` needs `LLM_API_KEY`. Run in extractive mode instead.")
            return
        from scripts.run_eval import run_benchmark_cmd

        with st.spinner("Running benchmark… this ingests the corpus once per configuration…"):
            try:
                results = run_benchmark_cmd(
                    dataset_path=dataset_path,
                    corpus_dir="data/corpus",
                    mode=mode,
                    reranker=use_rr,
                )
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")
                return
        st.session_state["bench_results"] = results

    for result in st.session_state["bench_results"] or []:
        st.markdown(f"**{result.config.label}**")
        st.dataframe([result.metrics.as_dict()], use_container_width=True)


def main() -> None:
    init_state()
    base = get_settings()

    if st.session_state["settings"] is None:
        st.session_state["settings"] = base

    st.set_page_config(page_title=base.app_title, page_icon=PAGE_ICON, layout="wide")
    st.title(f"{PAGE_ICON} {base.app_title}")

    sidebar()

    engine = current_engine()
    max_mb = st.session_state["settings"].max_upload_mb

    left, right = st.columns([1, 2], gap="large")
    with left:
        render_document_panel(engine, max_mb)
    with right:
        tabs = st.tabs(["💬 Chat", "📊 Evaluation", "⚙️ Settings"])
        with tabs[0]:
            chat_tab(engine)
        with tabs[1]:
            settings_tab(engine)
        with tabs[2]:
            st.subheader("Settings")
            st.info("Use the sidebar to change retrieval, chunking and model settings.")

    with st.expander("Recent ingestion log"):
        log = st.session_state["ingest_log"]
        if not log:
            st.caption("Nothing ingested in this session yet.")
        else:
            for r in log:
                st.markdown(
                    f"- **{r.status}** — {r.filename}: {r.message} "
                    f"({r.elapsed_s:.1f}s)"
                )


if __name__ == "__main__":
    main()