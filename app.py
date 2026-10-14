"""Streamlit entry point for the PDF RAG chatbot.

All application logic lives in ``src/``; this file renders state, custom design tokens,
and wires up user interactions with an enhanced Deep Tech Obsidian interface.
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
try:
    from src.config import get_settings, save_env_overrides  # noqa: E402
except ImportError:
    import importlib
    import src.config  # noqa: E402
    importlib.reload(src.config)
    from src.config import get_settings, save_env_overrides  # noqa: E402
from src.evaluation import coverage, format_comparison_table, load_dataset  # noqa: E402
from src.models import Answer, RetrievedChunk  # noqa: E402
from src.rag import RAGEngine  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.WARNING)

PAGE_ICON = "⚡"


# --------------------------------------------------------------------------
# Custom CSS Design System (Deep Tech Obsidian Cockpit)
# Strict adherence to Maestro color rules: uses slate, obsidian, electric cyan & emerald.
# --------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Inject bespoke styling for a modern, high-contrast, obsidian-themed AI workbench."""
    st.markdown(
        """
        <style>
        /* Base typography & variables */
        :root {
            --bg-obsidian: #080d1a;
            --surface-slate: #0f172a;
            --surface-card: #131d31;
            --surface-card-hover: #17243c;
            --border-subtle: rgba(56, 189, 248, 0.15);
            --border-strong: rgba(56, 189, 248, 0.35);
            --accent-cyan: #0ea5e9;
            --accent-cyan-light: #38bdf8;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }

        /* Top header telemetry strip */
        .telemetry-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            padding: 12px 18px;
            margin-bottom: 24px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(19, 29, 49, 0.65) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: space-between;
        }

        .telemetry-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.82rem;
            color: var(--text-secondary);
        }

        .telemetry-item strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .telemetry-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            display: inline-block;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }

        .telemetry-dot.amber {
            background-color: var(--accent-amber);
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
        }

        /* Document Library Cards */
        .doc-card {
            background-color: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            transition: all 0.2s ease-in-out;
        }

        .doc-card:hover {
            border-color: var(--accent-cyan-light);
            background-color: var(--surface-card-hover);
            transform: translateY(-1px);
        }

        .doc-title {
            font-weight: 600;
            font-size: 0.92rem;
            color: var(--text-primary);
            margin-bottom: 6px;
            word-break: break-all;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 9999px;
            background-color: rgba(56, 189, 248, 0.1);
            color: var(--accent-cyan-light);
            border: 1px solid rgba(56, 189, 248, 0.2);
            margin-right: 6px;
            margin-bottom: 4px;
        }

        .badge-pill.ocr {
            background-color: rgba(245, 158, 11, 0.12);
            color: var(--accent-amber);
            border-color: rgba(245, 158, 11, 0.3);
        }

        .badge-pill.emerald {
            background-color: rgba(16, 185, 129, 0.12);
            color: var(--accent-emerald);
            border-color: rgba(16, 185, 129, 0.3);
        }

        /* Citation & Grounding Cards */
        .citation-box {
            background: rgba(19, 29, 49, 0.6);
            border-left: 3px solid var(--accent-cyan);
            border-radius: 0 8px 8px 0;
            padding: 10px 14px;
            margin: 8px 0;
            font-size: 0.88rem;
        }

        .citation-box.weak {
            border-left-color: var(--accent-amber);
            background: rgba(245, 158, 11, 0.05);
        }

        /* Degraded / Abstention alert cards */
        .alert-card {
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 16px;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .alert-card.warning {
            background-color: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fef3c7;
        }

        .alert-card.info {
            background-color: rgba(14, 165, 233, 0.1);
            border: 1px solid rgba(14, 165, 233, 0.35);
            color: #e0f2fe;
        }

        /* Quick start inquiry chips */
        .starter-section {
            margin: 18px 0 24px 0;
            padding: 18px;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 12px;
            border: 1px dashed var(--border-subtle);
        }

        /* Micro progress similarity bar */
        .meter-container {
            width: 100%;
            height: 6px;
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
            margin: 6px 0;
        }

        .meter-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald));
        }

        /* Enhance tab bar visuals */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 16px;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(14, 165, 233, 0.12) !important;
            color: var(--accent-cyan-light) !important;
            border-bottom: 2px solid var(--accent-cyan) !important;
        }

        /* Refined chat input anchor */
        .stChatInputContainer {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("llm_test_result", None)


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
        # Forward LLM credentials from session — these are not cached in the engine
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
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
        merged.llm_api_key,
        merged.llm_base_url,
        merged.llm_model,
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
                ("llm_api_key", merged.llm_api_key),
                ("llm_base_url", merged.llm_base_url),
                ("llm_model", merged.llm_model),
                ("llm_temperature", merged.llm_temperature),
            ),
        )
        st.session_state["engine_key"] = key
    return st.session_state["engine"]


# --------------------------------------------------------------------------
# UI Components & Helpers
# --------------------------------------------------------------------------


def render_telemetry_bar(engine: RAGEngine) -> None:
    """Render top horizontal telemetry status ribbon."""
    status = engine.status()
    dot_class = "telemetry-dot" if status.llm_configured else "telemetry-dot amber"
    llm_label = status.llm_detail if status.llm_configured else "Extractive Mode (No API Key)"
    ocr_label = status.ocr_detail

    st.markdown(
        f"""
        <div class="telemetry-bar">
            <div class="telemetry-item">
                <span class="{dot_class}"></span>
                <span>Corpus: <strong>{status.doc_count} docs</strong> · <strong>{status.chunk_count} passages</strong></span>
            </div>
            <div class="telemetry-item">
                <span>⚡ Embeddings: <strong>{status.embedding_detail}</strong></span>
            </div>
            <div class="telemetry-item">
                <span>🧠 Engine: <strong>{llm_label}</strong></span>
            </div>
            <div class="telemetry-item">
                <span>👁️ OCR: <strong>{ocr_label}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _test_llm_connection(api_key: str, base_url: str, model: str) -> tuple[bool, str]:
    """Quick connection probe — returns (success, message)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=8.0, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        reply = (resp.choices[0].message.content or "").strip()
        return True, f"Connected · model replied: '{reply[:60]}'"
    except Exception as exc:
        return False, str(exc)[:200]


def sidebar() -> None:
    settings = st.session_state["settings"]
    base = get_settings()

    with st.sidebar:
        st.markdown("### ⚙️ Engine Control Panel")
        st.caption("Fine-tune retrieval thresholds, chunk geometries, and neural models.")

        # ── LLM / API Key Configuration ──────────────────────────────────
        llm_configured = bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model)
        llm_status_icon = "🟢" if llm_configured else "🟡"
        with st.expander(f"{llm_status_icon} LLM API Configuration", expanded=not llm_configured):
            st.caption(
                "Configure your OpenAI-compatible API endpoint. Changes are saved to `.env` "
                "and applied immediately — no restart needed."
            )

            llm_api_key = st.text_input(
                "API Key",
                value=settings.llm_api_key or "",
                type="password",
                placeholder="sk-…",
                help="Your OpenAI-compatible API key. Stored in `.env`.",
            )
            llm_base_url = st.text_input(
                "Base URL",
                value=settings.llm_base_url or "https://openrouter.ai/api/v1",
                placeholder="https://openrouter.ai/api/v1",
                help="The base URL for the OpenAI-compatible endpoint. Must end with `/v1`.",
            )
            llm_model = st.text_input(
                "Model ID",
                value=settings.llm_model or "openai/gpt-4o-mini",
                placeholder="openai/gpt-4o-mini",
                help="The model identifier string understood by the endpoint.",
            )
            llm_temperature = st.slider(
                "Temperature",
                0.0, 1.0,
                float(settings.llm_temperature),
                0.05,
                help="Sampling temperature. Lower = more deterministic & factual.",
            )

            c_test, c_save = st.columns(2)
            with c_test:
                if st.button("🔌 Test", use_container_width=True, help="Send a quick ping to the endpoint"):
                    with st.spinner("Testing connection…"):
                        ok, msg = _test_llm_connection(llm_api_key, llm_base_url, llm_model)
                    st.session_state["llm_test_result"] = (ok, msg)

            with c_save:
                if st.button("💾 Save to .env", use_container_width=True, type="primary",
                             help="Write these values to .env and activate immediately"):
                    save_env_overrides({
                        "LLM_API_KEY": llm_api_key,
                        "LLM_BASE_URL": llm_base_url,
                        "LLM_MODEL": llm_model,
                        "LLM_TEMPERATURE": str(llm_temperature),
                    })
                    fresh = get_settings(reload=True)
                    st.session_state["settings"] = fresh.with_overrides(
                        top_k=settings.top_k,
                        similarity_threshold=settings.similarity_threshold,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        embedding_provider=settings.embedding_provider,
                        use_reranker=settings.use_reranker,
                        max_upload_mb=settings.max_upload_mb,
                    )
                    st.session_state["engine"] = None  # force rebuild with new LLM
                    st.session_state["engine_key"] = None
                    st.session_state["llm_test_result"] = None
                    st.toast("✅ LLM credentials saved and engine reloaded!")
                    st.rerun()

            # Show test result if present
            test_res = st.session_state.get("llm_test_result")
            if test_res is not None:
                ok, msg = test_res
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ Connection failed: {msg}")

        # ── Retrieval Parameters ──────────────────────────────────────────
        with st.expander("🎯 Retrieval Parameters", expanded=True):
            top_k = st.slider(
                "Top-k Candidates",
                1,
                20,
                settings.top_k,
                help="Number of chunks retrieved from Chroma and presented to the answerer.",
            )
            threshold = st.slider(
                "Similarity Threshold",
                0.0,
                1.0,
                settings.similarity_threshold,
                0.01,
                help=(
                    "Passages scoring below this cosine similarity threshold are purged. "
                    "Raising it increases answer abstention for ungrounded queries."
                ),
            )

        with st.expander("✂️ Chunking Architecture", expanded=False):
            chunk_size = st.number_input(
                "Chunk Size (tokens)",
                128,
                2048,
                settings.chunk_size,
                step=64,
                help="Target token length for each document fragment.",
            )
            overlap = st.number_input(
                "Chunk Overlap (tokens)",
                0,
                512,
                settings.chunk_overlap,
                step=16,
                help="Sliding window token overlap between consecutive chunks.",
            )
            if overlap >= chunk_size:
                st.error("Overlap must be strictly smaller than chunk size.")
                overlap = max(0, chunk_size // 4)

        with st.expander("🤖 Embedding Models", expanded=False):
            provider = st.selectbox(
                "Embedding Engine",
                ["onnx", "sentence-transformers"],
                index=0 if settings.embedding_provider == "onnx" else 1,
                help=(
                    "`onnx`: High-speed zero-dependency model (~79MB ONNX runtime).\n"
                    "`sentence-transformers`: High-precision embeddings requiring PyTorch extras."
                ),
            )
            use_reranker = st.toggle(
                "Cross-Encoder Reranker",
                value=settings.use_reranker,
                help="Re-scores top semantic candidates with a heavy cross-encoder for peak precision.",
            )

        with st.expander("📦 Upload Constraints", expanded=False):
            max_mb = st.number_input(
                "Max Document Size (MB)",
                1,
                500,
                settings.max_upload_mb,
                help="Maximum allowed PDF file size for ingest processing.",
            )

        st.divider()

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.caption(f"**Index:** `{Path(base.chroma_persist_dir).name}`")
        with col_s2:
            llm_ok = bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model)
            st.caption(f"**LLM:** {'✅ Active' if llm_ok else '⚠️ Extractive'}")

        if st.button("💾 Apply Retrieval Settings", use_container_width=True, type="primary"):
            st.session_state["settings"] = st.session_state["settings"].with_overrides(
                top_k=int(top_k),
                similarity_threshold=float(threshold),
                chunk_size=int(chunk_size),
                chunk_overlap=int(overlap),
                embedding_provider=provider,
                use_reranker=use_reranker,
                max_upload_mb=int(max_mb),
            )
            st.toast("Retrieval configuration updated!")
            st.rerun()

        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_retrieval"] = None
            st.toast("Chat history cleared.")
            st.rerun()


def render_document_panel(engine: RAGEngine, max_mb: int) -> None:
    st.markdown("### 📥 Document Intake")
    st.caption("Upload native, messy, or scanned PDFs for OCR parsing and vectorization.")

    uploaded = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"File limit: up to {max_mb} MB per document.",
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button(
            f"⚡ Vectorize {len(uploaded)} Uploaded File(s)",
            type="primary",
            use_container_width=True,
        ):
            progress_bar = st.progress(0.0, text="Initializing ingestion pipeline...")
            success_count = 0
            for i, f in enumerate(uploaded):
                data = f.getvalue()
                report = engine.add_upload(
                    f.name,
                    data,
                    progress=lambda msg, frac: progress_bar.progress(frac, text=f"{f.name}: {msg}"),
                )
                st.session_state["ingest_log"].insert(0, report)
                if report.status == "indexed":
                    success_count += 1
                    ocr_note = f" · OCR on {len(report.ocr_pages)}p" if report.ocr_pages else ""
                    st.success(
                        f"✅ **{report.filename}** — {report.chunk_count} chunks, {report.page_count} pages{ocr_note}"
                    )
                elif report.status == "skipped":
                    st.info(f"↩️ **{report.filename}** — {report.message}")
                else:
                    st.error(f"❌ **{report.filename}** — {report.message}")

                for w in report.warnings[:2]:
                    st.warning(f"   ↳ {w}")

            progress_bar.empty()
            if success_count > 0:
                st.toast(f"Successfully vectorized {success_count} document(s)!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Indexed Corpus")
    docs = engine.list_documents()

    if not docs:
        st.markdown(
            """
            <div style="padding: 24px 16px; text-align: center; background: rgba(15, 23, 42, 0.4); border-radius: 10px; border: 1px dashed rgba(56, 189, 248, 0.2);">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">📂</div>
                <div style="font-weight: 500; color: #f8fafc; font-size: 0.95rem;">No documents in corpus</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Drag & drop PDFs above to start asking questions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"Currently managing **{len(docs)}** vectorized document(s):")

    def _render_doc_card(doc):
        with st.container(border=True):
            col_info, col_act = st.columns([4, 1])
            with col_info:
                st.markdown(f"**📄 {doc.filename}**")
                ocr_badge = f'<span class="badge-pill ocr">OCR: {doc.ocr_pages}p</span>' if doc.ocr_pages else ""
                st.markdown(
                    f"""
                    <div>
                        <span class="badge-pill">{doc.page_count} pages</span>
                        <span class="badge-pill emerald">{doc.chunk_count} chunks</span>
                        {ocr_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_act:
                if st.button("🗑️", key=f"rm_{doc.doc_id}", help=f"Purge {doc.filename} from index", use_container_width=True):
                    removed = engine.remove_document(doc.doc_id)
                    st.toast(f"Purged {removed} chunks from {doc.filename}")
                    st.rerun()

    # Show top 3 documents by default to keep page height compact
    preview_docs = docs[:3]
    remaining_docs = docs[3:]

    for d in preview_docs:
        _render_doc_card(d)

    if remaining_docs:
        with st.expander(f"📂 View remaining {len(remaining_docs)} document(s) in corpus...", expanded=False):
            for d in remaining_docs:
                _render_doc_card(d)


def render_citations(engine: RAGEngine, answer: Answer) -> None:
    """Render structured source grounding cards with confidence indicators."""
    if not answer.citations:
        return

    st.markdown("#### 🎯 Grounded Citations")
    for cit in answer.citations:
        grounded_status = "✅ Verified Grounding" if cit.grounded else "⚠️ Weakly Supported"
        badge_class = "emerald" if cit.grounded else "ocr"
        expander_title = (
            f"{cit.marker} {cit.filename} (p. {cit.page}) — Match: {cit.similarity:.2f} · {grounded_status}"
        )

        with st.expander(expander_title, expanded=False):
            st.markdown(
                f"""
                <div class="citation-box {'weak' if not cit.grounded else ''}">
                    <div style="margin-bottom: 6px;">
                        <span class="badge-pill {badge_class}">{grounded_status}</span>
                        <span class="badge-pill">Chunk ID: {cit.chunk_id}</span>
                        <span class="badge-pill">Cosine Match: {cit.similarity:.3f}</span>
                    </div>
                    <div style="font-style: italic; color: #e2e8f0; margin-top: 6px;">
                        "{cit.excerpt}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if engine is not None and cit.page:
                with st.expander("🔍 Inspect Full Page Context in Corpus"):
                    for ch in engine.page_excerpts(cit.doc_id, cit.page):
                        st.caption(f"Chunk `{ch.chunk_id}`:")
                        st.text(ch.text[:400] + ("…" if len(ch.text) > 400 else ""))


def render_retrieved(result) -> None:
    """Render retrieved semantic passages with similarity meters."""
    if result is None or not result.chunks:
        return

    latency_ms = result.latency_s * 1000
    with st.expander(f"⚡ Retrieved Passages ({len(result.chunks)}) · Ingestion Latency {latency_ms:.0f} ms"):
        if result.below_threshold:
            st.caption(
                f"ℹ️ Discarded **{result.below_threshold} candidate(s)** scoring below the similarity threshold."
            )

        for rc in result.chunks:
            score = similarity_of(rc)
            pct = int(min(score, 1.0) * 100)
            rerank_label = f" · Rerank Score: **{rc.rerank_score:.3f}**" if rc.rerank_score is not None else ""

            st.markdown(
                f"""
                <div style="margin-top: 8px; margin-bottom: 4px; font-size: 0.86rem; color: #f8fafc;">
                    <strong>{rc.chunk.filename}</strong> (Page {rc.chunk.page}) — Similarity: <strong>{rc.similarity:.3f}</strong>{rerank_label}
                </div>
                <div class="meter-container">
                    <div class="meter-fill" style="width: {pct}%;"></div>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-family: ui-monospace, monospace; background: rgba(0,0,0,0.25); padding: 6px 10px; border-radius: 6px; margin-bottom: 12px;">
                    {rc.chunk.text[:240].strip()}…
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_answer(answer: Answer) -> None:
    """Render answer text with styled state handling for errors, abstentions, and degraded modes."""
    if answer.error:
        st.error(f"❌ **Execution Error:** {answer.error}")
        return

    if answer.degraded_reason:
        st.markdown(
            f"""
            <div class="alert-card info">
                <strong>⚡ Extractive Mode Active</strong> (<em>{answer.degraded_reason}</em>)<br/>
                No external LLM API key detected in <code>.env</code>. The system synthesized this response
                deterministically by ranking and extracting passages directly from your PDF vector index.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(answer.text)
    elif answer.abstained:
        st.markdown(
            f"""
            <div class="alert-card warning">
                <strong>⚠️ Grounded Abstention:</strong> {answer.text}<br/>
                <span style="font-size: 0.8rem; opacity: 0.85;">(Diagnostic reason: <code>{answer.abstention_reason}</code>)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(answer.text)


def chat_tab(engine: RAGEngine) -> None:
    status = engine.status()

    if status.chunk_count == 0:
        st.markdown(
            """
            <div style="padding: 36px 20px; text-align: center; background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px dashed rgba(56, 189, 248, 0.25); margin-top: 12px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">📑</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">Your Knowledge Corpus is Empty</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; max-width: 480px; margin: 0 auto;">
                    Use the <strong>Document Intake</strong> panel on the left to upload PDF documents.
                    Once indexed, you can ask questions with full citations and grounding.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not status.llm_configured:
        st.markdown(
            """
            <div class="alert-card info" style="margin-bottom: 16px;">
                <strong>💡 Retrieval-Only Mode:</strong> No <code>LLM_API_KEY</code> is configured in <code>.env</code>.
                Answers are extracted directly from top matching vector chunks with verified citations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Empty chat welcome & quick starter prompts
    if not st.session_state["messages"]:
        st.markdown(
            """
            <div class="starter-section">
                <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin-bottom: 6px;">
                    💡 Suggested Inquiries for Your Corpus:
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 12px;">
                    Click a prompt below to execute an immediate grounded search across your indexed documents:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("📌 Summarize key conclusions & takeaways", use_container_width=True):
                st.session_state["pending_prompt"] = "Summarize the primary conclusions, findings, and key takeaways."
                st.rerun()
            if st.button("🔬 Extract technical methodologies & tools", use_container_width=True):
                st.session_state["pending_prompt"] = "What specific methodologies, architectures, or tools are detailed?"
                st.rerun()
        with p_col2:
            if st.button("📊 Extract numerical metrics & statistics", use_container_width=True):
                st.session_state["pending_prompt"] = "List all key quantitative metrics, benchmarks, and statistical results."
                st.rerun()
            if st.button("⚠️ Highlight constraints, risks & limitations", use_container_width=True):
                st.session_state["pending_prompt"] = "What limitations, assumptions, risks, or caveats are highlighted?"
                st.rerun()

    # Chat history rendering
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📚 Sources & Citations", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"- **{cit['marker']}** `{cit['filename']}` (page {cit['page']}) · *similarity {cit['similarity']:.2f}*"
                        )

    # Handle input or quick starter trigger
    suggested = st.session_state.pop("pending_prompt", None)
    question = st.chat_input("Ask a grounded question about your indexed documents...") or suggested

    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector index and evaluating grounded passages…"):
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
                {
                    "marker": c.marker,
                    "filename": c.filename,
                    "page": c.page,
                    "similarity": c.similarity,
                }
                for c in answer.citations
            ],
        }
    )


def settings_tab(engine: RAGEngine) -> None:
    st.markdown("### 📊 System Diagnostics & Benchmarks")
    status = engine.status()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Ingested", status.doc_count)
    with col2:
        st.metric("Vector Chunks", status.chunk_count)
    with col3:
        st.metric("Active Embedding", status.embedding_detail)

    with st.container(border=True):
        st.markdown("#### 🛠️ Runtime Infrastructure Details")
        infra_items = [
            ("Vector Store Location", str(engine.store.persist_dir)),
            ("Collection Identifier", str(engine.store.collection_name)),
            ("Language Model Endpoint", status.llm_detail),
            ("Optical Character Recognition", status.ocr_detail),
        ]
        for label, val in infra_items:
            st.markdown(f"- **{label}:** `{val}`")

    st.divider()
    st.markdown("### 📈 Evaluation Dataset Status")
    dataset_path = "data/eval_dataset.jsonl"
    if Path(dataset_path).exists():
        try:
            qs = load_dataset(dataset_path)
            st.code(coverage(qs).summary(), language="text")
        except Exception as exc:
            st.error(f"Could not read evaluation dataset: {exc}")
    else:
        st.info("No evaluation ground truth dataset found at `data/eval_dataset.jsonl`.")

    st.divider()
    st.markdown("### ⏱️ Chunking Strategy Benchmark")
    st.caption(
        "Runs deterministic evaluation comparing 256/32, 512/64, and 1024/128 token splits across the evaluation corpus."
    )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        mode = st.selectbox("Benchmark Synthesis Mode", ["extractive", "llm"])
    with b_col2:
        use_rr = st.checkbox("Include Cross-Encoder Reranker", value=False)

    if st.button("🚀 Execute Benchmark Suite", type="primary"):
        if mode == "llm" and not get_settings().llm_configured:
            st.error("`--mode llm` requires `LLM_API_KEY`. Execute in extractive mode instead.")
            return
        from scripts.run_eval import run_benchmark_cmd

        with st.spinner("Executing benchmark across configurations… this ingests the corpus for each configuration…"):
            try:
                results = run_benchmark_cmd(
                    dataset_path=dataset_path,
                    corpus_dir="data/corpus",
                    mode=mode,
                    reranker=use_rr,
                )
            except Exception as exc:
                st.error(f"Benchmark execution failed: {exc}")
                return
        st.session_state["bench_results"] = results
        st.toast("Benchmark completed!")

    for result in st.session_state["bench_results"] or []:
        st.markdown(f"**Configuration: `{result.config.label}`**")
        st.dataframe([result.metrics.as_dict()], use_container_width=True)


def main() -> None:
    init_state()
    base = get_settings()

    if st.session_state["settings"] is None:
        st.session_state["settings"] = base

    st.set_page_config(
        page_title=f"{base.app_title} · Research Cockpit",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    sidebar()

    engine = current_engine()
    max_mb = st.session_state["settings"].max_upload_mb

    # App Title & Header
    st.title(f"{PAGE_ICON} {base.app_title}")
    render_telemetry_bar(engine)

    left, right = st.columns([1, 2], gap="large")
    with left:
        render_document_panel(engine, max_mb)
    with right:
        tabs = st.tabs(["💬 Grounded Chat", "📊 Evaluation & Benchmarks", "ℹ️ Architecture"])
        with tabs[0]:
            chat_tab(engine)
        with tabs[1]:
            settings_tab(engine)
        with tabs[2]:
            st.markdown("### 🏛️ PDF RAG System Architecture")
            st.markdown(
                """
                - **Ingestion Pipeline**: Native PDF text extraction with PyMuPDF; automatic OCR fallback for scanned/degraded pages with Tesseract.
                - **Chunking Engine**: Configurable token-window chunking with sliding overlap and header/footer stripping.
                - **Vector Store**: Local ChromaDB persistence with cosine similarity distance indexing.
                - **Neural Reranking**: Optional cross-encoder reranking stage for high-precision query candidate reordering.
                - **Grounded Verification**: Overlap citation analysis and groundedness scoring to prevent hallucinations.
                """
            )

    with st.expander("📋 Session Ingestion Activity Log"):
        log = st.session_state["ingest_log"]
        if not log:
            st.caption("No ingest activity recorded in this session yet.")
        else:
            for r in log:
                icon = {"indexed": "✅", "skipped": "↩️", "failed": "❌"}.get(r.status, "•")
                st.markdown(
                    f"- {icon} **{r.filename}** — {r.message} "
                    f"*(runtime: {r.elapsed_s:.1f}s)*"
                )


if __name__ == "__main__":
    main()