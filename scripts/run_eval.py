"""Command-line interface for ingestion, evaluation and benchmarking.

Examples
--------
Ingest a directory of PDFs::

    python -m scripts.run_eval ingest --corpus data/corpus

Run the three-configuration chunking comparison (no API key needed)::

    python -m scripts.run_eval benchmark --corpus data/corpus

Run one configuration with a real LLM::

    python -m scripts.run_eval benchmark --corpus data/corpus --mode llm --chunk-size 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHUNKING_EXPERIMENTS, Settings, get_settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    BenchmarkConfig,
    coverage,
    export_all,
    format_comparison_table,
    load_dataset,
    run_benchmark,
)

log = logging.getLogger("run_eval")


def _progress(msg: str, frac: float) -> None:
    bar = "#" * int(frac * 30)
    sys.stderr.write(f"\r[{bar:<30}] {frac * 100:5.1f}%  {msg[:60]:<60}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def _setup(verbose: bool) -> Settings:
    settings = get_settings()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return settings


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from src.rag import RAGEngine

    settings = _setup(args.verbose)
    engine = RAGEngine(settings)
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"error: corpus directory not found: {corpus}", file=sys.stderr)
        return 2

    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        print(f"error: no PDFs in {corpus}", file=sys.stderr)
        return 2

    print(f"Ingesting {len(pdfs)} PDF(s) from {corpus}\n")
    failures = 0
    for i, pdf in enumerate(pdfs):
        report = engine.add_file(pdf, force=args.force, progress=_progress if args.progress else None)
        mark = {"indexed": "OK", "skipped": "--", "failed": "!!"}.get(report.status, "??")
        if report.status == "failed":
            failures += 1
        print(
            f"  [{mark}] {report.filename:<45} {report.message}"
            + (f"  ({report.elapsed_s:.1f}s)" if report.elapsed_s else "")
        )
        for w in report.warnings[:3]:
            print(f"        warning: {w}")

    print(f"\n{len(pdfs) - failures} indexed, {failures} failed.")
    st = engine.status()
    print(f"Index now holds {st.doc_count} document(s) / {st.chunk_count} chunk(s).")
    return 1 if failures else 0


def cmd_questions(args: argparse.Namespace) -> int:
    questions = load_dataset(args.dataset)
    cov = coverage(questions)
    print(cov.summary())
    if args.list:
        print()
        for q in questions:
            flag = "UNANSWERABLE" if q.should_abstain else "answerable"
            print(f"  {q.id:<22} [{flag:<11}] {q.category:<12} {q.question}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    settings = _setup(args.verbose)

    if args.dataset and not Path(args.dataset).exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    try:
        questions = load_dataset(args.dataset)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(coverage(questions).summary())
    if args.mode == "llm" and not settings.llm_configured:
        print(
            "\nerror: --mode llm requires LLM_API_KEY / LLM_BASE_URL / LLM_MODEL.\n"
            "       Falling back is not silent; re-run with --mode extractive.\n",
            file=sys.stderr,
        )
        return 2

    # Build the configuration matrix.
    if args.chunk_size:
        configs = [{"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}]
    else:
        configs = list(CHUNKING_EXPERIMENTS)

    if args.reranker:
        variants = [False, True]
    else:
        variants = [args.reranker or False]

    runs = []
    for cfg in configs:
        for use_rr in variants:
            label = f"cs{cfg['chunk_size']}" + ("+rr" if use_rr else "")
            print(f"\n=== running {label} (mode={args.mode}) ===", flush=True)
            runs.append(
                run_benchmark(
                    BenchmarkConfig(
                        dataset_path=Path(args.dataset),
                        corpus_dir=Path(args.corpus),
                        chunk_size=cfg["chunk_size"],
                        chunk_overlap=cfg["chunk_overlap"],
                        top_k=args.top_k,
                        similarity_threshold=args.threshold,
                        use_reranker=use_rr,
                        answer_mode=args.mode,
                        label=label,
                    ),
                    settings,
                    progress=_progress if args.progress else None,
                )
            )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(format_comparison_table(runs))

    notes = args.notes or ""
    if args.mode == "extractive":
        notes = (
            (notes + "\n\n" if notes else "")
            + "Answer mode: `extractive` (deterministic, no LLM). Answer-level metrics "
            "measure the retrieval + extractive baseline, not generative quality. "
            "Re-run with `--mode llm` to evaluate the generation path."
        )
    artifacts = export_all(runs, args.out, notes=notes)
    print("\nArtifacts written:")
    for kind, path in artifacts.items():
        print(f"  {kind:<24} {path}")
    return 0


def run_benchmark_cmd(
    *,
    dataset_path: str = "data/eval_dataset.jsonl",
    corpus_dir: str = "data/corpus",
    mode: str = "extractive",
    reranker: bool = False,
    top_k: int = 5,
    out: str = "results",
) -> list:
    """Run the chunking comparison programmatically (used by the Streamlit UI)."""
    settings = get_settings()
    results = []
    for cfg in CHUNKING_EXPERIMENTS:
        for use_rr in ([False, True] if reranker else [False]):
            label = f"cs{cfg['chunk_size']}" + ("+rr" if use_rr else "")
            results.append(
                run_benchmark(
                    BenchmarkConfig(
                        dataset_path=Path(dataset_path),
                        corpus_dir=Path(corpus_dir),
                        chunk_size=cfg["chunk_size"],
                        chunk_overlap=cfg["chunk_overlap"],
                        top_k=top_k,
                        use_reranker=use_rr,
                        answer_mode=mode,
                        label=label,
                    ),
                    settings,
                )
            )
    try:
        export_all(results, out)
    except Exception as exc:  # artefact export must not lose the results
        print(f"warning: could not write artefacts: {exc}", file=sys.stderr)
    return results


def cmd_status(args: argparse.Namespace) -> int:
    from src.rag import RAGEngine

    settings = _setup(args.verbose)
    engine = RAGEngine(settings)
    st = engine.status()
    print(f"Index      : {engine.store.persist_dir}")
    print(f"Collection : {engine.store.collection_name}")
    print(f"Documents  : {st.doc_count}  |  chunks: {st.chunk_count}")
    print(f"Embedding  : {st.embedding_detail}")
    print(f"LLM        : {st.llm_detail}")
    print(f"OCR        : {st.ocr_detail}")
    if st.doc_count:
        print("\nIndexed documents:")
        for d in engine.list_documents():
            print(f"  {d.filename:<45} {d.page_count:>4} pages  {d.chunk_count:>5} chunks")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Ingest PDFs, inspect datasets and run RAG evaluations.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Ingest a directory of PDFs into the index")
    pi.add_argument("--corpus", default="data/corpus")
    pi.add_argument("--force", action="store_true", help="Re-index even if unchanged")
    pi.add_argument("--no-progress", dest="progress", action="store_false")
    pi.set_defaults(func=cmd_ingest, progress=True)

    # questions
    pq = sub.add_parser("questions", help="Validate and summarise an eval dataset")
    pq.add_argument("--dataset", default="data/eval_dataset.jsonl")
    pq.add_argument("--list", action="store_true", help="Print every question")
    pq.set_defaults(func=cmd_questions)

    # benchmark
    pb = sub.add_parser("benchmark", help="Run the evaluation / chunking comparison")
    pb.add_argument("--dataset", default="data/eval_dataset.jsonl")
    pb.add_argument("--corpus", default="data/corpus")
    pb.add_argument("--out", default="results")
    pb.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    pb.add_argument("--chunk-size", type=int, default=None,
                    help="Run a single chunk size instead of the 3-config comparison")
    pb.add_argument("--chunk-overlap", type=int, default=64)
    pb.add_argument("--top-k", type=int, default=5)
    pb.add_argument("--threshold", type=float, default=0.0)
    pb.add_argument("--reranker", action="store_true",
                    help="Also run each config with the cross-encoder reranker")
    pb.add_argument("--notes", default="", help="Free-text notes added to the report")
    pb.add_argument("--no-progress", dest="progress", action="store_false")
    pb.set_defaults(func=cmd_benchmark, progress=True)

    # status
    ps = sub.add_parser("status", help="Show index and capability status")
    ps.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())