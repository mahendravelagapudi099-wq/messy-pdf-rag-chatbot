"""Result export: versioned JSON/CSV files, a comparison table and a chart."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .benchmark import BenchmarkResult
from .metrics import by_category

log = logging.getLogger(__name__)

# Metrics plotted in the comparison chart, in display order.
CHART_METRICS = (
    ("hit_rate_at_k", "Hit Rate@k"),
    ("mrr", "MRR"),
    ("answer_f1", "Answer F1"),
    ("citation_correctness", "Citation Correctness"),
    ("fact_recall", "Fact Recall"),
    ("unsupported_rate", "Unsupported Rate"),
    ("mean_latency_s", "Latency (s)"),
)


def results_root(base: Path | str = "results") -> Path:
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id(results: Sequence[BenchmarkResult]) -> str:
    """Versioned identifier, e.g. ``run-20261012T101500Z-cs512``."""
    suffix = results[0].config.label if len(results) == 1 else "compare"
    return f"run-{_stamp()}-{suffix}"


def to_json(result: BenchmarkResult, *, include_outcomes: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(result.config),
        "metrics": result.metrics.as_dict(),
        "dataset_coverage": result.dataset_coverage,
        "elapsed_s": round(result.elapsed_s, 2),
        "by_category": by_category(result.outcomes),
        "ingestion": [
            {
                "filename": r.filename,
                "status": r.status,
                "message": r.message,
                "pages": r.page_count,
                "chunks": r.chunk_count,
                "ocr_pages": r.ocr_pages,
                "warnings": r.warnings,
            }
            for r in result.ingestion
        ],
    }
    if include_outcomes:
        payload["outcomes"] = [
            {
                "question_id": o.question_id,
                "question": o.question,
                "hit": o.hit,
                "reciprocal_rank": round(o.reciprocal_rank, 4),
                "answer_f1": o.answer_f1,
                "fact_recall": o.fact_recall,
                "citation_correct": o.citation_correct,
                "abstained": o.abstained,
                "retrieved": o.retrieved_count,
                "latency_s": round(o.latency_s, 4),
                "answer": o.answer,
            }
            for o in result.outcomes
        ]
    return payload


def save_json(result: BenchmarkResult, out_dir: Path | str, *, include_outcomes: bool = True) -> Path:
    out_dir = results_root(out_dir)
    path = out_dir / f"{run_id([result])}.json"
    path.write_text(
        json.dumps(to_json(result, include_outcomes=include_outcomes), indent=2, default=str),
        encoding="utf-8",
    )
    return path


def comparison_rows(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    return [r.row() for r in results]


def save_csv(results: Sequence[BenchmarkResult], out_dir: Path | str) -> Path:
    """Write one row per configuration to a comparison CSV."""
    out_dir = results_root(out_dir)
    rows = comparison_rows(results)
    if not rows:
        raise ValueError("No results to write.")

    path = out_dir / f"{run_id(results)}.csv"
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def save_json_summary(results: Sequence[BenchmarkResult], out_dir: Path | str) -> Path:
    out_dir = results_root(out_dir)
    path = out_dir / f"{run_id(results)}_summary.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runs": comparison_rows(results),
                "datasets": [r.dataset_coverage for r in results],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def format_comparison_table(results: Sequence[BenchmarkResult]) -> str:
    """Render a Markdown comparison table."""
    rows = comparison_rows(results)
    if not rows:
        return "_No results._"

    keys = [
        "label", "chunk_size", "chunk_overlap", "reranker", "answer_mode",
        "hit_rate_at_k", "mrr", "answer_f1", "fact_recall",
        "citation_correctness", "unsupported_rate", "abstention_rate",
        "mean_latency_s", "n_questions",
    ]
    keys = [k for k in keys if k in rows[0]]

    header = "| " + " | ".join(k.replace("_", " ") for k in keys) + " |"
    sep = "|" + "|".join("---" for _ in keys) + "|"
    body = [
        "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows
    ]
    return "\n".join([header, sep, *body])


def save_chart(results: Sequence[BenchmarkResult], out_dir: Path | str) -> Optional[Path]:
    """Grouped bar chart comparing configurations across metrics."""
    if not results:
        return None

    rows = comparison_rows(results)
    labels = [str(r["label"]) for r in rows]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - charting is optional
        log.warning("Could not build chart: %s", exc)
        return None

    metrics = [(k, label) for k, label in CHART_METRICS if k in rows[0]]
    if not metrics:
        return None

    n_configs = len(rows)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = min(0.8 / max(n_configs, 1), 0.3)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * n_metrics), 4.5))
    for i, row in enumerate(rows):
        vals = [float(row.get(k, 0) or 0) for k, _ in metrics]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=str(row["label"]))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Chunking configuration comparison")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir = results_root(out_dir)
    path = out_dir / f"{run_id(results)}_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_markdown_report(
    results: Sequence[BenchmarkResult], out_dir: Path | str, extra_notes: str = ""
) -> Path:
    """Write a self-contained Markdown report next to the CSV/JSON."""
    out_dir = results_root(out_dir)
    path = out_dir / f"{run_id(results)}_report.md"

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset coverage",
        "",
        "```",
        results[0].dataset_coverage if results else "n/a",
        "```",
        "",
        "## Comparison",
        "",
        format_comparison_table(results),
        "",
        "## Per-run detail",
        "",
    ]
    for r in results:
        lines += [
            f"### {r.config.label or r.config.key()}",
            "",
            f"- chunking: {r.config.chunk_size} tokens / {r.config.chunk_overlap} overlap",
            f"- top_k: {r.config.top_k} | threshold: {r.config.similarity_threshold} "
            f"| reranker: {r.config.use_reranker}",
            f"- answer mode: {r.config.answer_mode} | elapsed: {r.elapsed_s:.1f}s",
            f"- ingestion: {sum(1 for x in r.ingestion if x.status == 'indexed')} indexed, "
            f"{sum(1 for x in r.ingestion if x.status == 'failed')} failed",
            "",
            "```json",
            json.dumps(r.metrics.as_dict(), indent=2),
            "```",
            "",
            "#### Metrics by question category",
            "",
            "| category | n | hit rate | fact recall |",
            "|---|---|---|---|",
        ]
        for cat, vals in by_category(r.outcomes).items():
            lines.append(f"| {cat} | {vals['n']} | {vals['hit_rate']:.3f} | {vals['fact_recall']:.3f} |")
        lines.append("")

    if extra_notes:
        lines += ["## Notes", "", extra_notes, ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_all(
    results: Sequence[BenchmarkResult], out_dir: Path | str, *, notes: str = ""
) -> dict[str, str]:
    """Write every artefact and return a map of artifact kind -> path."""
    artifacts = {
        "csv": str(save_csv(results, out_dir)),
        "json": str(save_json_summary(results, out_dir)),
        "markdown": str(save_markdown_report(results, out_dir, notes)),
    }
    for i, r in enumerate(results):
        artifacts[f"detail_{r.config.key()}"] = str(save_json(r, out_dir))
    chart = save_chart(results, out_dir)
    if chart:
        artifacts["chart"] = str(chart)
    return artifacts