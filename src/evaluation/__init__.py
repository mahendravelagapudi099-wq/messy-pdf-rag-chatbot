"""Evaluation framework: dataset, metrics, benchmark runner and reporting."""

from .benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark
from .dataset import (
    VALID_CATEGORIES,
    CoverageReport,
    DatasetError,
    EvalQuestion,
    coverage,
    dump_dataset,
    load_dataset,
)
from .metrics import Metrics, QuestionOutcome, aggregate, token_f1
from .report import export_all, format_comparison_table, save_csv, save_chart, save_json

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "VALID_CATEGORIES",
    "CoverageReport",
    "DatasetError",
    "EvalQuestion",
    "coverage",
    "dump_dataset",
    "load_dataset",
    "Metrics",
    "QuestionOutcome",
    "aggregate",
    "token_f1",
    "export_all",
    "format_comparison_table",
    "save_csv",
    "save_chart",
    "save_json",
]