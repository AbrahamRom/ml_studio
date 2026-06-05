"""Utilities for the ML Studio tabular AutoML pipeline."""

from .tasks import infer_target_task
from .automl_runner import run_target_automl
from .comparison import build_final_matrix
from .quality import (
    SHAPIRO_ALPHA,
    SHAPIRO_MAX_SAMPLES,
    analyze_data_quality,
    compute_variable_stats,
    variable_stats_to_csv,
    variable_stats_to_json,
)

__all__ = [
    "infer_target_task",
    "run_target_automl",
    "build_final_matrix",
    "analyze_data_quality",
    "compute_variable_stats",
    "variable_stats_to_csv",
    "variable_stats_to_json",
    "SHAPIRO_ALPHA",
    "SHAPIRO_MAX_SAMPLES",
]
