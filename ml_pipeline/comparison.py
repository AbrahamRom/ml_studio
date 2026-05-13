"""Comparison tables for independent target AutoML runs."""

from __future__ import annotations

import pandas as pd


def _best_by_direction(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df.empty:
        return df
    idx = (
        df.groupby("model_type")["metric_value"].idxmax()
        if direction == "max"
        else df.groupby("model_type")["metric_value"].idxmin()
    )
    return df.loc[idx].copy()


def build_final_matrix(target_results: dict) -> pd.DataFrame:
    """Build target x model_type matrix with the best metric per model family."""

    rows = {}
    for target, result in target_results.items():
        leaderboard = result.get("leaderboard")
        if leaderboard is None:
            leaderboard = result.get("leaderboard_df")
        if leaderboard is None or len(leaderboard) == 0:
            continue

        lb = pd.DataFrame(leaderboard).copy()
        if "model_type" not in lb.columns or "metric_value" not in lb.columns:
            continue

        direction = result.get("config", {}).get("direction", "max")
        best_rows = _best_by_direction(lb.dropna(subset=["metric_value"]), direction)
        rows[target] = {
            row["model_type"]: float(row["metric_value"])
            for _, row in best_rows.iterrows()
        }

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    if not matrix.empty:
        matrix.index.name = "Target"
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def build_target_summary(target_results: dict) -> pd.DataFrame:
    rows = []
    for target, result in target_results.items():
        metrics = result.get("holdout_metrics", {})
        config = result.get("config", {})
        rows.append(
            {
                "Target": target,
                "Tarea": config.get("task"),
                "MLJAR task": config.get("ml_task"),
                "Métrica primaria": config.get("primary_metric"),
                "Dirección": config.get("direction"),
                "Mejor modelo holdout": result.get("best_model_name"),
                "Tipo holdout": result.get("best_model_type"),
                "Métrica holdout": result.get("best_model_metric"),
                "Score holdout": metrics.get(config.get("primary_metric")),
                "Mejor modelo interno": result.get("internal_best_model_name"),
                "Tipo interno": result.get("internal_best_model_type"),
                "Métrica interna": config.get("primary_metric"),
                "Score interno": result.get("internal_best_metric_value"),
                "Train rows": result.get("train_rows"),
                "Test rows": result.get("test_rows"),
            }
        )
    return pd.DataFrame(rows)
