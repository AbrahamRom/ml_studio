"""Comparison tables for independent target AutoML runs."""

from __future__ import annotations

import pandas as pd


def _holdout_target_scale_stats(result: dict) -> dict:
    y_true = result.get("y_test")
    if y_true is None:
        prediction_frame = result.get("prediction_frame")
        if isinstance(prediction_frame, pd.DataFrame) and "y_true" in prediction_frame.columns:
            y_true = prediction_frame["y_true"]

    if y_true is None:
        return {}

    series = pd.Series(y_true).dropna()
    if series.empty:
        return {}

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}

    return {
        "Holdout min": float(numeric.min()),
        "Holdout max": float(numeric.max()),
        "Holdout media": float(numeric.mean()),
        "Holdout mediana": float(numeric.median()),
    }


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
    """Build target x model_type matrix with the best metric per model family.

    Uses the internal leaderboard (validation metric) for all model types, then
    supplements with holdout metrics from ``per_model_metrics`` for any model_type
    not present in the leaderboard (e.g. custom baselines like "Baseline (promedio)").
    """

    rows = {}
    for target, result in target_results.items():
        target_metrics: dict[str, float] = {}

        # 1. Leaderboard (internal validation metric)
        leaderboard = result.get("leaderboard")
        if leaderboard is None:
            leaderboard = result.get("leaderboard_df")
        if leaderboard is not None and len(leaderboard) > 0:
            lb = pd.DataFrame(leaderboard).copy()
            if "model_type" in lb.columns and "metric_value" in lb.columns:
                direction = result.get("config", {}).get("direction", "max")
                best_rows = _best_by_direction(lb.dropna(subset=["metric_value"]), direction)
                for _, row in best_rows.iterrows():
                    target_metrics[row["model_type"]] = float(row["metric_value"])

        # 2. Supplement with holdout metrics for model_types not in leaderboard
        per_model_metrics = result.get("per_model_metrics")
        if per_model_metrics is not None and len(per_model_metrics) > 0:
            pm = pd.DataFrame(per_model_metrics)
            config = result.get("config", {})
            primary_metric = config.get("primary_metric", "score_global")
            if "model_type" in pm.columns and primary_metric in pm.columns:
                for _, row in pm.iterrows():
                    mt = row["model_type"]
                    if mt not in target_metrics:
                        val = row.get(primary_metric)
                        if val is not None and pd.notna(val):
                            target_metrics[mt] = float(val)

        rows[target] = target_metrics

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


def _select_best_holdout_row(
    per_model_metrics: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series | None, str | None]:
    if per_model_metrics is None or per_model_metrics.empty:
        return None, None

    preferred_columns = [config.get("primary_metric"), "score_global"]
    numeric_columns = [
        column
        for column in per_model_metrics.columns
        if column not in {"model_name", "model_type", "model_class", "evaluation_error"}
        and pd.api.types.is_numeric_dtype(per_model_metrics[column])
    ]

    metric_column = next(
        (column for column in preferred_columns if column and column in numeric_columns),
        None,
    )
    if metric_column is None and numeric_columns:
        metric_column = numeric_columns[0]
    if metric_column is None:
        return None, None

    metric_values = pd.to_numeric(per_model_metrics[metric_column], errors="coerce")
    if metric_values.dropna().empty:
        return None, None

    direction = config.get("direction", "max")
    idx = metric_values.idxmax() if direction == "max" else metric_values.idxmin()
    return per_model_metrics.loc[idx], metric_column


def build_best_model_metrics(target_results: dict) -> pd.DataFrame:
    rows = []
    for target, result in target_results.items():
        per_model_metrics = result.get("per_model_metrics")
        if per_model_metrics is None or len(per_model_metrics) == 0:
            continue

        metrics_df = pd.DataFrame(per_model_metrics)
        best_name = result.get("best_model_name")
        best_row = None

        if best_name and "model_name" in metrics_df.columns:
            match = metrics_df.loc[metrics_df["model_name"] == best_name]
            if not match.empty:
                best_row = match.iloc[0]

        selected_metric = result.get("best_model_metric")
        if best_row is None:
            best_row, selected_metric = _select_best_holdout_row(
                metrics_df,
                result.get("config", {}),
            )

        if best_row is None:
            continue

        scale_stats = _holdout_target_scale_stats(result)
        metric_columns = [
            column
            for column in metrics_df.columns
            if column
            not in {
                "model_name",
                "model_type",
                "model_class",
                "evaluation_error",
            }
            and pd.api.types.is_numeric_dtype(metrics_df[column])
        ]

        # --- Best model row ---
        row = {
            "Target": target,
            "Mejor modelo holdout": best_row.get("model_name"),
            "Tipo holdout": best_row.get("model_type"),
        }
        if selected_metric:
            row["Métrica usada"] = selected_metric
        if scale_stats:
            row.update(scale_stats)
        for column in metric_columns:
            row[column] = best_row.get(column)

        # --- Columna Baseline (promedio) con su métrica líder ---
        baseline_match = metrics_df.loc[metrics_df["model_name"] == "Baseline (promedio)"]
        if not baseline_match.empty:
            bl_row = baseline_match.iloc[0]
            bl_metric = result.get("config", {}).get("primary_metric", "score_global")
            row["Baseline (promedio)"] = bl_row.get(bl_metric)

        rows.append(row)

    return pd.DataFrame(rows)
