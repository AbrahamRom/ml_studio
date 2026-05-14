"""Artifact paths and persistence helpers for AutoML runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd


DEFAULT_RUNS_DIR = Path("artifacts") / "automl_runs"


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "target"


def create_run_dir(base_dir: str | Path = DEFAULT_RUNS_DIR) -> tuple[str, Path]:
    root = Path(base_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = root / run_id
    suffix = 1
    while run_path.exists():
        suffix += 1
        run_path = root / f"{run_id}_{suffix}"
    run_path.mkdir(parents=True, exist_ok=False)
    return run_path.name, run_path


def target_dir(run_path: str | Path, target: str) -> Path:
    path = Path(run_path) / safe_slug(target)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str | Path, data: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=json_safe), encoding="utf-8")
    return out


def save_dataframe(path: str | Path, df: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


@lru_cache(maxsize=128)
def read_json(path: str | Path) -> dict | None:
    src = Path(path)
    if not src.exists():
        return None
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=128)
def read_dataframe(path: str | Path) -> pd.DataFrame:
    src = Path(path)
    if not src.exists():
        return pd.DataFrame()
    return pd.read_csv(src)


def list_runs(base_dir: str | Path = DEFAULT_RUNS_DIR) -> list[dict]:
    root = Path(base_dir)
    if not root.exists():
        return []

    runs = []
    for run_path in sorted(root.iterdir(), reverse=True):
        if not run_path.is_dir():
            continue
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path) or {}
        targets = list((manifest.get("targets") or {}).keys())
        runs.append(
            {
                "run_id": manifest.get("run_id", run_path.name),
                "path": run_path,
                "targets": sorted(targets),
                "errors": manifest.get("errors") or {},
            }
        )

    return runs


def load_run_manifest(run_path: str | Path) -> dict:
    path = Path(run_path) / "run_manifest.json"
    manifest = read_json(path)
    if manifest is None:
        raise FileNotFoundError(f"No run_manifest.json at {path}")
    return manifest


def _load_compare_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = read_dataframe(path)
    if df.empty:
        return df
    if "Target" in df.columns:
        df = df.set_index("Target")
    return df


def _infer_plot_paths(target_path: Path) -> dict:
    plots_dir = target_path / "plots"
    if not plots_dir.exists():
        return {}
    return {path.stem: str(path) for path in sorted(plots_dir.glob("*.html"))}


def load_target_artifacts(
    run_path: str | Path,
    target: str,
    manifest: dict | None = None,
) -> dict:
    run_path = Path(run_path)
    target_path = run_path / safe_slug(target)
    manifest = manifest or read_json(target_path / "target_manifest.json") or {}

    result = {**manifest}
    result.setdefault("target", target)
    result.setdefault("config", {})
    result.setdefault("feature_cols", manifest.get("feature_cols") or [])

    leaderboard_path = Path(result.get("leaderboard_path") or target_path / "leaderboard.csv")
    per_model_metrics_path = Path(
        result.get("per_model_metrics_path") or target_path / "per_model_metrics.csv"
    )
    predictions_path = Path(result.get("predictions_path") or target_path / "predictions.csv")
    metrics_path = Path(result.get("metrics_path") or target_path / "holdout_metrics.json")

    result["leaderboard_path"] = str(leaderboard_path)
    result["per_model_metrics_path"] = str(per_model_metrics_path)
    result["predictions_path"] = str(predictions_path)
    result["metrics_path"] = str(metrics_path)
    result.setdefault("results_path", str(target_path / "mljar"))

    plot_paths = result.get("plot_paths") or _infer_plot_paths(target_path)
    result["plot_paths"] = plot_paths

    leaderboard = read_dataframe(leaderboard_path)
    per_model_metrics = read_dataframe(per_model_metrics_path)
    prediction_frame = read_dataframe(predictions_path)
    holdout_metrics = read_json(metrics_path) or {}

    y_true = None
    y_pred = None
    row_index = None
    if not prediction_frame.empty:
        if "row_index" in prediction_frame.columns:
            row_index = prediction_frame["row_index"]
        if "y_true" in prediction_frame.columns:
            y_true = prediction_frame["y_true"]
        if "y_pred" in prediction_frame.columns:
            y_pred = prediction_frame["y_pred"]

    if row_index is None:
        row_index = prediction_frame.index if not prediction_frame.empty else pd.RangeIndex(0)

    predictions = (
        pd.Series(y_pred.to_numpy(), index=row_index, name="y_pred")
        if y_pred is not None
        else pd.Series(dtype=float)
    )
    y_test = (
        pd.Series(y_true.to_numpy(), index=row_index, name="y_true")
        if y_true is not None
        else None
    )

    proba_cols = [col for col in prediction_frame.columns if col.startswith("proba_")]
    proba = prediction_frame[proba_cols].to_numpy() if proba_cols else None

    return {
        **result,
        "automl": None,
        "leaderboard": leaderboard,
        "per_model_metrics": per_model_metrics,
        "holdout_metrics": holdout_metrics,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": y_test,
        "predictions": predictions,
        "prediction_frame": prediction_frame,
        "proba": proba,
    }


def load_automl_run(run_path: str | Path) -> dict:
    run_path = Path(run_path)
    manifest = load_run_manifest(run_path)
    targets = manifest.get("targets") or {}

    target_results = {}
    if targets:
        for target, target_manifest in targets.items():
            target_results[target] = load_target_artifacts(run_path, target, target_manifest)
    else:
        for target_manifest_path in sorted(run_path.glob("*/target_manifest.json")):
            target_manifest = read_json(target_manifest_path) or {}
            target_name = target_manifest.get("target", target_manifest_path.parent.name)
            target_results[target_name] = load_target_artifacts(
                run_path,
                target_name,
                target_manifest,
            )

    compare_df = _load_compare_table(manifest.get("final_matrix_path") or run_path / "final_matrix.csv")
    summary_df = read_dataframe(manifest.get("target_summary_path") or run_path / "target_summary.csv")
    best_metrics_df = read_dataframe(
        manifest.get("best_model_metrics_path") or run_path / "best_model_metrics.csv"
    )

    return {
        **manifest,
        "run_id": manifest.get("run_id", run_path.name),
        "base_path": str(run_path),
        "target_results": target_results,
        "compare_df": compare_df,
        "summary_df": summary_df,
        "best_model_metrics_df": best_metrics_df,
        "source": "loaded",
    }
