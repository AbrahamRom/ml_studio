"""Artifact paths and persistence helpers for AutoML runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
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
