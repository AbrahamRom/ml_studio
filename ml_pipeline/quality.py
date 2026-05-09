"""Small, serializable data quality report for AutoML runs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def analyze_data_quality(df: pd.DataFrame, targets: Iterable[str] | None = None) -> dict:
    """Return a JSON-safe quality summary used by training and reports."""

    target_set = set(targets or [])
    rows = []
    for col in df.columns:
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        null_count = int(series.isna().sum())
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "is_target": col in target_set,
            "null_count": null_count,
            "null_pct": float(null_count / max(len(df), 1) * 100),
            "unique_count": nunique,
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "min": _safe_number(series.min()),
                    "max": _safe_number(series.max()),
                    "mean": _safe_number(series.mean()),
                    "std": _safe_number(series.std()),
                }
            )
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                outliers = 0
            else:
                outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            row["outlier_count_iqr"] = outliers
            row["outlier_pct_iqr"] = float(outliers / max(len(df), 1) * 100)
        rows.append(row)

    target_issues = []
    for target in target_set:
        if target not in df.columns:
            target_issues.append({"target": target, "severity": "error", "message": "No existe en el dataset."})
            continue
        clean = df[target].dropna()
        if clean.nunique(dropna=True) < 2:
            target_issues.append(
                {
                    "target": target,
                    "severity": "error",
                    "message": "Tiene menos de 2 valores distintos después de eliminar nulos.",
                }
            )
        if clean.empty:
            target_issues.append(
                {"target": target, "severity": "error", "message": "No tiene valores no nulos."}
            )

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "total_nulls": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": rows,
        "target_issues": target_issues,
    }
