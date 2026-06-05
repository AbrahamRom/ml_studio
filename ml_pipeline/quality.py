"""Small, serializable data quality report for AutoML runs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ── Variable-level descriptive statistics ──────────────────────────────────────
# Maximum sample size accepted by scipy.stats.shapiro. Beyond this the test
# is unreliable, so we sub-sample with a fixed random_state for reproducibility.
SHAPIRO_MAX_SAMPLES = 5000
SHAPIRO_ALPHA = 0.05


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


# ── Per-variable statistics (min, max, median, mean, variance, Shapiro) ────────


def _classify_column(series: pd.Series) -> str:
    """Mirror the categorical/discrete/continuous classification used in the EDA page."""
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    if (
        series.nunique(dropna=True) <= 15
        and pd.api.types.is_integer_dtype(series)
    ):
        return "discrete"
    return "continuous"


def _shapiro_for_numeric(clean: pd.Series) -> dict:
    """Run the Shapiro-Wilk test on a numeric Series with no NaN values.

    Returns a JSON-safe dict with the test statistic, p-value, decision and any
    caveat that prevented the test from being applied (e.g. too few samples,
    constant column, or sub-sampling due to n > SHAPIRO_MAX_SAMPLES).
    """
    n = int(len(clean))
    if n < 3:
        return {
            "shapiro_W": None,
            "shapiro_p": None,
            "shapiro_is_normal": None,
            "shapiro_n_used": n,
            "shapiro_note": "n < 3: test no aplicable",
        }

    note = None
    sample = clean
    if n > SHAPIRO_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(clean.index, size=SHAPIRO_MAX_SAMPLES, replace=False)
        sample = clean.loc[idx]
        note = (
            f"n > {SHAPIRO_MAX_SAMPLES}: submuestreo reproducible a "
            f"{SHAPIRO_MAX_SAMPLES} observaciones (random_state=42)"
        )

    # Constant columns produce 0 variance — Shapiro-Wilk is undefined for them.
    std_sample = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    if not math.isfinite(std_sample) or std_sample == 0:
        return {
            "shapiro_W": None,
            "shapiro_p": None,
            "shapiro_is_normal": None,
            "shapiro_n_used": int(len(sample)),
            "shapiro_note": "varianza cero: test no aplicable",
        }

    try:
        w_stat, p_value = stats.shapiro(sample.to_numpy())
    except Exception as exc:  # scipy can raise for degenerate inputs
        return {
            "shapiro_W": None,
            "shapiro_p": None,
            "shapiro_is_normal": None,
            "shapiro_n_used": int(len(sample)),
            "shapiro_note": f"error al calcular Shapiro: {exc}",
        }

    return {
        "shapiro_W": _safe_number(float(w_stat)),
        "shapiro_p": _safe_number(float(p_value)),
        "shapiro_is_normal": bool(p_value > SHAPIRO_ALPHA),
        "shapiro_n_used": int(len(sample)),
        "shapiro_note": note,
    }


def compute_variable_stats(
    df: pd.DataFrame,
    targets: Iterable[str] | None = None,
    alpha: float = SHAPIRO_ALPHA,
) -> dict:
    """Compute per-variable descriptive statistics + Shapiro-Wilk normality test.

    The result is a JSON-safe dict with two parallel views:
      * ``columns``  — list of per-column dicts (long format, easy to iterate)
      * ``wide``     — pandas DataFrame in wide format (rows=columns, cols=metrics)
                       (not JSON-safe, but convenient for direct display)
    The function gracefully handles non-numeric columns, constant columns and
    sub-sampling for very large datasets (Shapiro-Wilk is capped at
    ``SHAPIRO_MAX_SAMPLES`` observations for reliability).
    """

    target_set = set(targets or [])
    rows: list[dict] = []
    wide_records: list[dict] = []

    for col in df.columns:
        series = df[col]
        classification = _classify_column(series)
        n_total = int(len(series))
        null_count = int(series.isna().sum())
        nunique = int(series.nunique(dropna=True))

        base = {
            "column": col,
            "dtype": str(series.dtype),
            "class": classification,
            "is_target": col in target_set,
            "count": n_total,
            "null_count": null_count,
            "null_pct": float(null_count / max(n_total, 1) * 100),
            "unique_count": nunique,
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            n_clean = int(len(clean))
            mean = _safe_number(clean.mean()) if n_clean else None
            median = _safe_number(clean.median()) if n_clean else None
            std = _safe_number(clean.std()) if n_clean > 1 else None
            variance = _safe_number(clean.var()) if n_clean > 1 else None
            min_v = _safe_number(clean.min()) if n_clean else None
            max_v = _safe_number(clean.max()) if n_clean else None

            shapiro = _shapiro_for_numeric(clean)

            numeric_block = {
                "min": min_v,
                "max": max_v,
                "mean": mean,
                "median": median,
                "std": std,
                "variance": variance,
                "n_non_null": n_clean,
                **shapiro,
            }
            # Allow per-call alpha override without mutating module constant.
            if alpha != SHAPIRO_ALPHA and shapiro["shapiro_p"] is not None:
                shapiro_p = shapiro["shapiro_p"]
                numeric_block["shapiro_is_normal"] = bool(shapiro_p > alpha)

            base.update(numeric_block)
        else:
            clean = series.dropna()
            n_clean = int(len(clean))
            mode_series = clean.mode()
            top = _safe_number(mode_series.iloc[0]) if not mode_series.empty else None
            freq = (
                int((clean == top).sum()) if top is not None and n_clean else 0
            )
            base.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "variance": None,
                    "n_non_null": n_clean,
                    "shapiro_W": None,
                    "shapiro_p": None,
                    "shapiro_is_normal": None,
                    "shapiro_n_used": None,
                    "shapiro_note": "no aplica a variables categóricas",
                    "top": top,
                    "freq": freq,
                }
            )

        rows.append(base)
        wide_records.append(
            {
                "column": base["column"],
                "class": base["class"],
                "count": base["count"],
                "null_pct": base["null_pct"],
                "min": base.get("min"),
                "max": base.get("max"),
                "mean": base.get("mean"),
                "median": base.get("median"),
                "variance": base.get("variance"),
                "std": base.get("std"),
                "shapiro_W": base.get("shapiro_W"),
                "shapiro_p": base.get("shapiro_p"),
                "shapiro_is_normal": base.get("shapiro_is_normal"),
                "shapiro_n_used": base.get("shapiro_n_used"),
            }
        )

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "alpha": float(alpha),
        "shapiro_max_samples": SHAPIRO_MAX_SAMPLES,
        "columns": rows,
        "wide": pd.DataFrame(wide_records),
    }


def variable_stats_to_csv(stats_dict: dict) -> bytes:
    """Serialize the wide ``DataFrame`` from ``compute_variable_stats`` to CSV bytes."""
    df_wide: pd.DataFrame = stats_dict["wide"]
    return df_wide.to_csv(index=False).encode("utf-8")


def variable_stats_to_json(stats_dict: dict) -> bytes:
    """Serialize the full stats dict (including wide frame) to JSON bytes."""
    payload = {
        "shape": stats_dict["shape"],
        "alpha": stats_dict["alpha"],
        "shapiro_max_samples": stats_dict["shapiro_max_samples"],
        "columns": stats_dict["columns"],
        "wide": stats_dict["wide"].replace({np.nan: None}).to_dict(orient="records"),
    }
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
