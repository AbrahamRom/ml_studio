"""Risk-based early-warning helpers for regression quality targets."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CONFIG_PATH = Path("config") / "quality_specs.json"
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20)


def normalize_quality_name(value: str) -> str:
    """Normalize target/spec names so config aliases can match dataset columns."""

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_quality_specs(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        return {}
    return json.loads(src.read_text(encoding="utf-8"))


def resolve_quality_spec(
    target: str,
    specs: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the matching quality specification for a target column."""

    specs = specs if specs is not None else load_quality_specs()
    if not specs:
        return None, None

    if target in specs:
        return target, specs[target]

    normalized_target = normalize_quality_name(target)
    for key, config in specs.items():
        candidates = [key, *(config.get("aliases") or [])]
        if normalized_target in {normalize_quality_name(candidate) for candidate in candidates}:
            return key, config

    return None, None


def _as_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' debe ser numerico.") from exc
    if not math.isfinite(number):
        raise ValueError(f"'{name}' debe ser finito.")
    return number


def _positive_width(value: Any, name: str) -> float:
    width = _as_float(value, name)
    if width <= 0:
        raise ValueError(f"'{name}' debe ser mayor que cero.")
    return width


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((float(low), float(high)) for low, high in intervals if high >= low)
    if not clean:
        return []

    merged = [clean[0]]
    for low, high in clean[1:]:
        prev_low, prev_high = merged[-1]
        if low <= prev_high:
            merged[-1] = (prev_low, max(prev_high, high))
        else:
            merged.append((low, high))
    return merged


def _spec_type(spec: dict[str, Any]) -> str:
    value = str(spec.get("type", "two_sided")).lower()
    aliases = {
        "range": "two_sided",
        "two-sided": "two_sided",
        "lower": "lower_only",
        "min": "lower_only",
        "minimum": "lower_only",
        "upper": "upper_only",
        "max": "upper_only",
        "maximum": "upper_only",
    }
    return aliases.get(value, value)


def build_danger_zone_intervals(config: dict[str, Any]) -> list[tuple[float, float]]:
    """Build in-spec danger-zone intervals from a target quality config."""

    spec = config.get("spec") or {}
    dz = config.get("danger_zone") or {}
    spec_type = _spec_type(spec)
    mode = str(dz.get("mode", "absolute")).lower()

    if spec_type == "two_sided":
        lower = _as_float(spec.get("lower"), "spec.lower")
        upper = _as_float(spec.get("upper"), "spec.upper")
        if upper <= lower:
            raise ValueError("spec.upper debe ser mayor que spec.lower.")

        if mode == "percent_of_spec_width":
            total_fraction = _as_float(dz.get("total_fraction", dz.get("fraction", 0.10)), "danger_zone.total_fraction")
            if total_fraction <= 0:
                raise ValueError("danger_zone.total_fraction debe ser mayor que cero.")
            total_width = (upper - lower) * total_fraction
            lower_width = _as_float(dz.get("lower_fraction", 0.5), "danger_zone.lower_fraction") * total_width
            upper_width = _as_float(dz.get("upper_fraction", 0.5), "danger_zone.upper_fraction") * total_width
        else:
            lower_width = _positive_width(dz.get("lower_width", dz.get("width")), "danger_zone.lower_width")
            upper_width = _positive_width(dz.get("upper_width", dz.get("width")), "danger_zone.upper_width")

        lower_width = min(lower_width, upper - lower)
        upper_width = min(upper_width, upper - lower)
        return _merge_intervals(
            [
                (lower, min(lower + lower_width, upper)),
                (max(upper - upper_width, lower), upper),
            ]
        )

    if spec_type == "lower_only":
        lower = _as_float(spec.get("lower"), "spec.lower")
        width = _positive_width(dz.get("lower_width", dz.get("width")), "danger_zone.width")
        return [(lower, lower + width)]

    if spec_type == "upper_only":
        upper = _as_float(spec.get("upper"), "spec.upper")
        width = _positive_width(dz.get("upper_width", dz.get("width")), "danger_zone.width")
        return [(upper - width, upper)]

    raise ValueError(f"Tipo de especificacion no soportado: {spec_type}")


def values_in_intervals(
    values: Any,
    intervals: list[tuple[float, float]],
    *,
    include_bounds: bool = True,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.zeros(arr.shape, dtype=bool)
    for low, high in intervals:
        if include_bounds:
            mask |= (arr >= low) & (arr <= high)
        else:
            mask |= (arr > low) & (arr < high)
    return mask


def values_oos(values: Any, config: dict[str, Any]) -> np.ndarray:
    spec = config.get("spec") or {}
    spec_type = _spec_type(spec)
    arr = np.asarray(values, dtype=float)
    inclusive = bool(spec.get("inclusive", True))

    if spec_type == "two_sided":
        lower = _as_float(spec.get("lower"), "spec.lower")
        upper = _as_float(spec.get("upper"), "spec.upper")
        if inclusive:
            return (arr < lower) | (arr > upper)
        return (arr <= lower) | (arr >= upper)

    if spec_type == "lower_only":
        lower = _as_float(spec.get("lower"), "spec.lower")
        return arr < lower if inclusive else arr <= lower

    if spec_type == "upper_only":
        upper = _as_float(spec.get("upper"), "spec.upper")
        return arr > upper if inclusive else arr >= upper

    raise ValueError(f"Tipo de especificacion no soportado: {spec_type}")


def probability_in_intervals(
    mu: float,
    residuals: Any,
    intervals: list[tuple[float, float]],
) -> float:
    residual_arr = clean_residuals(residuals)
    predictive_values = float(mu) + residual_arr
    return float(values_in_intervals(predictive_values, intervals).mean())


def probability_oos(mu: float, residuals: Any, config: dict[str, Any]) -> float:
    residual_arr = clean_residuals(residuals)
    predictive_values = float(mu) + residual_arr
    return float(values_oos(predictive_values, config).mean())


def clean_residuals(residuals: Any) -> np.ndarray:
    arr = np.asarray(residuals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No hay residuos de calibracion validos.")
    return arr


def fit_residual_uncertainty(y_true: Any, y_pred: Any) -> np.ndarray:
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    return clean_residuals(y_true_arr - y_pred_arr)


def predictive_interval(
    y_pred: Any,
    residuals: Any,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < coverage < 1:
        raise ValueError("coverage debe estar entre 0 y 1.")
    residual_arr = clean_residuals(residuals)
    alpha = 1.0 - coverage
    low_q, high_q = np.quantile(residual_arr, [alpha / 2.0, 1.0 - alpha / 2.0])
    pred_arr = np.asarray(y_pred, dtype=float)
    return pred_arr + low_q, pred_arr + high_q


def alert_thresholds(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("alert_thresholds") or {}
    p_dz = float(raw.get("p_dz", raw.get("p_dz_warning", 0.10)))
    p_oos = float(raw.get("p_oos", raw.get("p_oos_warning", 0.02)))
    return {
        "p_dz": p_dz,
        "p_oos": p_oos,
        "p_dz_watch": float(raw.get("p_dz_watch", p_dz)),
        "p_dz_critical": float(raw.get("p_dz_critical", max(0.50, p_dz * 2.0))),
        "p_oos_critical": float(raw.get("p_oos_critical", max(0.10, p_oos * 2.0))),
    }


def alert_tier(
    p_dz: float,
    p_oos: float,
    thresholds: dict[str, float],
    *,
    low_confidence: bool = False,
) -> str:
    if low_confidence:
        return "Low confidence"
    if p_oos >= thresholds["p_oos_critical"] or p_dz >= thresholds["p_dz_critical"]:
        return "Critical"
    if p_oos >= thresholds["p_oos"] or p_dz >= thresholds["p_dz"]:
        return "Warning"
    if p_dz >= thresholds["p_dz_watch"]:
        return "Watch"
    return "Normal"


def compute_early_warning_predictions(
    y_true: Any,
    y_pred: Any,
    residuals: Any,
    config: dict[str, Any],
    *,
    row_index: Any | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true e y_pred deben tener la misma longitud.")

    residual_arr = clean_residuals(residuals)
    dz_intervals = build_danger_zone_intervals(config)
    coverage = float((config.get("uncertainty") or {}).get("coverage", 0.90))
    pi_low, pi_high = predictive_interval(y_pred_arr, residual_arr, coverage)
    max_width = (config.get("uncertainty") or {}).get("max_interval_width")
    max_width = None if max_width is None else float(max_width)
    thresholds = alert_thresholds(config)

    rows = []
    actual_in_dz = values_in_intervals(y_true_arr, dz_intervals)
    actual_oos = values_oos(y_true_arr, config)
    index_values = list(row_index) if row_index is not None else list(range(len(y_true_arr)))

    for i, mu in enumerate(y_pred_arr):
        p_dz = probability_in_intervals(mu, residual_arr, dz_intervals)
        p_oos = probability_oos(mu, residual_arr, config)
        interval_width = float(pi_high[i] - pi_low[i])
        low_confidence = bool(max_width is not None and interval_width > max_width)
        alert = bool(p_dz >= thresholds["p_dz"] or p_oos >= thresholds["p_oos"])
        tier = alert_tier(p_dz, p_oos, thresholds, low_confidence=low_confidence)
        rows.append(
            {
                "row_index": index_values[i],
                "target": target,
                "y_true": float(y_true_arr[i]),
                "y_pred": float(mu),
                f"pi_low_{int(round(coverage * 100))}": float(pi_low[i]),
                f"pi_high_{int(round(coverage * 100))}": float(pi_high[i]),
                "interval_width": interval_width,
                "low_confidence": low_confidence,
                "p_dz": p_dz,
                "p_oos": p_oos,
                "risk_score": float(max(p_dz, p_oos)),
                "alert": alert,
                "alert_tier": tier,
                "actual_in_dz": bool(actual_in_dz[i]),
                "actual_oos": bool(actual_oos[i]),
                "actual_event": bool(actual_in_dz[i] or actual_oos[i]),
            }
        )

    return pd.DataFrame(rows)


def compute_alert_metrics(
    actual_event: Any,
    alert: Any,
    scores: Any | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(actual_event, dtype=bool).astype(int)
    y_pred = np.asarray(alert, dtype=bool).astype(int)
    if y_true.size == 0:
        return {
            "n": 0,
            "event_count": 0,
            "alert_count": 0,
            "high_risk_count": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "false_alerts_per_batch": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "labels": ["No event", "Event"],
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_alerts_per_batch = fp / len(y_true) if len(y_true) else 0.0

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "event_count": int(y_true.sum()),
        "alert_count": int(y_pred.sum()),
        "high_risk_count": int(y_pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "false_alerts_per_batch": float(false_alerts_per_batch),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "labels": ["No event", "Event"],
    }

    if scores is not None and len(np.unique(y_true)) == 2:
        score_arr = np.asarray(scores, dtype=float)
        if np.isfinite(score_arr).all():
            metrics["roc_auc"] = float(roc_auc_score(y_true, score_arr))
            metrics["pr_auc"] = float(average_precision_score(y_true, score_arr))

    return metrics


def compute_early_warning_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return compute_alert_metrics([], [])
    return compute_alert_metrics(
        predictions["actual_event"],
        predictions["alert"],
        predictions.get("risk_score"),
    )


def threshold_analysis(
    predictions: pd.DataFrame,
    *,
    p_oos_threshold: float,
    p_dz_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for threshold in p_dz_thresholds:
        alert = (predictions["p_dz"] >= threshold) | (predictions["p_oos"] >= p_oos_threshold)
        metrics = compute_alert_metrics(predictions["actual_event"], alert, predictions["risk_score"])
        rows.append(
            {
                "p_dz_threshold": threshold,
                "p_oos_threshold": p_oos_threshold,
                "alert_count": metrics["alert_count"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def recompute_early_warning_for_target(
    y_true: Any,
    y_pred: Any,
    config: dict[str, Any],
    *,
    residuals: Any | None = None,
    row_index: Any | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Recalcula Early Warning desde datos de holdout existentes, sin reentrenar.

    Usa residuos de calibracion si estan disponibles; si no, calcula
    residuos del holdout (y_true - y_pred) como aproximacion. Esto
    permite re-evaluar las alertas con especificaciones actualizadas
    sin necesidad de reentrenar el modelo.

    Parameters
    ----------
    y_true : array-like
        Valores reales del conjunto de prueba.
    y_pred : array-like
        Predicciones del modelo sobre el conjunto de prueba.
    config : dict
        Especificacion de calidad resuelta desde quality_specs.json.
    residuals : array-like, optional
        Residuos de calibracion (y_true_calib - y_pred_calib).
    row_index : array-like, optional
        Indices de fila originales.
    target : str, optional
        Nombre del target.

    Returns
    -------
    dict with keys:
        predictions : pd.DataFrame
        metrics : dict
        residuals : np.ndarray
        used_fallback_residuals : bool
        warnings : list[str]
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true e y_pred deben tener la misma longitud.")

    warnings_list: list[str] = []
    used_fallback = False
    residual_arr = None

    if residuals is not None:
        residual_arr = np.asarray(residuals, dtype=float).ravel()
        residual_arr = residual_arr[np.isfinite(residual_arr)]
        if residual_arr.size == 0:
            residual_arr = None

    if residual_arr is None:
        residual_arr = fit_residual_uncertainty(y_true_arr, y_pred_arr)
        used_fallback = True
        warnings_list.append(
            "No se encontraron residuos de calibracion. "
            "Usando residuos del conjunto de prueba como aproximacion. "
            "Las probabilidades pueden diferir ligeramente de una calibracion dedicada."
        )

    predictions = compute_early_warning_predictions(
        y_true_arr,
        y_pred_arr,
        residual_arr,
        config,
        row_index=row_index,
        target=target,
    )

    metrics = compute_early_warning_metrics(predictions)
    metrics.update(
        {
            "calibration_rows": 0 if used_fallback else int(len(residual_arr)),
            "residual_count": int(len(residual_arr)),
            "used_fallback_residuals": used_fallback,
        }
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "residuals": residual_arr,
        "used_fallback_residuals": used_fallback,
        "warnings": warnings_list,
    }


def spec_description(config: dict[str, Any]) -> str:
    spec = config.get("spec") or {}
    spec_type = _spec_type(spec)
    if spec_type == "two_sided":
        return f"{spec.get('lower')} - {spec.get('upper')}"
    if spec_type == "lower_only":
        return f">= {spec.get('lower')}"
    if spec_type == "upper_only":
        return f"<= {spec.get('upper')}"
    return spec_type
