"""Holdout metrics for per-target AutoML results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = None
    if not np.isnan(denom).all():
        mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    adjusted_r2 = None
    if n_features is not None and len(y_true) > (n_features + 1):
        adjusted_r2 = float(1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1))
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(smape_denom == 0, 0.0, np.abs(y_true - y_pred) / smape_denom)) * 200)
    scale = float(np.nanmean(np.abs(y_true)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(y_true))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    def _bounded_score(value: float | None, *, scale_value: float | None = None, upper_bound: float | None = None) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        if scale_value is not None:
            return float(1.0 / (1.0 + (value / scale_value)))
        if upper_bound is not None:
            return float(np.clip((value + upper_bound) / (2.0 * upper_bound), 0.0, 1.0))
        return float(np.clip(value, 0.0, 1.0))

    component_scores = {
        "r2_adjusted": _bounded_score(adjusted_r2 if adjusted_r2 is not None else r2, upper_bound=1.0),
        "rmse": _bounded_score(float(np.sqrt(mse)), scale_value=scale),
        "mae": _bounded_score(float(mean_absolute_error(y_true, y_pred)), scale_value=scale),
        "smape": _bounded_score(smape, scale_value=100.0),
        "mape": _bounded_score(mape, scale_value=100.0) if mape is not None else None,
    }
    weights = {
        "r2_adjusted": 0.40,
        "rmse": 0.25,
        "mae": 0.15,
        "smape": 0.15,
        "mape": 0.05,
    }
    present_scores = {
        name: score for name, score in component_scores.items() if score is not None
    }
    total_weight = sum(weights[name] for name in present_scores)
    global_score = None
    if total_weight > 0:
        global_score = float(
            sum(weights[name] * score for name, score in present_scores.items()) / total_weight
        )
    return {
        "r2": r2,
        "r2_adjusted": adjusted_r2,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape": mape,
        "smape": smape,
        "score_global": global_score,
    }


def classification_metrics(y_true, y_pred, proba=None) -> dict:
    labels = pd.unique(
        pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True).dropna()
    ).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classes": labels,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if proba is not None:
        try:
            proba_arr = np.asarray(proba)
            if len(labels) == 2 and proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_arr[:, 1]))
            elif len(labels) > 2 and proba_arr.ndim == 2 and proba_arr.shape[1] == len(labels):
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, proba_arr, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass
    return metrics


def compute_holdout_metrics(task: str, y_true, y_pred, proba=None, n_features: int | None = None) -> dict:
    if task == "regression":
        return regression_metrics(y_true, y_pred, n_features=n_features)
    return classification_metrics(y_true, y_pred, proba)
