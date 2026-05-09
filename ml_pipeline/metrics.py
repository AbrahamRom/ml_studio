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


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = None
    if not np.isnan(denom).all():
        mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape": mape,
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


def compute_holdout_metrics(task: str, y_true, y_pred, proba=None) -> dict:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred, proba)
