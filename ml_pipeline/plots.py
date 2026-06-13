"""Evaluation plot generation and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve


DARK = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font_color="#1e293b",
)


def _write(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path)


def save_regression_plots(out_dir: str | Path, target: str, y_true, y_pred) -> dict:
    out = Path(out_dir)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    min_v = min(np.nanmin(y_true), np.nanmin(y_pred))
    max_v = max(np.nanmax(y_true), np.nanmax(y_pred))

    actual = go.Figure()
    actual.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color="#5b6af0", size=5, opacity=0.65),
            name="Predicciones",
        )
    )
    actual.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            line=dict(color="#2dd4bf", dash="dash"),
            name="Perfecto",
        )
    )
    actual.update_layout(**DARK, title=f"Real vs predicho - {target}", xaxis_title="Real", yaxis_title="Predicho")

    residual = go.Figure(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color="#f59e0b", size=5, opacity=0.65),
        )
    )
    residual.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"))
    residual.update_layout(**DARK, title=f"Residuos - {target}", xaxis_title="Predicho", yaxis_title="Residuo")

    hist = px.histogram(
        x=residuals,
        nbins=40,
        color_discrete_sequence=["#5b6af0"],
        title=f"Distribución de residuos - {target}",
        labels={"x": "Residuo"},
    )
    hist.update_layout(**DARK)

    return {
        "actual_vs_pred": _write(actual, out / "actual_vs_pred.html"),
        "residuals": _write(residual, out / "residuals.html"),
        "error_distribution": _write(hist, out / "error_distribution.html"),
    }


def save_classification_plots(out_dir: str | Path, target: str, y_true, y_pred, proba=None) -> dict:
    out = Path(out_dir)
    labels = pd.unique(
        pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True).dropna()
    ).tolist()
    matrix = pd.crosstab(
        pd.Series(y_true, name="Real"),
        pd.Series(y_pred, name="Predicho"),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    cm = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=[f"Pred: {label}" for label in matrix.columns],
            y=[f"Real: {label}" for label in matrix.index],
            text=matrix.values,
            texttemplate="%{text}",
            colorscale=[[0, 'white'], [1, '#3b82f6']],
        )
    )
    cm.update_layout(**DARK, title=f"Matriz de confusión - {target}")
    paths = {"confusion_matrix": _write(cm, out / "confusion_matrix.html")}

    if proba is not None and len(labels) == 2:
        try:
            prob_pos = np.asarray(proba)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, prob_pos)
            precision, recall, _ = precision_recall_curve(y_true, prob_pos)

            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy", line=dict(color="#5b6af0")))
            roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#64748b"))
            roc.update_layout(**DARK, title=f"ROC - {target}", xaxis_title="FPR", yaxis_title="TPR")

            pr = go.Figure()
            pr.add_trace(go.Scatter(x=recall, y=precision, fill="tozeroy", line=dict(color="#2dd4bf")))
            pr.update_layout(**DARK, title=f"Precision-Recall - {target}", xaxis_title="Recall", yaxis_title="Precision")

            paths["roc_curve"] = _write(roc, out / "roc_curve.html")
            paths["precision_recall"] = _write(pr, out / "precision_recall.html")
        except Exception:
            pass

    return paths


def save_evaluation_plots(task: str, out_dir: str | Path, target: str, y_true, y_pred, proba=None) -> dict:
    if task == "regression":
        return save_regression_plots(out_dir, target, y_true, y_pred)
    return save_classification_plots(out_dir, target, y_true, y_pred, proba)
