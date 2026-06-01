import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline.early_warning import (
    alert_thresholds,
    build_danger_zone_intervals,
    load_quality_specs,
    resolve_quality_spec,
    spec_description,
    threshold_analysis,
)


DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)


def _format_intervals(intervals):
    return ", ".join(f"[{low:g}, {high:g}]" for low, high in intervals)


def _metric_value(metrics, key, default=0):
    value = metrics.get(key, default) if metrics else default
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    return value


def _early_warning_status(result, target, specs):
    _, spec = resolve_quality_spec(target, specs)
    predictions = result.get("early_warning_predictions")
    if predictions is None:
        predictions = pd.DataFrame()

    if spec is None:
        return "No spec", spec, predictions
    if not predictions.empty:
        return "Ready", spec, predictions
    if result.get("early_warning_error"):
        return "Error", spec, predictions
    return "Legacy run", spec, predictions


def _summary_rows(target_results, specs):
    rows = []
    for target, result in target_results.items():
        if (result.get("config") or {}).get("task") != "regression":
            continue
        status, spec, predictions = _early_warning_status(result, target, specs)
        metrics = result.get("early_warning_metrics") or {}
        rows.append(
            {
                "Target": target,
                "Status": status,
                "Spec": "-" if spec is None else spec_description(spec),
                "Rows": int(len(predictions)),
                "Alerts": int(_metric_value(metrics, "alert_count", 0)),
                "Events": int(_metric_value(metrics, "event_count", 0)),
                "TP": int(_metric_value(metrics, "true_positives", 0)),
                "FP": int(_metric_value(metrics, "false_positives", 0)),
                "FN": int(_metric_value(metrics, "false_negatives", 0)),
                "False alerts/batch": float(_metric_value(metrics, "false_alerts_per_batch", 0.0)),
                "Precision": float(_metric_value(metrics, "precision", 0.0)),
                "Recall": float(_metric_value(metrics, "recall", 0.0)),
                "F1": float(_metric_value(metrics, "f1", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _show_confusion_matrix(metrics, title):
    matrix = np.asarray(metrics.get("confusion_matrix") or [[0, 0], [0, 0]])
    text = [[str(int(value)) for value in row] for row in matrix]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Pred: No alert", "Pred: Alert"],
            y=["Actual: No event", "Actual: Event"],
            text=text,
            texttemplate="%{text}",
            colorscale="Reds",
        )
    )
    fig.update_layout(**DARK, title=title, height=390)
    st.plotly_chart(fig, use_container_width=True)


st.markdown("# 🚨 Early Warning")
st.caption("Risk-based quality alerts using danger zones, residual uncertainty, and calibrated probabilities.")

if st.session_state.automl_run is None:
    st.warning("Entrena o carga una corrida AutoML primero.")
    st.stop()

run = st.session_state.automl_run
target_results = run.get("target_results") or {}
specs = load_quality_specs()
regression_targets = [
    target
    for target, result in target_results.items()
    if (result.get("config") or {}).get("task") == "regression"
]

if not regression_targets:
    st.info("No hay targets de regresion disponibles para Early Warning.")
    st.stop()

if not specs:
    st.warning("No se encontro `config/quality_specs.json`. Agrega especificaciones para activar Early Warning.")
    st.stop()

summary = _summary_rows(target_results, specs)

st.markdown("### Overview")
if summary.empty:
    st.info("No hay targets de regresion con resultados disponibles.")
    st.stop()

st.dataframe(summary.round(4), use_container_width=True, hide_index=True)

target_options = ["All regression targets", *regression_targets]
selected_target = st.selectbox("Target", target_options)

if selected_target == "All regression targets":
    ready = summary.loc[summary["Status"] == "Ready"]
    if ready.empty:
        st.info("No hay artefactos Early Warning listos. Reentrena con calibracion y especificaciones configuradas.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Targets ready", f"{len(ready):,}")
        c2.metric("Total alerts", f"{int(ready['Alerts'].sum()):,}")
        c3.metric("Total events", f"{int(ready['Events'].sum()):,}")
        c4.metric("False negatives", f"{int(ready['FN'].sum()):,}")
    st.stop()

result = target_results[selected_target]
status, spec, predictions = _early_warning_status(result, selected_target, specs)
metrics = result.get("early_warning_metrics") or {}

st.divider()
st.markdown(f"### {selected_target}")

if spec is None:
    st.warning("Este target no tiene especificacion en `config/quality_specs.json`.")
    st.stop()

try:
    dz_intervals = build_danger_zone_intervals(spec)
    thresholds = alert_thresholds(spec)
except Exception as exc:
    st.error(f"La especificacion configurada no es valida: {exc}")
    st.stop()

s1, s2, s3, s4 = st.columns(4)
s1.metric("Specification", spec_description(spec))
s2.metric("Danger zone", _format_intervals(dz_intervals))
s3.metric("P(DZ) threshold", f"{thresholds['p_dz']:.0%}")
s4.metric("P(OOS) threshold", f"{thresholds['p_oos']:.0%}")

if status == "Error":
    st.error(f"No se pudieron generar los artefactos Early Warning: {result.get('early_warning_error')}")
    st.stop()

if predictions.empty:
    st.info(
        "Esta corrida no tiene residuos de calibracion ni predicciones Early Warning. "
        "Reentrena el target para generar `calibration_residuals.csv` y `early_warning_predictions.csv`."
    )
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alerts", f"{int(_metric_value(metrics, 'alert_count', 0)):,}")
c2.metric("High risk", f"{int(_metric_value(metrics, 'high_risk_count', 0)):,}")
c3.metric("Events", f"{int(_metric_value(metrics, 'event_count', 0)):,}")
c4.metric("False positives", f"{int(_metric_value(metrics, 'false_positives', 0)):,}")
c5.metric("False negatives", f"{int(_metric_value(metrics, 'false_negatives', 0)):,}")

st.metric(
    "False alerts per batch",
    f"{_metric_value(metrics, 'false_alerts_per_batch', 0.0):.3f}",
)

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Precision", f"{_metric_value(metrics, 'precision', 0.0):.3f}")
c7.metric("Recall", f"{_metric_value(metrics, 'recall', 0.0):.3f}")
c8.metric("Specificity", f"{_metric_value(metrics, 'specificity', 0.0):.3f}")
c9.metric("F1", f"{_metric_value(metrics, 'f1', 0.0):.3f}")
c10.metric("Balanced acc.", f"{_metric_value(metrics, 'balanced_accuracy', 0.0):.3f}")

c11, c12 = st.columns(2)
c11.metric("ROC-AUC", "-" if "roc_auc" not in metrics else f"{metrics['roc_auc']:.3f}")
c12.metric("PR-AUC", "-" if "pr_auc" not in metrics else f"{metrics['pr_auc']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Alert Table", "Confusion Matrix", "Threshold Analysis", "Risk Distribution"]
)

with tab1:
    interval_cols = [col for col in predictions.columns if col.startswith("pi_low_") or col.startswith("pi_high_")]
    display_cols = [
        "row_index",
        "y_true",
        "y_pred",
        *interval_cols,
        "p_dz",
        "p_oos",
        "risk_score",
        "alert",
        "alert_tier",
        "actual_in_dz",
        "actual_oos",
        "actual_event",
    ]
    display_cols = [col for col in display_cols if col in predictions.columns]
    st.dataframe(predictions[display_cols].round(5), use_container_width=True, hide_index=True)

with tab2:
    _show_confusion_matrix(metrics, f"Early Warning confusion matrix - {selected_target}")

with tab3:
    sweep = threshold_analysis(
        predictions,
        p_oos_threshold=thresholds["p_oos"],
        p_dz_thresholds=(0.05, 0.10, 0.20),
    )
    st.dataframe(sweep.round(4), use_container_width=True, hide_index=True)

with tab4:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=predictions["p_dz"],
            name="P(DZ)",
            opacity=0.75,
            marker_color="#f97316",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=predictions["p_oos"],
            name="P(OOS)",
            opacity=0.65,
            marker_color="#ef4444",
        )
    )
    fig.add_vline(x=thresholds["p_dz"], line=dict(color="#f97316", dash="dash"))
    fig.add_vline(x=thresholds["p_oos"], line=dict(color="#ef4444", dash="dot"))
    fig.update_layout(
        **DARK,
        title=f"Predictive risk distribution - {selected_target}",
        xaxis_title="Probability",
        yaxis_title="Batch count",
        barmode="overlay",
        height=390,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("Early Warning artifacts", expanded=False):
    rows = [
        {"Artifact": "quality_specs", "Path": "config/quality_specs.json"},
        {"Artifact": "calibration_residuals", "Path": result.get("calibration_residuals_path")},
        {"Artifact": "early_warning_predictions", "Path": result.get("early_warning_predictions_path")},
        {"Artifact": "early_warning_metrics", "Path": result.get("early_warning_metrics_path")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
