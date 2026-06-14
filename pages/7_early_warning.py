from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import json

from ml_pipeline.artifacts import save_dataframe, save_json, target_dir as _target_dir
from ml_pipeline.early_warning import (
    alert_thresholds,
    build_danger_zone_intervals,
    compute_alert_metrics,
    load_quality_specs,
    recompute_early_warning_for_target,
    resolve_quality_spec,
    spec_description,
    threshold_analysis,
)
from ml_pipeline.early_warning_report import generate_early_warning_report


DARK = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font={"color": "#1e293b"},
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
            colorscale=[[0, 'white'], [1, '#3b82f6']],
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
    ready = summary.loc[summary["Status"].isin(["Ready", "Legacy run"])]
    if ready.empty:
        st.info("No hay artefactos Early Warning listos. Reentrena con calibracion y especificaciones configuradas.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Targets available", f"{len(ready):,}")
    c2.metric("Total alerts", f"{int(ready['Alerts'].sum()):,}")
    c3.metric("Total events", f"{int(ready['Events'].sum()):,}")
    c4.metric("False negatives", f"{int(ready['FN'].sum()):,}")

    # --- Export report button ---
    if not ready.empty:
        report_data = generate_early_warning_report(
            target_results, specs, run.get("base_path", "")
        )
        report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Descargar reporte Early Warning (JSON)",
            data=report_json,
            file_name="early_warning_report.json",
            mime="application/json",
            use_container_width=True,
        )

    legacy = summary.loc[summary["Status"] == "Legacy run"]
    if not legacy.empty:
        st.info(
            f"{len(legacy)} target(s) en estado 'Legacy run'. "
            "Seleccionalos individualmente y presiona **Recalcular Early Warning** para generar las predicciones."
        )

    # --- Aggregated confusion matrices ---
    st.markdown("---")
    st.markdown("### Matrices de Confusion Agregadas")

    all_alerts = []
    all_actual_events = []
    all_dz_alerts = []
    all_actual_in_dz = []
    all_oos_alerts = []
    all_actual_oos = []

    for target_name in regression_targets:
        target_res = target_results.get(target_name)
        if target_res is None:
            continue
        preds = target_res.get("early_warning_predictions")
        if preds is None or preds.empty:
            continue
        status_str, target_spec, _ = _early_warning_status(target_res, target_name, specs)
        if status_str not in ("Ready", "Legacy run"):
            continue
        if target_spec is None:
            continue
        try:
            target_thresholds = alert_thresholds(target_spec)
        except Exception:
            continue

        p_dz_threshold = target_thresholds["p_dz"]
        p_oos_threshold = target_thresholds["p_oos"]

        if "alert" in preds.columns and "actual_event" in preds.columns:
            all_alerts.append(preds["alert"].astype(bool))
            all_actual_events.append(preds["actual_event"].astype(bool))

        if "p_dz" in preds.columns and "actual_in_dz" in preds.columns:
            all_dz_alerts.append(preds["p_dz"] >= p_dz_threshold)
            all_actual_in_dz.append(preds["actual_in_dz"].astype(bool))

        if "p_oos" in preds.columns and "actual_oos" in preds.columns:
            all_oos_alerts.append(preds["p_oos"] >= p_oos_threshold)
            all_actual_oos.append(preds["actual_oos"].astype(bool))

    if not all_alerts and not all_dz_alerts and not all_oos_alerts:
        st.info("No hay predicciones disponibles para generar matrices de confusion agregadas.")
        st.stop()

    cm_col1, cm_col2, cm_col3 = st.columns(3)

    with cm_col1:
        if all_alerts:
            combined_alerts = pd.concat(all_alerts, ignore_index=True)
            combined_events = pd.concat(all_actual_events, ignore_index=True)
            general_metrics = compute_alert_metrics(
                combined_events.to_numpy(),
                combined_alerts.to_numpy(),
                None,
            )
            _show_confusion_matrix(
                general_metrics,
                "Alertas Generales (alert vs actual_event)",
            )
        else:
            st.info("Sin datos para alertas generales.")

    with cm_col2:
        if all_dz_alerts:
            combined_dz_alert = pd.concat(all_dz_alerts, ignore_index=True)
            combined_dz_event = pd.concat(all_actual_in_dz, ignore_index=True)
            dz_metrics = compute_alert_metrics(
                combined_dz_event.to_numpy(),
                combined_dz_alert.to_numpy(),
                None,
            )
            _show_confusion_matrix(
                dz_metrics,
                "Zona de Peligro (p_dz ≥ threshold vs actual_in_dz)",
            )
        else:
            st.info("Sin datos para zona de peligro.")

    with cm_col3:
        if all_oos_alerts:
            combined_oos_alert = pd.concat(all_oos_alerts, ignore_index=True)
            combined_oos_event = pd.concat(all_actual_oos, ignore_index=True)
            oos_metrics = compute_alert_metrics(
                combined_oos_event.to_numpy(),
                combined_oos_alert.to_numpy(),
                None,
            )
            _show_confusion_matrix(
                oos_metrics,
                "OOS (p_oos ≥ threshold vs actual_oos)",
            )
        else:
            st.info("Sin datos para OOS.")

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

needs_recompute = predictions.empty

if needs_recompute:
    st.warning(
        "Esta corrida no tiene predicciones Early Warning. "
        "Puedes recalcularlas sin reentrenar usando las especificaciones actuales de `quality_specs.json` "
        "y los datos de holdout ya guardados."
    )

st.markdown("---")
recalc_col1, recalc_col2 = st.columns([2, 1])
with recalc_col1:
    if needs_recompute:
        st.caption("💡 Cambia `quality_specs.json` y presiona **Recalcular** para evaluar con nuevos limites, danger zones y thresholds.")
    else:
        st.caption("💡 Cambia `quality_specs.json` y presiona **Recalcular** para re-evaluar las alertas con los nuevos parametros sin reentrenar el modelo.")
with recalc_col2:
    do_recompute = st.button(
        "🔄 Recalcular Early Warning",
        use_container_width=True,
        type="primary",
    )

if do_recompute:
    pred_frame = result.get("prediction_frame")
    if pred_frame is None or pred_frame.empty or "y_true" not in pred_frame or "y_pred" not in pred_frame:
        st.error("No se encontraron predicciones de holdout (`predictions.csv`). Reentrena el target.")
        st.stop()

    y_true = pred_frame["y_true"].to_numpy()
    y_pred = pred_frame["y_pred"].to_numpy()
    row_index = pred_frame["row_index"].to_numpy() if "row_index" in pred_frame.columns else None

    calibration_residuals = result.get("calibration_residuals")
    if calibration_residuals is None:
        calib_frame = result.get("calibration_residuals_frame")
        if calib_frame is not None and not calib_frame.empty and "residual" in calib_frame:
            calibration_residuals = calib_frame["residual"].to_numpy()

    with st.spinner("Recalculando Early Warning con las especificaciones actuales..."):
        try:
            ew_result = recompute_early_warning_for_target(
                y_true=y_true,
                y_pred=y_pred,
                config=spec,
                residuals=calibration_residuals,
                row_index=row_index,
                target=selected_target,
            )
        except Exception as exc:
            st.error(f"Error al recalcular: {exc}")
            st.stop()

    predictions = ew_result["predictions"]
    metrics = ew_result["metrics"]
    residuals_arr = ew_result.get("residuals")

    run_path = Path(run.get("base_path", ""))
    if run_path.exists():
        t_dir = _target_dir(run_path, selected_target)
        residual_df = pd.DataFrame(
            {
                "row_index": row_index if row_index is not None else range(len(y_true)),
                "target": selected_target,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
            }
        )
        calib_path = save_dataframe(t_dir / "calibration_residuals.csv", residual_df)
        pred_path = save_dataframe(t_dir / "early_warning_predictions.csv", predictions)
        metrics_path = save_json(t_dir / "early_warning_metrics.json", metrics)
        result["calibration_residuals_path"] = str(calib_path)
        result["early_warning_predictions_path"] = str(pred_path)
        result["early_warning_metrics_path"] = str(metrics_path)

    result["calibration_residuals"] = residuals_arr
    result["early_warning_predictions"] = predictions
    result["early_warning_metrics"] = metrics
    result["early_warning_error"] = None
    if "quality_spec_key" not in result:
        result["quality_spec_key"] = selected_target

    if ew_result.get("warnings"):
        for warning_msg in ew_result["warnings"]:
            st.warning(warning_msg)

    st.success("Early Warning recalculado exitosamente. Recargando...")
    st.rerun()

if predictions.empty:
    st.info("Presiona **Recalcular Early Warning** para generar las predicciones con los specs actuales.")
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

# --- Global export button (always available when there is data) ---
ready_global = summary.loc[summary["Status"].isin(["Ready", "Legacy run"])]
if not ready_global.empty:
    report_data_global = generate_early_warning_report(
        target_results, specs, run.get("base_path", "")
    )
    report_json_global = json.dumps(report_data_global, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Descargar reporte completo Early Warning (JSON)",
        data=report_json_global,
        file_name="early_warning_report.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("Early Warning artifacts", expanded=False):
    rows = [
        {"Artifact": "quality_specs", "Path": "config/quality_specs.json"},
        {"Artifact": "calibration_residuals", "Path": result.get("calibration_residuals_path")},
        {"Artifact": "early_warning_predictions", "Path": result.get("early_warning_predictions_path")},
        {"Artifact": "early_warning_metrics", "Path": result.get("early_warning_metrics_path")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
