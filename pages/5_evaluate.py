import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ml_pipeline.automl_runner import predict_with_model
from ml_pipeline.early_warning import load_column_display_names, load_quality_specs, resolve_display_name

DARK = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font={"color": "#1e293b"},
)

st.markdown("# 🔬 Evaluación de Modelos")

if st.session_state.automl_run is None:
    st.warning("⚠️ Entrena AutoML primero.")
    st.stop()

run = st.session_state.automl_run
target_results = run["target_results"]
targets = list(target_results.keys())
_specs = load_quality_specs()
_col_names = load_column_display_names()

target = st.selectbox("Target a evaluar", targets)

with st.expander("🌍 Global Normalized Observed vs Predicted", expanded=False):
    norm_dfs = []
    for tgt, res in target_results.items():
        cfg = res.get("config", {})
        if cfg.get("task") != "regression":
            continue

        y_t = res.get("y_test")
        y_p = res.get("predictions")
        if y_t is None or y_p is None:
            continue

        qs = res.get("quality_spec")
        if qs is None:
            continue

        spec = qs.get("spec", {})
        stype = spec.get("type", "two_sided")

        if stype == "two_sided":
            lower = spec.get("lower")
            upper = spec.get("upper")
            if lower is None or upper is None or upper <= lower:
                continue
            lsl, usl = float(lower), float(upper)

        elif stype == "upper_only":
            upper = spec.get("upper")
            if upper is None or float(upper) <= 0:
                continue
            lsl, usl = 0.0, float(upper)

        elif stype == "lower_only":
            lower = spec.get("lower")
            if lower is None or float(lower) <= 0:
                continue
            lsl = float(lower)
            unit = qs.get("unit", "")
            tgt_lower = tgt.lower()
            if unit == "%" or "percent" in tgt_lower:
                usl = 100.0
            else:
                usl = 2.0 * lsl

        else:
            continue

        y_t_a = np.asarray(y_t, dtype=float)
        y_p_a = np.asarray(y_p, dtype=float)
        y_t_n = (y_t_a - lsl) / (usl - lsl)
        y_p_n = (y_p_a - lsl) / (usl - lsl)

        norm_dfs.append(
            pd.DataFrame({
                "observed_norm": y_t_n,
                "predicted_norm": y_p_n,
                "target": tgt,
            })
        )

    if norm_dfs:
        combined = pd.concat(norm_dfs, ignore_index=True)

        targets_ordered = combined["target"].unique()
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet
        color_map = {t: palette[i % len(palette)] for i, t in enumerate(targets_ordered)}

        fig = go.Figure()
        for tgt in targets_ordered:
            mask = combined["target"] == tgt
            fig.add_trace(
                go.Scatter(
                    x=combined.loc[mask, "observed_norm"],
                    y=combined.loc[mask, "predicted_norm"],
                    mode="markers",
                    marker=dict(color=color_map[tgt], size=5, opacity=0.5),
                    name=resolve_display_name(tgt, _specs, _col_names),
                    hovertemplate="<b>%{text}</b><br>Obs: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>",
                    text=[tgt] * mask.sum(),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#2dd4bf", dash="dash", width=2),
                name="Perfecto",
            )
        )
        fig.update_layout(
            **DARK,
            title="Observed vs Predicted (normalizado por spec por target)",
            xaxis_title="Observado normalizado",
            yaxis_title="Predicho normalizado",
            height=500,
        )
        fig.update_xaxes(range=[-0.05, 1.05])
        fig.update_yaxes(range=[-0.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        total_points = len(combined)
        n_targets = combined["target"].nunique()
        st.caption(f"{total_points:,} puntos · {n_targets} targets")
    else:
        st.info("No hay targets de regresión con spec válida para el gráfico normalizado.")

result = target_results[target]
config = result["config"]


def _resolve_saved_predictions(result):
    prediction_frame = result.get("prediction_frame")
    if prediction_frame is not None and prediction_frame.empty:
        prediction_frame = None

    y_true = result.get("y_test")
    y_pred = result.get("predictions")
    proba = result.get("proba")
    row_index = None

    if prediction_frame is not None:
        if "row_index" in prediction_frame.columns:
            row_index = prediction_frame["row_index"]
        if y_true is None and "y_true" in prediction_frame.columns:
            y_true = prediction_frame["y_true"]
        if y_pred is None and "y_pred" in prediction_frame.columns:
            y_pred = prediction_frame["y_pred"]
        if proba is None:
            proba_cols = [col for col in prediction_frame.columns if col.startswith("proba_")]
            if proba_cols:
                proba = prediction_frame[proba_cols].to_numpy()

    return y_true, y_pred, proba, row_index, prediction_frame


y_true, y_pred, proba, row_index, prediction_frame = _resolve_saved_predictions(result)
best_model_name = result.get("best_model_name")
selected_metrics = result["holdout_metrics"]

per_model_metrics = result.get("per_model_metrics")
if per_model_metrics is not None and best_model_name and "model_name" in per_model_metrics.columns:
    match = per_model_metrics.loc[per_model_metrics["model_name"] == best_model_name]
    if not match.empty:
        selected_metrics = match.iloc[0].to_dict()

selected_model_name = best_model_name
if result.get("automl") is not None and result.get("X_test") is not None:
    model_pred, model_proba, selected_model_name = predict_with_model(
        result.get("automl"),
        best_model_name,
        result["X_test"],
        config["task"],
    )
    if model_pred is not None:
        y_pred = model_pred
        proba = model_proba
        y_true = result.get("y_test")

if y_pred is None and result.get("predictions") is not None:
    y_pred = result.get("predictions")
if y_true is None and result.get("y_test") is not None:
    y_true = result.get("y_test")

if y_true is None or y_pred is None:
    st.error("No se encontraron predicciones guardadas para este target.")
    st.stop()

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

metrics = selected_metrics

_target_dn = resolve_display_name(target, _specs, _col_names)

st.markdown(
    f"**Target:** `{_target_dn}` · **Tarea:** `{config['ml_task']}` · "
    f"**Mejor modelo según holdout real:** `{selected_model_name}`"
)
st.caption(f"Reporte mljar: `{result['results_path']}`")
st.divider()

if config["task"] == "classification":
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

    classes = metrics.get("classes") or pd.unique(pd.Series(y_true)).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    c2.metric("F1 weighted", f"{metrics.get('f1', 0):.4f}")
    c3.metric("F3", f"{metrics.get('f3', 0):.4f}")
    c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    c5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔲 Matriz", "📋 Reporte", "📈 Curvas", "🎯 Predicciones"]
    )

    with tab1:
        normalize_cm = st.checkbox("Normalizar por fila", value=False)
        cm_plot = cm.astype(float)
        if normalize_cm:
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)
            text = [[f"{value:.1%}" for value in row] for row in cm_plot]
        else:
            text = [[str(int(value)) for value in row] for row in cm_plot]
        fig = go.Figure(
            go.Heatmap(
                z=cm_plot,
                x=[f"Pred: {value}" for value in classes],
                y=[f"Real: {value}" for value in classes],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, 'white'], [1, '#3b82f6']],
            )
        )
        fig.update_layout(**DARK, title=f"Matriz de confusión - {_target_dn}", height=430)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        report = pd.DataFrame(metrics["classification_report"]).T.round(4)
        st.dataframe(report, use_container_width=True)

    with tab3:
        if proba is None:
            st.info("El mejor modelo no expuso probabilidades para curvas ROC/PR.")
        elif len(classes) == 2:
            try:
                prob_pos = np.asarray(proba)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, prob_pos)
                precision, recall, _ = precision_recall_curve(y_true, prob_pos)
                roc_auc = auc(fpr, tpr)
                col_roc, col_pr = st.columns(2)
                with col_roc:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            fill="tozeroy",
                            name=f"AUC = {roc_auc:.3f}",
                            line=dict(color="#5b6af0"),
                        )
                    )
                    fig_roc.add_shape(
                        type="line",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(dash="dash", color="#64748b"),
                    )
                    fig_roc.update_layout(**DARK, title="ROC", height=380)
                    st.plotly_chart(fig_roc, use_container_width=True)
                with col_pr:
                    fig_pr = go.Figure(
                        go.Scatter(x=recall, y=precision, fill="tozeroy", line=dict(color="#2dd4bf"))
                    )
                    fig_pr.update_layout(**DARK, title="Precision-Recall", height=380)
                    st.plotly_chart(fig_pr, use_container_width=True)
            except Exception as exc:
                st.warning(f"No se pudieron calcular las curvas: {exc}")
        else:
            st.info("Las curvas ROC/PR interactivas se muestran solo para clasificación binaria.")

    with tab4:
        if result.get("X_test") is not None:
            preview = result["X_test"].copy().reset_index(drop=True)
        elif st.session_state.df is not None and row_index is not None:
            preview = st.session_state.df.reindex(row_index).reset_index(drop=True)
        elif prediction_frame is not None:
            preview = prediction_frame.copy().reset_index(drop=True)
        else:
            preview = pd.DataFrame()

        if not preview.empty:
            preview["y_real"] = y_true
            preview["y_pred"] = y_pred
            preview["correcto"] = preview["y_real"] == preview["y_pred"]
            st.dataframe(preview.head(100), use_container_width=True)
        else:
            st.info("No hay features guardadas; se muestran solo las predicciones.")
            st.dataframe(
                pd.DataFrame({"y_real": y_true, "y_pred": y_pred}).head(100),
                use_container_width=True,
            )

else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Score global",
        "-" if metrics.get("score_global") is None else f"{metrics['score_global']:.4f}",
    )
    c2.metric(
        "R² ajustado",
        "-" if metrics.get("r2_adjusted") is None else f"{metrics['r2_adjusted']:.4f}",
    )
    c3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    c4.metric("SMAPE", f"{metrics.get('smape', 0):.2f}%")

    c5, c6 = st.columns(2)
    c5.metric("R²", f"{metrics.get('r2', 0):.4f}")
    c6.metric("MAE", f"{metrics.get('mae', 0):.4f}")

    if metrics.get("mape") is not None:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")

    residuals = y_true - y_pred
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Real vs Pred", "📉 Residuos", "📊 Distribución Error", "🎯 Predicciones"]
    )

    with tab1:
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker=dict(color="#5b6af0", size=5, opacity=0.65),
                name="Predicciones",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                line=dict(color="#2dd4bf", dash="dash", width=2),
                name="Perfecto",
            )
        )
        fig.update_layout(**DARK, title=f"Real vs predicho - {_target_dn}", height=440)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Residuos vs predicho", "Residuos vs índice"))
        fig2.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(color="#5b6af0", size=4, opacity=0.65),
            ),
            row=1,
            col=1,
        )
        fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=1)
        fig2.add_trace(
            go.Scatter(
                x=list(range(len(residuals))),
                y=residuals,
                mode="markers",
                marker=dict(color="#f59e0b", size=4, opacity=0.65),
            ),
            row=1,
            col=2,
        )
        fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=2)
        fig2.update_layout(**DARK, height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.histogram(
            x=residuals,
            nbins=50,
            color_discrete_sequence=["#5b6af0"],
            labels={"x": "Error"},
            title="Distribución de residuos",
        )
        fig3.update_layout(**DARK, height=360)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        if result.get("X_test") is not None:
            preview = result["X_test"].copy().reset_index(drop=True).head(100)
        elif st.session_state.df is not None and row_index is not None:
            preview = st.session_state.df.reindex(row_index).reset_index(drop=True).head(100)
        elif prediction_frame is not None:
            preview = prediction_frame.copy().reset_index(drop=True).head(100)
        else:
            preview = pd.DataFrame()

        if not preview.empty:
            preview[f"{target}_real"] = y_true[: len(preview)]
            preview[f"{target}_pred"] = np.round(y_pred[: len(preview)], 4)
            preview["error"] = np.round(y_true[: len(preview)] - y_pred[: len(preview)], 4)
            st.dataframe(preview, use_container_width=True)
        else:
            st.info("No hay features guardadas; se muestran solo las predicciones.")
            st.dataframe(
                pd.DataFrame(
                    {
                        f"{target}_real": y_true[:100],
                        f"{target}_pred": np.round(y_pred[:100], 4),
                        "error": np.round(y_true[:100] - y_pred[:100], 4),
                    }
                ),
                use_container_width=True,
            )

st.divider()
with st.expander("Archivos guardados para este target", expanded=False):
    plot_paths = result.get("plot_paths") or {}
    rows = [{"Artefacto": name, "Ruta": path} for name, path in plot_paths.items()]
    rows.extend(
        [
            {"Artefacto": "leaderboard", "Ruta": result["leaderboard_path"]},
            {"Artefacto": "predicciones", "Ruta": result["predictions_path"]},
            {"Artefacto": "métricas", "Ruta": result["metrics_path"]},
        ]
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Global export button
# ---------------------------------------------------------------------------
st.divider()


def _build_evaluate_report(run, target, result, config, metrics, y_true, y_pred, proba, per_model_metrics, best_model_name):
    report = {
        "project_info": {
            "run_id": run["run_id"],
            "timestamp": run.get("timestamp") or datetime.now().isoformat(),
            "description": run.get("description", ""),
            "base_path": str(run.get("base_path", "")),
        },
        "target": target,
        "config": dict(config),
        "best_model": {
            "name": best_model_name,
            "type": result.get("best_model_type"),
        },
    }

    report["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if config["task"] == "classification":
        classes = metrics.get("classes") or sorted(pd.unique(pd.Series(y_t))).tolist()
        report["classification_report"] = metrics.get("classification_report")
        report["classes"] = classes
        report["confusion_matrix"] = metrics.get("confusion_matrix")
        if proba is not None:
            report["probability_shape"] = list(np.asarray(proba).shape)
    else:
        residuals = y_t - y_p
        report["residuals"] = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "percentile_25": float(np.percentile(residuals, 25)),
            "percentile_75": float(np.percentile(residuals, 75)),
            "mean_abs": float(np.mean(np.abs(residuals))),
        }
        report["predictions_summary"] = {
            "y_true": {"mean": float(np.mean(y_t)), "std": float(np.std(y_t)), "min": float(np.min(y_t)), "max": float(np.max(y_t))},
            "y_pred": {"mean": float(np.mean(y_p)), "std": float(np.std(y_p)), "min": float(np.min(y_p)), "max": float(np.max(y_p))},
            "n": int(len(y_t)),
        }

    report["artifacts"] = {
        "mljar_report": result.get("results_path"),
        "leaderboard": result.get("leaderboard_path"),
        "predictions": result.get("predictions_path"),
        "metrics": result.get("metrics_path"),
    }

    return report


report_data = _build_evaluate_report(
    run, target, result, config, metrics, y_true, y_pred, proba,
    per_model_metrics, selected_model_name,
)
report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
st.download_button(
    label="📥 Descargar reporte de Evaluación (JSON)",
    data=report_json,
    file_name="evaluation_report.json",
    mime="application/json",
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Global report for all targets
# ---------------------------------------------------------------------------
def _build_global_evaluate_report(run, target_results):
    targets_data = {}
    for tgt, res in target_results.items():
        cfg = res.get("config", {})
        m = res.get("holdout_metrics") or {}
        y_t = res.get("y_test")
        y_p = res.get("predictions")
        proba = res.get("proba")

        entry = {
            "config": dict(cfg),
            "best_model": {
                "name": res.get("best_model_name"),
                "type": res.get("best_model_type"),
            },
            "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float, str))},
        }

        if y_t is not None and y_p is not None:
            y_t_a = np.asarray(y_t)
            y_p_a = np.asarray(y_p)
            if cfg.get("task") == "classification":
                classes = m.get("classes") or sorted(pd.unique(pd.Series(y_t_a))).tolist()
                entry["classification_report"] = m.get("classification_report")
                entry["classes"] = classes
                entry["confusion_matrix"] = m.get("confusion_matrix")
                if proba is not None:
                    entry["probability_shape"] = list(np.asarray(proba).shape)
            else:
                residuals = y_t_a - y_p_a
                entry["residuals"] = {
                    "mean": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                    "min": float(np.min(residuals)),
                    "max": float(np.max(residuals)),
                    "percentile_25": float(np.percentile(residuals, 25)),
                    "percentile_75": float(np.percentile(residuals, 75)),
                    "mean_abs": float(np.mean(np.abs(residuals))),
                }
                entry["predictions_summary"] = {
                    "y_true": {"mean": float(np.mean(y_t_a)), "std": float(np.std(y_t_a)), "min": float(np.min(y_t_a)), "max": float(np.max(y_t_a))},
                    "y_pred": {"mean": float(np.mean(y_p_a)), "std": float(np.std(y_p_a)), "min": float(np.min(y_p_a)), "max": float(np.max(y_p_a))},
                    "n": int(len(y_t_a)),
                }

        entry["artifacts"] = {
            "leaderboard": res.get("leaderboard_path"),
            "predictions": res.get("predictions_path"),
            "metrics": res.get("metrics_path"),
        }
        targets_data[tgt] = entry

    report = {
        "project_info": {
            "run_id": run["run_id"],
            "timestamp": run.get("timestamp") or datetime.now().isoformat(),
            "description": run.get("description", ""),
            "base_path": str(run.get("base_path", "")),
        },
        "targets": targets_data,
        "target_count": len(target_results),
    }
    return report


global_report_data = _build_global_evaluate_report(run, target_results)
global_report_json = json.dumps(global_report_data, indent=2, ensure_ascii=False)
st.download_button(
    label="📥 Descargar reporte global de todos los targets (JSON)",
    data=global_report_json,
    file_name="global_evaluation_report.json",
    mime="application/json",
    use_container_width=True,
)
