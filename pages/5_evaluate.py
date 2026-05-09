import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font_color="#e2e8f0",
    gridcolor="#252a38",
)

st.markdown("# 🔬 Evaluación de Modelos")

if st.session_state.automl_run is None:
    st.warning("⚠️ Entrena AutoML primero.")
    st.stop()

run = st.session_state.automl_run
target_results = run["target_results"]
targets = list(target_results.keys())

target = st.selectbox("Target a evaluar", targets)
result = target_results[target]
config = result["config"]
metrics = result["holdout_metrics"]
y_true = result["y_test"].to_numpy()
y_pred = result["predictions"].to_numpy()
proba = result.get("proba")

st.markdown(
    f"**Target:** `{target}` · **Tarea:** `{config['ml_task']}` · "
    f"**Mejor modelo:** `{result.get('best_model_name')}`"
)
st.caption(f"Reporte mljar: `{result['results_path']}`")
st.divider()

if config["task"] == "classification":
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

    classes = metrics.get("classes") or pd.unique(pd.Series(y_true)).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    c2.metric("F1 weighted", f"{metrics.get('f1', 0):.4f}")
    c3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    c4.metric("Recall", f"{metrics.get('recall', 0):.4f}")

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
                colorscale="Blues",
            )
        )
        fig.update_layout(**DARK, title=f"Matriz de confusión - {target}", height=430)
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
        preview = result["X_test"].copy().reset_index(drop=True)
        preview["y_real"] = y_true
        preview["y_pred"] = y_pred
        preview["correcto"] = preview["y_real"] == preview["y_pred"]
        st.dataframe(preview.head(100), use_container_width=True)

else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{metrics.get('r2', 0):.4f}")
    c2.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    c3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    c4.metric("MAPE", "-" if metrics.get("mape") is None else f"{metrics['mape']:.2f}%")

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
        fig.update_layout(**DARK, title=f"Real vs predicho - {target}", height=440)
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
        preview = result["X_test"].copy().reset_index(drop=True).head(100)
        preview[f"{target}_real"] = y_true[:100]
        preview[f"{target}_pred"] = np.round(y_pred[:100], 4)
        preview["error"] = np.round(y_true[:100] - y_pred[:100], 4)
        st.dataframe(preview, use_container_width=True)

st.divider()
with st.expander("Archivos guardados para este target", expanded=False):
    rows = [{"Artefacto": name, "Ruta": path} for name, path in result["plot_paths"].items()]
    rows.extend(
        [
            {"Artefacto": "leaderboard", "Ruta": result["leaderboard_path"]},
            {"Artefacto": "predicciones", "Ruta": result["predictions_path"]},
            {"Artefacto": "métricas", "Ruta": result["metrics_path"]},
        ]
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
