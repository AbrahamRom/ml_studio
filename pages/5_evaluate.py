import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils.pagination import paginated_dataframe

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

st.markdown("# 🔬 Evaluación de Modelos")

has_automl = st.session_state.automl_run is not None
has_dl = bool(st.session_state.get("dl_results"))

if not has_automl and not has_dl:
    st.warning("⚠️ Entrena modelos primero.")
    st.stop()

# ── Model source selector ──────────────────────────────────────────────────────
source_options = []
if has_automl:
    source_options.append("AutoML")
if has_dl:
    source_options.append("Deep Learning")

model_source = st.radio("Fuente del modelo", source_options, horizontal=True, key="eval_source")

# ── AutoML evaluation ──────────────────────────────────────────────────────────
if model_source == "AutoML":
    run = st.session_state.automl_run
    target_results = run["target_results"]
    targets = list(target_results.keys())

    target = st.selectbox("Target a evaluar", targets, key="eval_aml_target")
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
            normalize_cm = st.checkbox("Normalizar por fila", value=False, key="eval_norm_cm")
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
            paginated_dataframe(report, key="eval_class_report", height=350)

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
                            go.Scatter(x=fpr, y=tpr, fill="tozeroy", name=f"AUC = {roc_auc:.3f}", line=dict(color="#5b6af0"))
                        )
                        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#64748b"))
                        fig_roc.update_layout(**DARK, title="ROC", height=380)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    with col_pr:
                        fig_pr = go.Figure(go.Scatter(x=recall, y=precision, fill="tozeroy", line=dict(color="#2dd4bf")))
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
            paginated_dataframe(preview, key="eval_pred_preview", height=400)

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score global", "-" if metrics.get("score_global") is None else f"{metrics['score_global']:.4f}")
        c2.metric("R² ajustado", "-" if metrics.get("r2_adjusted") is None else f"{metrics['r2_adjusted']:.4f}")
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
            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(color="#5b6af0", size=5, opacity=0.65), name="Predicciones"))
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#2dd4bf", dash="dash", width=2), name="Perfecto"))
            fig.update_layout(**DARK, title=f"Real vs predicho - {target}", height=440)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Residuos vs predicho", "Residuos vs índice"))
            fig2.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(color="#5b6af0", size=4, opacity=0.65)), row=1, col=1)
            fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=list(range(len(residuals))), y=residuals, mode="markers", marker=dict(color="#f59e0b", size=4, opacity=0.65)), row=1, col=2)
            fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=2)
            fig2.update_layout(**DARK, height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            fig3 = px.histogram(x=residuals, nbins=50, color_discrete_sequence=["#5b6af0"], labels={"x": "Error"}, title="Distribución de residuos")
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
        paginated_dataframe(pd.DataFrame(rows), key="eval_artifacts", height=250, hide_index=True)

# ── Deep Learning evaluation ───────────────────────────────────────────────────
else:
    dl_results = st.session_state.dl_results

    dl_options = []
    for i, r in enumerate(dl_results):
        label = f"{r['model_type'].upper()}"
        if r.get("target"):
            label += f" → {r['target']}"
        else:
            label += " (unsupervised)"
        if r.get("hpo"):
            label += " [HPO]"
        label += f" (val_rmse={r['val_rmse']:.4f})"
        dl_options.append((label, i))

    selected_label, selected_idx = st.selectbox("Modelo DL a evaluar", dl_options, format_func=lambda x: x[0], key="eval_dl_model")
    dl_result = dl_results[selected_idx]

    model_type = dl_result["model_type"]
    target = dl_result.get("target")
    val_rmse = dl_result["val_rmse"]
    train_rmse = dl_result["train_rmse"]
    history = dl_result["history"]
    config = dl_result.get("config", {})

    st.markdown(f"**Modelo:** `{model_type.upper()}` · **Target:** `{target or '(unsupervised)'}` · **HPO:** `{'✅' if dl_result.get('hpo') else '❌'}`")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train RMSE", f"{train_rmse:.4f}")
    c2.metric("Val RMSE", f"{val_rmse:.4f}")
    c3.metric("Gap", f"{abs(val_rmse - train_rmse):.4f}")
    c4.metric("Epochs", len(history.get("train_loss", [])))

    if model_type in ("autoencoder", "vae"):
        tab1, tab2, tab3 = st.tabs(["📈 Training history", "🔍 Reconstrucción", "📐 Latent space"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["train_loss"], name="Train loss", line=dict(color="#5b6af0")))
            fig.add_trace(go.Scatter(y=history["val_loss"], name="Val loss", line=dict(color="#2dd4bf")))
            fig.update_layout(title="Loss por epoch", xaxis_title="Epoch", yaxis_title="Loss", **DARK, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            train_recon = dl_result.get("train_reconstructed")
            val_recon = dl_result.get("val_reconstructed")
            if train_recon is not None:
                n_features = min(5, train_recon.shape[1])
                for i in range(n_features):
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=train_recon[:200, i], name="Reconstruido", line=dict(color="#2dd4bf"), mode="lines"))
                    df = st.session_state.df
                    feature_cols = dl_result.get("feature_cols", [])
                    if i < len(feature_cols):
                        orig = df[feature_cols[i]].values.astype(float)
                        fig2.add_trace(go.Scatter(y=orig[:200], name="Original", line=dict(color="#5b6af0", dash="dash"), mode="lines"))
                    fig2.update_layout(title=f"Feature {i}", **DARK, height=280, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            from sklearn.decomposition import PCA
            encoded = dl_result.get("train_encoded")
            if encoded is not None:
                if encoded.shape[1] > 2:
                    pca = PCA(n_components=2)
                    encoded_2d = pca.fit_transform(encoded)
                else:
                    encoded_2d = encoded[:, :2]
                fig3 = px.scatter(x=encoded_2d[:, 0], y=encoded_2d[:, 1], labels={"x": "Component 1", "y": "Component 2"}, title="Latent space (PCA 2D)", color_discrete_sequence=["#5b6af0"])
                fig3.update_layout(**DARK, height=400)
                st.plotly_chart(fig3, use_container_width=True)

    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Training history", "🔮 Predicciones vs Reales", "📊 Scatter pred vs real", "🎯 Preview"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["train_loss"], name="Train loss", line=dict(color="#5b6af0")))
            fig.add_trace(go.Scatter(y=history["val_loss"], name="Val loss", line=dict(color="#2dd4bf")))
            fig.update_layout(title="Loss por epoch", xaxis_title="Epoch", yaxis_title="Loss", **DARK, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            val_pred = dl_result.get("val_predictions", [])
            if target and len(val_pred) > 0:
                df = st.session_state.df
                n_val = len(val_pred)
                y_val_actual = df.iloc[-n_val:][target].values if target else []
                if len(y_val_actual) > len(val_pred):
                    y_val_actual = y_val_actual[:len(val_pred)]

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(y=val_pred[:200], name="Predicción", line=dict(color="#2dd4bf"), mode="lines"))
                if len(y_val_actual) >= 200:
                    fig4.add_trace(go.Scatter(y=y_val_actual[:200], name="Real", line=dict(color="#5b6af0", dash="dash"), mode="lines"))
                fig4.update_layout(title="Predicciones vs Reales (val set)", **DARK, height=350)
                st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            val_pred = dl_result.get("val_predictions", [])
            if target and len(val_pred) > 0:
                df = st.session_state.df
                n_val = len(val_pred)
                y_val_actual = df.iloc[-n_val:][target].values if target else []
                if len(y_val_actual) > len(val_pred):
                    y_val_actual = y_val_actual[:len(val_pred)]

                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=y_val_actual, y=val_pred[:len(y_val_actual)], mode="markers", marker=dict(color="#5b6af0", opacity=0.6)))
                if len(y_val_actual) > 0:
                    min_v = min(y_val_actual.min(), val_pred[:len(y_val_actual)].min())
                    max_v = max(y_val_actual.max(), val_pred[:len(y_val_actual)].max())
                    fig5.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#f43f5e", dash="dash"), name="Perfect prediction"))
                fig5.update_layout(title="Predicción vs Real", xaxis_title="Real", yaxis_title="Predicción", **DARK, height=400)
                st.plotly_chart(fig5, use_container_width=True)

        with tab4:
            val_pred = dl_result.get("val_predictions", [])
            if target and len(val_pred) > 0:
                df = st.session_state.df
                n_val = len(val_pred)
                y_val_actual = df.iloc[-n_val:][target].values if target else []
                if len(y_val_actual) > len(val_pred):
                    y_val_actual = y_val_actual[:len(val_pred)]

                preview = pd.DataFrame({
                    "y_real": y_val_actual[:100],
                    "y_pred": np.round(val_pred[:100], 4),
                    "error": np.round(y_val_actual[:100] - val_pred[:100], 4),
                })
                paginated_dataframe(preview, key="eval_dl_preview", height=400)

    st.divider()
    st.markdown("### 📋 Configuración usada")
    st.json(config)
