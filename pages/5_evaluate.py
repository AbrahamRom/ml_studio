import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK = dict(
    paper_bgcolor="#0d0f14", plot_bgcolor="#141720",
    font_color="#e2e8f0", gridcolor="#252a38",
)

st.markdown("# 🔬 Evaluación de Modelos")

if st.session_state.trained_models is None:
    st.warning("⚠️ Entrena los modelos primero.")
    st.stop()

models   = st.session_state.trained_models
task     = st.session_state.task_type
targets  = st.session_state.target_cols
is_multi = st.session_state.multioutput
best     = st.session_state.best_model

# ── Model selector ─────────────────────────────────────────────────────────────
model_name = st.selectbox(
    "Modelo a evaluar",
    list(models.keys()),
    index=list(models.keys()).index(best) if best in models else 0,
)
res      = models[model_name]
pipe     = res["pipeline"]
X_test   = res["X_test"]
y_test   = res["y_test"]
preds_df = res["predictions"]

st.markdown(
    f'**Modelo seleccionado:** `{model_name}` '
    f'{"⭐ (mejor)" if model_name == best else ""}',
    unsafe_allow_html=True,
)
st.divider()

# ── Select target for multioutput ──────────────────────────────────────────────
if is_multi:
    target_sel = st.selectbox("Target a analizar", targets)
    y_true_s   = y_test[target_sel].values
    y_pred_s   = preds_df[target_sel].values
else:
    target_sel = targets[0]
    y_true_s   = y_test.values
    y_pred_s   = preds_df.values

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
if task == "classification":
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc, precision_recall_curve,
        ConfusionMatrixDisplay,
    )

    classes     = np.unique(np.concatenate([y_true_s, y_pred_s]))
    cm          = confusion_matrix(y_true_s, y_pred_s, labels=classes)
    report_dict = classification_report(y_true_s, y_pred_s, output_dict=True, zero_division=0)
    report_df   = pd.DataFrame(report_dict).T.round(3)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔲 Matriz Confusión", "📋 Reporte", "📈 ROC / PR", "🎯 Predicciones"]
    )

    with tab1:
        st.markdown(f"### Matriz de Confusión — `{target_sel}`")
        # Normalize option
        normalize_cm = st.checkbox("Normalizar (porcentajes)", value=False)
        cm_plot = cm.astype(float)
        if normalize_cm:
            cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
            fmt_vals = [[f"{v:.1%}" for v in row] for row in cm_plot]
        else:
            fmt_vals = [[str(int(v)) for v in row] for row in cm_plot]

        fig = go.Figure(go.Heatmap(
            z=cm_plot,
            x=[f"Pred: {c}" for c in classes],
            y=[f"Real: {c}" for c in classes],
            colorscale="Blues",
            text=fmt_vals,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(**DARK, height=420, title="Confusion Matrix",
                          margin=dict(t=50, b=30, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Reporte de Clasificación")
        c1, c2, c3 = st.columns(3)
        for col_w, key in zip([c1, c2, c3], ["accuracy", "macro avg", "weighted avg"]):
            if key in report_dict:
                val = report_dict[key]
                metric_val = val if isinstance(val, float) else val.get("f1-score", 0)
                with col_w:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">{key.upper()}</div>'
                        f'<div class="val">{metric_val:.3f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.dataframe(report_df, use_container_width=True)

    with tab3:
        binary = len(classes) == 2
        if binary:
            try:
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if is_multi:
                        col_idx = list(targets).index(target_sel)
                        # For MultiOutputClassifier, predict_proba returns list
                        prob_pos = proba[col_idx][:, 1]
                    else:
                        prob_pos = proba[:, 1]

                    fpr, tpr, _ = roc_curve(y_true_s, prob_pos)
                    roc_auc     = auc(fpr, tpr)
                    prec, rec, _ = precision_recall_curve(y_true_s, prob_pos)

                    col_roc, col_pr = st.columns(2)
                    with col_roc:
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
                                                     name=f"AUC = {roc_auc:.3f}",
                                                     line=dict(color="#5b6af0", width=2)))
                        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                          line=dict(dash="dash", color="#64748b"))
                        fig_roc.update_layout(**DARK, title="ROC Curve", height=380,
                                              xaxis_title="FPR", yaxis_title="TPR")
                        st.plotly_chart(fig_roc, use_container_width=True)

                    with col_pr:
                        fig_pr = go.Figure()
                        fig_pr.add_trace(go.Scatter(x=rec, y=prec, fill="tozeroy",
                                                    name="PR Curve",
                                                    line=dict(color="#2dd4bf", width=2)))
                        fig_pr.update_layout(**DARK, title="Precision-Recall Curve", height=380,
                                             xaxis_title="Recall", yaxis_title="Precision")
                        st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info("Este modelo no soporta probabilidades (predict_proba).")
            except Exception as e:
                st.warning(f"No se pudo calcular curva ROC: {e}")
        else:
            st.info("ROC/PR curves disponibles solo para clasificación binaria.")

    with tab4:
        st.markdown("### Muestra de predicciones")
        preview = X_test.copy().reset_index(drop=True)
        preview["y_real"] = y_true_s
        preview["y_pred"] = y_pred_s
        preview["✅"] = (preview["y_real"] == preview["y_pred"]).map({True: "✅", False: "❌"})
        st.dataframe(preview.head(50), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
else:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2   = r2_score(y_true_s, y_pred_s)
    mae  = mean_absolute_error(y_true_s, y_pred_s)
    rmse = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
    residuals = y_true_s - y_pred_s

    # Top metrics
    c1, c2, c3 = st.columns(3)
    for col_w, label, val in zip([c1, c2, c3],
                                  ["R²", "MAE", "RMSE"],
                                  [r2, mae, rmse]):
        with col_w:
            color = "#2dd4bf" if label == "R²" else "#f59e0b"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label} — {target_sel}</div>'
                f'<div class="val" style="color:{color}">{val:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Real vs Pred", "📉 Residuos", "📊 Distribución Error", "🎯 Predicciones"]
    )

    with tab1:
        min_v = min(y_true_s.min(), y_pred_s.min())
        max_v = max(y_true_s.max(), y_pred_s.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true_s, y=y_pred_s, mode="markers",
            marker=dict(color="#5b6af0", size=5, opacity=0.6),
            name="Predicciones",
        ))
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", line=dict(color="#2dd4bf", dash="dash", width=2),
            name="Perfecto",
        ))
        fig.update_layout(**DARK, title=f"Real vs Predicho — {target_sel}", height=440,
                          xaxis_title="Real", yaxis_title="Predicho")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=("Residuos vs Predicho", "Residuos vs Índice"))
        fig2.add_trace(
            go.Scatter(x=y_pred_s, y=residuals, mode="markers",
                       marker=dict(color="#5b6af0", size=4, opacity=0.6), name="Residuos"),
            row=1, col=1,
        )
        fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=1)
        fig2.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, mode="markers",
                       marker=dict(color="#f59e0b", size=4, opacity=0.6), name="Por índice"),
            row=1, col=2,
        )
        fig2.add_hline(y=0, line=dict(color="#2dd4bf", dash="dash"), row=1, col=2)
        fig2.update_layout(**DARK, height=400, showlegend=False,
                           margin=dict(t=60, b=20, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            fig3 = px.histogram(x=residuals, nbins=50,
                                color_discrete_sequence=["#5b6af0"],
                                title="Distribución de Residuos",
                                labels={"x": "Error"})
            fig3.update_layout(**DARK, height=360)
            st.plotly_chart(fig3, use_container_width=True)
        with col_b:
            # Q-Q plot approximation
            from scipy import stats as scipy_stats
            (osm, osr), _ = scipy_stats.probplot(residuals)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                      marker=dict(color="#2dd4bf", size=4), name="Q-Q"))
            m, b = np.polyfit(osm, osr, 1)
            fig4.add_trace(go.Scatter(x=osm, y=m*np.array(osm)+b,
                                      mode="lines", line=dict(color="#f43f5e", dash="dash"),
                                      name="Normal ref"))
            fig4.update_layout(**DARK, title="Q-Q Plot Residuos", height=360,
                               xaxis_title="Theoretical Quantiles",
                               yaxis_title="Sample Quantiles")
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.markdown("### Muestra de predicciones")
        preview = X_test.copy().reset_index(drop=True).head(50)
        preview[f"{target_sel}_real"] = y_true_s[:50]
        preview[f"{target_sel}_pred"] = y_pred_s[:50].round(2)
        preview["error"]              = (y_true_s[:50] - y_pred_s[:50]).round(3)
        preview["% error"]            = (preview["error"] / (y_true_s[:50] + 1e-9) * 100).round(2)
        st.dataframe(preview, use_container_width=True)

    # Multioutput: show all targets at once
    if is_multi:
        st.divider()
        st.markdown("### 📊 Resumen Multioutput — Todos los targets")
        rows = []
        for t in targets:
            yt = y_test[t].values
            yp = preds_df[t].values
            rows.append({
                "Target":  t,
                "R²":     round(r2_score(yt, yp), 4),
                "MAE":    round(mean_absolute_error(yt, yp), 4),
                "RMSE":   round(np.sqrt(mean_squared_error(yt, yp)), 4),
            })
        summary_df = pd.DataFrame(rows).set_index("Target")
        st.dataframe(summary_df, use_container_width=True)

        fig_multi = go.Figure()
        for metric, color in zip(["R²", "MAE", "RMSE"],
                                  ["#5b6af0", "#2dd4bf", "#f59e0b"]):
            fig_multi.add_trace(go.Bar(
                name=metric,
                x=summary_df.index,
                y=summary_df[metric],
                marker_color=color,
            ))
        fig_multi.update_layout(**DARK, barmode="group", height=380,
                                title="Métricas por Target",
                                margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig_multi, use_container_width=True)
