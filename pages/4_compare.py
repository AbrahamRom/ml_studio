import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.pagination import paginated_dataframe

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)
PALETTE = ["#5b6af0", "#2dd4bf", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"]

st.markdown("# 📊 Comparar Modelos")

has_automl = st.session_state.automl_run is not None
has_dl = bool(st.session_state.get("dl_results"))

if not has_automl and not has_dl:
    st.warning("⚠️ Entrena modelos primero (AutoML o Deep Learning).")
    st.stop()

st.caption("Comparación entre AutoML y Deep Learning.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏁 Matriz comparativa", "📉 Barras", "📁 Artefactos AutoML", "🧮 Métricas detalladas"]
)

# ── Build combined comparison ──────────────────────────────────────────────────
def _get_automl_data():
    if not has_automl:
        return {}, None, None
    run = st.session_state.automl_run
    return run["target_results"], run.get("compare_df"), run.get("summary_df")

def _get_dl_data():
    if not has_dl:
        return []
    return st.session_state.dl_results

target_results, compare_df, summary_df = _get_automl_data()
dl_results = _get_dl_data()

with tab1:
    st.markdown("### Tabla final target × modelo")
    st.caption("Cada celda contiene la mejor métrica del modelo para ese target.")

    all_targets = set()
    if target_results:
        all_targets.update(target_results.keys())
    for r in dl_results:
        if r.get("target"):
            all_targets.add(r["target"])

    if all_targets:
        selected_target = st.selectbox("Target", sorted(all_targets), key="compare_combined_target")

        rows = []

        if target_results and selected_target in target_results:
            tr = target_results[selected_target]
            per_model = tr.get("per_model_metrics")
            if per_model is not None and not per_model.empty:
                for _, row_data in per_model.iterrows():
                    rows.append({
                        "Modelo": row_data.get("model_name", "N/A"),
                        "Tipo": row_data.get("model_type", "N/A"),
                        "Fuente": "AutoML",
                        "Score": row_data.get("score_global", row_data.get(tr["config"]["primary_metric"], "")),
                        "RMSE": row_data.get("rmse", ""),
                        "R²": row_data.get("r2", ""),
                        "F1": row_data.get("f1", ""),
                        "Accuracy": row_data.get("accuracy", ""),
                    })

        for r in dl_results:
            if r.get("target") == selected_target or (not r.get("target") and r["model_type"] in ("autoencoder", "vae")):
                label = r["model_type"].upper()
                if r.get("hpo"):
                    label += " (HPO)"
                rows.append({
                    "Modelo": label,
                    "Tipo": r["model_type"],
                    "Fuente": "Deep Learning",
                    "Score": "",
                    "RMSE": f"{r['val_rmse']:.4f}",
                    "R²": "",
                    "F1": "",
                    "Accuracy": "",
                })

        if rows:
            df_comp = pd.DataFrame(rows)
            paginated_dataframe(df_comp, key="compare_dl", height=350, hide_index=True)
        else:
            st.info("No hay resultados para este target.")

        if summary_df is not None and not summary_df.empty:
            st.markdown("### Resumen AutoML por target")
            paginated_dataframe(summary_df, key="compare_summary", height=300, hide_index=True)

    else:
        st.info("No hay targets configurados.")

with tab2:
    if target_results:
        target = st.selectbox("Target para gráfica", list(target_results.keys()), key="bar_target_combined")
        tr = target_results[target]
        per_model = tr.get("per_model_metrics")

        if per_model is not None and not per_model.empty:
            metric_col = tr["config"]["primary_metric"]
            if metric_col not in per_model.columns:
                metric_col = "score_global"
            if metric_col not in per_model.columns:
                numeric_cols = [c for c in per_model.columns if c not in {"model_name", "model_type", "evaluation_error"} and pd.api.types.is_numeric_dtype(per_model[c])]
                metric_col = numeric_cols[0] if numeric_cols else None

            if metric_col:
                plot_data = per_model[["model_name", metric_col]].copy()
                plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
                plot_data = plot_data.sort_values(metric_col, ascending=(tr["config"]["direction"] == "min"))
                plot_data = plot_data.head(15)

                direction = tr["config"]["direction"]
                best_idx = plot_data[metric_col].idxmin() if direction == "min" else plot_data[metric_col].idxmax()

                colors = [
                    "#2dd4bf" if idx == best_idx else "#5b6af0"
                    for idx in plot_data.index
                ]

                fig = go.Figure(
                    go.Bar(
                        x=plot_data["model_name"],
                        y=plot_data[metric_col],
                        marker_color=colors,
                        text=[f"{v:.4f}" for v in plot_data[metric_col]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **DARK,
                    title=f"{target} · {metric_col} por modelo",
                    height=420,
                    margin=dict(t=50, b=80, l=10, r=10),
                    yaxis=dict(gridcolor="#252a38"),
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Entrena AutoML para ver gráficas comparativas.")

with tab3:
    if target_results:
        st.markdown("### Artefactos de la corrida AutoML")
        rows = []
        for target, result in target_results.items():
            rows.append(
                {
                    "Target": target,
                    "MLJAR report": result["results_path"],
                    "Leaderboard": result["leaderboard_path"],
                    "Predicciones": result["predictions_path"],
                    "Métricas": result["metrics_path"],
                }
            )
        paginated_dataframe(pd.DataFrame(rows), key="compare_artifacts", height=300, hide_index=True)
    else:
        st.info("No hay corrida AutoML disponible.")

with tab4:
    if target_results:
        target = st.selectbox("Target con métricas detalladas", list(target_results.keys()), key="detailed_metrics_target_combined")
        result = target_results[target]
        per_model_metrics = result.get("per_model_metrics")

        st.markdown(f"### `{target}` · métricas de holdout por modelo")
        st.caption("Se calculan con las predicciones de cada modelo interno sobre el holdout.")

        if per_model_metrics is None or per_model_metrics.empty:
            st.info("No hay métricas por modelo disponibles para este target.")
        else:
            detailed_df = per_model_metrics.copy()
            numeric_metric_columns = [
                column
                for column in detailed_df.columns
                if column not in {"model_name", "model_type", "evaluation_error"}
                and pd.api.types.is_numeric_dtype(detailed_df[column])
            ]

            default_sort_column = None
            if "score_global" in numeric_metric_columns:
                default_sort_column = "score_global"
            elif result["config"]["task"] == "regression" and result["config"]["primary_metric"] in numeric_metric_columns:
                default_sort_column = result["config"]["primary_metric"]
            elif numeric_metric_columns:
                default_sort_column = numeric_metric_columns[0]

            if numeric_metric_columns:
                sort_column = st.selectbox(
                    "Ordenar por",
                    numeric_metric_columns,
                    index=numeric_metric_columns.index(default_sort_column) if default_sort_column in numeric_metric_columns else 0,
                    key=f"sort_detailed_metrics_{target}",
                )
                lower_is_better = {"rmse", "mae", "mape", "mse", "smape"}
                ascending = sort_column in lower_is_better or (
                    sort_column == result["config"]["primary_metric"] and result["config"]["direction"] == "min"
                )
                if sort_column == "score_global":
                    ascending = False
                detailed_df = detailed_df.sort_values(sort_column, ascending=ascending)

            paginated_dataframe(detailed_df.round(4), key="compare_detailed", height=400, hide_index=True)
    else:
        st.info("No hay corrida AutoML disponible.")
