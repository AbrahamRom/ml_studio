import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline.comparison import (
    build_best_model_metrics,
    build_final_matrix,
    build_target_summary,
)

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)
PALETTE = ["#5b6af0", "#2dd4bf", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"]

st.markdown("# 📊 Comparar Modelos")

if st.session_state.automl_run is None:
    st.warning("⚠️ Entrena AutoML primero.")
    st.stop()

run = st.session_state.automl_run
target_results = run["target_results"]
compare_df = run.get("compare_df")
summary_df = run.get("summary_df")
best_metrics_df = run.get("best_model_metrics_df")
targets = list(target_results.keys())

if compare_df is None or compare_df.empty:
    compare_df = build_final_matrix(target_results)
if summary_df is None or summary_df.empty:
    summary_df = build_target_summary(target_results)
required_scale_cols = {"Holdout min", "Holdout max", "Holdout media", "Holdout mediana"}
if best_metrics_df is None or best_metrics_df.empty or not required_scale_cols.issubset(
    best_metrics_df.columns
):
    best_metrics_df = build_best_model_metrics(target_results)

st.caption(f"Corrida `{run['run_id']}` · Artefactos `{run['base_path']}`")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏁 Matriz final", "🏅 Leaderboard", "📉 Barras", "📁 Artefactos", "🧮 Métricas por modelo"]
)


def _direction_for(target: str) -> str:
    return target_results[target]["config"]["direction"]


def _primary_metric_for(target: str) -> str:
    return target_results[target]["config"]["primary_metric"]


with tab1:
    st.markdown("### Tabla final target × tipo de modelo")
    st.caption("Cada celda contiene el mejor valor del tipo de modelo para ese target, usando el holdout real.")

    def highlight_row(row):
        target = row.name
        values = pd.to_numeric(row, errors="coerce").dropna()
        if values.empty:
            return [""] * len(row)
        best_value = values.max() if _direction_for(target) == "max" else values.min()
        return [
            "background-color:#1a2540;color:#2dd4bf;font-weight:bold;"
            if pd.notna(value) and value == best_value
            else ""
            for value in row
        ]

    st.dataframe(compare_df.round(4).style.apply(highlight_row, axis=1), use_container_width=True)

    st.markdown("### Resumen por target")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if best_metrics_df is not None and not best_metrics_df.empty:
        st.markdown("### Mejor modelo + métricas (holdout real)")
        st.caption(
            "Cada fila resume el modelo ganador por target con todas sus métricas de holdout y la escala del target."
        )
        st.dataframe(best_metrics_df.round(4), use_container_width=True, hide_index=True)

    st.markdown("### Heatmap de desempeño (mejor modelo por target)")
    st.caption("Cada celda es el valor de la métrica para el mejor modelo de cada target en holdout.")

    if best_metrics_df is not None and not best_metrics_df.empty:
        _exclude = {
            "Target", "Mejor modelo holdout", "Tipo holdout", "Métrica usada",
            "Holdout min", "Holdout max", "Holdout media", "Holdout mediana",
        }
        _metric_cols = [
            c for c in best_metrics_df.columns
            if c not in _exclude and pd.api.types.is_numeric_dtype(best_metrics_df[c])
        ]
        if _metric_cols:
            _defaults = [c for c in ["r2", "nrmse", "nmae", "rmse", "mae", "smape", "score_global"] if c in _metric_cols]
            _sel = st.multiselect("Métricas para el heatmap", _metric_cols, default=_defaults, key="hm_metrics")
            if _sel:
                _labels = {
                    "r2": "R²", "r2_adjusted": "R² Ajustado", "rmse": "RMSE", "mae": "MAE",
                    "mse": "MSE", "mape": "MAPE", "smape": "SMAPE",
                    "nrmse": "NRMSE", "nmae": "NMAE", "score_global": "Score Global",
                }
                _data = best_metrics_df.set_index("Target")[_sel].rename(columns={c: _labels.get(c, c) for c in _sel})
                fig = go.Figure(
                    go.Heatmap(
                        z=_data.values,
                        x=_data.columns,
                        y=_data.index,
                        text=_data.round(4).values,
                        texttemplate="%{text}",
                        colorscale="thermal",
                        hovertemplate="Target: %{y}<br>%{x}: %{text}<extra></extra>",
                    )
                )
                fig.update_layout(
                    **DARK, title="Métricas por target",
                    height=200 + 40 * len(_data),
                    xaxis=dict(tickangle=-45),
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    target = st.selectbox("Target", targets, key="compare_target")
    result = target_results[target]
    leaderboard = result.get("leaderboard")
    if leaderboard is None or leaderboard.empty:
        st.info("No hay leaderboard guardado para este target.")
    else:
        leaderboard = leaderboard.copy()
        direction = _direction_for(target)
        ascending = direction == "min"
        leaderboard["metric_value"] = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
        leaderboard = leaderboard.sort_values("metric_value", ascending=ascending)

        st.markdown(
            f"### `{target}` · leaderboard interno de mljar · métrica `{_primary_metric_for(target)}`"
        )
        st.dataframe(leaderboard.round(4), use_container_width=True, hide_index=True)

        best = leaderboard.iloc[0] if not leaderboard.empty else None
        if best is not None:
            st.success(
                f"Mejor candidato interno: **{best['name']}** · tipo `{best['model_type']}` · "
                f"score `{best['metric_value']:.4f}`"
            )

with tab3:
    target = st.selectbox("Target para gráfica", targets, key="bar_target")
    row = compare_df.loc[target].dropna()
    direction = _direction_for(target)
    row = row.sort_values(ascending=(direction == "min"))

    colors = [PALETTE[0] if i == 0 else "#334155" for i in range(len(row))]
    fig = go.Figure(
        go.Bar(
            x=row.index,
            y=row.values,
            marker_color=colors,
            text=[f"{value:.4f}" for value in row.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        **DARK,
        title=f"{target} · mejor {_primary_metric_for(target)} por tipo de modelo",
        height=420,
        margin=dict(t=50, b=30, l=10, r=10),
        yaxis=dict(gridcolor="#252a38"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Artefactos de la corrida")
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab5:
    target = st.selectbox("Target con métricas detalladas", targets, key="detailed_metrics_target")
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

        st.dataframe(detailed_df.round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Global export button
# ---------------------------------------------------------------------------
st.divider()


def _build_compare_report(run, target_results, targets, compare_df, summary_df, best_metrics_df):
    report = {
        "project_info": {
            "run_id": run["run_id"],
            "timestamp": run.get("timestamp") or datetime.now().isoformat(),
            "description": run.get("description", ""),
            "base_path": str(run.get("base_path", "")),
        },
        "comparison": {},
        "targets": {},
    }

    if compare_df is not None and not compare_df.empty:
        report["comparison"]["final_matrix"] = {
            "description": "Best metric value per (target, model_type) on holdout",
            "data": compare_df.to_dict(),
            "shape": list(compare_df.shape),
            "targets": list(compare_df.index),
            "model_types": list(compare_df.columns),
        }

    if summary_df is not None and not summary_df.empty:
        report["comparison"]["target_summary"] = summary_df.to_dict(orient="records")

    if best_metrics_df is not None and not best_metrics_df.empty:
        report["comparison"]["best_model_metrics"] = best_metrics_df.to_dict(orient="records")

    for t in targets:
        res = target_results[t]
        te = {"config": dict(res["config"])}
        te["best_model"] = {
            "name": res.get("best_model_name"),
            "type": res.get("best_model_type"),
        }
        lb = res.get("leaderboard")
        if lb is not None and not lb.empty:
            te["leaderboard"] = lb.to_dict(orient="records")
        pm = res.get("per_model_metrics")
        if pm is not None and not pm.empty:
            te["per_model_metrics"] = pm.to_dict(orient="records")
        te["artifacts"] = {
            "mljar_report": res.get("results_path"),
            "leaderboard": res.get("leaderboard_path"),
            "predictions": res.get("predictions_path"),
            "metrics": res.get("metrics_path"),
        }
        report["targets"][t] = te

    return report


report_data = _build_compare_report(run, target_results, targets, compare_df, summary_df, best_metrics_df)
report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
st.download_button(
    label="📥 Descargar reporte de Comparación (JSON)",
    data=report_json,
    file_name="comparison_report.json",
    mime="application/json",
    use_container_width=True,
)
