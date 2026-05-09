import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
compare_df = run["compare_df"]
summary_df = run["summary_df"]
targets = list(target_results.keys())

st.caption(f"Corrida `{run['run_id']}` · Artefactos `{run['base_path']}`")

tab1, tab2, tab3, tab4 = st.tabs(["🏁 Matriz final", "🏅 Leaderboard", "📉 Barras", "📁 Artefactos"])


def _direction_for(target: str) -> str:
    return target_results[target]["config"]["direction"]


def _primary_metric_for(target: str) -> str:
    return target_results[target]["config"]["primary_metric"]


with tab1:
    st.markdown("### Tabla final target × tipo de modelo")
    st.caption("Cada celda contiene el mejor valor del tipo de modelo para ese target.")

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

with tab2:
    target = st.selectbox("Target", targets, key="compare_target")
    result = target_results[target]
    leaderboard = result["leaderboard"].copy()
    direction = _direction_for(target)
    ascending = direction == "min"
    leaderboard["metric_value"] = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
    leaderboard = leaderboard.sort_values("metric_value", ascending=ascending)

    st.markdown(f"### `{target}` · métrica `{_primary_metric_for(target)}`")
    st.dataframe(leaderboard.round(4), use_container_width=True, hide_index=True)

    best = leaderboard.iloc[0] if not leaderboard.empty else None
    if best is not None:
        st.success(
            f"Mejor candidato: **{best['name']}** · tipo `{best['model_type']}` · "
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
