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
PALETTE = ["#5b6af0", "#2dd4bf", "#f59e0b", "#f43f5e", "#a78bfa",
           "#34d399", "#fb923c", "#60a5fa", "#e879f9", "#4ade80"]

st.markdown("# 📊 Comparar Modelos")

if st.session_state.trained_models is None:
    st.warning("⚠️ Entrena los modelos primero.")
    st.stop()

models   = st.session_state.trained_models
task     = st.session_state.task_type
targets  = st.session_state.target_cols
is_multi = st.session_state.multioutput
best     = st.session_state.best_model
cdf      = st.session_state.compare_df  # index = model name

model_names = list(models.keys())

# ── Metric selector ────────────────────────────────────────────────────────────
all_metrics = cdf.columns.tolist()
default_metrics = all_metrics[:3]
sel_metrics = st.multiselect("Métricas a comparar", all_metrics, default=default_metrics)

if not sel_metrics:
    st.info("Selecciona al menos una métrica.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🏅 Ranking", "📉 Barras", "🕸️ Radar", "📦 Por Target"])

# ── TAB 1: Ranking table ───────────────────────────────────────────────────────
with tab1:
    st.markdown("### Tabla de ranking")

    display = cdf[sel_metrics].copy().reset_index()
    display.insert(0, "🏆", ["⭐" if n == best else "" for n in display["Modelo"]])

    # color best row
    def highlight_best(row):
        color = "background-color: #1a2540; color: #2dd4bf; font-weight:bold;" if row["Modelo"] == best else ""
        return [color] * len(row)

    st.dataframe(
        display.style.apply(highlight_best, axis=1),
        use_container_width=True, hide_index=True,
    )

    # Per-metric winners
    st.markdown("### 🥇 Ganadores por métrica")
    cols_w = st.columns(len(sel_metrics))
    for i, m in enumerate(sel_metrics):
        if m in cdf.columns:
            if task == "regression" and m in ["MAE (avg)", "RMSE (avg)", "MAE", "RMSE"]:
                winner = cdf[m].idxmin()
                sign   = "↓ min"
            else:
                winner = cdf[m].idxmax()
                sign   = "↑ max"
            val = cdf.loc[winner, m]
            with cols_w[i]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{m} ({sign})</div>'
                    f'<div class="val" style="font-size:1rem">{winner}</div>'
                    f'<div style="color:#64748b;font-size:.8rem">{val:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── TAB 2: Bar charts ─────────────────────────────────────────────────────────
with tab2:
    for metric in sel_metrics:
        if metric not in cdf.columns:
            continue
        vals  = cdf[metric].sort_values(ascending=False)
        colors = [PALETTE[0] if n == best else PALETTE[2] for n in vals.index]
        fig = go.Figure(go.Bar(
            x=vals.index, y=vals.values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in vals.values],
            textposition="outside",
        ))
        fig.update_layout(
            **DARK, title=f"Comparación — {metric}",
            height=380, margin=dict(t=50, b=30, l=10, r=10),
            yaxis=dict(gridcolor="#252a38"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Radar chart ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Radar de métricas normalizadas")
    # Normalize to [0,1] per metric (higher = better always after flip)
    radar_df = cdf[sel_metrics].copy()
    norm_df  = radar_df.copy()
    for m in sel_metrics:
        col = radar_df[m]
        rng = col.max() - col.min()
        if rng == 0:
            norm_df[m] = 1.0
        elif task == "regression" and m in ["MAE (avg)", "RMSE (avg)", "MAE", "RMSE"]:
            norm_df[m] = 1 - (col - col.min()) / rng  # flip: lower is better
        else:
            norm_df[m] = (col - col.min()) / rng

    fig = go.Figure()
    for i, name in enumerate(model_names):
        vals_r = norm_df.loc[name, sel_metrics].tolist()
        vals_r += [vals_r[0]]  # close polygon
        cats = sel_metrics + [sel_metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_r, theta=cats, fill="toself",
            name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fillcolor=PALETTE[i % len(PALETTE)].replace(")", ",0.15)").replace("rgb", "rgba")
                       if PALETTE[i % len(PALETTE)].startswith("rgb") else PALETTE[i % len(PALETTE)] + "26",
            opacity=0.85,
        ))
    fig.update_layout(
        **DARK, height=520,
        polar=dict(
            bgcolor="#141720",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#252a38", color="#64748b"),
            angularaxis=dict(gridcolor="#252a38", color="#e2e8f0"),
        ),
        legend=dict(bgcolor="#0d0f14", bordercolor="#252a38", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Per-target breakdown (multioutput only) ────────────────────────────
with tab4:
    if not is_multi:
        st.info("Esta vista sólo aplica para modelos **multioutput** (múltiples targets).")
    else:
        st.markdown("### Métricas por target individual")
        # Gather per-target metrics from raw stored metrics
        for t in targets:
            st.markdown(f"#### 🎯 Target: `{t}`")
            rows = []
            for mname, res in models.items():
                row = {"Modelo": mname}
                for k, v in res["metrics"].items():
                    if k.endswith(f"_{t}"):
                        label = k.replace(f"_{t}", "")
                        row[label] = round(v, 4)
                rows.append(row)
            t_df = pd.DataFrame(rows).set_index("Modelo")
            if t_df.empty or t_df.shape[1] == 0:
                st.info("No hay métricas per-target disponibles para este target.")
                continue

            # bar per metric
            fig = make_subplots(rows=1, cols=len(t_df.columns),
                                subplot_titles=t_df.columns.tolist())
            for ci, metric in enumerate(t_df.columns, 1):
                vals = t_df[metric].sort_values(ascending=False)
                colors = [PALETTE[0] if n == best else "#334155" for n in vals.index]
                fig.add_trace(
                    go.Bar(x=vals.index, y=vals.values,
                           marker_color=colors,
                           text=[f"{v:.3f}" for v in vals.values],
                           textposition="outside", showlegend=False),
                    row=1, col=ci,
                )
            fig.update_layout(**DARK, height=320, margin=dict(t=50, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
