import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_pipeline.quality import (
    compute_variable_stats,
    variable_stats_to_csv,
    variable_stats_to_json,
)

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

st.markdown("# 🔍 EDA & Calidad de Datos")

if st.session_state.df is None:
    st.warning("⚠️ Carga un dataset primero en la sección Dataset.")
    st.stop()

df = st.session_state.df
targets = st.session_state.target_cols or []

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Resumen", "📈 Distribuciones", "🔗 Correlaciones", "❗ Calidad", "📐 Estadísticas"]
)

# ── TAB 1: Resumen ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Tipos de columnas")
    type_map = {}
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category" or pd.api.types.is_string_dtype(df[c]):
            type_map[c] = "categorical"
        elif df[c].nunique() <= 15 and df[c].dtype in [np.int64, np.int32, int]:
            type_map[c] = "discrete"
        else:
            type_map[c] = "continuous"

    rows = []
    for c in df.columns:
        rows.append({
            "Columna": c,
            "Tipo dtype": str(df[c].dtype),
            "Clase": type_map[c],
            "# Únicos": df[c].nunique(),
            "% Nulos": f"{df[c].isnull().mean()*100:.1f}%",
            "Min": df[c].min() if type_map[c] != "categorical" else "-",
            "Max": df[c].max() if type_map[c] != "categorical" else "-",
            "Media": f"{df[c].mean():.2f}" if type_map[c] != "categorical" else "-",
            "Target": "🎯" if c in targets else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Estadísticas descriptivas")
    st.dataframe(df.describe().T.round(3), use_container_width=True)

# ── TAB 2: Distribuciones ──────────────────────────────────────────────────────
with tab2:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        st.markdown("#### Variables numéricas")
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            batch = num_cols[i:i+cols_per_row]
            cols  = st.columns(len(batch))
            for j, col_name in enumerate(batch):
                with cols[j]:
                    color = "#5b6af0" if col_name not in targets else "#2dd4bf"
                    fig = px.histogram(
                        df, x=col_name, nbins=40,
                        color_discrete_sequence=[color],
                        title=f"{'🎯 ' if col_name in targets else ''}{col_name}",
                    )
                    fig.update_layout(**DARK, height=260, margin=dict(t=36, b=10, l=10, r=10),
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.markdown("#### Variables categóricas")
        for col_name in cat_cols:
            vc = df[col_name].value_counts().head(20)
            fig = px.bar(x=vc.index, y=vc.values,
                         labels={"x": col_name, "y": "Frecuencia"},
                         title=col_name, color_discrete_sequence=["#5b6af0"])
            fig.update_layout(**DARK, height=280, margin=dict(t=36, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    if targets and num_cols:
        st.markdown("#### Targets vs Features (scatter)")
        feat   = st.selectbox("Feature X", [c for c in num_cols if c not in targets])
        target_sel = st.selectbox("Target Y", targets)
        fig = px.scatter(df, x=feat, y=target_sel, trendline="ols",
                         color_discrete_sequence=["#2dd4bf"],
                         opacity=0.6, title=f"{feat} vs {target_sel}")
        fig.update_layout(**DARK, height=380)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Correlaciones ───────────────────────────────────────────────────────
with tab3:
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.info("Se necesitan al menos 2 columnas numéricas.")
    else:
        corr = num_df.corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
        ))
        fig.update_layout(**DARK, height=520, title="Matriz de correlación",
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        if targets:
            st.markdown("#### Correlación con los targets")
            target_num = [t for t in targets if t in num_df.columns]
            if target_num:
                corr_targets = num_df.corr()[target_num].drop(index=target_num, errors="ignore")
                fig2 = px.imshow(corr_targets.T, color_continuous_scale="RdBu",
                                 color_continuous_midpoint=0, text_auto=True,
                                 title="Features vs Targets")
                fig2.update_layout(**DARK, height=300)
                st.plotly_chart(fig2, use_container_width=True)

# ── TAB 4: Calidad ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Nulos por columna")
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    fig = px.bar(x=null_pct.index, y=null_pct.values,
                 labels={"x": "Columna", "y": "% Nulos"},
                 color=null_pct.values,
                 color_continuous_scale=["#22c55e", "#f59e0b", "#f43f5e"],
                 title="Porcentaje de valores nulos")
    fig.update_layout(**DARK, height=350, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Outliers (IQR)")
    num_cols2 = df.select_dtypes(include="number").columns.tolist()
    outlier_rows = []
    for c in num_cols2:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[c] < Q1 - 1.5*IQR) | (df[c] > Q3 + 1.5*IQR)).sum()
        outlier_rows.append({"Columna": c, "# Outliers": n_out,
                              "% Outliers": f"{n_out/len(df)*100:.1f}%"})
    st.dataframe(pd.DataFrame(outlier_rows).sort_values("# Outliers", ascending=False),
                 use_container_width=True)

    col_box = st.selectbox("Boxplot de:", num_cols2)
    fig3 = px.box(df, y=col_box, color_discrete_sequence=["#5b6af0"])
    fig3.update_layout(**DARK, height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 5: Estadísticas (min, max, mediana, media, varianza, Shapiro-Wilk) ───
with tab5:
    st.markdown("### Estadísticas detalladas por variable")
    st.caption(
        "Para cada columna del dataset se calculan mínimo, máximo, media, mediana, "
        "varianza y el test de normalidad **Shapiro-Wilk** (α = 0.05). "
        "Las variables categóricas no aplican al test de normalidad."
    )

    ctrl_a, ctrl_b, ctrl_c = st.columns([1, 1, 2])
    with ctrl_a:
        only_numeric = st.toggle(
            "Solo variables numéricas",
            value=False,
            help="Si está activo, excluye las columnas categóricas/discretas-int del resumen.",
        )
    with ctrl_b:
        alpha = st.select_slider(
            "Nivel de significancia (α)",
            options=[0.01, 0.025, 0.05, 0.10],
            value=0.05,
            help="Umbral del p-valor para declarar la distribución como normal.",
        )

    # Cache the heavy computation per dataset identity to avoid recomputing
    # Shapiro-Wilk on every widget interaction. The cache key includes α so the
    # decision column updates when the slider changes.
    @st.cache_data(show_spinner="Calculando estadísticas por variable…")
    def _cached_variable_stats(df_id: int, df_hash: str, _df: pd.DataFrame, alpha_: float):
        return compute_variable_stats(_df, targets, alpha=alpha_)

    df_id = id(df)
    df_hash = pd.util.hash_pandas_object(df, index=True).sum().__int__()
    stats = _cached_variable_stats(df_id, df_hash, df, float(alpha))

    wide: pd.DataFrame = stats["wide"].copy()

    if only_numeric:
        wide = wide[wide["class"].isin(["continuous", "discrete"])].reset_index(drop=True)

    # Format the decision column with an emoji + label for readability in the UI.
    def _fmt_norm(value):
        if value is True:
            return "✅ Normal"
        if value is False:
            return "❌ No normal"
        return "⚠️ N/A"

    display = wide.rename(
        columns={
            "column": "Columna",
            "class": "Tipo",
            "count": "n",
            "null_pct": "% Nulos",
            "min": "Mínimo",
            "max": "Máximo",
            "mean": "Media",
            "median": "Mediana",
            "variance": "Varianza",
            "std": "Desv. estándar",
            "shapiro_W": "Shapiro W",
            "shapiro_p": "Shapiro p-valor",
            "shapiro_is_normal": "¿Normal?",
            "shapiro_n_used": "n (Shapiro)",
        }
    )
    if "¿Normal?" in display.columns:
        display["¿Normal?"] = display["¿Normal?"].map(_fmt_norm)

    # Pretty numeric formatting (3 decimals, leave ints alone).
    for col_name in ["Mínimo", "Máximo", "Media", "Mediana", "Varianza",
                     "Desv. estándar", "Shapiro W", "Shapiro p-valor", "% Nulos"]:
        if col_name in display.columns:
            display[col_name] = display[col_name].apply(
                lambda v: (f"{v:.4f}" if pd.notnull(v) and isinstance(v, float) else v)
            )
    if "n" in display.columns:
        display["n"] = display["n"].astype("Int64")
    if "n (Shapiro)" in display.columns:
        display["n (Shapiro)"] = display["n (Shapiro)"].astype("Int64")

    st.dataframe(display, use_container_width=True, height=420)

    # ── Notas sobre el test de Shapiro-Wilk ───────────────────────────────────
    notes = [r for r in stats["columns"] if r.get("shapiro_note")]
    if notes:
        with st.expander("📝 Notas del test de Shapiro-Wilk"):
            for r in notes:
                st.markdown(
                    f"- **{r['column']}** ({r['class']}, n={r.get('shapiro_n_used')}): "
                    f"{r['shapiro_note']}"
                )

    # ── Descargas estructuradas (CSV / JSON) ──────────────────────────────────
    st.markdown("#### Descargar resumen")
    csv_bytes = variable_stats_to_csv(stats)
    json_bytes = variable_stats_to_json(stats)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇️ Descargar CSV (formato ancho)",
            data=csv_bytes,
            file_name="variable_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="⬇️ Descargar JSON (completo)",
            data=json_bytes,
            file_name="variable_stats.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("ℹ️ Detalles de las métricas"):
        st.markdown(
            """
            - **Mínimo / Máximo** — valores extremos observados (sin nulos).
            - **Media** — promedio aritmético.
            - **Mediana** — percentil 50, robusta a outliers.
            - **Varianza** — medida de dispersión al cuadrado (`var(ddof=1)`).
            - **Shapiro-Wilk** — test de hipótesis de normalidad. Hipótesis nula
              H₀: la muestra proviene de una distribución normal. Se considera
              "normal" cuando `p > α`. SciPy limita el test a 5000 observaciones;
              si el dataset supera ese tamaño se aplica submuestreo reproducible
              (`random_state=42`).
            - **Variables categóricas** — el test de Shapiro-Wilk no aplica, se
              reportan `moda` (top) y su frecuencia.
            """
        )
