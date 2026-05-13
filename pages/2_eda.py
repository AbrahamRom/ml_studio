import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen", "📈 Distribuciones", "🔗 Correlaciones", "❗ Calidad"])

# ── TAB 1: Resumen ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Tipos de columnas")
    type_map = {}
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
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
