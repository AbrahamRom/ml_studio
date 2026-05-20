import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from utils.pagination import paginated_dataframe
from utils.preprocessing import RobustScaler

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Resumen", "📈 Distribuciones", "🔗 Correlaciones", "❗ Calidad", "📐 Estadística"])

# ── TAB 1: Resumen ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Tipos de columnas")
    type_map = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            type_map[c] = "continuous" if df[c].nunique() > 15 else "discrete"
        else:
            type_map[c] = "categorical"

    rows = []
    for c in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        rows.append({
            "Columna": c,
            "Tipo dtype": str(df[c].dtype),
            "Clase": type_map[c],
            "# Únicos": df[c].nunique(),
            "# Nulos": int(df[c].isnull().sum()),
            "% Nulos": f"{df[c].isnull().mean()*100:.1f}%",
            "Min": df[c].min() if is_numeric else "-",
            "Max": df[c].max() if is_numeric else "-",
            "Media": f"{df[c].mean():.2f}" if is_numeric else "-",
            "Target": "🎯" if c in targets else "",
        })
    paginated_dataframe(pd.DataFrame(rows), key="eda_resumen", height=400)

    st.markdown("### Estadísticas descriptivas")
    paginated_dataframe(df.describe().T.round(3), key="eda_stats", height=300)

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
        slope, intercept, _, _, _ = stats.linregress(df[feat], df[target_sel])
        x_range = np.linspace(df[feat].min(), df[feat].max(), 100)
        fig = px.scatter(df, x=feat, y=target_sel, color_discrete_sequence=["#2dd4bf"], opacity=0.6, title=f"{feat} vs {target_sel}")
        fig.add_trace(go.Scatter(x=x_range, y=slope*x_range+intercept, mode="lines", name="Tendencia", line=dict(color="#f59e0b", width=2)))
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
    null_counts = df.isnull().sum().sort_values(ascending=False)
    null_pct = (null_counts / len(df) * 100)
    null_df = pd.DataFrame({
        "Columna": null_counts.index,
        "# Nulos": null_counts.values,
        "% Nulos": (null_pct.values).round(1),
    }).reset_index(drop=True)
    paginated_dataframe(null_df, key="eda_nulls", height=300, hide_index=True)

    fig = px.bar(x=null_pct.index, y=null_pct.values,
                 labels={"x": "Columna", "y": "% Nulos"},
                 color=null_pct.values,
                 color_continuous_scale=["#22c55e", "#f59e0b", "#f43f5e"],
                 title="Porcentaje de valores nulos",
                 text=null_pct.values.round(1).astype(str) + "%")
    fig.update_layout(**DARK, height=350, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Outliers (IQR)")
    num_cols2 = df.select_dtypes(include="number").columns.tolist()
    if num_cols2:
        outlier_rows = []
        for c in num_cols2:
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            n_out = ((df[c] < Q1 - 1.5*IQR) | (df[c] > Q3 + 1.5*IQR)).sum()
            outlier_rows.append({"Columna": c, "# Outliers": n_out,
                                  "% Outliers": f"{n_out/len(df)*100:.1f}%"})
        paginated_dataframe(pd.DataFrame(outlier_rows).sort_values("# Outliers", ascending=False),
                     key="eda_outliers", height=300, hide_index=True)

        col_box = st.selectbox("Boxplot de:", num_cols2)
        fig3 = px.box(df, y=col_box, color_discrete_sequence=["#5b6af0"])
        fig3.update_layout(**DARK, height=320)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No hay columnas numéricas para calcular outliers.")

# ── TAB 5: Estadística ─────────────────────────────────────────────────────────
with tab5:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── 5.1: Estadísticas Descriptivas Extendidas ──────────────────────────────
    st.markdown("### 📊 Estadísticas Descriptivas Extendidas")
    st.caption("Métricas completas para variables numéricas.")

    if num_cols:
        stats_rows = []
        for c in num_cols:
            clean = df[c].dropna()
            if len(clean) < 2:
                continue
            mean = clean.mean()
            std = clean.std()
            variance = clean.var()
            median = clean.median()
            mode_val = clean.mode().iloc[0] if not clean.mode().empty else np.nan
            cv = (std / abs(mean) * 100) if mean != 0 else float("inf")
            skew = clean.skew()
            kurt = clean.kurtosis()
            stats_rows.append({
                "Columna": c,
                "N": len(clean),
                "Media": round(mean, 4),
                "Mediana": round(median, 4),
                "Moda": round(mode_val, 4) if pd.api.types.is_numeric_dtype(type(mode_val)) else mode_val,
                "Varianza": round(variance, 4),
                "Desv. Estándar": round(std, 4),
                "CV (%)": round(cv, 2) if cv != float("inf") else "∞",
                "Mín": round(clean.min(), 4),
                "Máx": round(clean.max(), 4),
                "Rango": round(clean.max() - clean.min(), 4),
                "Q1": round(clean.quantile(0.25), 4),
                "Q3": round(clean.quantile(0.75), 4),
                "IQR": round(clean.quantile(0.75) - clean.quantile(0.25), 4),
                "Asimetría": round(skew, 4),
                "Curtosis": round(kurt, 4),
            })
        paginated_dataframe(pd.DataFrame(stats_rows), key="eda_ext_stats", height=400, hide_index=True)
    else:
        st.info("No hay columnas numéricas.")

    st.divider()

    # ── 5.2: Pruebas de Normalidad ─────────────────────────────────────────────
    st.markdown("### 🔔 Pruebas de Normalidad")
    st.caption("Evalúa si los datos siguen una distribución normal (Gaussiana).")

    if num_cols:
        normal_col = st.selectbox("Selecciona columna para test de normalidad:", num_cols, key="normal_col_sel")
        clean = df[normal_col].dropna()

        if len(clean) >= 3:
            c1, c2, c3 = st.columns(3)
            with c1:
                if len(clean) <= 5000:
                    sw_stat, sw_p = stats.shapiro(clean)
                    st.metric("Shapiro-Wilk", f"p={sw_p:.4f}")
                    st.caption("Normal" if sw_p > 0.05 else "No normal")
                else:
                    st.info("Shapiro-Wilk: requiere ≤ 5000 muestras")
            with c2:
                ks_stat, ks_p = stats.kstest(clean, "norm", args=(clean.mean(), clean.std()))
                st.metric("Kolmogorov-Smirnov", f"p={ks_p:.4f}")
                st.caption("Normal" if ks_p > 0.05 else "No normal")
            with c3:
                aa_result = stats.anderson(clean, dist="norm")
                st.metric("Anderson-Darling", f"A²={aa_result.statistic:.4f}")
                is_normal_aa = aa_result.statistic < aa_result.critical_values[2]
                st.caption("Normal" if is_normal_aa else "No normal")

            st.divider()
            st.markdown("#### Visualización de Normalidad")
            v1, v2 = st.columns(2)
            with v1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=clean, nbinsx=50, name="Datos", marker_color="#5b6af0", opacity=0.7, histnorm="probability density"))
                x_range = np.linspace(clean.min(), clean.max(), 200)
                fig_hist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, clean.mean(), clean.std()), name="Curva Normal", line=dict(color="#2dd4bf", width=2)))
                fig_hist.update_layout(title=f"Histograma + Curva Normal: {normal_col}", xaxis_title=normal_col, yaxis_title="Densidad", **DARK, height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
            with v2:
                qq = stats.probplot(clean, dist="norm")
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Datos", marker=dict(color="#5b6af0", size=4)))
                fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0]*qq[0][0]+qq[1][1], mode="lines", name="Referencia", line=dict(color="#2dd4bf", width=2)))
                fig_qq.update_layout(title=f"QQ-Plot: {normal_col}", xaxis_title="Cuantiles Teóricos", yaxis_title="Cuantiles Muestrales", **DARK, height=350)
                st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.info("Se necesitan al menos 3 valores no nulos.")
    else:
        st.info("No hay columnas numéricas.")

    st.divider()

    # ── 5.3: Pruebas de Hipótesis ──────────────────────────────────────────────
    st.markdown("### 🧪 Pruebas de Hipótesis")
    st.caption("Compara distribuciones entre grupos o contra un valor de referencia.")

    if num_cols:
        hyp_type = st.radio("Tipo de prueba", ["t-test (1 muestra)", "t-test (2 muestras)", "ANOVA (≥3 grupos)", "Mann-Whitney U (no paramétrica)"], horizontal=True, key="hyp_type")

        if hyp_type == "t-test (1 muestra)":
            col_1s = st.selectbox("Columna a testear:", num_cols, key="hyp_1s_col")
            ref_val = st.number_input("Valor de referencia (H₀: μ = )", value=0.0, key="hyp_1s_ref")
            clean = df[col_1s].dropna()
            if len(clean) >= 2:
                t_stat, p_val = stats.ttest_1samp(clean, ref_val)
                c1, c2, c3 = st.columns(3)
                c1.metric("t-statistic", f"{t_stat:.4f}")
                c2.metric("p-value", f"{p_val:.4f}")
                c3.metric("Resultado", "Rechazar H₀" if p_val < 0.05 else "No rechazar H₀")
                st.caption(f"La media de '{col_1s}' {'difiere significativamente' if p_val < 0.05 else 'NO difiere significativamente'} de {ref_val} (α=0.05)")

        elif hyp_type == "t-test (2 muestras)":
            col_2s = st.selectbox("Variable numérica:", num_cols, key="hyp_2s_col")
            group_col = st.selectbox("Variable de agrupación (categórica):", cat_cols + [c for c in num_cols if df[c].nunique() <= 10], key="hyp_2s_group")
            groups = df.groupby(group_col)[col_2s].apply(list)
            groups = {k: [v for v in vals if pd.notna(v)] for k, vals in groups.items() if len([v for v in vals if pd.notna(v)]) >= 2}
            if len(groups) == 2:
                g1, g2 = list(groups.values())
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                c1, c2, c3 = st.columns(3)
                c1.metric("t-statistic", f"{t_stat:.4f}")
                c2.metric("p-value", f"{p_val:.4f}")
                c3.metric("Resultado", "Rechazar H₀" if p_val < 0.05 else "No rechazar H₀")
                st.caption(f"Las medias entre grupos de '{group_col}' {'difieren significativamente' if p_val < 0.05 else 'NO difieren significativamente'} (α=0.05)")
            else:
                st.info(f"Se necesitan exactamente 2 grupos. '{group_col}' tiene {len(groups)} grupos válidos.")

        elif hyp_type == "ANOVA (≥3 grupos)":
            col_anova = st.selectbox("Variable numérica:", num_cols, key="hyp_anova_col")
            group_anova = st.selectbox("Variable de agrupación:", cat_cols + [c for c in num_cols if df[c].nunique() >= 3], key="hyp_anova_group")
            groups_anova = df.groupby(group_anova)[col_anova].apply(list)
            groups_anova = {k: [v for v in vals if pd.notna(v)] for k, vals in groups_anova.items() if len([v for v in vals if pd.notna(v)]) >= 2}
            if len(groups_anova) >= 3:
                f_stat, p_val = stats.f_oneway(*list(groups_anova.values()))
                c1, c2, c3 = st.columns(3)
                c1.metric("F-statistic", f"{f_stat:.4f}")
                c2.metric("p-value", f"{p_val:.4f}")
                c3.metric("Resultado", "Rechazar H₀" if p_val < 0.05 else "No rechazar H₀")
                st.caption(f"Al menos un grupo de '{group_anova}' {'tiene media diferente' if p_val < 0.05 else 'NO tiene media diferente'} (α=0.05)")
            else:
                st.info(f"Se necesitan ≥3 grupos. '{group_anova}' tiene {len(groups_anova)} grupos válidos.")

        else:
            col_mw = st.selectbox("Variable numérica:", num_cols, key="hyp_mw_col")
            group_mw = st.selectbox("Variable de agrupación (2 grupos):", cat_cols + [c for c in num_cols if df[c].nunique() == 2], key="hyp_mw_group")
            groups_mw = df.groupby(group_mw)[col_mw].apply(list)
            groups_mw = {k: [v for v in vals if pd.notna(v)] for k, vals in groups_mw.items() if len([v for v in vals if pd.notna(v)]) >= 2}
            if len(groups_mw) == 2:
                g1, g2 = list(groups_mw.values())
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                c1, c2, c3 = st.columns(3)
                c1.metric("U-statistic", f"{u_stat:.4f}")
                c2.metric("p-value", f"{p_val:.4f}")
                c3.metric("Resultado", "Rechazar H₀" if p_val < 0.05 else "No rechazar H₀")
                st.caption(f"Las distribuciones entre grupos de '{group_mw}' {'difieren significativamente' if p_val < 0.05 else 'NO difieren significativamente'} (α=0.05)")
            else:
                st.info(f"Se necesitan exactamente 2 grupos.")
    else:
        st.info("No hay columnas numéricas.")

    st.divider()

    # ── 5.4: Correlación Avanzada ──────────────────────────────────────────────
    st.markdown("### 🔗 Correlación Avanzada")
    st.caption("Pearson para variables cuantitativas, Chi-cuadrado para variables cualitativas.")

    corr_type = st.radio("Tipo de correlación", ["Pearson (numérico-numérico)", "Chi-cuadrado (categórico-categórico)", "Punto-Biserial (numérico-categórico)"], horizontal=True, key="corr_adv_type")

    if corr_type == "Pearson (numérico-numérico)":
        if len(num_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                pearson_x = st.selectbox("Variable X:", num_cols, key="pearson_x")
            with c2:
                pearson_y = st.selectbox("Variable Y:", [c for c in num_cols if c != pearson_x], key="pearson_y")
            clean_pairs = df[[pearson_x, pearson_y]].dropna()
            if len(clean_pairs) >= 3:
                r, p = stats.pearsonr(clean_pairs[pearson_x], clean_pairs[pearson_y])
                c1, c2, c3 = st.columns(3)
                c1.metric("r (Pearson)", f"{r:.4f}")
                c2.metric("p-value", f"{p:.4f}")
                strength = "Fuerte" if abs(r) > 0.7 else "Moderada" if abs(r) > 0.4 else "Débil"
                direction = "Positiva" if r > 0 else "Negativa"
                c3.metric("Relación", f"{strength} {direction}")
                st.caption(f"{'Hay' if p < 0.05 else 'NO hay'} correlación lineal significativa entre '{pearson_x}' y '{pearson_y}' (α=0.05)")

                slope_p, intercept_p, _, _, _ = stats.linregress(clean_pairs[pearson_x], clean_pairs[pearson_y])
                x_range_p = np.linspace(clean_pairs[pearson_x].min(), clean_pairs[pearson_x].max(), 100)
                fig_scatter = px.scatter(clean_pairs, x=pearson_x, y=pearson_y, color_discrete_sequence=["#5b6af0"], title=f"{pearson_x} vs {pearson_y}")
                fig_scatter.add_trace(go.Scatter(x=x_range_p, y=slope_p*x_range_p+intercept_p, mode="lines", name="Tendencia", line=dict(color="#f59e0b", width=2)))
                fig_scatter.update_layout(**DARK, height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Se necesitan ≥3 pares de datos completos.")
        else:
            st.info("Se necesitan ≥2 columnas numéricas.")

    elif corr_type == "Chi-cuadrado (categórico-categórico)":
        if len(cat_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                chi_x = st.selectbox("Variable X:", cat_cols, key="chi_x")
            with c2:
                chi_y = st.selectbox("Variable Y:", [c for c in cat_cols if c != chi_x], key="chi_y")
            contingency = pd.crosstab(df[chi_x], df[chi_y])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                c1, c2, c3 = st.columns(3)
                c1.metric("χ²", f"{chi2:.4f}")
                c2.metric("p-value", f"{p:.4f}")
                c3.metric("Resultado", "Dependientes" if p < 0.05 else "Independientes")
                st.caption(f"Las variables '{chi_x}' y '{chi_y}' {'son dependientes' if p < 0.05 else 'son independientes'} (α=0.05)")

                st.markdown("#### Tabla de contingencia")
                st.dataframe(contingency, use_container_width=True)

                fig_chi = px.imshow(contingency, text_auto=True, color_continuous_scale="Blues", title=f"Contingencia: {chi_x} × {chi_y}")
                fig_chi.update_layout(**DARK, height=400)
                st.plotly_chart(fig_chi, use_container_width=True)
            else:
                st.info("Se necesitan ≥2 categorías en cada variable.")
        else:
            st.info("Se necesitan ≥2 columnas categóricas.")

    else:
        if num_cols and cat_cols:
            c1, c2 = st.columns(2)
            with c1:
                pb_num = st.selectbox("Variable numérica:", num_cols, key="pb_num")
            with c2:
                pb_cat = st.selectbox("Variable categórica (2 grupos):", [c for c in cat_cols if df[c].nunique() == 2] + [c for c in num_cols if df[c].nunique() == 2], key="pb_cat")
            groups_pb = df.groupby(pb_cat)[pb_num].apply(list)
            groups_pb = {k: [v for v in vals if pd.notna(v)] for k, vals in groups_pb.items() if len([v for v in vals if pd.notna(v)]) >= 2}
            if len(groups_pb) == 2:
                g1, g2 = list(groups_pb.values())
                all_vals = g1 + g2
                binary = [0]*len(g1) + [1]*len(g2)
                r_pb, p_pb = stats.pointbiserialr(binary, all_vals)
                c1, c2, c3 = st.columns(3)
                c1.metric("r_pb", f"{r_pb:.4f}")
                c2.metric("p-value", f"{p_pb:.4f}")
                c3.metric("Relación", "Significativa" if p_pb < 0.05 else "No significativa")
                st.caption(f"{'Hay' if p_pb < 0.05 else 'NO hay'} asociación significativa entre '{pb_num}' y '{pb_cat}' (α=0.05)")
            else:
                st.info(f"Se necesitan exactamente 2 grupos. '{pb_cat}' tiene {len(groups_pb)} grupos.")
        else:
            st.info("Se necesitan columnas numéricas y categóricas.")

    st.divider()

    # ── 5.5: Análisis de Residuos ──────────────────────────────────────────────
    st.markdown("### 📉 Análisis de Residuos (Regresión Lineal)")
    st.caption("Valida los supuestos de un modelo de regresión lineal simple.")

    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            res_x = st.selectbox("Variable independiente (X):", num_cols, key="res_x")
        with c2:
            res_y = st.selectbox("Variable dependiente (Y):", [c for c in num_cols if c != res_x], key="res_y")
        clean_res = df[[res_x, res_y]].dropna()
        if len(clean_res) >= 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_res[res_x], clean_res[res_y])
            y_pred = slope * clean_res[res_x] + intercept
            residuals = clean_res[res_y] - y_pred

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pendiente", f"{slope:.4f}")
            c2.metric("R²", f"{r_value**2:.4f}")
            c3.metric("p-value", f"{p_value:.4f}")
            c4.metric("Std Error", f"{std_err:.4f}")

            st.divider()
            st.markdown("#### Supuestos de Regresión")

            sv1, sv2, sv3 = st.columns(3)
            with sv1:
                st.markdown("**1. Normalidad de Residuos**")
                sw_res, sw_p_res = stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan)
                if not np.isnan(sw_p_res):
                    st.caption(f"Shapiro-Wilk p={sw_p_res:.4f}")
                    st.caption("✅ Normal" if sw_p_res > 0.05 else "❌ No normal")
                else:
                    st.caption("Demasiadas muestras para Shapiro-Wilk")
                fig_res_hist = px.histogram(residuals, nbins=30, title="Distribución de Residuos", color_discrete_sequence=["#5b6af0"])
                fig_res_hist.update_layout(**DARK, height=280)
                st.plotly_chart(fig_res_hist, use_container_width=True)

            with sv2:
                st.markdown("**2. Homocedasticidad**")
                fig_resid = px.scatter(x=y_pred, y=residuals, title="Residuos vs Predichos", labels={"x": "Valores Predichos", "y": "Residuos"}, color_discrete_sequence=["#2dd4bf"])
                fig_resid.add_hline(y=0, line_dash="dash", line_color="#f43f5e")
                fig_resid.update_layout(**DARK, height=280)
                st.plotly_chart(fig_resid, use_container_width=True)
                st.caption("Varianza constante si los puntos se distribuyen aleatoriamente alrededor de 0")

            with sv3:
                st.markdown("**3. Independencia (Durbin-Watson)**")
                diff = np.diff(residuals)
                dw = np.sum(diff**2) / np.sum(residuals**2)
                st.metric("Durbin-Watson", f"{dw:.4f}")
                st.caption("✅ Independiente" if 1.5 < dw < 2.5 else "⚠️ Posible autocorrelación")

            st.divider()
            st.markdown("#### QQ-Plot de Residuos")
            qq_res = stats.probplot(residuals, dist="norm")
            fig_qq_res = go.Figure()
            fig_qq_res.add_trace(go.Scatter(x=qq_res[0][0], y=qq_res[0][1], mode="markers", name="Residuos", marker=dict(color="#5b6af0", size=4)))
            fig_qq_res.add_trace(go.Scatter(x=qq_res[0][0], y=qq_res[1][0]*qq_res[0][0]+qq_res[1][1], mode="lines", name="Referencia", line=dict(color="#2dd4bf", width=2)))
            fig_qq_res.update_layout(title="QQ-Plot de Residuos", xaxis_title="Cuantiles Teóricos", yaxis_title="Cuantiles Muestrales", **DARK, height=350)
            st.plotly_chart(fig_qq_res, use_container_width=True)
        else:
            st.info("Se necesitan ≥5 pares de datos completos.")
    else:
        st.info("Se necesitan ≥2 columnas numéricas.")

    st.divider()

    # ── 5.6: Análisis de Componentes Principales (PCA) ─────────────────────────
    st.markdown("### 🧬 Análisis de Componentes Principales (PCA)")
    st.caption("Reduce la dimensionalidad identificando las direcciones de mayor varianza.")

    if len(num_cols) >= 2:
        from sklearn.decomposition import PCA
        from utils.preprocessing import RobustScaler

        pca_cols = st.multiselect("Columnas para PCA:", num_cols, default=num_cols[:min(6, len(num_cols))], key="pca_cols")
        if len(pca_cols) >= 2:
            pca_data = df[pca_cols].dropna()
            if len(pca_data) >= len(pca_cols):
                st.markdown("#### Tipo de Escalado")
                scaler_type = st.radio("Método de escalado:", ["StandardScaler (media=0, var=1)", "RobustScaler (mediana, IQR)", "MinMaxScaler (rango [0,1])"], horizontal=True, key="pca_scaler_type")

                if scaler_type == "StandardScaler (media=0, var=1)":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    pca_data_scaled = pd.DataFrame(scaler.fit_transform(pca_data), columns=pca_cols, index=pca_data.index)
                    scaler_label = "StandardScaler"
                elif scaler_type == "RobustScaler (mediana, IQR)":
                    scaler = RobustScaler()
                    pca_data_scaled = scaler.fit_transform(pca_data)
                    scaler_label = "RobustScaler"
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    pca_data_scaled = pd.DataFrame(scaler.fit_transform(pca_data), columns=pca_cols, index=pca_data.index)
                    scaler_label = "MinMaxScaler"

                st.info(f"Escalador activo: **{scaler_label}** — {'Usa mediana e IQR, robusto a outliers' if 'Robust' in scaler_label else 'Estandariza usando media y desviación estándar' if 'Standard' in scaler_label else 'Escala al rango [0, 1]'}")

                n_components = st.slider("Número de componentes", 2, min(len(pca_cols), 10), min(3, len(pca_cols)), key="pca_ncomp")
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(pca_data_scaled)

                c1, c2 = st.columns(2)
                c1.metric("Varianza total explicada", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")
                c2.metric("Componentes", n_components)

                st.markdown("#### Varianza explicada por componente")
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(n_components)], y=pca.explained_variance_ratio_*100, name="Varianza individual", marker_color="#5b6af0"))
                cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
                fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_components)], y=cum_var, mode="lines+markers", name="Varianza acumulada", line=dict(color="#2dd4bf", width=3), marker=dict(size=8)))
                fig_var.update_layout(title="Varianza Explicada", xaxis_title="Componente Principal", yaxis_title="Varianza (%)", **DARK, height=350)
                st.plotly_chart(fig_var, use_container_width=True)

                st.markdown("#### Loadings (Pesos de cada feature en los componentes)")
                loadings_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=pca_cols)
                paginated_dataframe(loadings_df.round(4), key="pca_loadings", height=300)

                if n_components >= 2:
                    st.markdown("#### Proyección 2D (PC1 vs PC2)")
                    pca_2d_df = pd.DataFrame({"PC1": pca_result[:, 0], "PC2": pca_result[:, 1]}, index=pca_data.index)
                    pca_2d_df["_idx"] = pca_2d_df.index.astype(str)
                    fig_pca = px.scatter(pca_2d_df, x="PC1", y="PC2", title="PCA: PC1 vs PC2", color_discrete_sequence=["#5b6af0"], hover_data=["_idx"])
                    fig_pca.update_layout(**DARK, height=400)
                    st.plotly_chart(fig_pca, use_container_width=True)

                st.markdown("#### Matriz de componentes")
                fig_pca_heat = px.imshow(pca.components_, x=pca_cols, y=[f"PC{i+1}" for i in range(n_components)], color_continuous_scale="RdBu", text_auto=True, title="Loadings Heatmap")
                fig_pca_heat.update_layout(**DARK, height=max(300, len(pca_cols)*30))
                st.plotly_chart(fig_pca_heat, use_container_width=True)

                if "Robust" in scaler_label:
                    st.divider()
                    st.markdown("#### Parámetros del RobustScaler")
                    params = scaler.get_params()
                    params_df = pd.DataFrame({
                        "Columna": pca_cols,
                        "Mediana": [round(params["median"][c], 4) for c in pca_cols],
                        "Q1": [round(params["q1"][c], 4) for c in pca_cols],
                        "Q3": [round(params["q3"][c], 4) for c in pca_cols],
                        "IQR": [round(params["iqr"][c], 4) for c in pca_cols],
                    })
                    paginated_dataframe(params_df, key="robust_params", height=250, hide_index=True)
            else:
                st.info("Se necesitan más filas completas que columnas seleccionadas.")
        else:
            st.info("Selecciona al menos 2 columnas.")
    else:
        st.info("Se necesitan ≥2 columnas numéricas para PCA.")
