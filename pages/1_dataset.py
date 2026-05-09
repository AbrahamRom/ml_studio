import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.markdown("# 📂 Dataset")
st.markdown('<div class="section-header">Load & Configure</div>', unsafe_allow_html=True)

# ── Sample datasets ────────────────────────────────────────────────────────────
SAMPLES = {
    "🏠 Housing (Regression multioutput)": "housing_multi",
    "🌸 Iris (Clasificación)":             "iris",
    "💳 Credit Risk (Clasificación)":      "credit",
    "🌡️ Energy (Regresión multioutput)":   "energy",
    "🍷 Wine Quality (Regresión)":         "wine",
}

def load_sample(name):
    from sklearn.datasets import load_iris, load_wine, fetch_california_housing
    if name == "iris":
        d = load_iris(as_frame=True)
        df = d.frame
        return df
    elif name == "wine":
        d = load_wine(as_frame=True)
        df = d.frame
        return df
    elif name == "housing_multi":
        rng = np.random.default_rng(42)
        n = 500
        area    = rng.uniform(40, 300, n)
        rooms   = rng.integers(1, 8, n)
        age     = rng.integers(0, 50, n)
        price   = area * 1500 + rooms * 20000 - age * 800 + rng.normal(0, 15000, n)
        rent    = area * 6   + rooms * 80    - age * 3   + rng.normal(0, 60, n)
        quality = (price / price.max() * 8 + rng.normal(0, 0.5, n)).clip(1, 10).round(1)
        return pd.DataFrame({
            "area_m2": area.round(1), "rooms": rooms,
            "age_years": age, "price_eur": price.round(),
            "rent_eur": rent.round(1), "quality_score": quality,
        })
    elif name == "credit":
        rng = np.random.default_rng(0)
        n = 600
        income  = rng.uniform(10000, 100000, n)
        debt    = rng.uniform(0, 50000, n)
        score   = rng.integers(300, 850, n)
        default = ((debt / income > 0.4) | (score < 500)).astype(int)
        fraud   = rng.binomial(1, 0.05, n)
        return pd.DataFrame({
            "income": income.round(), "debt": debt.round(),
            "credit_score": score, "age": rng.integers(22, 70, n),
            "default": default, "fraud_flag": fraud,
        })
    elif name == "energy":
        rng = np.random.default_rng(7)
        n = 500
        compact = rng.uniform(0.6, 0.98, n)
        area    = rng.uniform(50, 250, n)
        glaz    = rng.uniform(0, 0.4, n)
        heat    = compact * -50 + area * 0.3 + glaz * 20 + rng.normal(0, 2, n) + 30
        cool    = compact * -20 + area * 0.2 + glaz * 30 + rng.normal(0, 2, n) + 15
        return pd.DataFrame({
            "compactness": compact.round(3), "surface_area": area.round(1),
            "glazing_area": glaz.round(3),
            "heating_load": heat.round(2), "cooling_load": cool.round(2),
        })

# ── UI ─────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Cargar dataset propio")
    uploaded = st.file_uploader("Sube un CSV / Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                sep = st.selectbox("Separador", [",", ";", "\t", "|"])
                df = pd.read_csv(uploaded, sep=sep)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"✅ Cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al leer: {e}")

with col2:
    st.markdown("#### O usa un dataset de ejemplo")
    sample_label = st.selectbox("Dataset de ejemplo", list(SAMPLES.keys()))
    if st.button("Cargar ejemplo"):
        key = SAMPLES[sample_label]
        st.session_state.df = load_sample(key)
        st.success("✅ Dataset de ejemplo cargado")

# ── Preview & config ───────────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val">{df.shape[0]:,}</div><div class="label">Filas</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{df.shape[1]}</div><div class="label">Columnas</div></div>', unsafe_allow_html=True)
    with c3:
        nulls = df.isnull().sum().sum()
        st.markdown(f'<div class="metric-card"><div class="val">{nulls}</div><div class="label">Nulos totales</div></div>', unsafe_allow_html=True)
    with c4:
        dupes = df.duplicated().sum()
        st.markdown(f'<div class="metric-card"><div class="val">{dupes}</div><div class="label">Duplicados</div></div>', unsafe_allow_html=True)

    st.markdown("### Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ Configurar Targets y Tarea")

    col_a, col_b = st.columns(2)
    with col_a:
        targets = st.multiselect(
            "Variables objetivo (target/s)",
            options=df.columns.tolist(),
            default=st.session_state.target_cols or [],
            help="Selecciona 1 target para modelo normal, 2+ para multioutput"
        )
    with col_b:
        task = st.selectbox(
            "Tipo de tarea",
            ["classification", "regression"],
            index=0 if st.session_state.task_type == "classification" else 1
            if st.session_state.task_type else 0
        )

    if targets:
        is_multi = len(targets) > 1
        badge = '<span class="tag teal">MULTIOUTPUT</span>' if is_multi else '<span class="tag">SINGLE TARGET</span>'
        st.markdown(f"**Modo:** {badge}", unsafe_allow_html=True)

        if st.button("✅ Confirmar configuración"):
            st.session_state.target_cols = targets
            st.session_state.task_type   = task
            st.session_state.multioutput = is_multi
            # Reset downstream state
            st.session_state.trained_models = None
            st.session_state.best_model     = None
            st.session_state.compare_df     = None
            st.session_state.setup_done     = None
            st.success(f"Configurado: {len(targets)} target(s) · tarea = {task}")
