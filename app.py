import streamlit as st
from utils.styles import load_global_css

st.set_page_config(
    page_title="ML Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme ───────────────────────────────────────────────────────────────
load_global_css()

# ── Session state init ─────────────────────────────────────────────────────────
for key in ["df", "task_type", "target_cols", "target_configs", "mode",
            "trained_models", "best_model", "setup_done", "automl_run",
            "compare_df", "multioutput"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ML Studio")
    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "📂  Dataset":        "1_dataset",
        "🔍  EDA & Quality":  "2_eda",
        "🏋️  Train Models":   "3_train",
        "📊  Compare":        "4_compare",
        "🔬  Evaluate":       "5_evaluate",
        "🧠  Explainability": "6_explain",
        "🚨  Early Warning":  "7_early_warning",
    }

    if "page" not in st.session_state:
        st.session_state.page = "1_dataset"

    for label, key in pages.items():
        active = st.session_state.page == key
        btn_label = f"▶ {label}" if active else label
        if st.button(btn_label, use_container_width=True, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

    st.divider()
    # Quick status
    st.markdown('<div class="section-header">Pipeline Status</div>', unsafe_allow_html=True)
    checks = {
        "Dataset loaded":   st.session_state.df is not None,
        "Targets selected": st.session_state.target_cols is not None,
        "Models trained":   st.session_state.automl_run is not None,
    }
    for label, done in checks.items():
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {label}")

# ── Route to page ──────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "1_dataset":
    exec(open("pages/1_dataset.py").read())
elif page == "2_eda":
    exec(open("pages/2_eda.py").read())
elif page == "3_train":
    exec(open("pages/3_train.py").read())
elif page == "4_compare":
    exec(open("pages/4_compare.py").read())
elif page == "5_evaluate":
    exec(open("pages/5_evaluate.py").read())
elif page == "6_explain":
    exec(open("pages/6_explain.py").read())
elif page == "7_early_warning":
    exec(open("pages/7_early_warning.py").read())
