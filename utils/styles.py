from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
GLOBAL_CSS_PATH = ROOT_DIR / "assets" / "global.css"


def load_global_css(css_path: Path = GLOBAL_CSS_PATH) -> None:
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
