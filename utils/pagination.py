"""Reusable pagination component for Streamlit dataframes."""

import streamlit as st
import pandas as pd


def paginated_dataframe(df: pd.DataFrame, key: str = "table", height: int = 400, **kwargs):
    """Display a DataFrame with pagination controls.
    
    Args:
        df: DataFrame to display
        key: Unique key for session state
        height: Height of the dataframe component
        **kwargs: Additional kwargs passed to st.dataframe
    """
    if df is None or df.empty:
        st.info("No hay datos para mostrar.")
        return

    total_rows = len(df)
    page_size_options = [10, 25, 50, 100, 200, "Todas"]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        page_size = st.selectbox(
            "Filas por página",
            options=page_size_options,
            index=1,
            key=f"{key}_page_size",
        )
    
    if page_size == "Todas":
        page_size = total_rows
    
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    
    with col2:
        if total_pages > 1:
            current_page = st.slider(
                "Página",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"{key}_page",
            )
        else:
            current_page = 1

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    page_df = df.iloc[start_idx:end_idx]
    
    st.caption(f"Mostrando {start_idx + 1}-{end_idx} de {total_rows} filas")
    st.dataframe(page_df, use_container_width=True, height=height, **kwargs)
