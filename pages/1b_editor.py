import streamlit as st
import pandas as pd
import numpy as np
from utils.pagination import paginated_dataframe

st.markdown("# 🛠️ Editor de Dataset")
st.markdown('<div class="section-header">Editar, eliminar y crear columnas</div>', unsafe_allow_html=True)

# ── Guard clause ───────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.warning("⚠️ Carga un dataset primero en la sección **Dataset**.")
    st.stop()

df = st.session_state.df.copy()

# ── State for edit tracking ────────────────────────────────────────────────────
if "edit_history" not in st.session_state:
    st.session_state.edit_history = []

# ── Summary bar ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="val">{df.shape[0]:,}</div><div class="label">Filas</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="val">{df.shape[1]}</div><div class="label">Columnas</div></div>', unsafe_allow_html=True)
with c3:
    edits = len(st.session_state.edit_history)
    st.markdown(f'<div class="metric-card"><div class="val">{edits}</div><div class="label">Ediciones</div></div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_delete, tab_nulls, tab_edit, tab_create, tab_derived = st.tabs([
    "🗑️  Eliminar columnas",
    "🧹  Limpiar nulos",
    "✏️  Editar columnas",
    "➕  Crear columna vacía",
    "🔗  Derivar columna",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Delete columns
# ═══════════════════════════════════════════════════════════════════════════════
with tab_delete:
    st.markdown("### Selecciona columnas para eliminar")

    cols_to_delete = st.multiselect(
        "Columnas a eliminar",
        options=df.columns.tolist(),
        help="Las columnas seleccionadas se eliminarán del dataset.",
    )

    if cols_to_delete:
        st.markdown(f"**Se eliminarán {len(cols_to_delete)} columna(s):**")
        for c in cols_to_delete:
            dtype = df[c].dtype
            nulls = df[c].isnull().sum()
            st.caption(f"`{c}` — dtype: `{dtype}`, nulos: `{nulls}`")

        if st.button("🗑️  Eliminar seleccionadas", type="primary"):
            st.session_state.df = df.drop(columns=cols_to_delete)
            st.session_state.edit_history.append({
                "action": "delete",
                "columns": cols_to_delete,
            })
            st.success(f"✅ {len(cols_to_delete)} columna(s) eliminada(s).")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Clean nulls (dropna / fillna)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_nulls:
    st.markdown("### Manejo de valores nulos")

    null_summary = df.isnull().sum()
    cols_with_nulls = null_summary[null_summary > 0]

    if cols_with_nulls.empty:
        st.success("✅ No hay valores nulos en el dataset.")
    else:
        st.markdown(f"**{len(cols_with_nulls)} columna(s) con valores nulos:**")
        null_df = pd.DataFrame({
            "Columna": cols_with_nulls.index,
            "Nulos": cols_with_nulls.values,
            "%": (cols_with_nulls.values / len(df) * 100).round(2),
        })
        paginated_dataframe(null_df, key="editor_nulls", height=300, hide_index=True)

        st.divider()
        st.markdown("#### Opciones de limpieza")

        clean_mode = st.radio(
            "Método",
            ["Eliminar filas", "Eliminar columnas", "Rellenar valores"],
            horizontal=True,
            key="null_clean_mode",
        )

        if clean_mode == "Eliminar filas":
            scope = st.radio(
                "Alcance",
                ["Todo el dataset", "Solo columnas seleccionadas"],
                horizontal=True,
                key="dropna_scope",
            )

            selected_cols_drop = []
            if scope == "Solo columnas seleccionadas":
                selected_cols_drop = st.multiselect(
                    "Columnas a considerar",
                    options=cols_with_nulls.index.tolist(),
                    default=cols_with_nulls.index.tolist(),
                )

            how = st.radio("Criterio", ["any (al menos un nulo)", "all (todos nulos)"], horizontal=True, key="dropna_how")
            how_val = "any" if how == "any (al menos un nulo)" else "all"

            preview_rows = df.dropna(subset=selected_cols_drop if selected_cols_drop else None, how=how_val)
            rows_to_drop = len(df) - len(preview_rows)

            if rows_to_drop > 0:
                st.warning(f"Se eliminarán **{rows_to_drop:,} fila(s)**. Quedarán **{len(preview_rows):,} fila(s)**.")
            else:
                st.info("No se eliminaría ninguna fila con esta configuración.")

            if st.button("🧹  Eliminar filas con nulos", type="primary", key="dropna_btn"):
                if rows_to_drop > 0:
                    st.session_state.df = preview_rows.reset_index(drop=True)
                    st.session_state.edit_history.append({
                        "action": "dropna_rows",
                        "how": how_val,
                        "columns": selected_cols_drop if selected_cols_drop else "all",
                        "rows_dropped": rows_to_drop,
                    })
                    st.success(f"✅ {rows_to_drop:,} fila(s) eliminada(s).")
                    st.rerun()

        elif clean_mode == "Eliminar columnas":
            threshold_pct = st.slider(
                "Umbral: eliminar columnas con más de X% de nulos",
                min_value=0, max_value=100, value=50, step=5,
                key="dropcol_threshold",
            )

            cols_to_drop = null_summary[null_summary / len(df) * 100 > threshold_pct]
            if len(cols_to_drop) > 0:
                st.warning(f"Se eliminarán **{len(cols_to_drop)} columna(s):**")
                for col_name, count in cols_to_drop.items():
                    pct = count / len(df) * 100
                    st.caption(f"`{col_name}` — {count} nulos ({pct:.1f}%)")
            else:
                st.info("No hay columnas que superen el umbral.")

            if st.button("🧹  Eliminar columnas con nulos", type="primary", key="dropcol_btn"):
                if len(cols_to_drop) > 0:
                    st.session_state.df = df.drop(columns=cols_to_drop.index)
                    st.session_state.edit_history.append({
                        "action": "dropna_cols",
                        "threshold_pct": threshold_pct,
                        "columns_dropped": cols_to_drop.index.tolist(),
                    })
                    st.success(f"✅ {len(cols_to_drop)} columna(s) eliminada(s).")
                    st.rerun()

        else:
            fill_scope = st.radio(
                "Alcance",
                ["Todo el dataset", "Solo columnas seleccionadas"],
                horizontal=True,
                key="fillna_scope",
            )

            selected_cols_fill = []
            if fill_scope == "Solo columnas seleccionadas":
                selected_cols_fill = st.multiselect(
                    "Columnas a rellenar",
                    options=cols_with_nulls.index.tolist(),
                    default=cols_with_nulls.index.tolist(),
                )

            fill_method = st.radio(
                "Método de relleno",
                ["Valor fijo", "Media", "Mediana", "Moda", "Forward fill", "Backward fill"],
                horizontal=True,
                key="fillna_method",
            )

            fill_value = None
            if fill_method == "Valor fijo":
                fill_type = st.radio("Tipo", ["Número", "Texto"], horizontal=True, key="fillna_val_type")
                if fill_type == "Número":
                    fill_value = st.number_input("Valor", value=0.0, key="fillna_num_val")
                else:
                    fill_value = st.text_input("Valor", value="", key="fillna_str_val")

            if st.button("🧹  Rellenar nulos", type="primary", key="fillna_btn"):
                try:
                    filled = df.copy()
                    target_cols = selected_cols_fill if selected_cols_fill else filled.columns.tolist()

                    for col in target_cols:
                        if fill_method == "Valor fijo":
                            filled[col] = filled[col].fillna(fill_value)
                        elif fill_method == "Media":
                            if pd.api.types.is_numeric_dtype(filled[col]):
                                filled[col] = filled[col].fillna(filled[col].mean())
                        elif fill_method == "Mediana":
                            if pd.api.types.is_numeric_dtype(filled[col]):
                                filled[col] = filled[col].fillna(filled[col].median())
                        elif fill_method == "Moda":
                            mode_val = filled[col].mode()
                            if not mode_val.empty:
                                filled[col] = filled[col].fillna(mode_val.iloc[0])
                        elif fill_method == "Forward fill":
                            filled[col] = filled[col].fillna(method="ffill")
                        elif fill_method == "Backward fill":
                            filled[col] = filled[col].fillna(method="bfill")

                    n_filled = sum(df[col].isnull().sum() - filled[col].isnull().sum() for col in target_cols)
                    st.session_state.df = filled
                    st.session_state.edit_history.append({
                        "action": "fillna",
                        "method": fill_method,
                        "columns": target_cols if selected_cols_fill else "all",
                        "values_filled": int(n_filled),
                    })
                    st.success(f"✅ {int(n_filled)} valor(es) nulo(s) rellenado(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al rellenar: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Edit columns (rename, type convert)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.markdown("### Renombrar o cambiar tipo de columna")

    col_to_edit = st.selectbox(
        "Columna a editar",
        options=df.columns.tolist(),
    )

    if col_to_edit:
        col_data = df[col_to_edit]

        st.markdown(f"**Información de `{col_to_edit}`**")
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Tipo", str(col_data.dtype))
        with info_cols[1]:
            st.metric("Nulos", int(col_data.isnull().sum()))
        with info_cols[2]:
            st.metric("Únicos", int(col_data.nunique(dropna=True)))
        with info_cols[3]:
            st.metric("Filas", len(col_data))

        st.divider()

        edit_mode = st.radio("Modo de edición", ["Renombrar", "Convertir tipo"], horizontal=True)

        if edit_mode == "Renombrar":
            new_name = st.text_input("Nuevo nombre", value=col_to_edit)
            if new_name and new_name != col_to_edit:
                if new_name in df.columns:
                    st.error(f"Ya existe una columna llamada `{new_name}`.")
                elif st.button("✏️  Renombrar", type="primary"):
                    st.session_state.df = df.rename(columns={col_to_edit: new_name})
                    st.session_state.edit_history.append({
                        "action": "rename",
                        "old": col_to_edit,
                        "new": new_name,
                    })
                    st.success(f"✅ Renombrada `{col_to_edit}` → `{new_name}`.")
                    st.rerun()

        else:
            type_options = {
                "Numérico (float)": "float64",
                "Entero (Int64 nullable)": "Int64",
                "Texto (string)": "string",
                "Booleano (boolean)": "boolean",
                "Categórico (category)": "category",
                "Datetime": "datetime64[ns]",
            }

            current_type = str(col_data.dtype)
            selected_type = st.selectbox(
                "Convertir a",
                options=list(type_options.values()),
                index=list(type_options.values()).index(current_type) if current_type in type_options.values() else 0,
                format_func=lambda x: [k for k, v in type_options.items() if v == x][0],
            )

            if st.button("🔄  Convertir tipo", type="primary"):
                try:
                    converted = df.copy()
                    if selected_type == "datetime64[ns]":
                        converted[col_to_edit] = pd.to_datetime(df[col_to_edit])
                    elif selected_type == "boolean":
                        converted[col_to_edit] = df[col_to_edit].astype("boolean")
                    elif selected_type == "category":
                        converted[col_to_edit] = df[col_to_edit].astype("category")
                    elif selected_type == "Int64":
                        converted[col_to_edit] = pd.to_numeric(df[col_to_edit], errors="coerce").astype("Int64")
                    elif selected_type == "float64":
                        converted[col_to_edit] = pd.to_numeric(df[col_to_edit], errors="coerce")
                    elif selected_type == "string":
                        converted[col_to_edit] = df[col_to_edit].astype("string")

                    st.session_state.df = converted
                    st.session_state.edit_history.append({
                        "action": "type_convert",
                        "column": col_to_edit,
                        "from": current_type,
                        "to": selected_type,
                    })
                    st.success(f"✅ `{col_to_edit}` convertida a `{selected_type}`.")
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo convertir: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Create empty column
# ═══════════════════════════════════════════════════════════════════════════════
with tab_create:
    st.markdown("### Crear una nueva columna desde cero")

    new_col_name = st.text_input("Nombre de la columna", key="new_col_empty")

    if new_col_name:
        if new_col_name in df.columns:
            st.error(f"Ya existe una columna llamada `{new_col_name}`.")
        else:
            fill_mode = st.radio(
                "Valor de relleno",
                ["Ceros (0)", "Unos (1)", "Nulos (NaN)", "Texto vacío", "Valor personalizado"],
                horizontal=True,
            )

            custom_val = None
            if fill_mode == "Valor personalizado":
                custom_type = st.radio("Tipo del valor", ["Número", "Texto"], horizontal=True)
                if custom_type == "Número":
                    custom_val = st.number_input("Valor", value=0.0)
                else:
                    custom_val = st.text_input("Valor", value="")

            if st.button("➕  Crear columna", type="primary"):
                if fill_mode == "Ceros (0)":
                    df[new_col_name] = 0
                elif fill_mode == "Unos (1)":
                    df[new_col_name] = 1
                elif fill_mode == "Nulos (NaN)":
                    df[new_col_name] = np.nan
                elif fill_mode == "Texto vacío":
                    df[new_col_name] = ""
                elif fill_mode == "Valor personalizado":
                    df[new_col_name] = custom_val

                st.session_state.df = df
                st.session_state.edit_history.append({
                    "action": "create_empty",
                    "column": new_col_name,
                    "fill": fill_mode,
                })
                st.success(f"✅ Columna `{new_col_name}` creada.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Derive column from existing ones
# ═══════════════════════════════════════════════════════════════════════════════
with tab_derived:
    st.markdown("### Crear columna a partir de otras")

    derive_mode = st.radio(
        "Método",
        ["Expresión Python", "Interfaz visual", "Operaciones matemáticas"],
        horizontal=True,
        key="derive_mode",
    )

    # ── Sub-tab: Python expression ─────────────────────────────────────────
    if derive_mode == "Expresión Python":
        st.info(
            "Escribe una expresión Python que se evaluará sobre el DataFrame. "
            "Usa `df['columna']` para acceder a los datos. "
            "La expresión debe producir una serie del mismo largo que el DataFrame."
        )

        derived_name = st.text_input("Nombre de la nueva columna", key="derived_name_py")

        # Show available columns as reference
        st.markdown("**Columnas disponibles:**")
        col_chips = ", ".join([f"`{c}`" for c in df.columns])
        st.caption(col_chips)

        expression = st.text_area(
            "Expresión Python",
            value="",
            height=120,
            placeholder="Ejemplo: df['A'].apply(lambda x: 0 if x > 5 else 1)",
            key="py_expr",
        )

        # Quick examples
        with st.expander("📖 Ejemplos de expresiones"):
            st.markdown("""
```python
# Binaria: 0 si A > 5, sino 1
df['A'].apply(lambda x: 0 if x > 5 else 1)

# Basada en rango: categorizar valores
pd.cut(df['edad'], bins=[0, 18, 35, 60, 100], labels=['niño', 'joven', 'adulto', 'mayor'])

# Operación entre columnas
df['precio'] / df['cantidad']

# Conditional con numpy
np.where(df['A'] > df['B'], 'alto', 'bajo')

# String operations
df['nombre'].str.upper()

# Booleana
(df['edad'] >= 18) & (df['activo'] == 1)
```
""")

        if derived_name and expression:
            if derived_name in df.columns:
                st.error(f"Ya existe una columna llamada `{derived_name}`.")
            else:
                if st.button("🔗  Crear columna desde expresión", type="primary"):
                    try:
                        result = eval(expression, {"df": df, "pd": pd, "np": np})
                        if isinstance(result, pd.Series):
                            if len(result) != len(df):
                                st.error(f"La expresión produce {len(result)} valores pero se necesitan {len(df)}.")
                            else:
                                df[derived_name] = result
                                st.session_state.df = df
                                st.session_state.edit_history.append({
                                    "action": "derive_python",
                                    "column": derived_name,
                                    "expression": expression,
                                })
                                st.success(f"✅ Columna `{derived_name}` creada.")
                                st.rerun()
                        elif isinstance(result, (list, np.ndarray)):
                            if len(result) != len(df):
                                st.error(f"La expresión produce {len(result)} valores pero se necesitan {len(df)}.")
                            else:
                                df[derived_name] = list(result)
                                st.session_state.df = df
                                st.session_state.edit_history.append({
                                    "action": "derive_python",
                                    "column": derived_name,
                                    "expression": expression,
                                })
                                st.success(f"✅ Columna `{derived_name}` creada.")
                                st.rerun()
                        else:
                            # Scalar: broadcast
                            df[derived_name] = result
                            st.session_state.df = df
                            st.session_state.edit_history.append({
                                "action": "derive_python",
                                "column": derived_name,
                                "expression": expression,
                            })
                            st.success(f"✅ Columna `{derived_name}` creada.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error al evaluar la expresión: {e}")

    # ── Sub-tab: Visual interface ──────────────────────────────────────────
    elif derive_mode == "Interfaz visual":
        derived_name_vis = st.text_input("Nombre de la nueva columna", key="derived_name_vis")

        source_col = st.selectbox(
            "Columna origen",
            options=df.columns.tolist(),
            key="vis_source_col",
        )

        if source_col and derived_name_vis:
            if derived_name_vis in df.columns:
                st.error(f"Ya existe una columna llamada `{derived_name_vis}`.")
            else:
                # Preview source column
                src_dtype = str(df[source_col].dtype)
                is_numeric = pd.api.types.is_numeric_dtype(df[source_col])

                st.markdown(f"**Columna origen:** `{source_col}` (tipo: `{src_dtype}`)")

                def parse_output(v):
                    try:
                        return float(v) if '.' in v else int(v)
                    except (ValueError, TypeError):
                        return v

                if is_numeric:
                    condition_type = st.radio(
                        "Tipo de condición",
                        ["Mayor que", "Menor que", "Mayor o igual", "Menor o igual", "Igual a", "Distinto de", "Entre valores", "Fuera de rango"],
                        horizontal=True,
                        key="vis_cond_type",
                    )

                    threshold_a = st.number_input("Valor umbral", value=0.0, key="vis_threshold_a")
                    threshold_b = None
                    if condition_type in ("Entre valores", "Fuera de rango"):
                        threshold_b = st.number_input("Segundo valor", value=0.0, key="vis_threshold_b")

                    # Output values
                    st.markdown("**Valores de salida:**")
                    out_cols = st.columns(2)
                    with out_cols[0]:
                        val_true = st.text_input("Si cumple condición", value="1", key="vis_val_true")
                    with out_cols[1]:
                        val_false = st.text_input("Si no cumple", value="0", key="vis_val_false")

                    if st.button("🔗  Crear columna", type="primary", key="vis_create_btn"):
                        try:
                            vt = parse_output(val_true)
                            vf = parse_output(val_false)
                            col = df[source_col]

                            if condition_type == "Mayor que":
                                mask = col > threshold_a
                            elif condition_type == "Menor que":
                                mask = col < threshold_a
                            elif condition_type == "Mayor o igual":
                                mask = col >= threshold_a
                            elif condition_type == "Menor o igual":
                                mask = col <= threshold_a
                            elif condition_type == "Igual a":
                                mask = col == threshold_a
                            elif condition_type == "Distinto de":
                                mask = col != threshold_a
                            elif condition_type == "Entre valores":
                                lo, hi = min(threshold_a, threshold_b), max(threshold_a, threshold_b)
                                mask = (col >= lo) & (col <= hi)
                            elif condition_type == "Fuera de rango":
                                lo, hi = min(threshold_a, threshold_b), max(threshold_a, threshold_b)
                                mask = (col < lo) | (col > hi)

                            df[derived_name_vis] = np.where(mask, vt, vf)
                            st.session_state.df = df
                            st.session_state.edit_history.append({
                                "action": "derive_visual",
                                "column": derived_name_vis,
                                "source": source_col,
                                "condition": condition_type,
                                "threshold_a": threshold_a,
                                "threshold_b": threshold_b,
                                "val_true": vt,
                                "val_false": vf,
                            })
                            st.success(f"✅ Columna `{derived_name_vis}` creada.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                else:
                    # Categorical/text conditions
                    unique_vals = df[source_col].dropna().unique()[:50].tolist()
                    condition_cat = st.radio(
                        "Condición",
                        ["Es igual a", "No es igual a", "Contiene texto", "Está en lista"],
                        horizontal=True,
                        key="vis_cat_cond",
                    )

                    if condition_cat == "Está en lista":
                        selected_cats = st.multiselect(
                            "Selecciona valores",
                            options=unique_vals,
                            key="vis_cat_list",
                        )
                    elif condition_cat == "Contiene texto":
                        search_text = st.text_input("Texto a buscar", key="vis_search_text")
                    else:
                        match_value = st.selectbox("Valor", options=unique_vals, key="vis_match_val")

                    out_cols2 = st.columns(2)
                    with out_cols2[0]:
                        val_true2 = st.text_input("Si cumple", value="1", key="vis_val_true2")
                    with out_cols2[1]:
                        val_false2 = st.text_input("Si no cumple", value="0", key="vis_val_false2")

                    if st.button("🔗  Crear columna", type="primary", key="vis_cat_create_btn"):
                        try:
                            vt2 = parse_output(val_true2)
                            vf2 = parse_output(val_false2)
                            col = df[source_col].astype(str)

                            if condition_cat == "Es igual a":
                                mask = col == str(match_value)
                            elif condition_cat == "No es igual a":
                                mask = col != str(match_value)
                            elif condition_cat == "Contiene texto":
                                mask = col.str.contains(str(search_text), na=False, case=False)
                            elif condition_cat == "Está en lista":
                                mask = col.isin([str(v) for v in selected_cats])

                            df[derived_name_vis] = np.where(mask, vt2, vf2)
                            st.session_state.df = df
                            st.session_state.edit_history.append({
                                "action": "derive_visual_cat",
                                "column": derived_name_vis,
                                "source": source_col,
                                "condition": condition_cat,
                                "val_true": vt2,
                                "val_false": vf2,
                            })
                            st.success(f"✅ Columna `{derived_name_vis}` creada.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ── Sub-tab: Math operations ───────────────────────────────────────────
    else:
        derived_name_math = st.text_input("Nombre de la nueva columna", key="derived_name_math")

        if derived_name_math and derived_name_math in df.columns:
            st.error(f"Ya existe una columna llamada `{derived_name_math}`.")
        elif derived_name_math:
            col_a = st.selectbox("Columna A", options=df.columns.tolist(), key="math_col_a")
            col_b = st.selectbox("Columna B", options=df.columns.tolist(), key="math_col_b")

            operation = st.selectbox(
                "Operación",
                ["A + B", "A - B", "A * B", "A / B", "A % B", "A ** B (potencia)",
                 "abs(A - B)", "max(A, B)", "min(A, B)", "A / (A + B)"],
                key="math_op",
            )

            handle_errors = st.checkbox("Reemplazar infinitos/errores con NaN", value=True)

            if st.button("🔗  Crear columna", type="primary", key="math_create_btn"):
                try:
                    a = pd.to_numeric(df[col_a], errors="coerce")
                    b = pd.to_numeric(df[col_b], errors="coerce")

                    if operation == "A + B":
                        result = a + b
                    elif operation == "A - B":
                        result = a - b
                    elif operation == "A * B":
                        result = a * b
                    elif operation == "A / B":
                        result = a / b
                    elif operation == "A % B":
                        result = a % b
                    elif operation == "A ** B (potencia)":
                        result = a ** b
                    elif operation == "abs(A - B)":
                        result = (a - b).abs()
                    elif operation == "max(A, B)":
                        result = pd.concat([a, b], axis=1).max(axis=1)
                    elif operation == "min(A, B)":
                        result = pd.concat([a, b], axis=1).min(axis=1)
                    elif operation == "A / (A + B)":
                        result = a / (a + b)

                    if handle_errors:
                        result = result.replace([np.inf, -np.inf], np.nan)

                    df[derived_name_math] = result
                    st.session_state.df = df
                    st.session_state.edit_history.append({
                        "action": "derive_math",
                        "column": derived_name_math,
                        "col_a": col_a,
                        "col_b": col_b,
                        "operation": operation,
                    })
                    st.success(f"✅ Columna `{derived_name_math}` creada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Edit history ───────────────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Historial de ediciones"):
    if st.session_state.edit_history:
        for i, edit in enumerate(st.session_state.edit_history, 1):
            action = edit.get("action", "")
            if action == "delete":
                label = f"🗑️  Eliminar columnas: {', '.join(edit.get('columns', []))}"
            elif action == "dropna_rows":
                label = f"🧹  Eliminar {edit.get('rows_dropped', 0):,} fila(s) con nulos (how={edit.get('how', 'any')})"
            elif action == "dropna_cols":
                label = f"🧹  Eliminar {len(edit.get('columns_dropped', []))} columna(s) con >{edit.get('threshold_pct', 0)}% nulos"
            elif action == "fillna":
                label = f"🧹  Rellenar {edit.get('values_filled', 0):,} nulo(s) con {edit.get('method', '?')}"
            elif action == "rename":
                label = f"✏️  Renombrar `{edit.get('old', '?')}` → `{edit.get('new', '?')}`"
            elif action == "type_convert":
                label = f"🔄  Convertir `{edit.get('column', '?')}`: `{edit.get('from', '?')}` → `{edit.get('to', '?')}`"
            elif action == "create_empty":
                label = f"➕  Crear columna vacía `{edit.get('column', '?')}` (relleno: {edit.get('fill', '?')})"
            elif action == "derive_python":
                label = f"🔗  Derivar `{edit.get('column', '?')}` con expresión Python"
            elif action == "derive_visual":
                label = f"🔗  Derivar `{edit.get('column', '?')}` de `{edit.get('source', '?')}` (condición: {edit.get('condition', '?')})"
            elif action == "derive_visual_cat":
                label = f"🔗  Derivar `{edit.get('column', '?')}` de `{edit.get('source', '?')}` (categórica: {edit.get('condition', '?')})"
            elif action == "derive_math":
                op = edit.get("operation", "?")
                op = op.replace("A", edit.get("col_a", "A")).replace("B", edit.get("col_b", "B"))
                label = f"🔗  Derivar `{edit.get('column', '?')}`: {op}"
            else:
                label = action
            st.caption(f"{i}. {label}")
    else:
        st.caption("Sin ediciones aún.")

    if st.session_state.edit_history and st.button("🔄  Restaurar dataset original", type="secondary"):
        st.warning("Esto descartará TODAS las ediciones. ¿Continuar?")
        if st.button("Sí, restaurar original", type="primary"):
            st.info("Para restaurar, recarga el dataset en la página **Dataset**.")

# ── Save edited dataset ────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💾 Guardar dataset editado")

save_filename = st.text_input("Nombre del archivo", value="dataset_editado.csv", key="save_filename")
save_format = st.radio("Formato", ["CSV", "CSV con punto y coma", "Excel"], horizontal=True, key="save_format")

if st.button("💾  Guardar archivo", type="primary", key="save_dataset_btn"):
    try:
        import os
        from pathlib import Path

        if not save_filename.endswith((".csv", ".xlsx")):
            save_filename += ".csv" if save_format != "Excel" else ".xlsx"

        save_dir = Path(".")
        save_path = save_dir / save_filename

        total_rows = len(st.session_state.df)
        total_cols = len(st.session_state.df.columns)

        if save_format == "CSV":
            st.session_state.df.to_csv(save_path, index=False, encoding="utf-8")
        elif save_format == "CSV con punto y coma":
            st.session_state.df.to_csv(save_path, index=False, encoding="utf-8", sep=";")
        elif save_format == "Excel":
            st.session_state.df.to_excel(save_path, index=False)

        st.success(f"✅ Dataset guardado: `{save_path}` ({total_rows:,} filas × {total_cols} columnas)")

        with open(save_path, "rb") as f:
            st.download_button(
                label="⬇️  Descargar archivo",
                data=f,
                file_name=save_filename,
                mime="application/octet-stream",
                key="download_edited_dataset",
            )
    except Exception as e:
        st.error(f"Error al guardar: {e}")

# ── Current dataset preview ────────────────────────────────────────────────────
st.divider()
st.markdown(f"### Dataset actual ({len(st.session_state.df):,} filas, {len(st.session_state.df.columns)} columnas)")
paginated_dataframe(st.session_state.df, key="editor_preview", height=450)
