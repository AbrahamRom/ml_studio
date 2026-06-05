from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline.automl_runner import resolve_model_by_name

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

# ---------------------------------------------------------------------------
# Compatibility patches for numpy 2.x / pandas 3.x
# Some XAI libraries (shap 0.52.0, lime) call .copy(deep=True) on numpy arrays,
# which fails in numpy 2.x. This patch makes it a no-op.
# ---------------------------------------------------------------------------
_original_ndarray_copy = np.ndarray.copy


def _patched_ndarray_copy(self, order="C", deep=None):
    return _original_ndarray_copy(self, order=order)


# numpy 2.x raises TypeError: cannot set 'copy' attribute of immutable type 'numpy.ndarray'.
# We wrap the assignment in try/except so the patch applies silently on older numpy
# and is safely skipped on numpy 2.x (where ndarray.copy is read-only).
try:
    np.ndarray.copy = _patched_ndarray_copy
except (TypeError, AttributeError):
    pass

st.markdown("# 🧠 Explainabilidad")

if st.session_state.automl_run is None:
    st.warning("⚠️ Entrena AutoML primero.")
    st.stop()

run = st.session_state.automl_run
target_results = run["target_results"]
targets = list(target_results.keys())

target = st.selectbox("Target a explicar", targets)
result = target_results[target]
config = result["config"]
automl = result.get("automl")
X_test = result.get("X_test")
feature_cols = result.get("feature_cols") or []
has_model = automl is not None and X_test is not None
is_classification = config["task"] == "classification"

st.markdown(
    f"**Target:** `{target}` · **Mejor modelo según holdout real:** `{result.get('best_model_name')}` · "
    f"**Tipo:** `{result.get('best_model_type')}`"
)
st.caption(f"Reporte mljar: `{result['results_path']}`")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📄 Reporte MLJAR",
        "🔀 Permutation Importance",
        "🎯 SHAP",
        "🔍 LIME",
        "📈 PDP / ICE",
        "📊 Leaderboard",
        "🔮 Predicción Manual",
    ]
)

# ---------------------------------------------------------------------------
# Helper: prepare numeric-only DataFrame for XAI methods
# ---------------------------------------------------------------------------
def _prepare_numeric_xai(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Convert categorical/string columns to numeric (int) for XAI use.

    The mljar model's predict() expects categorical features as integer or string
    types, not float. Returns a DataFrame with:
    - Numeric columns preserved as float
    - Label-encoded categorical columns as int
    The original column order is preserved.
    """
    from sklearn.preprocessing import LabelEncoder

    prepared = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            prepared[col] = pd.to_numeric(series, errors="coerce").fillna(0)
        elif series.nunique(dropna=True) <= 2:
            # Binary categorical: map to 0/1 as int
            categories = series.dropna().unique().tolist()
            mapping = {cat: i for i, cat in enumerate(sorted(categories))}
            prepared[col] = series.map(mapping).fillna(0).astype(int)
        else:
            # Multi-categorical: label encode as int
            le = LabelEncoder()
            non_null = series.dropna()
            if len(non_null) > 0:
                encoded = le.fit_transform(non_null)
                prepared[col] = series.map(
                    dict(zip(non_null, encoded))
                ).fillna(0).astype(int)
            else:
                prepared[col] = 0

    return prepared


def _get_individual_model(automl, best_model_name: str):
    """Get the best individual fitted model from the AutoML ensemble.

    Returns a callable model suitable for predictions.
    """
    model = resolve_model_by_name(automl, best_model_name)
    if model is not None:
        return model
    return automl


def _predict_with_original_dtypes(model, X_np: np.ndarray, original_df: pd.DataFrame):
    """Call model.predict() on X_np after restoring original column dtypes.

    numpy arrays homogenize all types to float64 when the DataFrame has mixed
    dtypes. The mljar model expects categorical columns as int (not float), so
    we recast each column to its original dtype before calling predict.
    """
    X_df = pd.DataFrame(X_np, columns=original_df.columns, index=original_df.index)
    for col in original_df.columns:
        orig_dtype = original_df[col].dtype
        if orig_dtype != X_df[col].dtype:
            try:
                X_df[col] = X_df[col].astype(orig_dtype)
            except (ValueError, TypeError):
                pass  # keep as-is if casting fails

    if hasattr(model, "predict"):
        return model.predict(X_df)
    elif callable(model):
        return model(X_df)
    else:
        raise ValueError("Model does not expose predict() and is not callable.")


def _manual_partial_dependence(
    model,
    X: pd.DataFrame,
    feature_idx: int,
    grid: np.ndarray,
    kind: str = "average",
):
    """Compute PDP / ICE manually without requiring sklearn-compatible estimator.

    Works with any model that exposes predict() or __call__().
    """
    X_np = X.values  # (n_samples, n_features) — may homogenize to float
    n_samples = X_np.shape[0]
    n_grid = len(grid)

    if kind == "individual":
        # ICE: return predictions for each sample × each grid value
        ice = np.zeros((n_samples, n_grid))
        for j, val in enumerate(grid):
            X_perturbed = X_np.copy()
            X_perturbed[:, feature_idx] = val
            preds = _predict_with_original_dtypes(model, X_perturbed, X)
            ice[:, j] = np.asarray(preds).ravel()
        return {"average": ice.mean(axis=0, keepdims=True), "individual": ice, "values": grid}
    else:
        # PDP: average over all samples for each grid value
        pdp = np.zeros(n_grid)
        for j, val in enumerate(grid):
            X_perturbed = X_np.copy()
            X_perturbed[:, feature_idx] = val
            preds = _predict_with_original_dtypes(model, X_perturbed, X)
            pdp[j] = np.asarray(preds).ravel().mean()
        return {"average": pdp[np.newaxis, :], "individual": None, "values": grid}


# ---------------------------------------------------------------------------
# TAB 1 – Reporte MLJAR (sin cambios)
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### Artefactos explicativos generados por mljar")
    report_path = Path(result["results_path"])
    readme = report_path / "README.md"
    if readme.exists():
        st.info(f"Reporte principal: `{readme}`")
    else:
        st.info("mljar no generó un README principal para este target.")

    image_files = sorted(
        [
            path
            for path in report_path.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and any(token in path.name.lower() for token in ["importance", "shap", "tree", "learning"])
        ]
    )
    if not image_files:
        st.caption("No se encontraron imágenes de importancia/SHAP en el reporte guardado.")
    else:
        selected_image = st.selectbox(
            "Gráfica guardada",
            image_files,
            format_func=lambda p: str(p.relative_to(report_path)),
        )
        st.image(str(selected_image), use_column_width=True)

# ---------------------------------------------------------------------------
# TAB 2 – Permutation Importance (sin cambios)
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### Permutation Importance del mejor AutoML")
    st.caption("Mide cuánto cae la métrica cuando se mezcla una feature en el holdout.")
    max_repeats = 20
    n_repeats = st.slider("Repeticiones", 3, max_repeats, 8)
    scoring = "neg_root_mean_squared_error" if config["task"] == "regression" else "f1_weighted"
    if not has_model:
        st.info("El modelo no está disponible en memoria. Esto puede ocurrir si la corrida se guardó antes de que se implementara la persistencia del modelo. Entrena nuevamente los modelos para habilitar esta funcionalidad.")
    else:
        if st.button("⚙️ Calcular importancia", use_container_width=True):
            with st.spinner("Calculando permutation importance..."):
                try:
                    from sklearn.inspection import permutation_importance

                    perm = permutation_importance(
                        automl,
                        X_test,
                        result["y_test"],
                        scoring=scoring,
                        n_repeats=n_repeats,
                        random_state=42,
                        n_jobs=-1,
                    )
                    perm_df = pd.DataFrame(
                        {
                            "Feature": feature_cols,
                            "Importance": perm.importances_mean,
                            "Std": perm.importances_std,
                        }
                    ).sort_values("Importance", ascending=False)
                    st.session_state[f"perm_{run['run_id']}_{target}"] = perm_df
                except Exception as exc:
                    st.error(f"No se pudo calcular permutation importance: {exc}")

    perm_key = f"perm_{run['run_id']}_{target}"
    if perm_key in st.session_state:
        perm_df = st.session_state[perm_key]
        max_top = min(50, len(perm_df))
        top_n = st.slider("Top N", 1, max_top, min(20, max_top))
        plot_df = perm_df.head(top_n)
        fig = go.Figure(
            go.Bar(
                x=plot_df["Importance"][::-1],
                y=plot_df["Feature"][::-1],
                orientation="h",
                error_x=dict(array=plot_df["Std"][::-1].tolist(), color="#f59e0b"),
                marker_color="#5b6af0",
            )
        )
        fig.update_layout(
            **DARK,
            title=f"Permutation importance - {target}",
            height=max(360, top_n * 24),
            xaxis_title=f"Caída de score ({scoring})",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(perm_df.round(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 – SHAP manual (cálculo a demanda) – FIXED v2
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### SHAP Values – cálculo manual")
    st.caption(
        "Calcula valores SHAP sobre el conjunto de holdout usando un **KernelExplainer** "
        "(model-agnostic, basado en una muestra representativa del fondo). "
        "Las columnas categóricas se codifican automáticamente a numéricas para el cálculo."
    )

    if not has_model:
        st.info("El modelo no está disponible en memoria. Esto puede ocurrir si la corrida se guardó antes de que se implementara la persistencia del modelo. Entrena nuevamente los modelos para habilitar esta funcionalidad.")
    else:
        if st.button("🎯 Calcular SHAP", use_container_width=True):
            with st.spinner("Calculando SHAP values (puede tomar varios minutos)..."):
                try:
                    import shap

                    # Prepare numeric-only data for SHAP
                    X_numeric = _prepare_numeric_xai(X_test, feature_cols)

                    # Usar .sample() de pandas en lugar de shap.sample()
                    # (shap.sample() tiene problemas con numpy 2.x)
                    background_size = min(100, len(X_numeric))
                    background = X_numeric.sample(n=background_size, random_state=42)

                    # Obtener el modelo individual (no el ensemble AutoML)
                    individual_model = _get_individual_model(automl, result.get("best_model_name"))

                    def _shap_predict_wrapper(x: np.ndarray):
                        """Restore original dtypes before passing to model.

                        Always return the full predict_proba output (2D) to avoid
                        shap 0.52.0 internal array construction issues with 1D return
                        values when combined with numpy 2.x.
                        """
                        x_df = pd.DataFrame(x, columns=background.columns)
                        for col in background.columns:
                            orig_dtype = background[col].dtype
                            if orig_dtype != x_df[col].dtype:
                                try:
                                    x_df[col] = x_df[col].astype(orig_dtype)
                                except (ValueError, TypeError):
                                    pass
                        if is_classification and hasattr(individual_model, "predict_proba"):
                            return individual_model.predict_proba(x_df)
                        else:
                            return individual_model.predict(x_df)

                    model_predict = _shap_predict_wrapper

                    # Limit evaluation set for speed
                    eval_size = min(200, len(X_numeric))
                    X_eval = X_numeric.iloc[:eval_size]

                    # shap 0.52.0 + numpy 2.x has known compatibility issues with
                    # KernelExplainer. Try multiple strategies in order of preference.
                    shap_values = None

                    # Strategy 1: TreeExplainer (fast, works if model has tree structures)
                    try:
                        from shap import TreeExplainer
                        if hasattr(individual_model, "get_booster") or hasattr(individual_model, "feature_importances_"):
                            tree_explainer = TreeExplainer(individual_model)
                            shap_values_tree = tree_explainer.shap_values(X_eval)
                            if shap_values_tree is not None:
                                shap_values = shap_values_tree
                    except Exception:
                        pass

                    # Strategy 2: PermutationExplainer (model-agnostic, robust)
                    if shap_values is None:
                        try:
                            from shap import PermutationExplainer
                            perm_explainer = PermutationExplainer(
                                model_predict, background.values
                            )
                            shap_values_perm = perm_explainer(X_eval.values)
                            if shap_values_perm is not None:
                                # Convert Explanation object to raw array
                                shap_values = shap_values_perm.values
                        except Exception:
                            pass

                    # Strategy 3: KernelExplainer with automatic nsamples
                    if shap_values is None:
                        try:
                            explainer_ke = shap.KernelExplainer(model_predict, background.values)
                            # nsamples="auto" lets shap decide
                            shap_values_ke = explainer_ke.shap_values(X_eval.values)
                            if shap_values_ke is not None:
                                shap_values = shap_values_ke
                        except Exception:
                            pass

                    if shap_values is None:
                        st.error("No se pudieron calcular valores SHAP con ninguno de los métodos disponibles. Esto puede deberse a incompatibilidades entre shap 0.52.0 y numpy 2.x.")
                        st.stop()

                    st.session_state[f"shap_{run['run_id']}_{target}"] = {
                        "shap_values": shap_values,
                        "X_eval": X_eval,
                        "feature_names": list(X_numeric.columns),
                        "explainer": None,  # stored to avoid pickle issues
                    }
                    st.success(f"SHAP calculado sobre {eval_size} muestras de holdout.")
                except Exception as exc:
                    st.error(f"Error calculando SHAP: {exc}")

    shap_key = f"shap_{run['run_id']}_{target}"
    if shap_key in st.session_state:
        shap_data = st.session_state[shap_key]
        shap_values = shap_data["shap_values"]
        X_eval = shap_data["X_eval"]
        fnames = shap_data["feature_names"]
        explainer = shap_data.get("explainer")

        st.markdown("#### SHAP Summary (Beeswarm)")
        st.caption("Cada punto es una observación; el color indica el valor de la feature (rojo = alto, azul = bajo).")

        # Determinar si es multiclase o binaria/regresión
        if isinstance(shap_values, list):
            # Multiclass: mostrar el summary de la clase 1 (o permitir seleccionar)
            n_classes = len(shap_values)
            class_idx = st.selectbox("Clase SHAP a visualizar", list(range(n_classes)),
                                     format_func=lambda i: f"Clase {i}")
            sv = shap_values[class_idx]
            st.caption(f"Mostrando SHAP para la **Clase {class_idx}**")
        else:
            sv = shap_values

        # Beeswarm con Plotly
        shap_df = pd.DataFrame(sv, columns=fnames)
        # Calcular importancia global como mean(|SHAP|) para ordenar
        global_imp = shap_df.abs().mean().sort_values(ascending=False)
        top_shap = min(20, len(global_imp))
        top_features = global_imp.index[:top_shap].tolist()

        # Preparar datos para el beeswarm
        rows = []
        for feat in top_features:
            vals = shap_df[feat].values
            feat_vals = X_eval[feat].values if feat in X_eval.columns else np.zeros_like(vals)

            # Normalizar valores de feature para color: percentil rank
            feat_sorted = np.sort(feat_vals)
            color_norm = np.searchsorted(feat_sorted, feat_vals) / max(len(feat_sorted), 1)

            for v, c in zip(vals, color_norm):
                rows.append({"Feature": feat, "SHAP value": v, "Feature value": c})

        beeswarm_df = pd.DataFrame(rows)

        fig = go.Figure()
        for feat in top_features:
            subset = beeswarm_df[beeswarm_df["Feature"] == feat]
            fig.add_trace(
                go.Scatter(
                    x=subset["SHAP value"],
                    y=[feat] * len(subset),
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=subset["Feature value"],
                        colorscale="RdYlBu_r",
                        showscale=False,
                        opacity=0.7,
                    ),
                    hovertemplate="SHAP: %{x:.4f}<br>Feature: %{y}<extra></extra>",
                    name=feat,
                )
            )

        fig.update_layout(
            **DARK,
            title=f"SHAP Summary – {target}",
            height=max(360, top_shap * 28),
            yaxis=dict(autorange="reversed"),
            xaxis_title="SHAP value (impacto en la salida del modelo)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # SHAP Bar (importancia global media)
        st.markdown("#### Importancia Global (mean |SHAP|)")
        imp_series = shap_df.abs().mean().sort_values(ascending=False).head(top_shap)
        fig_bar = go.Figure(
            go.Bar(
                x=imp_series.values[::-1],
                y=imp_series.index[::-1],
                orientation="h",
                marker_color="#a855f7",
            )
        )
        fig_bar.update_layout(
            **DARK,
            title=f"Mean |SHAP| – {target}",
            height=max(300, top_shap * 24),
            xaxis_title="mean |SHAP value|",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Waterfall para una instancia específica
        st.markdown("#### Waterfall para una instancia específica")
        total_samples = len(X_eval)
        instance_idx = st.number_input(
            "Índice de instancia en el holdout (0 – {})".format(total_samples - 1),
            min_value=0,
            max_value=total_samples - 1,
            value=0,
        )
        if st.button("📊 Mostrar Waterfall", use_container_width=True):
            with st.spinner("Generando waterfall..."):
                try:
                    # Construir waterfall manual con Plotly
                    instance_shap = sv[instance_idx]
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        if isinstance(shap_values, list):
                            base_val = base_value[class_idx]
                        else:
                            base_val = base_value[0]
                    else:
                        base_val = base_value

                    # Ordenar features por |SHAP|
                    order = np.argsort(np.abs(instance_shap))[::-1]
                    # Mostrar top 15
                    order = order[:15]
                    ordered_shap = instance_shap[order]
                    ordered_names = [fnames[i] for i in order]

                    fig_wf = go.Figure(
                        go.Waterfall(
                            name="SHAP",
                            orientation="v",
                            measure=["relative"] * len(ordered_shap),
                            x=ordered_names,
                            y=ordered_shap.tolist(),
                            text=[f"{s:+.4f}" for s in ordered_shap],
                            textposition="outside",
                            connector={"line": {"color": "#64748b", "dash": "dot"}},
                            decreasing={"marker": {"color": "#3b82f6"}},
                            increasing={"marker": {"color": "#ef4444"}},
                            totals={"marker": {"color": "#a855f7"}},
                        )
                    )
                    fig_wf.add_hline(
                        y=0, line_dash="dash", line_color="#94a3b8", opacity=0.5
                    )
                    fig_wf.update_layout(
                        **DARK,
                        title=f"Waterfall SHAP – Instancia {instance_idx} – {target}",
                        height=500,
                        yaxis_title="Contribución SHAP",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)
                except Exception as exc:
                    st.error(f"Error generando waterfall: {exc}")

        # Tabla de SHAP para la instancia seleccionada
        st.markdown("#### Valores SHAP detallados")
        instance_sv = sv[instance_idx]
        sv_detail = pd.DataFrame(
            {"Feature": fnames, "SHAP value": instance_sv}
        ).sort_values("SHAP value", key=abs, ascending=False)
        st.dataframe(sv_detail.round(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 – LIME (explicaciones locales) – FIXED v2
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### LIME – Explicaciones locales")
    st.caption(
        "Selecciona una instancia del holdout y genera una explicación local "
        "con **LIME** (Local Interpretable Model-agnostic Explanations). "
        "Las columnas categóricas se codifican automáticamente a numéricas."
    )

    if not has_model:
        st.info("El modelo no está disponible en memoria. Esto puede ocurrir si la corrida se guardó antes de que se implementara la persistencia del modelo. Entrena nuevamente los modelos para habilitar esta funcionalidad.")
    else:
        # Preparar datos numéricos para LIME
        X_numeric = _prepare_numeric_xai(X_test, feature_cols)
        numeric_feature_names = list(X_numeric.columns)
        total_samples = len(X_test)
        lime_instance_idx = st.number_input(
            "Índice de instancia a explicar (0 – {})".format(total_samples - 1),
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            key="lime_idx",
        )

        # Obtener el modelo individual para predicciones
        individual_model = _get_individual_model(automl, result.get("best_model_name"))

        if st.button("🔍 Explicar con LIME", use_container_width=True):
            with st.spinner("Generando explicación LIME..."):
                try:
                    from lime.lime_tabular import LimeTabularExplainer

                    training_data = X_numeric.values
                    feature_names = numeric_feature_names

                    if is_classification:
                        class_names = [str(c) for c in sorted(result["y_test"].unique())]
                        mode = "classification"
                    else:
                        class_names = ["target"]
                        mode = "regression"

                    explainer = LimeTabularExplainer(
                        training_data,
                        feature_names=feature_names,
                        class_names=class_names,
                        mode=mode,
                        random_state=42,
                    )

                    instance = X_numeric.iloc[lime_instance_idx].values

                    def _lime_predict_wrapper(x: np.ndarray):
                        """Restore original dtypes before passing to model."""
                        x_df = pd.DataFrame(x, columns=feature_names)
                        for col in X_numeric.columns:
                            orig_dtype = X_numeric[col].dtype
                            if orig_dtype != x_df[col].dtype:
                                try:
                                    x_df[col] = x_df[col].astype(orig_dtype)
                                except (ValueError, TypeError):
                                    pass
                        if is_classification:
                            if hasattr(individual_model, "predict_proba"):
                                return individual_model.predict_proba(x_df)
                            return automl.predict_proba(x_df)
                        else:
                            if hasattr(individual_model, "predict"):
                                return individual_model.predict(x_df)
                            return automl.predict(x_df)

                    exp = explainer.explain_instance(
                        instance, _lime_predict_wrapper, num_features=15, top_labels=(3 if is_classification else None)
                    )

                    if is_classification:
                        pred_class = exp.predict_proba.argmax()
                        st.markdown(
                            f"##### Clase predicha: **{class_names[pred_class]}** "
                            f"(prob: {exp.predict_proba[pred_class]:.2%})"
                        )
                        exp_list = exp.as_list(label=pred_class)
                    else:
                        pred_value = _lime_predict_wrapper(instance.reshape(1, -1))[0]
                        st.markdown(
                            f"##### Predicción: **{pred_value:.4f}**"
                        )
                        exp_list = exp.as_list()

                    # Convertir a DataFrame para visualización
                    lime_df = pd.DataFrame(exp_list, columns=["Feature", "Weight"])
                    lime_df["AbsWeight"] = lime_df["Weight"].abs()
                    lime_df = lime_df.sort_values("AbsWeight", ascending=True)

                    fig = go.Figure(
                        go.Bar(
                            x=lime_df["Weight"],
                            y=lime_df["Feature"],
                            orientation="h",
                            marker_color=[
                                "#ef4444" if w >= 0 else "#3b82f6"
                                for w in lime_df["Weight"]
                            ],
                        )
                    )
                    fig.update_layout(
                        **DARK,
                        title=f"LIME – Instancia {lime_instance_idx} – {target}",
                        height=max(360, len(lime_df) * 28),
                        xaxis_title="Peso (contribución a la predicción)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        lime_df[["Feature", "Weight"]].round(5),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Mostrar tabla con los valores reales de la instancia (versión original)
                    st.markdown("##### Valores reales de la instancia")
                    instance_vals = X_test.iloc[lime_instance_idx][feature_cols].to_frame("Valor")
                    st.dataframe(instance_vals, use_container_width=True)

                except ImportError:
                    st.error(
                        "La librería `lime` no está instalada. "
                        "Ejecuta `pip install lime` e intenta de nuevo."
                    )
                except Exception as exc:
                    st.error(f"Error generando explicación LIME: {exc}")

# ---------------------------------------------------------------------------
# TAB 5 – PDP / ICE (Partial Dependence / Individual Conditional Expectation) – FIXED v2
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("### Partial Dependence Plots (PDP) / ICE")
    st.caption(
        "Muestra cómo cambia la predicción promedio al variar una feature (PDP) "
        "y las expectativas condicionales individuales (ICE). "
        "Cálculo manual (compatible con cualquier modelo de mljar)."
    )

    if not has_model:
        st.info("El modelo no está disponible en memoria. Esto puede ocurrir si la corrida se guardó antes de que se implementara la persistencia del modelo. Entrena nuevamente los modelos para habilitar esta funcionalidad.")
    else:
        # Obtener el modelo individual para PDP/ICE
        individual_model = _get_individual_model(automl, result.get("best_model_name"))
        best_model_name = result.get("best_model_name")

        # Seleccionar feature para PDP
        sel_feature = st.selectbox(
            "Feature a analizar",
            feature_cols,
            index=0,
            key="pdp_feature",
        )

        # Opciones avanzadas
        with st.expander("⚙️ Opciones avanzadas", expanded=False):
            show_ice = st.checkbox("Mostrar líneas ICE individuales", value=True)
            ice_sample = st.slider(
                "Máximo de líneas ICE a mostrar",
                5, 100, 30,
                help="Para no saturar la gráfica, solo se muestran N líneas ICE aleatorias.",
            )
            pdp_grid_size = st.slider(
                "Tamaño de la grid de valores",
                10, 100, 50,
                help="Cantidad de puntos equidistantes para evaluar la feature.",
            )

        if st.button("📈 Generar PDP / ICE", use_container_width=True):
            with st.spinner("Calculando PDP / ICE..."):
                try:
                    # Preparar datos numéricos para PDP
                    X_numeric = _prepare_numeric_xai(X_test, feature_cols)

                    # Obtener índice de la feature en el DataFrame numérico
                    if sel_feature not in X_numeric.columns:
                        st.error(
                            f"La feature '{sel_feature}' no está disponible en formato numérico "
                            f"(es probablemente categórica). Las features categóricas no son "
                            f"soportadas para PDP/ICE. Selecciona una feature numérica."
                        )
                        st.stop()

                    feature_idx = list(X_numeric.columns).index(sel_feature)
                    feature_values = X_numeric[sel_feature].dropna()

                    if feature_values.empty:
                        st.error("La feature seleccionada no tiene valores válidos en holdout.")
                        st.stop()

                    # Crear grid de valores
                    val_min = float(feature_values.min())
                    val_max = float(feature_values.max())
                    grid = np.linspace(val_min, val_max, pdp_grid_size)

                    # Calcular PDP / ICE manualmente
                    pdp_results = _manual_partial_dependence(
                        individual_model,
                        X_numeric,
                        feature_idx=feature_idx,
                        grid=grid,
                        kind="individual" if show_ice else "average",
                    )
                    pdp_values = pdp_results["average"][0]
                    pdp_grid = pdp_results["values"]
                    ice_values = pdp_results.get("individual")

                    # Crear figura
                    fig = go.Figure()

                    # Líneas ICE
                    if ice_values is not None and ice_values.shape[0] > 0:
                        n_samples = ice_values.shape[0]
                        sample_idx = np.random.choice(
                            n_samples,
                            min(ice_sample, n_samples),
                            replace=False,
                        )
                        for i in sample_idx:
                            fig.add_trace(
                                go.Scatter(
                                    x=pdp_grid,
                                    y=ice_values[i],
                                    mode="lines",
                                    line=dict(color="rgba(100, 150, 255, 0.15)", width=1),
                                    showlegend=False,
                                    hovertemplate="ICE %{x:.4f}<br>%{y:.4f}<extra></extra>",
                                )
                            )

                    # Línea PDP principal
                    fig.add_trace(
                        go.Scatter(
                            x=pdp_grid,
                            y=pdp_values,
                            mode="lines+markers",
                            line=dict(color="#f59e0b", width=3),
                            marker=dict(size=6, color="#f59e0b"),
                            name="PDP promedio",
                            hovertemplate="Valor: %{x:.4f}<br>Predicción: %{y:.4f}<extra></extra>",
                        )
                    )

                    # Distribución de la feature en el fondo (rug plot en el eje x)
                    hist_data = feature_values.sample(min(500, len(feature_values))).values
                    fig.add_trace(
                        go.Histogram(
                            x=hist_data,
                            yaxis="y2",
                            marker_color="rgba(255, 255, 255, 0.1)",
                            showlegend=False,
                            nbinsx=40,
                            name="Distribución",
                            hovertemplate="Frecuencia: %{y}<extra></extra>",
                        )
                    )

                    fig.update_layout(
                        **DARK,
                        title=f"PDP / ICE – {sel_feature} – {target}",
                        height=500,
                        xaxis_title=sel_feature,
                        yaxis_title="Predicción promedio",
                        yaxis=dict(domain=[0, 0.85]),
                        yaxis2=dict(
                            domain=[0.85, 1],
                            overlaying="y",
                            side="right",
                            showticklabels=False,
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de datos PDP
                    pdp_table = pd.DataFrame(
                        {"Valor": pdp_grid, "Predicción promedio": pdp_values}
                    )
                    st.dataframe(pdp_table.round(5), use_container_width=True, hide_index=True)

                    # Interpretación
                    st.markdown("##### Interpretación")
                    pdp_min = pdp_grid[np.argmin(pdp_values)]
                    pdp_max = pdp_grid[np.argmax(pdp_values)]
                    range_pdp = pdp_values.max() - pdp_values.min()

                    st.markdown(
                        f"- **Rango de efecto:** la predicción promedio varía en **{range_pdp:.4f}** "
                        f"al cambiar `{sel_feature}` de {pdp_min:.4f} a {pdp_max:.4f}."
                    )
                    if abs(range_pdp) < 0.01:
                        st.info("⚠️ El PDP muestra un efecto muy pequeño. Esta feature tiene baja influencia.")

                except Exception as exc:
                    st.error(f"Error generando PDP/ICE: {exc}")

# ---------------------------------------------------------------------------
# TAB 6 – Leaderboard (sin cambios sustanciales)
# ---------------------------------------------------------------------------
with tab6:
    leaderboard = result.get("leaderboard")
    if leaderboard is None or leaderboard.empty:
        st.info("No hay leaderboard guardado para este target.")
    else:
        leaderboard = leaderboard.copy()
        direction = config["direction"]
        leaderboard["metric_value"] = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
        leaderboard = leaderboard.sort_values("metric_value", ascending=(direction == "min"))
        st.dataframe(leaderboard.round(4), use_container_width=True, hide_index=True)

        model_type_counts = leaderboard["model_type"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=model_type_counts.index,
                y=model_type_counts.values,
                marker_color="#2dd4bf",
                text=model_type_counts.values,
                textposition="outside",
            )
        )
        fig.update_layout(**DARK, title="Cantidad de candidatos por tipo", height=360)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7 – Predicción Manual (sin cambios)
# ---------------------------------------------------------------------------
with tab7:
    st.markdown("### Predicción manual")
    st.caption("Completa las primeras features editables; el resto se rellena con mediana o moda del holdout.")

    if not has_model:
        st.info("El modelo no está disponible en memoria. Esto puede ocurrir si la corrida se guardó antes de que se implementara la persistencia del modelo. Entrena nuevamente los modelos para habilitar esta funcionalidad.")
    else:
        editable_features = feature_cols[: min(12, len(feature_cols))]
        input_values = {}
        groups = [editable_features[i : i + 3] for i in range(0, len(editable_features), 3)]
        for group in groups:
            cols = st.columns(len(group))
            for col_w, feature in zip(cols, group):
                series = X_test[feature]
                with col_w:
                    if pd.api.types.is_numeric_dtype(series):
                        clean = series.dropna()
                        if clean.empty:
                            min_value = max_value = value = 0.0
                        else:
                            min_value = float(clean.min())
                            max_value = float(clean.max())
                            value = float(clean.median())
                        step = (max_value - min_value) / 100 if max_value > min_value else 1.0
                        input_values[feature] = st.number_input(
                            feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=value,
                            step=step,
                        )
                    else:
                        options = series.dropna().astype(str).value_counts().head(50).index.tolist()
                        if not options:
                            options = [""]
                        input_values[feature] = st.selectbox(feature, options)

        for feature in feature_cols:
            if feature in input_values:
                continue
            series = X_test[feature]
            if pd.api.types.is_numeric_dtype(series):
                median = series.dropna().median()
                input_values[feature] = 0.0 if pd.isna(median) else float(median)
            else:
                mode = series.mode(dropna=True)
                input_values[feature] = mode.iloc[0] if not mode.empty else ""

        if st.button("🚀 Predecir", use_container_width=True):
            row_df = pd.DataFrame([input_values])[feature_cols]
            try:
                prediction = automl.predict(row_df)[0]
                st.markdown("#### Resultado")
                st.markdown(
                    f'<div class="metric-card"><div class="label">🎯 {target}</div>'
                    f'<div class="val">{prediction}</div></div>',
                    unsafe_allow_html=True,
                )
                if is_classification:
                    try:
                        probabilities = automl.predict_proba(row_df)[0]
                        st.markdown("##### Probabilidades")
                        for idx, probability in enumerate(probabilities):
                            st.progress(float(probability), text=f"Clase {idx}: {probability:.1%}")
                    except Exception:
                        pass
            except Exception as exc:
                st.error(f"Error en predicción manual: {exc}")