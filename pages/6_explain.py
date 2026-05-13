from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

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
automl = result["automl"]
X_test = result["X_test"]
feature_cols = result["feature_cols"]

st.markdown(
    f"**Target:** `{target}` · **Mejor modelo según holdout real:** `{result.get('best_model_name')}` · "
    f"**Tipo:** `{result.get('best_model_type')}`"
)
st.caption(f"Reporte mljar: `{result['results_path']}`")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Reporte MLJAR", "🔀 Permutation Importance", "📊 Leaderboard", "🔮 Predicción Manual"]
)

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
        selected_image = st.selectbox("Gráfica guardada", image_files, format_func=lambda p: str(p.relative_to(report_path)))
        st.image(str(selected_image), use_column_width=True)

with tab2:
    st.markdown("### Permutation Importance del mejor AutoML")
    st.caption("Mide cuánto cae la métrica cuando se mezcla una feature en el holdout.")
    max_repeats = 20
    n_repeats = st.slider("Repeticiones", 3, max_repeats, 8)
    scoring = "neg_root_mean_squared_error" if config["task"] == "regression" else "f1_weighted"

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

with tab3:
    leaderboard = result["leaderboard"].copy()
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

with tab4:
    st.markdown("### Predicción manual")
    st.caption("Completa las primeras features editables; el resto se rellena con mediana o moda del holdout.")

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
            if config["task"] == "classification":
                try:
                    probabilities = automl.predict_proba(row_df)[0]
                    st.markdown("##### Probabilidades")
                    for idx, probability in enumerate(probabilities):
                        st.progress(float(probability), text=f"Clase {idx}: {probability:.1%}")
                except Exception:
                    pass
        except Exception as exc:
            st.error(f"Error en predicción manual: {exc}")
