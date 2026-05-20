from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.pagination import paginated_dataframe

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

st.markdown("# 🧠 Explainabilidad")

has_automl = st.session_state.automl_run is not None
has_dl = bool(st.session_state.get("dl_results"))

if not has_automl and not has_dl:
    st.warning("⚠️ Entrena modelos primero (AutoML o Deep Learning).")
    st.stop()

source_options = []
if has_automl:
    source_options.append("AutoML")
if has_dl:
    source_options.append("Deep Learning")

model_source = st.radio("Fuente del modelo", source_options, horizontal=True, key="explain_source")

# ── AutoML explainability ──────────────────────────────────────────────────────
if model_source == "AutoML":
    run = st.session_state.automl_run
    target_results = run["target_results"]
    targets = list(target_results.keys())

    target = st.selectbox("Target a explicar", targets, key="explain_aml_target")
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
        n_repeats = st.slider("Repeticiones", 3, max_repeats, 8, key="aml_perm_repeats")
        scoring = "neg_root_mean_squared_error" if config["task"] == "regression" else "f1_weighted"

        if st.button("⚙️ Calcular importancia", use_container_width=True, key="aml_calc_perm"):
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
            top_n = st.slider("Top N", 1, max_top, min(20, max_top), key="aml_perm_topn")
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
            paginated_dataframe(perm_df.round(5), key="explain_perm_auto", height=350, hide_index=True)

    with tab3:
        leaderboard = result["leaderboard"].copy()
        direction = config["direction"]
        leaderboard["metric_value"] = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
        leaderboard = leaderboard.sort_values("metric_value", ascending=(direction == "min"))
        paginated_dataframe(leaderboard.round(4), key="explain_lb", height=350, hide_index=True)

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
                            key=f"aml_input_{feature}",
                        )
                    else:
                        options = series.dropna().astype(str).value_counts().head(50).index.tolist()
                        if not options:
                            options = [""]
                        input_values[feature] = st.selectbox(feature, options, key=f"aml_input_{feature}")

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

        if st.button("🚀 Predecir", use_container_width=True, key="aml_predict"):
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

# ── Deep Learning explainability ───────────────────────────────────────────────
else:
    dl_results = st.session_state.dl_results

    dl_options = []
    for i, r in enumerate(dl_results):
        label = f"{r['model_type'].upper()}"
        if r.get("target"):
            label += f" → {r['target']}"
        else:
            label += " (unsupervised)"
        if r.get("hpo"):
            label += " [HPO]"
        dl_options.append((label, i))

    selected_label, selected_idx = st.selectbox("Modelo DL a explicar", dl_options, format_func=lambda x: x[0], key="explain_dl_model")
    dl_result = dl_results[selected_idx]

    model_type = dl_result["model_type"]
    target = dl_result.get("target")
    feature_cols = dl_result.get("feature_cols", [])
    model = dl_result.get("model")
    scaler = dl_result.get("scaler")
    config = dl_result.get("config", {})

    st.markdown(f"**Modelo:** `{model_type.upper()}` · **Target:** `{target or '(unsupervised)'}` · **HPO:** `{'✅' if dl_result.get('hpo') else '❌'}`")
    st.divider()

    if model_type in ("lstm", "gru"):
        tab1, tab2, tab3 = st.tabs(["🔀 Permutation Importance", "🔮 Predicción Manual", "📐 Feature stats"])

        with tab1:
            st.markdown("### Permutation Importance (DL)")
            st.caption("Mide cuánto sube el RMSE cuando se mezcla una feature en el validation set.")

            max_repeats = 10
            n_repeats = st.slider("Repeticiones", 1, max_repeats, 5, key="dl_perm_repeats")

            if st.button("⚙️ Calcular importancia DL", use_container_width=True, key="dl_calc_perm"):
                with st.spinner("Calculando permutation importance para DL..."):
                    try:
                        import torch
                        from ml_pipeline.deep_learning import prepare_sequences

                        df = st.session_state.df
                        val_start = dl_result.get("val_start", int(len(df) * 0.8))
                        val_data = df.iloc[val_start:].copy()
                        y_val = val_data[target].values.astype(float)

                        base_pred = []
                        seq_len = config.get("seq_length", 10)
                        for i in range(seq_len, len(val_data)):
                            window = val_data.iloc[i - seq_len : i][feature_cols].values.astype(float)
                            window = np.nan_to_num(window, nan=0.0)
                            if scaler:
                                window = scaler.transform(window)
                            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(x).item()
                            base_pred.append(pred)

                        base_rmse = np.sqrt(np.mean((y_val[seq_len:] - np.array(base_pred)) ** 2))

                        importances = []
                        for feat_idx, feat_name in enumerate(feature_cols):
                            perm_errors = []
                            for _ in range(n_repeats):
                                perm_data = val_data.copy()
                                perm_data[feat_name] = perm_data[feat_name].sample(frac=1, random_state=None).values
                                perm_pred = []
                                for i in range(seq_len, len(perm_data)):
                                    window = perm_data.iloc[i - seq_len : i][feature_cols].values.astype(float)
                                    window = np.nan_to_num(window, nan=0.0)
                                    if scaler:
                                        window = scaler.transform(window)
                                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                                    with torch.no_grad():
                                        pred = model(x).item()
                                    perm_pred.append(pred)
                                perm_rmse = np.sqrt(np.mean((y_val[seq_len:] - np.array(perm_pred)) ** 2))
                                perm_errors.append(perm_rmse)
                            importances.append(np.mean(perm_errors) - base_rmse)

                        perm_df = pd.DataFrame({
                            "Feature": feature_cols,
                            "Importance": importances,
                        }).sort_values("Importance", ascending=False)
                        st.session_state[f"dl_perm_{selected_idx}"] = perm_df
                    except Exception as exc:
                        st.error(f"No se pudo calcular permutation importance: {exc}")

            perm_key = f"dl_perm_{selected_idx}"
            if perm_key in st.session_state:
                perm_df = st.session_state[perm_key]
                max_top = min(50, len(perm_df))
                top_n = st.slider("Top N", 1, max_top, min(20, max_top), key="dl_perm_topn")
                plot_df = perm_df.head(top_n)
                fig = go.Figure(
                    go.Bar(
                        x=plot_df["Importance"][::-1],
                        y=plot_df["Feature"][::-1],
                        orientation="h",
                        marker_color="#2dd4bf",
                    )
                )
                fig.update_layout(
                    **DARK,
                    title=f"Permutation importance - {model_type.upper()} → {target}",
                    height=max(360, top_n * 24),
                    xaxis_title="Aumento en RMSE",
                )
                st.plotly_chart(fig, use_container_width=True)
                paginated_dataframe(perm_df.round(5), key="explain_perm_dl_seq", height=350, hide_index=True)

        with tab2:
            st.markdown("### Predicción manual con DL")
            st.caption("Ingresa valores para las features. Se usa la ventana más reciente para predecir.")

            df = st.session_state.df
            input_values = {}
            editable_features = feature_cols[: min(12, len(feature_cols))]
            groups = [editable_features[i : i + 3] for i in range(0, len(editable_features), 3)]
            for group in groups:
                cols = st.columns(len(group))
                for col_w, feature in zip(cols, group):
                    series = df[feature]
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
                                key=f"dl_input_{feature}",
                            )
                        else:
                            options = series.dropna().astype(str).value_counts().head(50).index.tolist()
                            if not options:
                                options = [""]
                            input_values[feature] = st.selectbox(feature, options, key=f"dl_input_{feature}")

            for feature in feature_cols:
                if feature in input_values:
                    continue
                series = df[feature]
                if pd.api.types.is_numeric_dtype(series):
                    median = series.dropna().median()
                    input_values[feature] = 0.0 if pd.isna(median) else float(median)
                else:
                    mode = series.mode(dropna=True)
                    input_values[feature] = mode.iloc[0] if not mode.empty else ""

            if st.button("🚀 Predecir con DL", use_container_width=True, key="dl_predict"):
                try:
                    import torch
                    seq_len = config.get("seq_length", 10)
                    row = pd.DataFrame([input_values])[feature_cols]
                    values = row.values.astype(float)
                    values = np.nan_to_num(values, nan=0.0)
                    if scaler:
                        values = scaler.transform(values)

                    x = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        prediction = model(x).item()

                    st.markdown("#### Resultado")
                    st.markdown(
                        f'<div class="metric-card"><div class="label">🎯 {target}</div>'
                        f'<div class="val">{prediction:.4f}</div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Error en predicción DL: {exc}")

        with tab3:
            st.markdown("### Estadísticas de las features")
            st.caption("Distribución de las features usadas en el modelo DL.")

            for feature in feature_cols[:10]:
                series = df[feature]
                if pd.api.types.is_numeric_dtype(series):
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=series.dropna(), nbinsx=50, marker_color="#5b6af0"))
                    fig.update_layout(title=feature, **DARK, height=250, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

    else:
        tab1, tab2, tab3 = st.tabs(["🔀 Permutation Importance", "🔍 Reconstrucción análisis", "📐 Latent space"])

        with tab1:
            st.markdown("### Permutation Importance (Autoencoder/VAE)")
            st.caption("Mide cuánto sube el error de reconstrucción cuando se mezcla una feature.")

            max_repeats = 10
            n_repeats = st.slider("Repeticiones", 1, max_repeats, 5, key="dl_ae_perm_repeats")

            if st.button("⚙️ Calcular importancia", use_container_width=True, key="dl_ae_calc_perm"):
                with st.spinner("Calculando permutation importance..."):
                    try:
                        import torch

                        df = st.session_state.df
                        data = df[feature_cols].values.astype(float)
                        data = np.nan_to_num(data, nan=0.0)
                        if scaler:
                            data = scaler.transform(data)

                        x = torch.tensor(data, dtype=torch.float32)
                        with torch.no_grad():
                            base_recon = model(x).numpy()
                        base_mse = np.mean((data - base_recon) ** 2, axis=0)

                        importances = []
                        for feat_idx, feat_name in enumerate(feature_cols):
                            perm_errors = []
                            for _ in range(n_repeats):
                                perm_data = data.copy()
                                np.random.shuffle(perm_data[:, feat_idx])
                                x_perm = torch.tensor(perm_data, dtype=torch.float32)
                                with torch.no_grad():
                                    perm_recon = model(x_perm).numpy()
                                perm_mse = np.mean((perm_data - perm_recon) ** 2, axis=0)
                                perm_errors.append(perm_mse[feat_idx] - base_mse[feat_idx])
                            importances.append(np.mean(perm_errors))

                        perm_df = pd.DataFrame({
                            "Feature": feature_cols,
                            "Importance": importances,
                        }).sort_values("Importance", ascending=False)
                        st.session_state[f"dl_ae_perm_{selected_idx}"] = perm_df
                    except Exception as exc:
                        st.error(f"No se pudo calcular permutation importance: {exc}")

            perm_key = f"dl_ae_perm_{selected_idx}"
            if perm_key in st.session_state:
                perm_df = st.session_state[perm_key]
                max_top = min(50, len(perm_df))
                top_n = st.slider("Top N", 1, max_top, min(20, max_top), key="dl_ae_perm_topn")
                plot_df = perm_df.head(top_n)
                fig = go.Figure(
                    go.Bar(
                        x=plot_df["Importance"][::-1],
                        y=plot_df["Feature"][::-1],
                        orientation="h",
                        marker_color="#f59e0b",
                    )
                )
                fig.update_layout(
                    **DARK,
                    title=f"Permutation importance - {model_type.upper()}",
                    height=max(360, top_n * 24),
                    xaxis_title="Aumento en MSE de reconstrucción",
                )
                st.plotly_chart(fig, use_container_width=True)
                paginated_dataframe(perm_df.round(5), key="explain_perm_dl_unsup", height=350, hide_index=True)

        with tab2:
            st.markdown("### Análisis de reconstrucción")
            st.caption("Comparación entre datos originales y reconstruidos.")

            train_recon = dl_result.get("train_reconstructed")
            if train_recon is not None:
                n_features = min(5, train_recon.shape[1])
                df = st.session_state.df
                for i in range(n_features):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=train_recon[:200, i], name="Reconstruido", line=dict(color="#2dd4bf"), mode="lines"))
                    if i < len(feature_cols):
                        orig = df[feature_cols[i]].values.astype(float)
                        fig.add_trace(go.Scatter(y=orig[:200], name="Original", line=dict(color="#5b6af0", dash="dash"), mode="lines"))
                    fig.update_layout(title=f"Feature {i} ({feature_cols[i] if i < len(feature_cols) else 'N/A'})", **DARK, height=280, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

                recon_error = dl_result.get("reconstruction_error")
                if recon_error is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=recon_error, nbinsx=50, marker_color="#f43f5e"))
                    fig2.update_layout(title="Distribución del error de reconstrucción", **DARK, height=350)
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            from sklearn.decomposition import PCA

            encoded = dl_result.get("train_encoded")
            if encoded is not None:
                if encoded.shape[1] > 2:
                    pca = PCA(n_components=2)
                    encoded_2d = pca.fit_transform(encoded)
                else:
                    encoded_2d = encoded[:, :2]

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=encoded_2d[:, 0], y=encoded_2d[:, 1], mode="markers", marker=dict(color="#5b6af0", opacity=0.6)))
                fig3.update_layout(
                    title="Latent space (PCA 2D)",
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    **DARK,
                    height=400,
                )
                st.plotly_chart(fig3, use_container_width=True)

                st.caption(f"Varianza explicada por componentes: {pca.explained_variance_ratio_ if encoded.shape[1] > 2 else 'N/A (2D original)'}")
