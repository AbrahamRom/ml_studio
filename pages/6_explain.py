import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

DARK = dict(
    paper_bgcolor="#0d0f14", plot_bgcolor="#141720",
    font_color="#e2e8f0", gridcolor="#252a38",
)
PALETTE = ["#5b6af0", "#2dd4bf", "#f59e0b", "#f43f5e", "#a78bfa",
           "#34d399", "#fb923c", "#60a5fa"]

st.markdown("# 🧠 Explainabilidad")

if st.session_state.trained_models is None:
    st.warning("⚠️ Entrena los modelos primero.")
    st.stop()

models   = st.session_state.trained_models
task     = st.session_state.task_type
targets  = st.session_state.target_cols
is_multi = st.session_state.multioutput
best     = st.session_state.best_model

model_name = st.selectbox(
    "Modelo a explicar",
    list(models.keys()),
    index=list(models.keys()).index(best) if best in models else 0,
)
res          = models[model_name]
pipe         = res["pipeline"]
X_train      = res["X_train"]
X_test       = res["X_test"]
feature_cols = res["feature_cols"]

st.markdown(f'**Modelo:** `{model_name}` {"⭐" if model_name == best else ""}')
st.divider()

# ── Extract base estimator ──────────────────────────────────────────────────────
def get_base_estimator(pipe, is_multi):
    """Extract the core estimator from a Pipeline (possibly MultiOutput-wrapped)."""
    model_step = pipe.named_steps.get("model", None)
    if model_step is None:
        return None
    if is_multi and hasattr(model_step, "estimators_"):
        # MultiOutput: return first sub-estimator
        return model_step.estimators_[0]
    if is_multi and hasattr(model_step, "estimator"):
        return model_step.estimator
    return model_step

base_est = get_base_estimator(pipe, is_multi)

# ── Permutation importance (model-agnostic) ────────────────────────────────────
def compute_permutation_importance(pipe, X_test, y_test_arr, task, n_repeats=10, seed=42):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics     import r2_score, accuracy_score

    scoring = "r2" if task == "regression" else "accuracy"
    result  = permutation_importance(
        pipe, X_test, y_test_arr,
        n_repeats=n_repeats, random_state=seed, scoring=scoring, n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": result.importances_mean,
        "Std":        result.importances_std,
    }).sort_values("Importance", ascending=False)
    return imp_df

# ── Native feature importance (tree-based) ────────────────────────────────────
def get_native_importance(pipe, is_multi, feature_cols):
    model_step = pipe.named_steps.get("model")
    est = base_est

    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = est.coef_
        if coef.ndim > 1:
            imp = np.abs(coef).mean(axis=0)
        else:
            imp = np.abs(coef)
    else:
        return None

    # If MultiOutput, aggregate from all sub-estimators
    if is_multi and hasattr(model_step, "estimators_"):
        all_imp = []
        for sub in model_step.estimators_:
            if hasattr(sub, "feature_importances_"):
                all_imp.append(sub.feature_importances_)
            elif hasattr(sub, "coef_"):
                c = sub.coef_
                all_imp.append(np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c))
        if all_imp:
            imp = np.mean(all_imp, axis=0)

    n = min(len(imp), len(feature_cols))
    return pd.DataFrame({
        "Feature":    list(feature_cols[:n]),
        "Importance": imp[:n],
    }).sort_values("Importance", ascending=False)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Feature Importance",
    "🔀 Permutation Importance",
    "📈 Partial Dependence",
    "🎲 SHAP (TreeExplainer)",
    "🔮 Predicción Manual",
])

# ── TAB 1: Native feature importance ──────────────────────────────────────────
with tab1:
    st.markdown("### Importancia nativa del modelo")
    native_df = get_native_importance(pipe, is_multi, feature_cols)

    if native_df is None:
        st.info("Este modelo no expone importancia nativa de features. "
                "Usa 'Permutation Importance' para una alternativa model-agnostic.")
    else:
        top_n = st.slider("Top N features", 5, min(50, len(native_df)), 20, key="topn_native")
        plot_df = native_df.head(top_n)

        fig = go.Figure(go.Bar(
            x=plot_df["Importance"][::-1],
            y=plot_df["Feature"][::-1],
            orientation="h",
            marker=dict(
                color=plot_df["Importance"][::-1],
                colorscale=[[0, "#252a38"], [1, "#5b6af0"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in plot_df["Importance"][::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            **DARK, height=max(350, top_n * 22),
            title=f"Feature Importance — {model_name}",
            margin=dict(t=50, b=20, l=10, r=80),
            xaxis=dict(gridcolor="#252a38"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative importance
        cumulative = native_df["Importance"].cumsum() / native_df["Importance"].sum()
        n80 = (cumulative <= 0.8).sum() + 1
        st.markdown(
            f'<span class="tag teal">{n80} features</span> explican el **80%** de la importancia total.',
            unsafe_allow_html=True,
        )

        fig_cum = go.Figure(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative.values * 100,
            mode="lines+markers",
            line=dict(color="#2dd4bf", width=2),
            fill="tozeroy",
            fillcolor="rgba(45,212,191,0.08)",
        ))
        fig_cum.add_hline(y=80, line=dict(color="#f43f5e", dash="dash"),
                          annotation_text="80%", annotation_position="right")
        fig_cum.update_layout(**DARK, title="Importancia Acumulada (%)", height=320,
                              xaxis_title="# de Features", yaxis_title="% Importancia acumulada")
        st.plotly_chart(fig_cum, use_container_width=True)

# ── TAB 2: Permutation importance ─────────────────────────────────────────────
with tab2:
    st.markdown("### Permutation Importance (model-agnostic)")
    st.info("Mide cuánto empeora el modelo al mezclar aleatoriamente cada feature. "
            "Funciona con cualquier modelo.")

    if is_multi:
        target_pi = st.selectbox("Target para permutation importance", targets, key="pi_target")
        y_perm    = res["y_test"][target_pi].values
    else:
        y_perm = res["y_test"].values

    n_rep = st.slider("Repeticiones", 3, 20, 8, key="pi_reps")

    if st.button("⚙️ Calcular Permutation Importance"):
        with st.spinner("Calculando..."):
            try:
                perm_df = compute_permutation_importance(pipe, X_test, y_perm, task, n_repeats=n_rep)
                st.session_state[f"perm_df_{model_name}"] = perm_df
            except Exception as e:
                st.error(f"Error: {e}")

    perm_key = f"perm_df_{model_name}"
    if perm_key in st.session_state and st.session_state[perm_key] is not None:
        perm_df = st.session_state[perm_key]
        top_n2  = st.slider("Top N", 5, min(40, len(perm_df)), 20, key="topn_perm")
        plot_p  = perm_df.head(top_n2)

        fig = go.Figure(go.Bar(
            x=plot_p["Importance"][::-1],
            y=plot_p["Feature"][::-1],
            orientation="h",
            error_x=dict(array=plot_p["Std"][::-1].tolist(), color="#f59e0b"),
            marker=dict(
                color=plot_p["Importance"][::-1],
                colorscale=[[0, "#252a38"], [0.5, "#5b6af0"], [1, "#2dd4bf"]],
            ),
        ))
        fig.update_layout(
            **DARK, height=max(350, top_n2 * 22),
            title="Permutation Importance (± std)",
            margin=dict(t=50, b=20, l=10, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Partial Dependence Plots ───────────────────────────────────────────
with tab3:
    st.markdown("### Partial Dependence Plots (PDP)")
    st.info("Muestra el efecto marginal promedio de una feature sobre la predicción.")

    num_feats = X_test.select_dtypes(include="number").columns.tolist()
    if not num_feats:
        st.warning("No hay features numéricas para PDP.")
    else:
        pdp_feat = st.selectbox("Feature para PDP", num_feats, key="pdp_feat")

        if st.button("📈 Generar PDP"):
            with st.spinner("Calculando PDP..."):
                try:
                    feat_idx = list(X_test.columns).index(pdp_feat)
                    grid     = np.linspace(X_test[pdp_feat].quantile(0.02),
                                          X_test[pdp_feat].quantile(0.98), 60)
                    preds_grid = []
                    X_copy = X_test.values.copy()
                    for val in grid:
                        Xg = X_copy.copy()
                        Xg[:, feat_idx] = val
                        Xg_df = pd.DataFrame(Xg, columns=X_test.columns)
                        p = pipe.predict(Xg_df)
                        if is_multi:
                            preds_grid.append(p.mean(axis=0))  # avg across samples
                        else:
                            preds_grid.append(np.mean(p))
                    preds_grid = np.array(preds_grid)

                    if is_multi:
                        fig = go.Figure()
                        for ti, t in enumerate(targets):
                            fig.add_trace(go.Scatter(
                                x=grid, y=preds_grid[:, ti],
                                mode="lines", name=t,
                                line=dict(color=PALETTE[ti % len(PALETTE)], width=2),
                            ))
                        fig.update_layout(**DARK, title=f"PDP — {pdp_feat} (todos los targets)",
                                         height=420, xaxis_title=pdp_feat,
                                         yaxis_title="Predicción promedio")
                    else:
                        fig = go.Figure(go.Scatter(
                            x=grid, y=preds_grid, mode="lines",
                            fill="tozeroy", fillcolor="rgba(91,106,240,0.1)",
                            line=dict(color="#5b6af0", width=2),
                            name="PDP",
                        ))
                        fig.update_layout(**DARK, title=f"PDP — {pdp_feat}", height=380,
                                         xaxis_title=pdp_feat, yaxis_title="Predicción")

                    # Rug plot: actual distribution
                    fig.add_trace(go.Scatter(
                        x=X_test[pdp_feat].values,
                        y=[fig.data[0].y.min()] * len(X_test),
                        mode="markers",
                        marker=dict(color="#f59e0b", size=3, opacity=0.3, symbol="line-ns-open"),
                        name="Distribución real",
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calculando PDP: {e}")

# ── TAB 4: SHAP ────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### SHAP — TreeExplainer")
    st.info(
        "SHAP (SHapley Additive exPlanations) descompone cada predicción en la contribución "
        "de cada feature. TreeExplainer funciona con modelos basados en árboles (RF, XGBoost, LightGBM...)."
    )

    try:
        import shap

        tree_models = {"RandomForest", "ExtraTrees", "GradientBoosting",
                       "XGB", "LGBM", "DecisionTree"}

        est_name = type(base_est).__name__ if base_est else ""
        is_tree  = any(tm.lower() in est_name.lower() for tm in tree_models)

        if not is_tree:
            st.warning(
                f"`{est_name}` no es un modelo basado en árboles. "
                "SHAP TreeExplainer no está disponible. "
                "Usa KernelExplainer (muy lento) o selecciona un modelo de árbol."
            )
        else:
            n_shap = st.slider("Muestras para SHAP", 50, min(500, len(X_test)), 100, key="n_shap")

            if is_multi:
                shap_target = st.selectbox("Target para SHAP", targets, key="shap_target")
                shap_idx    = list(targets).index(shap_target)
            else:
                shap_target = targets[0]
                shap_idx    = 0

            if st.button("⚡ Calcular SHAP values"):
                with st.spinner("Calculando SHAP values..."):
                    try:
                        X_shap = X_test.iloc[:n_shap].copy()

                        # Need to pass through scaler if present
                        if "scaler" in pipe.named_steps:
                            X_shap_scaled = pd.DataFrame(
                                pipe.named_steps["scaler"].transform(X_shap),
                                columns=feature_cols,
                            )
                        else:
                            X_shap_scaled = X_shap

                        model_step = pipe.named_steps["model"]
                        if is_multi:
                            sub_est = model_step.estimators_[shap_idx]
                        else:
                            sub_est = model_step

                        explainer  = shap.TreeExplainer(sub_est)
                        shap_vals  = explainer.shap_values(X_shap_scaled)

                        # For binary classifiers, shap_values returns list [class0, class1]
                        if isinstance(shap_vals, list):
                            shap_arr = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
                        else:
                            shap_arr = shap_vals

                        st.session_state[f"shap_{model_name}_{shap_target}"] = {
                            "shap_arr": shap_arr,
                            "X_shap":   X_shap_scaled,
                        }
                        st.success("✅ SHAP values calculados")
                    except Exception as e:
                        st.error(f"Error SHAP: {e}")

            shap_key = f"shap_{model_name}_{shap_target}"
            if shap_key in st.session_state and st.session_state[shap_key] is not None:
                shap_data = st.session_state[shap_key]
                shap_arr  = shap_data["shap_arr"]
                X_shap    = shap_data["X_shap"]

                mean_abs  = np.abs(shap_arr).mean(axis=0)
                shap_df   = pd.DataFrame({
                    "Feature":          list(feature_cols[:len(mean_abs)]),
                    "Mean |SHAP|":      mean_abs,
                }).sort_values("Mean |SHAP|", ascending=False)

                top_k = st.slider("Top K features", 5, min(30, len(shap_df)), 15, key="shap_topk")

                # Bar: mean |SHAP|
                plot_s = shap_df.head(top_k)
                fig_sb = go.Figure(go.Bar(
                    x=plot_s["Mean |SHAP|"][::-1],
                    y=plot_s["Feature"][::-1],
                    orientation="h",
                    marker=dict(
                        color=plot_s["Mean |SHAP|"][::-1],
                        colorscale=[[0, "#252a38"], [1, "#5b6af0"]],
                    ),
                    text=[f"{v:.4f}" for v in plot_s["Mean |SHAP|"][::-1]],
                    textposition="outside",
                ))
                fig_sb.update_layout(
                    **DARK, title=f"SHAP Mean |value| — {shap_target}",
                    height=max(320, top_k * 24),
                    margin=dict(t=50, b=20, l=10, r=80),
                )
                st.plotly_chart(fig_sb, use_container_width=True)

                # Beeswarm-style scatter per top feature
                st.markdown("#### Distribución de SHAP values por feature (top features)")
                top_features = shap_df["Feature"].head(top_k).tolist()
                feat_indices = [list(feature_cols).index(f) for f in top_features if f in feature_cols]

                fig_bee = go.Figure()
                for fi, feat in zip(feat_indices, top_features):
                    sv   = shap_arr[:, fi]
                    fval = X_shap.iloc[:, fi].values
                    fig_bee.add_trace(go.Scatter(
                        x=sv,
                        y=[feat] * len(sv),
                        mode="markers",
                        marker=dict(
                            color=fval,
                            colorscale="RdBu",
                            size=5,
                            opacity=0.7,
                            colorbar=dict(title="Feature value") if feat == top_features[0] else None,
                            showscale=(feat == top_features[0]),
                        ),
                        name=feat,
                        showlegend=False,
                    ))
                fig_bee.add_vline(x=0, line=dict(color="#64748b", dash="dash"))
                fig_bee.update_layout(
                    **DARK, title="SHAP Beeswarm — efecto de cada feature",
                    height=max(380, top_k * 25),
                    xaxis_title="SHAP value (impacto en predicción)",
                    margin=dict(t=50, b=20, l=10, r=30),
                )
                st.plotly_chart(fig_bee, use_container_width=True)

                # Waterfall for single observation
                st.markdown("#### Waterfall — explicación de una predicción individual")
                obs_idx = st.number_input("Índice de observación", 0, len(X_shap)-1, 0, key="obs_idx")
                sv_obs  = shap_arr[obs_idx]
                top_obs = np.argsort(np.abs(sv_obs))[::-1][:10]
                feats_w = [feature_cols[i] for i in top_obs]
                vals_w  = sv_obs[top_obs]
                colors_w = ["#5b6af0" if v > 0 else "#f43f5e" for v in vals_w]

                fig_wf = go.Figure(go.Bar(
                    x=vals_w[::-1],
                    y=feats_w[::-1],
                    orientation="h",
                    marker_color=colors_w[::-1],
                    text=[f"{v:+.4f}" for v in vals_w[::-1]],
                    textposition="outside",
                ))
                fig_wf.add_vline(x=0, line=dict(color="#64748b", width=1))
                fig_wf.update_layout(
                    **DARK, title=f"Waterfall — observación #{obs_idx}",
                    height=380, margin=dict(t=50, b=20, l=10, r=80),
                    xaxis_title="Contribución SHAP",
                )
                st.plotly_chart(fig_wf, use_container_width=True)

    except ImportError:
        st.warning("📦 SHAP no está instalado. Ejecuta `pip install shap` para habilitar esta sección.")

# ── TAB 5: Manual prediction ───────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔮 Predicción manual — ingresa valores")
    st.info("Ingresa valores para las features y obtén una predicción del modelo en tiempo real.")

    # Show only numeric features (simplified UI)
    num_feats_manual = X_test.select_dtypes(include="number").columns.tolist()[:12]
    other_feats      = [c for c in feature_cols if c not in num_feats_manual]

    input_vals = {}
    col_groups = [num_feats_manual[i:i+3] for i in range(0, len(num_feats_manual), 3)]
    for group in col_groups:
        cols = st.columns(len(group))
        for col_w, feat in zip(cols, group):
            with col_w:
                mn  = float(X_test[feat].min())
                mx  = float(X_test[feat].max())
                med = float(X_test[feat].median())
                input_vals[feat] = st.number_input(
                    feat, min_value=mn, max_value=mx, value=med,
                    step=(mx - mn) / 100, key=f"manual_{feat}",
                )
    # Fill other features with median
    for feat in other_feats:
        try:
            input_vals[feat] = float(X_test[feat].median())
        except Exception:
            input_vals[feat] = 0.0

    if st.button("🚀 Predecir", use_container_width=True):
        try:
            row_df = pd.DataFrame([input_vals])[feature_cols]
            pred   = pipe.predict(row_df)
            st.markdown("#### Resultado")
            if is_multi:
                for i, t in enumerate(targets):
                    val = pred[0][i]
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:8px">'
                        f'<div class="label">🎯 {t}</div>'
                        f'<div class="val">{val:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                val = pred[0]
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">🎯 {targets[0]}</div>'
                    f'<div class="val">{val:.4f if isinstance(val, float) else val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if task == "classification" and hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(row_df)
                    st.markdown("##### Probabilidades")
                    if is_multi:
                        for i, t in enumerate(targets):
                            proba_t = proba[i][0]
                            for ci, p in enumerate(proba_t):
                                st.progress(float(p), text=f"{t} — clase {ci}: {p:.1%}")
                    else:
                        for ci, p in enumerate(proba[0]):
                            st.progress(float(p), text=f"Clase {ci}: {p:.1%}")
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Error en predicción: {e}")
