import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.markdown("# 🏋️ Entrenar Modelos")

if st.session_state.df is None or st.session_state.target_cols is None:
    st.warning("⚠️ Carga el dataset y configura los targets primero.")
    st.stop()

df      = st.session_state.df.copy()
targets = st.session_state.target_cols
task    = st.session_state.task_type
is_multi = st.session_state.multioutput

# ── Info banner ────────────────────────────────────────────────────────────────
badge = '<span class="tag teal">MULTIOUTPUT</span>' if is_multi else '<span class="tag">SINGLE TARGET</span>'
st.markdown(
    f"**Tarea:** `{task}` &nbsp; {badge} &nbsp; "
    f"**Targets:** `{'`, `'.join(targets)}`",
    unsafe_allow_html=True,
)

st.divider()

# ── Model catalog ──────────────────────────────────────────────────────────────
CLF_MODELS = {
    "Logistic Regression":      "lr",
    "Random Forest":            "rf",
    "XGBoost":                  "xgboost",
    "LightGBM":                 "lightgbm",
    "Decision Tree":            "dt",
    "K-Nearest Neighbors":      "knn",
    "Gradient Boosting":        "gbc",
    "Extra Trees":              "et",
    "SVM (RBF)":                "svm",
    "Naive Bayes":              "nb",
}
REG_MODELS = {
    "Linear Regression":        "lr",
    "Ridge":                    "ridge",
    "Lasso":                    "lasso",
    "Random Forest":            "rf",
    "XGBoost":                  "xgboost",
    "LightGBM":                 "lightgbm",
    "Gradient Boosting":        "gbr",
    "Extra Trees":              "et",
    "Decision Tree":            "dt",
    "KNN":                      "knn",
}
catalog = CLF_MODELS if task == "classification" else REG_MODELS

# ── Sidebar options ────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Opciones de entrenamiento")

col1, col2, col3 = st.columns(3)
with col1:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
with col2:
    cv_folds  = st.slider("CV Folds", 2, 10, 5)
with col3:
    normalize = st.checkbox("Normalizar features", value=True)

st.markdown("#### Selecciona los modelos a comparar")
selected_names = st.multiselect(
    "Modelos",
    list(catalog.keys()),
    default=["Random Forest", "XGBoost", "LightGBM",
             "Logistic Regression" if task == "classification" else "Linear Regression"],
)

if is_multi:
    st.info(
        "**Modo Multioutput:** Se usará `MultiOutputClassifier` / `MultiOutputRegressor` "
        "de sklearn para predecir todos los targets simultáneamente. "
        "PyCaret se ejecuta por separado para cada target para obtener métricas individuales.",
        icon="ℹ️"
    )

# ── Train button ───────────────────────────────────────────────────────────────
st.divider()
if not selected_names:
    st.warning("Selecciona al menos un modelo.")
    st.stop()

if st.button("🚀 Entrenar modelos", use_container_width=True):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing   import StandardScaler, LabelEncoder
    from sklearn.pipeline        import Pipeline as SKPipeline
    from sklearn.multioutput     import MultiOutputClassifier, MultiOutputRegressor
    from sklearn.metrics         import (
        accuracy_score, f1_score, roc_auc_score,
        r2_score, mean_absolute_error, mean_squared_error,
    )
    import sklearn.ensemble, sklearn.linear_model, sklearn.tree
    import sklearn.neighbors, sklearn.svm, sklearn.naive_bayes

    # ── Build sklearn estimator from short name ────────────────────────────────
    def get_estimator(short_name, task):
        from sklearn.ensemble  import (RandomForestClassifier, RandomForestRegressor,
                                       GradientBoostingClassifier, GradientBoostingRegressor,
                                       ExtraTreesClassifier, ExtraTreesRegressor)
        from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                          Ridge, Lasso)
        from sklearn.tree      import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm       import SVC, SVR
        from sklearn.naive_bayes import GaussianNB

        try:
            from xgboost import XGBClassifier, XGBRegressor
            XGB_OK = True
        except ImportError:
            XGB_OK = False
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            LGB_OK = True
        except ImportError:
            LGB_OK = False

        clf_map = {
            "lr":       LogisticRegression(max_iter=500),
            "rf":       RandomForestClassifier(n_estimators=100, random_state=42),
            "dt":       DecisionTreeClassifier(random_state=42),
            "knn":      KNeighborsClassifier(),
            "gbc":      GradientBoostingClassifier(random_state=42),
            "et":       ExtraTreesClassifier(n_estimators=100, random_state=42),
            "svm":      SVC(probability=True, random_state=42),
            "nb":       GaussianNB(),
        }
        reg_map = {
            "lr":       LinearRegression(),
            "ridge":    Ridge(),
            "lasso":    Lasso(max_iter=5000),
            "rf":       RandomForestRegressor(n_estimators=100, random_state=42),
            "dt":       DecisionTreeRegressor(random_state=42),
            "knn":      KNeighborsRegressor(),
            "gbr":      GradientBoostingRegressor(random_state=42),
            "et":       ExtraTreesRegressor(n_estimators=100, random_state=42),
        }
        if XGB_OK:
            clf_map["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
            reg_map["xgboost"] = XGBRegressor(random_state=42, verbosity=0)
        if LGB_OK:
            clf_map["lightgbm"] = LGBMClassifier(random_state=42, verbose=-1)
            reg_map["lightgbm"] = LGBMRegressor(random_state=42, verbose=-1)

        return clf_map[short_name] if task == "classification" else reg_map[short_name]

    # ── Prepare data ───────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in targets]
    X = pd.get_dummies(df[feature_cols], drop_first=True)

    if task == "classification":
        y_data = {}
        encoders = {}
        for t in targets:
            if df[t].dtype == object:
                le = LabelEncoder()
                y_data[t] = le.fit_transform(df[t])
                encoders[t] = le
            else:
                y_data[t] = df[t].values
        y_df = pd.DataFrame(y_data)
    else:
        y_df = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_df, test_size=test_size, random_state=42
    )

    results  = {}   # name -> {metrics, pipeline, predictions}
    progress = st.progress(0, text="Iniciando entrenamiento...")
    total    = len(selected_names)

    for idx, model_name in enumerate(selected_names):
        progress.progress((idx) / total, text=f"Entrenando {model_name}...")
        short = catalog[model_name]
        try:
            estimator = get_estimator(short, task)
        except KeyError:
            st.warning(f"⚠️ {model_name} no disponible (librería no instalada).")
            continue

        # Build pipeline
        steps = []
        if normalize:
            steps.append(("scaler", StandardScaler()))

        if is_multi:
            if task == "classification":
                mo = MultiOutputClassifier(estimator, n_jobs=-1)
            else:
                mo = MultiOutputRegressor(estimator, n_jobs=-1)
            steps.append(("model", mo))
        else:
            steps.append(("model", estimator))

        pipe = SKPipeline(steps)
        y_tr = y_train.values if is_multi else y_train.values.ravel()
        y_te = y_test.values  if is_multi else y_test.values.ravel()

        pipe.fit(X_train, y_tr)
        preds = pipe.predict(X_test)

        # ── Compute metrics ────────────────────────────────────────────────────
        metrics = {}
        if task == "classification":
            if is_multi:
                accs, f1s = [], []
                for i, t in enumerate(targets):
                    accs.append(accuracy_score(y_te[:, i], preds[:, i]))
                    f1s.append(f1_score(y_te[:, i], preds[:, i], average="weighted", zero_division=0))
                metrics["Accuracy (avg)"] = np.mean(accs)
                metrics["F1 (avg)"]       = np.mean(f1s)
                for i, t in enumerate(targets):
                    metrics[f"Acc_{t}"]   = accs[i]
                    metrics[f"F1_{t}"]    = f1s[i]
            else:
                avg = "binary" if len(np.unique(y_te)) == 2 else "weighted"
                metrics["Accuracy"] = accuracy_score(y_te, preds)
                metrics["F1"]       = f1_score(y_te, preds, average=avg, zero_division=0)
                try:
                    prob = pipe.predict_proba(X_test)
                    if prob.shape[1] == 2:
                        metrics["AUC"] = roc_auc_score(y_te, prob[:, 1])
                    else:
                        metrics["AUC"] = roc_auc_score(y_te, prob, multi_class="ovr", average="weighted")
                except Exception:
                    pass
        else:
            if is_multi:
                r2s, maes, rmses = [], [], []
                for i, t in enumerate(targets):
                    r2s.append(r2_score(y_te[:, i], preds[:, i]))
                    maes.append(mean_absolute_error(y_te[:, i], preds[:, i]))
                    rmses.append(np.sqrt(mean_squared_error(y_te[:, i], preds[:, i])))
                metrics["R² (avg)"]   = np.mean(r2s)
                metrics["MAE (avg)"]  = np.mean(maes)
                metrics["RMSE (avg)"] = np.mean(rmses)
                for i, t in enumerate(targets):
                    metrics[f"R²_{t}"]   = r2s[i]
                    metrics[f"MAE_{t}"]  = maes[i]
            else:
                metrics["R²"]   = r2_score(y_te, preds)
                metrics["MAE"]  = mean_absolute_error(y_te, preds)
                metrics["RMSE"] = np.sqrt(mean_squared_error(y_te, preds))

        results[model_name] = {
            "metrics":     metrics,
            "pipeline":    pipe,
            "X_train":     X_train,
            "X_test":      X_test,
            "y_train":     pd.DataFrame(y_tr, columns=targets) if is_multi else pd.Series(y_tr.ravel()),
            "y_test":      pd.DataFrame(y_te, columns=targets) if is_multi else pd.Series(y_te.ravel()),
            "predictions": pd.DataFrame(preds, columns=targets) if is_multi else pd.Series(preds.ravel()),
            "feature_cols": X.columns.tolist(),
        }

    progress.progress(1.0, text="✅ Entrenamiento completo")

    # Determine best model by first key metric
    def primary_score(m):
        metrics = m["metrics"]
        if task == "classification":
            return metrics.get("Accuracy (avg)", metrics.get("Accuracy", 0))
        else:
            return metrics.get("R² (avg)", metrics.get("R²", -999))

    best_name = max(results, key=lambda n: primary_score(results[n]))

    st.session_state.trained_models = results
    st.session_state.best_model     = best_name

    # Build compare dataframe
    rows = []
    for name, res in results.items():
        row = {"Modelo": name}
        row.update({k: round(v, 4) for k, v in res["metrics"].items()
                    if not k.startswith("Acc_") and not k.startswith("F1_")
                    and not k.startswith("R²_") and not k.startswith("MAE_")})
        rows.append(row)
    st.session_state.compare_df = pd.DataFrame(rows).set_index("Modelo")

    st.success(f"🏆 Mejor modelo: **{best_name}**")
    st.dataframe(st.session_state.compare_df, use_container_width=True)

elif st.session_state.trained_models:
    st.success(f"✅ Modelos ya entrenados. Mejor: **{st.session_state.best_model}**")
    st.dataframe(st.session_state.compare_df, use_container_width=True)
    st.info("Ve a las secciones Compare / Evaluate / Explainability para más análisis.")
