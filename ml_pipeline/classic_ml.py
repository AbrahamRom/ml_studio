"""Classic ML models: Baseline, Linear, Decision Tree, Random Forest, Extra Trees, LightGBM, Xgboost, CatBoost, Neural Network, Nearest Neighbors."""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, confusion_matrix

CLASSIC_MODELS = {
    "Baseline": {"classification": DummyClassifier, "regression": DummyRegressor},
    "Linear": {"classification": LogisticRegression, "regression": LinearRegression},
    "Decision Tree": {"classification": DecisionTreeClassifier, "regression": DecisionTreeRegressor},
    "Random Forest": {"classification": RandomForestClassifier, "regression": RandomForestRegressor},
    "Extra Trees": {"classification": ExtraTreesClassifier, "regression": ExtraTreesRegressor},
    "LightGBM": {"classification": None, "regression": None},
    "Xgboost": {"classification": None, "regression": None},
    "CatBoost": {"classification": None, "regression": None},
    "Neural Network": {"classification": MLPClassifier, "regression": MLPRegressor},
    "Nearest Neighbors": {"classification": KNeighborsClassifier, "regression": KNeighborsRegressor},
}

HPO_SEARCH_SPACES = {
    "Baseline": {},
    "Linear": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2", "l1", "elasticnet", None],
    },
    "Decision Tree": {
        "max_depth": [None, 3, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Extra Trees": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 3, 5, 7, 10],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 50],
    },
    "Xgboost": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "CatBoost": {
        "iterations": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth": [3, 5, 7, 10],
        "l2_leaf_reg": [1, 3, 5, 10],
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50), (100, 100), (200, 100)],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "activation": ["relu", "tanh", "logistic"],
    },
    "Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 10, 15, 20],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
}

DEFAULT_PARAMS = {
    "Baseline": {},
    "Linear": {"C": 1.0, "penalty": "l2"},
    "Decision Tree": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"},
    "Random Forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    "Extra Trees": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    "LightGBM": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31, "min_child_samples": 20},
    "Xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "colsample_bytree": 1.0},
    "CatBoost": {"iterations": 100, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3},
    "Neural Network": {"hidden_layer_sizes": (100,), "alpha": 0.001, "learning_rate_init": 0.001, "activation": "relu"},
    "Nearest Neighbors": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
}

try:
    import lightgbm as lgb
    CLASSIC_MODELS["LightGBM"]["classification"] = lgb.LGBMClassifier
    CLASSIC_MODELS["LightGBM"]["regression"] = lgb.LGBMRegressor
except ImportError:
    pass

try:
    import xgboost as xgb
    CLASSIC_MODELS["Xgboost"]["classification"] = xgb.XGBClassifier
    CLASSIC_MODELS["Xgboost"]["regression"] = xgb.XGBRegressor
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CLASSIC_MODELS["CatBoost"]["classification"] = CatBoostClassifier
    CLASSIC_MODELS["CatBoost"]["regression"] = CatBoostRegressor
except ImportError:
    pass


def get_available_models():
    available = []
    for name, estimators in CLASSIC_MODELS.items():
        if estimators["classification"] is not None or estimators["regression"] is not None:
            available.append(name)
    return available


def _prepare_data(df, feature_cols, target_col, task, test_size, random_state):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X = X.dropna()
    y = y.loc[X.index]
    X = X.fillna(0)
    is_classification = task in ("classification", "binary_classification")
    le = None
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler, le, is_classification


def _get_model_kwargs(model_name, is_classification, random_state, params=None):
    params = params or {}
    kwargs = {}
    if model_name == "Baseline":
        kwargs["strategy"] = "most_frequent" if is_classification else "mean"
    elif model_name == "Linear":
        kwargs["max_iter"] = 1000
        if "C" in params: kwargs["C"] = params["C"]
        if "penalty" in params and params["penalty"]: kwargs["penalty"] = params["penalty"]
        if params.get("penalty") in ("l1", "elasticnet"): kwargs["solver"] = "saga"
    elif model_name == "Decision Tree":
        if "max_depth" in params: kwargs["max_depth"] = params["max_depth"]
        if "min_samples_split" in params: kwargs["min_samples_split"] = params["min_samples_split"]
        if "min_samples_leaf" in params: kwargs["min_samples_leaf"] = params["min_samples_leaf"]
        if "criterion" in params: kwargs["criterion"] = params["criterion"]
    elif model_name in ("Random Forest", "Extra Trees"):
        if "n_estimators" in params: kwargs["n_estimators"] = params["n_estimators"]
        if "max_depth" in params: kwargs["max_depth"] = params["max_depth"]
        if "min_samples_split" in params: kwargs["min_samples_split"] = params["min_samples_split"]
        if "min_samples_leaf" in params: kwargs["min_samples_leaf"] = params["min_samples_leaf"]
        kwargs["random_state"] = random_state
        kwargs["n_jobs"] = -1
    elif model_name == "LightGBM":
        if "n_estimators" in params: kwargs["n_estimators"] = params["n_estimators"]
        if "learning_rate" in params: kwargs["learning_rate"] = params["learning_rate"]
        if "max_depth" in params: kwargs["max_depth"] = params["max_depth"]
        if "num_leaves" in params: kwargs["num_leaves"] = params["num_leaves"]
        if "min_child_samples" in params: kwargs["min_child_samples"] = params["min_child_samples"]
        kwargs["verbose"] = -1
        kwargs["random_state"] = random_state
    elif model_name == "Xgboost":
        if "n_estimators" in params: kwargs["n_estimators"] = params["n_estimators"]
        if "learning_rate" in params: kwargs["learning_rate"] = params["learning_rate"]
        if "max_depth" in params: kwargs["max_depth"] = params["max_depth"]
        if "subsample" in params: kwargs["subsample"] = params["subsample"]
        if "colsample_bytree" in params: kwargs["colsample_bytree"] = params["colsample_bytree"]
        kwargs["verbosity"] = 0
        kwargs["random_state"] = random_state
    elif model_name == "CatBoost":
        if "iterations" in params: kwargs["iterations"] = params["iterations"]
        if "learning_rate" in params: kwargs["learning_rate"] = params["learning_rate"]
        if "depth" in params: kwargs["depth"] = params["depth"]
        if "l2_leaf_reg" in params: kwargs["l2_leaf_reg"] = params["l2_leaf_reg"]
        kwargs["verbose"] = False
        kwargs["random_state"] = random_state
    elif model_name == "Neural Network":
        if "hidden_layer_sizes" in params: kwargs["hidden_layer_sizes"] = params["hidden_layer_sizes"]
        if "alpha" in params: kwargs["alpha"] = params["alpha"]
        if "learning_rate_init" in params: kwargs["learning_rate_init"] = params["learning_rate_init"]
        if "activation" in params: kwargs["activation"] = params["activation"]
        kwargs["max_iter"] = 500
        kwargs["random_state"] = random_state
    elif model_name == "Nearest Neighbors":
        if "n_neighbors" in params: kwargs["n_neighbors"] = params["n_neighbors"]
        if "weights" in params: kwargs["weights"] = params["weights"]
        if "metric" in params: kwargs["metric"] = params["metric"]
    return kwargs


def train_classic_model(df, feature_cols, target_col, model_name, task="classification", test_size=0.2, random_state=42, params=None, progress_callback=None):
    X_train_scaled, X_val_scaled, y_train, y_val, scaler, le, is_classification = _prepare_data(df, feature_cols, target_col, task, test_size, random_state)
    model_class = CLASSIC_MODELS[model_name][task] if is_classification else CLASSIC_MODELS[model_name].get("regression")
    if model_class is None:
        raise ValueError(f"Modelo '{model_name}' no disponible para {task}. Instala las dependencias necesarias.")
    kwargs = _get_model_kwargs(model_name, is_classification, random_state, params)
    model = model_class(**kwargs)
    if model_name == "CatBoost":
        model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val))
    else:
        model.fit(X_train_scaled, y_train)
    if progress_callback:
        progress_callback(1, 1, 0, 0)
    y_val_pred = model.predict(X_val_scaled)
    y_train_pred = model.predict(X_train_scaled)
    if is_classification:
        train_score = accuracy_score(y_train, y_train_pred)
        val_score = accuracy_score(y_val, y_val_pred)
        train_precision = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
        val_precision = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        try:
            train_proba = model.predict_proba(X_train_scaled)
            val_proba = model.predict_proba(X_val_scaled)
        except Exception:
            train_proba = val_proba = None
    else:
        train_score = r2_score(y_train, y_train_pred)
        val_score = r2_score(y_val, y_val_pred)
        train_precision = val_precision = None
        train_recall = val_recall = None
        train_f1 = val_f1 = None
        train_proba = val_proba = None
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le if is_classification else None,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_name": model_name,
        "task": task,
        "train_score": train_score,
        "val_score": val_score,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_precision": train_precision,
        "val_precision": val_precision,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "train_proba": train_proba,
        "val_proba": val_proba,
        "params": params,
    }


def optimize_classic_hyperparameters(df, feature_cols, target_col, model_name, task="classification", test_size=0.2, random_state=42, n_trials=30, progress_callback=None):
    search_space = HPO_SEARCH_SPACES.get(model_name, {})
    if not search_space:
        return {"best_config": None, "best_val_score": 0, "trials": [], "message": "No hay hiperparámetros para optimizar."}

    is_classification = task in ("classification", "binary_classification")
    trials = []
    best_val_score = 0 if is_classification else float("inf")
    best_config = None

    for trial in range(n_trials):
        trial_params = {}
        for param, values in search_space.items():
            trial_params[param] = random.choice(values)

        try:
            result = train_classic_model(df, feature_cols, target_col, model_name, task=task, test_size=test_size, random_state=random_state, params=trial_params)
            val_score = result["val_score"]
            val_rmse = result["val_rmse"]

            if is_classification:
                is_better = val_score > best_val_score
            else:
                is_better = val_rmse < best_val_score

            if is_better:
                best_val_score = val_score
                best_config = trial_params.copy()

            trials.append({"trial": trial, "val_score": val_score, "val_rmse": val_rmse, "config": trial_params, "error": False})
        except Exception as e:
            trials.append({"trial": trial, "val_score": 0 if is_classification else float("inf"), "val_rmse": float("inf"), "config": trial_params, "error": True, "error_msg": str(e)})

        if progress_callback:
            progress_callback(trial + 1, n_trials)

    return {"best_config": best_config, "best_val_score": best_val_score, "trials": trials}
