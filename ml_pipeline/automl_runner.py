"""Run one mljar-supervised AutoML job per target."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import save_dataframe, save_json, target_dir
from .metrics import compute_holdout_metrics
from .plots import save_evaluation_plots
from .tasks import normalize_target_config


os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ml_studio_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_studio_matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

FULL_MLJAR_ALGORITHMS = [
    "Baseline",
    "Linear",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "LightGBM",
    "Xgboost",
    "CatBoost",
    "Neural Network",
    "Nearest Neighbors",
]


def _train_test_split(X, y, task: str, test_size: float, random_state: int):
    stratify = None
    if task == "classification":
        counts = y.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = y
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _best_leaderboard_row(leaderboard: pd.DataFrame, direction: str) -> dict:
    if leaderboard.empty or "metric_value" not in leaderboard:
        return {}
    metric_values = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
    if metric_values.dropna().empty:
        return {}
    idx = metric_values.idxmax() if direction == "max" else metric_values.idxmin()
    return leaderboard.loc[idx].to_dict()


def _model_display_name(model, fallback_index: int) -> str:
    get_name = getattr(model, "get_name", None)
    if callable(get_name):
        try:
            value = get_name()
            if value:
                return str(value)
        except Exception:
            pass
    for attribute_name in ("name", "model_name", "model_id"):
        value = getattr(model, attribute_name, None)
        if value:
            return str(value)
    return f"model_{fallback_index}"


def _model_family_name(model) -> str:
    learner_params = getattr(model, "learner_params", None)
    if isinstance(learner_params, dict):
        family = learner_params.get("model_type")
        if family:
            return str(family)

    model_type = getattr(model, "model_type", None)
    if model_type:
        return str(model_type)

    return model.__class__.__name__


def _collect_model_metrics(
    automl,
    X_test: pd.DataFrame,
    y_test,
    task: str,
    n_features: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    models = list(getattr(automl, "_models", []) or [])
    stacked_models = list(getattr(automl, "_stacked_models", []) or [])
    all_models = models + stacked_models

    seen_names: set[str] = set()
    for index, model in enumerate(all_models, start=1):
        model_name = _model_display_name(model, index)
        if model_name in seen_names:
            continue
        seen_names.add(model_name)

        row = {
            "model_name": model_name,
            "model_type": _model_family_name(model),
            "model_class": model.__class__.__name__,
        }

        try:
            y_pred = model.predict(X_test)
            proba = None
            proba_classes = None
            if task == "classification" and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    classes_attr = getattr(model, "classes_", None)
                    if classes_attr is not None:
                        proba_classes = list(classes_attr)
                except Exception:
                    proba = None
                    proba_classes = None

            row.update(
                compute_holdout_metrics(
                    task,
                    y_test,
                    y_pred,
                    proba,
                    n_features=n_features,
                    proba_classes=proba_classes,
                )
            )
        except Exception as exc:
            row["evaluation_error"] = str(exc)

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df

    metric_order = [
        "score_global",
        "r2_adjusted",
        "r2",
        "rmse",
        "mae",
        "mape",
        "smape",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "roc_auc_ovr",
    ]
    ordered_columns = [
        column
        for column in ["model_name", "model_type", *metric_order, "evaluation_error"]
        if column in metrics_df.columns
    ]
    remaining_columns = [column for column in metrics_df.columns if column not in ordered_columns]
    return metrics_df[ordered_columns + remaining_columns]


def run_target_automl(
    df: pd.DataFrame,
    target: str,
    config: dict,
    *,
    all_targets: list[str] | None = None,
    run_path: str | Path,
    test_size: float = 0.2,
    total_time_limit: int = 180,
    mode: str = "Perform",
    algorithms: list[str] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Train one independent AutoML model for a target and persist artifacts."""

    from supervised.automl import AutoML

    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe en el dataset.")

    config = normalize_target_config(config)
    if config.get("ml_task") == "invalid":
        raise ValueError(f"Target '{target}' no es entrenable: {config.get('reason', '')}")

    feature_cols = [c for c in df.columns if c not in set(all_targets or [target])]
    if not feature_cols:
        raise ValueError("No quedan columnas feature después de excluir los targets.")

    data = df.loc[df[target].notna(), feature_cols + [target]].copy()
    if data[target].nunique(dropna=True) < 2:
        raise ValueError(f"Target '{target}' tiene menos de 2 valores distintos no nulos.")

    X = data[feature_cols]
    y = data[target]
    if config["task"] == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"Target '{target}' debe ser numérico para entrenar regresión.")

    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, config["task"], test_size, random_state
    )

    t_dir = target_dir(run_path, target)
    mljar_path = t_dir / "mljar"
    plots_path = t_dir / "plots"

    automl = AutoML(
        results_path=str(mljar_path),
        total_time_limit=int(total_time_limit),
        mode=mode,
        ml_task=config["ml_task"],
        algorithms=algorithms or FULL_MLJAR_ALGORITHMS,
        train_ensemble=True,
        stack_models="auto",
        eval_metric=config["primary_metric"],
        validation_strategy="auto",
        explain_level="auto",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test)
    proba = None
    proba_classes = None
    if config["task"] == "classification":
        try:
            proba = automl.predict_proba(X_test)
            classes_attr = getattr(automl, "classes_", None)
            if classes_attr is not None:
                proba_classes = list(classes_attr)
        except Exception:
            proba = None
            proba_classes = None

    holdout_metrics = compute_holdout_metrics(
        config["task"],
        y_test,
        y_pred,
        proba,
        n_features=len(feature_cols),
        proba_classes=proba_classes,
    )
    leaderboard = automl.get_leaderboard(original_metric_values=True)
    per_model_metrics = _collect_model_metrics(
        automl,
        X_test,
        y_test,
        config["task"],
        n_features=len(feature_cols),
    )
    best_row = _best_leaderboard_row(leaderboard, config["direction"])

    pred_df = pd.DataFrame(
        {
            "row_index": X_test.index,
            "target": target,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
        }
    )
    if proba is not None:
        proba_df = pd.DataFrame(proba).add_prefix("proba_")
        pred_df = pd.concat([pred_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

    leaderboard_path = save_dataframe(t_dir / "leaderboard.csv", leaderboard)
    per_model_metrics_path = save_dataframe(t_dir / "per_model_metrics.csv", per_model_metrics)
    predictions_path = save_dataframe(t_dir / "predictions.csv", pred_df)
    metrics_path = save_json(t_dir / "holdout_metrics.json", holdout_metrics)
    plot_paths = save_evaluation_plots(config["task"], plots_path, target, y_test, y_pred, proba)

    target_manifest = {
        "target": target,
        "config": config,
        "feature_cols": feature_cols,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results_path": str(mljar_path),
        "leaderboard_path": str(leaderboard_path),
        "per_model_metrics_path": str(per_model_metrics_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "plot_paths": plot_paths,
        "best_model_name": best_row.get("name"),
        "best_model_type": best_row.get("model_type"),
        "best_metric_value": best_row.get("metric_value"),
    }
    save_json(t_dir / "target_manifest.json", target_manifest)

    return {
        **target_manifest,
        "automl": automl,
        "leaderboard": leaderboard,
        "per_model_metrics": per_model_metrics,
        "holdout_metrics": holdout_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": pd.Series(y_pred, index=X_test.index, name="y_pred"),
        "prediction_frame": pred_df,
        "proba": proba,
    }
