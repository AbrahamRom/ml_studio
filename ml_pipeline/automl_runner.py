"""Run one mljar-supervised AutoML job per target."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
import math
from sklearn.model_selection import train_test_split

from .artifacts import save_dataframe, save_json, target_dir
from .early_warning import (
    compute_early_warning_metrics,
    compute_early_warning_predictions,
    fit_residual_uncertainty,
    load_quality_specs,
    resolve_quality_spec,
)
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


def _train_test_split(
    X, y, task: str, test_size: float, random_state: int, split_method: str = "random", time_col: str | None = None
):
    """Support random split (default) and temporal split by `time_col`.

    Temporal split: keep all rows with `time_col` <= cutoff for train and > cutoff for test.
    The cutoff year/value is chosen so that the test set contains at least `test_size` proportion
    of the samples. If `time_col` is missing/invalid or the temporal split would produce empty
    train/test partitions, falls back to random `train_test_split`.
    """
    # Helper: perform original random split with stratify when appropriate
    def _random_split():
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

    if split_method != "temporal":
        return _random_split()

    # Temporal split requested
    if time_col is None:
        warnings.warn("Temporal split requested but no time_col provided; falling back to random split")
        return _random_split()

    if time_col not in X.columns:
        warnings.warn(
            f"time_col '{time_col}' not found in feature columns; falling back to random split"
        )
        return _random_split()

    if X[time_col].isna().any():
        warnings.warn(f"time_col '{time_col}' contains nulls; falling back to random split")
        return _random_split()

    # Build combined frame to keep alignment
    df_comb = X.copy()
    df_comb = df_comb.assign(_target=y)

    # Unique sorted values for cutoff selection
    try:
        unique_vals = sorted(pd.Series(df_comb[time_col].dropna().unique()).tolist())
    except Exception:
        warnings.warn(f"Could not sort values of '{time_col}'; falling back to random split")
        return _random_split()

    if not unique_vals:
        warnings.warn(f"No valid values in '{time_col}'; falling back to random split")
        return _random_split()

    n_total = len(df_comb)
    required_test_n = int(math.ceil(test_size * n_total))

    cutoff = None
    for val in unique_vals:
        test_n = int((df_comb[time_col] > val).sum())
        if test_n >= required_test_n:
            cutoff = val
            break

    if cutoff is None:
        # Can't reach required proportion by year cutoff; choose closest (oldest)
        cutoff = unique_vals[0]
        warnings.warn(
            f"No cutoff value yields required test_size; using closest cutoff='{cutoff}' (test may be smaller than requested)"
        )

    train_mask = df_comb[time_col] <= cutoff
    test_mask = df_comb[time_col] > cutoff

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        warnings.warn(
            "Temporal split produced empty train or test partition; falling back to random split"
        )
        return _random_split()

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    return X_train, X_test, y_train, y_test


def _train_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    calibration_size: float,
    random_state: int,
):
    """Reserve part of the training data for residual uncertainty calibration."""

    empty_X = X.iloc[0:0].copy()
    empty_y = y.iloc[0:0].copy()
    if calibration_size <= 0 or len(X) < 5:
        return X, empty_X, y, empty_y

    calibration_count = int(math.ceil(len(X) * calibration_size))
    if calibration_count < 1 or len(X) - calibration_count < 2:
        return X, empty_X, y, empty_y

    stratify = None
    if task == "classification":
        counts = y.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = y

    try:
        return train_test_split(
            X,
            y,
            test_size=calibration_count,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        try:
            return train_test_split(
                X,
                y,
                test_size=calibration_count,
                random_state=random_state,
            )
        except ValueError:
            return X, empty_X, y, empty_y


def _best_leaderboard_row(leaderboard: pd.DataFrame, direction: str) -> dict:
    if leaderboard.empty or "metric_value" not in leaderboard:
        return {}
    metric_values = pd.to_numeric(leaderboard["metric_value"], errors="coerce")
    if metric_values.dropna().empty:
        return {}
    idx = metric_values.idxmax() if direction == "max" else metric_values.idxmin()
    return leaderboard.loc[idx].to_dict()


def _best_holdout_row(per_model_metrics: pd.DataFrame, config: dict) -> dict:
    if per_model_metrics.empty:
        return {}

    preferred_columns = [config.get("primary_metric"), "score_global"]
    numeric_columns = [
        column
        for column in per_model_metrics.columns
        if column not in {"model_name", "model_type", "model_class", "evaluation_error"}
        and pd.api.types.is_numeric_dtype(per_model_metrics[column])
    ]

    metric_column = next(
        (column for column in preferred_columns if column and column in numeric_columns),
        None,
    )
    if metric_column is None:
        metric_column = numeric_columns[0] if numeric_columns else None
    if metric_column is None:
        return {}

    direction = config.get("direction", "max")
    metric_values = pd.to_numeric(per_model_metrics[metric_column], errors="coerce")
    if metric_values.dropna().empty:
        return {}

    idx = metric_values.idxmax() if direction == "max" else metric_values.idxmin()
    best_row = per_model_metrics.loc[idx].to_dict()
    best_row["selected_metric"] = metric_column
    return best_row


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


def _all_tracked_models(automl) -> list:
    models = list(getattr(automl, "_models", []) or [])
    stacked_models = list(getattr(automl, "_stacked_models", []) or [])
    return models + stacked_models


def _coerce_1d_prediction(output):
    if output is None:
        return None

    if isinstance(output, pd.DataFrame):
        if output.shape[1] == 1:
            return output.iloc[:, 0].to_numpy()
        return output.to_numpy().ravel()

    if isinstance(output, pd.Series):
        return output.to_numpy()

    return np.asarray(output).ravel()


def resolve_model_by_name(automl, model_name: str | None):
    if not model_name:
        return None

    for index, model in enumerate(_all_tracked_models(automl), start=1):
        if _model_display_name(model, index) == model_name:
            return model
    return None


def predict_with_model(automl, model_name: str | None, X_test: pd.DataFrame, task: str):
    model = resolve_model_by_name(automl, model_name)
    if model is None:
        return None, None, None

    y_pred = _coerce_1d_prediction(model.predict(X_test))
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

    return y_pred, proba, proba_classes


def _collect_model_metrics(
    automl,
    X_test: pd.DataFrame,
    y_test,
    task: str,
    n_features: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    seen_names: set[str] = set()
    for index, model in enumerate(_all_tracked_models(automl), start=1):
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
            y_pred = _coerce_1d_prediction(model.predict(X_test))
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
        "f3",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
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
    calibration_size: float = 0.2,
    total_time_limit: int = 180,
    mode: str = "Perform",
    algorithms: list[str] | None = None,
    random_state: int = 42,
    split_method: str = "random",
    time_col: str | None = None,
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

    X_train_full, X_test, y_train_full, y_test = _train_test_split(
        X,
        y,
        config["task"],
        test_size,
        random_state,
        split_method=split_method,
        time_col=time_col,
    )
    X_train, X_calib, y_train, y_calib = _train_calibration_split(
        X_train_full,
        y_train_full,
        config["task"],
        calibration_size,
        random_state,
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

    y_pred = _coerce_1d_prediction(automl.predict(X_test))
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

    leaderboard = automl.get_leaderboard(original_metric_values=True)
    per_model_metrics = _collect_model_metrics(
        automl,
        X_test,
        y_test,
        config["task"],
        n_features=len(feature_cols),
    )
    best_row = _best_leaderboard_row(leaderboard, config["direction"])
    best_holdout_row = _best_holdout_row(per_model_metrics, config)
    best_model_name = best_holdout_row.get("model_name")

    selected_y_pred, selected_proba, selected_proba_classes = predict_with_model(
        automl,
        best_model_name,
        X_test,
        config["task"],
    )
    if selected_y_pred is not None:
        y_pred = selected_y_pred
        if config["task"] == "classification":
            proba = selected_proba
            proba_classes = selected_proba_classes

    holdout_metrics = compute_holdout_metrics(
        config["task"],
        y_test,
        y_pred,
        proba,
        n_features=len(feature_cols),
        proba_classes=proba_classes,
    )

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

    calibration_residuals = None
    early_warning_predictions = pd.DataFrame()
    early_warning_metrics = {}
    early_warning_error = None
    calibration_residuals_path = None
    early_warning_predictions_path = None
    early_warning_metrics_path = None
    quality_spec_key = None
    quality_spec = None

    if config["task"] == "regression":
        specs = load_quality_specs()
        quality_spec_key, quality_spec = resolve_quality_spec(target, specs)
        if quality_spec is not None:
            try:
                calib_pred, _, _ = predict_with_model(
                    automl,
                    best_model_name,
                    X_calib,
                    config["task"],
                )
                if calib_pred is None and not X_calib.empty:
                    calib_pred = _coerce_1d_prediction(automl.predict(X_calib))
                if calib_pred is None or X_calib.empty:
                    raise ValueError("No hay particion de calibracion disponible.")

                calibration_residuals = fit_residual_uncertainty(y_calib, calib_pred)
                residual_df = pd.DataFrame(
                    {
                        "row_index": X_calib.index,
                        "target": target,
                        "y_true": y_calib.to_numpy(),
                        "y_pred": calib_pred,
                        "residual": calibration_residuals,
                    }
                )
                early_warning_predictions = compute_early_warning_predictions(
                    y_test,
                    y_pred,
                    calibration_residuals,
                    quality_spec,
                    row_index=X_test.index,
                    target=target,
                )
                early_warning_metrics = compute_early_warning_metrics(early_warning_predictions)
                early_warning_metrics.update(
                    {
                        "quality_spec_key": quality_spec_key,
                        "calibration_rows": int(len(X_calib)),
                        "residual_count": int(len(calibration_residuals)),
                    }
                )
                calibration_residuals_path = save_dataframe(
                    t_dir / "calibration_residuals.csv",
                    residual_df,
                )
                early_warning_predictions_path = save_dataframe(
                    t_dir / "early_warning_predictions.csv",
                    early_warning_predictions,
                )
                early_warning_metrics_path = save_json(
                    t_dir / "early_warning_metrics.json",
                    early_warning_metrics,
                )
            except Exception as exc:
                early_warning_error = str(exc)

    if not early_warning_predictions.empty:
        ew_cols = [
            col
            for col in early_warning_predictions.columns
            if col not in {"row_index", "target", "y_true", "y_pred"}
        ]
        if ew_cols:
            pred_df = pred_df.merge(
                early_warning_predictions[["row_index", *ew_cols]],
                on="row_index",
                how="left",
            )

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
        "calibration_rows": int(len(X_calib)),
        "test_rows": int(len(X_test)),
        "results_path": str(mljar_path),
        "leaderboard_path": str(leaderboard_path),
        "per_model_metrics_path": str(per_model_metrics_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "calibration_residuals_path": str(calibration_residuals_path) if calibration_residuals_path else None,
        "early_warning_predictions_path": str(early_warning_predictions_path) if early_warning_predictions_path else None,
        "early_warning_metrics_path": str(early_warning_metrics_path) if early_warning_metrics_path else None,
        "early_warning_error": early_warning_error,
        "quality_spec_key": quality_spec_key,
        "plot_paths": plot_paths,
        "best_model_name": best_model_name,
        "best_model_type": best_holdout_row.get("model_type"),
        "best_metric_value": best_holdout_row.get(best_holdout_row.get("selected_metric")),
        "best_model_metric": best_holdout_row.get("selected_metric"),
        "internal_best_model_name": best_row.get("name"),
        "internal_best_model_type": best_row.get("model_type"),
        "internal_best_metric_value": best_row.get("metric_value"),
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
        "calibration_residuals": calibration_residuals,
        "early_warning_predictions": early_warning_predictions,
        "early_warning_metrics": early_warning_metrics,
        "quality_spec": quality_spec,
    }
