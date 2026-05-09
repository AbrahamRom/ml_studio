"""Target task inference for tabular supervised learning."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


CLASSIFICATION_UNIQUE_LIMIT = 20


@dataclass(frozen=True)
class TargetTask:
    task: str
    ml_task: str
    primary_metric: str
    direction: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _is_integer_like_numeric(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty or not is_numeric_dtype(clean):
        return False
    return bool((clean == clean.astype("int64")).all())


def infer_target_task(series: pd.Series) -> dict:
    """Infer the mljar task for a single target column.

    Returns a JSON-friendly dictionary with Streamlit-facing task metadata.
    """

    clean = series.dropna()
    unique_count = int(clean.nunique(dropna=True))

    if unique_count < 2:
        return TargetTask(
            task="invalid",
            ml_task="invalid",
            primary_metric="",
            direction="",
            reason="El target tiene menos de 2 valores distintos.",
        ).to_dict()

    if unique_count == 2:
        return TargetTask(
            task="classification",
            ml_task="binary_classification",
            primary_metric="f1",
            direction="max",
            reason="Tiene exactamente 2 clases distintas.",
        ).to_dict()

    if (
        is_bool_dtype(clean)
        or isinstance(clean.dtype, pd.CategoricalDtype)
        or clean.dtype == object
        or clean.dtype == "string"
    ):
        return TargetTask(
            task="classification",
            ml_task="multiclass_classification",
            primary_metric="f1",
            direction="max",
            reason="Es una variable categórica con más de 2 clases.",
        ).to_dict()

    if is_integer_dtype(clean) or _is_integer_like_numeric(clean):
        if unique_count <= CLASSIFICATION_UNIQUE_LIMIT:
            return TargetTask(
                task="classification",
                ml_task="multiclass_classification",
                primary_metric="f1",
                direction="max",
                reason=(
                    f"Es numérica discreta con {unique_count} valores únicos "
                    f"(<= {CLASSIFICATION_UNIQUE_LIMIT})."
                ),
            ).to_dict()

    return TargetTask(
        task="regression",
        ml_task="regression",
        primary_metric="rmse",
        direction="min",
        reason="Es numérica continua o tiene alta cardinalidad.",
    ).to_dict()


def normalize_target_config(config: dict) -> dict:
    """Fill required task metadata after a manual override."""

    ml_task = config.get("ml_task")
    if ml_task in {"binary_classification", "multiclass_classification"}:
        return {
            **config,
            "task": "classification",
            "primary_metric": "f1",
            "direction": "max",
        }
    if ml_task == "regression":
        return {
            **config,
            "task": "regression",
            "primary_metric": "rmse",
            "direction": "min",
        }
    return config


def task_label(ml_task: str) -> str:
    labels = {
        "binary_classification": "Clasificación binaria",
        "multiclass_classification": "Clasificación multiclase",
        "regression": "Regresión",
        "invalid": "No entrenable",
    }
    return labels.get(ml_task, ml_task)
