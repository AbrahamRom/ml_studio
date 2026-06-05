import json

import numpy as np
import pandas as pd

from ml_pipeline.automl_runner import _collect_model_metrics, predict_with_model, resolve_model_by_name
from ml_pipeline.artifacts import (
    create_run_dir,
    list_runs,
    load_automl_run,
    save_dataframe,
    save_json,
)
from ml_pipeline.comparison import build_final_matrix
from ml_pipeline.metrics import regression_metrics
from ml_pipeline.quality import (
    SHAPIRO_ALPHA,
    analyze_data_quality,
    compute_variable_stats,
    variable_stats_to_csv,
    variable_stats_to_json,
)
from ml_pipeline.tasks import infer_target_task


# Lightweight local approx helper so the tests don't depend on pytest.
def pytest_approx(value, rel=1e-9, atol=1e-9):
    class _Approx:
        def __init__(self, value, rel, atol):
            self.value = value
            self.rel = rel
            self.atol = atol

        def __eq__(self, other):
            return abs(other - self.value) <= max(self.atol, self.rel * abs(self.value))

        def __repr__(self):
            return f"approx({self.value})"

    return _Approx(value, rel, atol)


def test_infer_target_task_binary_classification():
    result = infer_target_task(pd.Series([0, 1, 0, 1, None]))

    assert result["task"] == "classification"
    assert result["ml_task"] == "binary_classification"
    assert result["primary_metric"] == "f1"
    assert result["direction"] == "max"


def test_infer_target_task_regression_for_continuous_numeric():
    result = infer_target_task(pd.Series([1.1, 2.3, 4.8, 8.2, 13.5]))

    assert result["task"] == "regression"
    assert result["ml_task"] == "regression"
    assert result["primary_metric"] == "rmse"
    assert result["direction"] == "min"


def test_quality_report_flags_constant_target():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [1, 1, 1]})

    report = analyze_data_quality(df, ["target"])

    assert report["shape"] == {"rows": 3, "columns": 2}
    assert report["target_issues"][0]["severity"] == "error"


def test_create_run_dir_and_save_json(tmp_path):
    run_id, run_path = create_run_dir(tmp_path)
    out = save_json(run_path / "manifest.json", {"run_id": run_id})

    assert run_path.exists()
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("{")


def test_list_runs_reads_run_manifest(tmp_path):
    run_id, run_path = create_run_dir(tmp_path / "automl_runs")
    save_json(run_path / "run_manifest.json", {"run_id": run_id, "targets": {"t1": {"target": "t1"}}})

    runs = list_runs(tmp_path / "automl_runs")

    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["targets"] == ["t1"]


def test_load_automl_run_loads_saved_artifacts(tmp_path):
    run_id, run_path = create_run_dir(tmp_path / "automl_runs")
    target = "target_a"
    target_path = run_path / target
    target_path.mkdir(parents=True, exist_ok=True)

    save_dataframe(
        target_path / "leaderboard.csv",
        pd.DataFrame(
            [{"name": "model_a", "model_type": "Linear", "metric_value": 0.91}]
        ),
    )
    save_dataframe(
        target_path / "per_model_metrics.csv",
        pd.DataFrame(
            [{"model_name": "model_a", "model_type": "Linear", "accuracy": 0.91}]
        ),
    )
    save_dataframe(
        target_path / "predictions.csv",
        pd.DataFrame(
            {
                "row_index": [0, 1],
                "target": [target, target],
                "y_true": [1, 0],
                "y_pred": [1, 0],
                "proba_0": [0.1, 0.9],
                "proba_1": [0.9, 0.1],
            }
        ),
    )
    save_json(
        target_path / "holdout_metrics.json",
        {"accuracy": 1.0, "classes": [0, 1], "classification_report": {"0": {"precision": 1.0}}},
    )

    target_manifest = {
        "target": target,
        "config": {
            "task": "classification",
            "ml_task": "binary_classification",
            "primary_metric": "f1",
            "direction": "max",
        },
        "feature_cols": ["x"],
        "train_rows": 8,
        "test_rows": 2,
        "results_path": str(target_path / "mljar"),
        "leaderboard_path": str(target_path / "leaderboard.csv"),
        "predictions_path": str(target_path / "predictions.csv"),
        "metrics_path": str(target_path / "holdout_metrics.json"),
        "plot_paths": {},
        "best_model_name": "model_a",
        "best_model_type": "Linear",
        "best_metric_value": 0.91,
        "best_model_metric": "f1",
        "internal_best_model_name": "model_a",
        "internal_best_model_type": "Linear",
        "internal_best_metric_value": 0.91,
    }
    save_json(target_path / "target_manifest.json", target_manifest)

    save_dataframe(
        run_path / "final_matrix.csv",
        pd.DataFrame({"Target": [target], "Linear": [0.91]}),
    )
    save_dataframe(
        run_path / "target_summary.csv",
        pd.DataFrame({"Target": [target], "Tarea": ["classification"]}),
    )
    save_dataframe(
        run_path / "best_model_metrics.csv",
        pd.DataFrame({"Target": [target], "accuracy": [1.0]}),
    )

    save_json(
        run_path / "run_manifest.json",
        {
            "run_id": run_id,
            "base_path": str(run_path),
            "targets": {target: target_manifest},
            "errors": {},
            "settings": {"mode": "Perform"},
            "quality_report_path": str(run_path / "quality_report.json"),
            "final_matrix_path": str(run_path / "final_matrix.csv"),
            "target_summary_path": str(run_path / "target_summary.csv"),
            "best_model_metrics_path": str(run_path / "best_model_metrics.csv"),
        },
    )

    run = load_automl_run(run_path)
    result = run["target_results"][target]

    assert run["run_id"] == run_id
    assert not result["leaderboard"].empty
    assert "y_true" in result["prediction_frame"].columns
    assert result["y_test"] is not None


def test_build_final_matrix_respects_metric_direction():
    target_results = {
        "class_target": {
            "config": {"direction": "max"},
            "leaderboard": pd.DataFrame(
                [
                    {"model_type": "Linear", "metric_value": 0.71},
                    {"model_type": "Linear", "metric_value": 0.75},
                    {"model_type": "Random Forest", "metric_value": 0.8},
                ]
            ),
        },
        "reg_target": {
            "config": {"direction": "min"},
            "leaderboard": pd.DataFrame(
                [
                    {"model_type": "Linear", "metric_value": 12.0},
                    {"model_type": "Linear", "metric_value": 10.0},
                    {"model_type": "Random Forest", "metric_value": 9.0},
                ]
            ),
        },
    }

    matrix = build_final_matrix(target_results)

    assert matrix.loc["class_target", "Linear"] == 0.75
    assert matrix.loc["reg_target", "Linear"] == 10.0
    assert matrix.loc["reg_target", "Random Forest"] == 9.0


def test_regression_metrics_include_adjusted_r2_and_smape():
    metrics = regression_metrics([3.0, 5.0, 7.0, 9.0], [2.5, 5.5, 6.5, 9.5], n_features=2)

    assert metrics["r2"] > 0
    assert metrics["r2_adjusted"] is not None
    assert metrics["smape"] >= 0
    assert metrics["score_global"] is not None
    assert 0 <= metrics["score_global"] <= 1


def test_regression_metrics_global_score_rewards_perfect_predictions():
    metrics = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], n_features=2)

    assert metrics["score_global"] == 1.0


def test_collect_model_metrics_builds_per_model_table_for_regression():
    class DummyModel:
        def __init__(self, name: str, model_type: str):
            self.name = name
            self.learner_params = {"model_type": model_type}

        def predict(self, X):
            return pd.Series([1.0, 2.0, 3.0, 4.0], index=X.index)

        def get_name(self):
            return self.name

    class DummyAutoML:
        def __init__(self):
            self._models = [DummyModel("model_a", "Linear")]
            self._stacked_models = []

    X_test = pd.DataFrame({"x": [10, 20, 30, 40]})
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0])

    table = _collect_model_metrics(DummyAutoML(), X_test, y_test, "regression", n_features=1)

    assert list(table["model_name"]) == ["model_a"]
    assert list(table["model_type"]) == ["Linear"]
    assert list(table["model_class"]) == ["DummyModel"]
    assert table.loc[0, "score_global"] == 1.0
    assert table.loc[0, "r2_adjusted"] == 1.0
    assert table.loc[0, "smape"] == 0.0


def test_resolve_model_by_name_and_predict_with_model_uses_holdout_winner():
    class DummyModel:
        def __init__(self, name: str, value: float, model_type: str = "Linear"):
            self.name = name
            self.value = value
            self.learner_params = {"model_type": model_type}

        def predict(self, X):
            return pd.Series([self.value] * len(X), index=X.index)

        def get_name(self):
            return self.name

    class DummyAutoML:
        def __init__(self):
            self._models = [DummyModel("internal_best", 10.0), DummyModel("holdout_best", 3.0)]
            self._stacked_models = []

    automl = DummyAutoML()
    X_test = pd.DataFrame({"x": [1, 2, 3]})

    model = resolve_model_by_name(automl, "holdout_best")
    assert model is not None
    assert model.get_name() == "holdout_best"

    y_pred, proba, proba_classes = predict_with_model(automl, "holdout_best", X_test, "regression")

    assert proba is None
    assert proba_classes is None
    assert list(y_pred) == [3.0, 3.0, 3.0]


# ── compute_variable_stats ─────────────────────────────────────────────────────


def test_compute_variable_stats_numeric_and_categorical_columns():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            # ~ Normal distribution (large n)
            "normal_var": rng.standard_normal(n),
            # Heavily skewed
            "skewed_var": rng.exponential(scale=2.0, size=n),
            # Constant
            "const_var": [7.0] * n,
            # 2 non-null values out of n (rest NaN) → Shapiro no aplicable (n < 3).
            "tiny_var": [1.0, 2.0] + [np.nan] * (n - 2),
            "category": rng.choice(["A", "B", "C"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    stats = compute_variable_stats(df, targets=["target"])

    by_col = {r["column"]: r for r in stats["columns"]}
    assert set(by_col) == {"normal_var", "skewed_var", "const_var", "tiny_var", "category", "target"}

    # Numeric values match pandas aggregations (within tolerance).
    nv = by_col["normal_var"]
    expected_mean = float(df["normal_var"].mean())
    expected_median = float(df["normal_var"].median())
    expected_var = float(df["normal_var"].var(ddof=1))
    assert nv["mean"] == pytest_approx(expected_mean)
    assert nv["median"] == pytest_approx(expected_median)
    assert nv["variance"] == pytest_approx(expected_var)
    assert nv["min"] == float(df["normal_var"].min())
    assert nv["max"] == float(df["normal_var"].max())
    assert nv["is_target"] is False

    # Shapiro results must be populated and self-consistent.
    assert nv["shapiro_W"] is not None
    assert nv["shapiro_p"] is not None
    assert nv["shapiro_n_used"] == n
    assert nv["shapiro_is_normal"] == (nv["shapiro_p"] > SHAPIRO_ALPHA)

    # Constant column → no Shapiro (varianza cero).
    cv = by_col["const_var"]
    assert cv["variance"] == 0.0
    assert cv["shapiro_p"] is None
    assert "varianza" in (cv["shapiro_note"] or "").lower()

    # Tiny series (< 3 non-null) → Shapiro not applicable.
    tv = by_col["tiny_var"]
    assert tv["n_non_null"] == 2
    assert tv["shapiro_p"] is None
    assert tv["shapiro_W"] is None

    # Categorical column → stats are None, mode/freq populated, no normality.
    cat = by_col["category"]
    assert cat["class"] == "categorical"
    assert cat["mean"] is None and cat["median"] is None and cat["variance"] is None
    assert cat["shapiro_p"] is None
    assert cat["top"] in {"A", "B", "C"}
    assert cat["freq"] == int((df["category"] == cat["top"]).sum())

    # Target flag must be carried through.
    assert by_col["target"]["is_target"] is True
    assert by_col["normal_var"]["is_target"] is False

    # Wide DataFrame mirrors long format with one row per column.
    wide = stats["wide"]
    assert len(wide) == len(df.columns)
    assert list(wide["column"]) == list(df.columns)


def test_compute_variable_stats_subsamples_for_large_series():
    rng = np.random.default_rng(123)
    n = 8000  # > SHAPIRO_MAX_SAMPLES (5000)
    df = pd.DataFrame({"big": rng.standard_normal(n)})

    stats = compute_variable_stats(df)
    big = next(r for r in stats["columns"] if r["column"] == "big")

    assert big["shapiro_n_used"] == 5000
    assert big["shapiro_W"] is not None
    assert "submuestreo" in (big["shapiro_note"] or "").lower()
    # Sub-sampling must be reproducible.
    stats_again = compute_variable_stats(df)
    big_again = next(r for r in stats_again["columns"] if r["column"] == "big")
    assert big_again["shapiro_W"] == big["shapiro_W"]


def test_compute_variable_stats_alpha_override_changes_decision():
    # Exponential data is clearly non-normal, so its Shapiro p-value is very
    # small. Use that to assert that the decision flips when alpha moves from
    # a value *above* the p-value to a value *below* it.
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"x": rng.exponential(scale=1.0, size=50)})

    loose = compute_variable_stats(df, alpha=0.5)   # p ≪ 0.5 → is_normal = False
    strict = compute_variable_stats(df, alpha=0.0)  # p > 0.0 → is_normal = True

    loose_row = next(r for r in loose["columns"] if r["column"] == "x")
    strict_row = next(r for r in strict["columns"] if r["column"] == "x")

    # p-valor y W deben ser idénticos, solo cambia la decisión.
    assert loose_row["shapiro_p"] == strict_row["shapiro_p"]
    assert loose_row["shapiro_W"] == strict_row["shapiro_W"]
    assert loose_row["shapiro_is_normal"] is False
    assert strict_row["shapiro_is_normal"] is True


def test_variable_stats_to_csv_and_json_are_well_formed():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": ["x", "y"] * 5,
        }
    )
    stats = compute_variable_stats(df, targets=["a"])

    csv_bytes = variable_stats_to_csv(stats)
    assert csv_bytes.startswith(b"column,")
    decoded = csv_bytes.decode("utf-8")
    assert "shapiro_p" in decoded
    assert "a" in decoded and "b" in decoded

    json_bytes = variable_stats_to_json(stats)
    payload = json.loads(json_bytes.decode("utf-8"))
    assert payload["shape"] == {"rows": 10, "columns": 2}
    assert payload["alpha"] == SHAPIRO_ALPHA
    assert {c["column"] for c in payload["columns"]} == {"a", "b"}
    assert {row["column"] for row in payload["wide"]} == {"a", "b"}
    # a is numeric, b is categorical.
    a_row = next(c for c in payload["columns"] if c["column"] == "a")
    b_row = next(c for c in payload["columns"] if c["column"] == "b")
    assert a_row["shapiro_p"] is not None
    assert b_row["shapiro_p"] is None
