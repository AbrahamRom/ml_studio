import pandas as pd

from ml_pipeline.automl_runner import _collect_model_metrics, predict_with_model, resolve_model_by_name
from ml_pipeline.artifacts import create_run_dir, save_json
from ml_pipeline.comparison import build_final_matrix
from ml_pipeline.metrics import regression_metrics
from ml_pipeline.quality import analyze_data_quality
from ml_pipeline.tasks import infer_target_task


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
