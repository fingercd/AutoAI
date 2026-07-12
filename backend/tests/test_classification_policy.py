import numpy as np
import pytest

from backend.app.parsers import load_modeling_csv
from backend.app.classification_policy import resolve_evaluation_policy
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_grouped_data_factory_preserves_repeat_index_groups(tmp_path):
    path = write_grouped_classification_csv(
        tmp_path / "grouped.csv",
        groups_per_class=5,
        repeats=2,
        feature_count=12,
    )
    dataset = load_modeling_csv(path)
    assert dataset.frame["Repeat_index"].nunique() == 10
    assert set(dataset.labels) == {"A", "B"}


def test_external_policy_defaults_to_eight_two_and_forbids_cv():
    policy = resolve_evaluation_policy({}, has_external_test=True)
    assert (policy.split_train, policy.split_valid, policy.split_test) == (8, 2, 0)
    assert policy.strategy == "external_test_holdout"
    assert policy.cv_allowed is False


def test_internal_policy_defaults_to_eight_one_one():
    policy = resolve_evaluation_policy({}, has_external_test=False)
    assert (policy.split_train, policy.split_valid, policy.split_test) == (8, 1, 1)
    assert policy.strategy == "stratified_holdout"


def test_external_policy_rejects_cross_validation():
    with pytest.raises(ValueError, match="独立测试集.*交叉验证"):
        resolve_evaluation_policy(
            {"split_mode": "leave_one_repeat_index_cv"},
            has_external_test=True,
        )


def test_classification_metrics_include_balanced_accuracy():
    import backend.app.training as training

    metrics = training._classification_metrics_payload(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 0, 0, 1]),
        ["A", "B"],
    )
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_traditional_selection_uses_balanced_accuracy(monkeypatch):
    import backend.app.training as training

    candidates = [
        training.TrainConfig(model_type="svm", svm_c=1.0),
        training.TrainConfig(model_type="svm", svm_c=10.0),
    ]

    class FakeModel:
        def __init__(self, c):
            self.c = c

        def fit(self, x, y):
            return self

    monkeypatch.setattr(training, "_traditional_candidate_configs", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(training, "build_traditional_model", lambda config, y, class_count: FakeModel(config.svm_c))

    def fake_evaluate(model, x, y, indices, labels):
        return {
            "accuracy": 0.5,
            "balanced_accuracy": 0.4 if model.c == 1.0 else 0.8,
            "macro_f1": 0.95 if model.c == 1.0 else 0.80,
            "true": [0, 1],
            "pred": [0, 1],
            "probabilities": [[0.9, 0.1], [0.1, 0.9]],
        }

    monkeypatch.setattr(training, "_evaluate_traditional_model", fake_evaluate)
    selection = training._select_traditional_config(
        training.TrainConfig(model_type="svm"),
        "svm",
        np.zeros((2, 4), dtype=np.float32),
        np.asarray([0, 1]),
        np.zeros((2, 4), dtype=np.float32),
        np.asarray([0, 1]),
        ["A", "B"],
    )

    assert selection.config.svm_c == 10.0
    assert selection.valid_balanced_accuracy == pytest.approx(0.8)
    selected_rows = [row for row in selection.search_rows if row["is_selected"]]
    assert len(selected_rows) == 1
    assert '"svm_c": 10.0' in selected_rows[0]["params_json"]


def test_final_traditional_fit_uses_only_train_and_valid(monkeypatch):
    import backend.app.training as training

    fit_rows = []

    class SpyModel:
        def fit(self, x, y):
            fit_rows.append(np.asarray(x)[:, 0].astype(int).tolist())
            return self

    monkeypatch.setattr(training, "build_traditional_model", lambda *args, **kwargs: SpyModel())
    x_raw = np.tile(np.arange(5, dtype=np.float32).reshape(-1, 1), (1, 4))
    y = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
    _, normalizer, final_fit_indices = training._fit_final_traditional_model(
        selected_config=training.TrainConfig(model_type="svm"),
        model_type="svm",
        x_raw=x_raw,
        y=y,
        train_valid_indices=[0, 1, 2],
        normalization="none",
        label_names=["A", "B"],
    )

    assert fit_rows == [[0, 1, 2]]
    assert set(final_fit_indices.tolist()) == {0, 1, 2}
    assert set(final_fit_indices.tolist()).isdisjoint({3, 4})
    assert normalizer["mode"] == "none"
