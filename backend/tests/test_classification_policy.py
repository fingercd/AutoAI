import numpy as np
import pytest

from backend.app.parsers import load_modeling_csv
from backend.app.classification_policy import resolve_evaluation_policy
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_grouped_data_factory_preserves_sample_id_groups(tmp_path):
    path = write_grouped_classification_csv(
        tmp_path / "grouped.csv",
        groups_per_class=5,
        repeats=2,
        feature_count=12,
    )
    dataset = load_modeling_csv(path)
    assert dataset.frame["Sample_ID"].nunique() == 10
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


def test_external_policy_keeps_leave_one_as_primary_data_audit():
    policy = resolve_evaluation_policy(
        {"split_mode": "leave_one_sample_id_cv", "split_train": 8, "split_valid": 2, "split_test": 0},
        has_external_test=True,
    )
    assert policy.strategy == "leave_one_sample_id_cv_with_external_test"
    assert policy.cv_allowed is True


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
    x_raw = np.zeros((10, 4), dtype=np.float32)
    y = np.asarray([0] * 5 + [1] * 5)
    sample_id = np.asarray([str(index) for index in range(10)])
    splits = {"train": list(range(8)), "valid": [8, 9], "test": []}

    class FakeModel:
        def __init__(self, c):
            self.c = c

        def fit(self, x, y):
            return self

    monkeypatch.setattr(training, "_traditional_candidate_configs", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(training, "_grouped_inner_cv_indices", lambda **_kwargs: [(np.asarray([0, 1, 5, 6]), np.asarray([2, 7]))] * 5)
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
        x_raw,
        y,
        sample_id,
        splits,
        ["A", "B"],
    )

    assert selection.config.svm_c == 10.0
    assert selection.valid_balanced_accuracy == pytest.approx(0.8)
    selected_rows = [row for row in selection.search_rows if row["is_selected"]]
    assert len(selected_rows) == 1
    assert '"svm_c": 10.0' in selected_rows[0]["params_json"]


def test_random_forest_selection_uses_grouped_inner_balanced_accuracy_not_oob(monkeypatch):
    import backend.app.training as training

    candidates = [
        training.TrainConfig(model_type="random_forest", random_forest_max_depth=3, random_forest_oob_score=True),
        training.TrainConfig(model_type="random_forest", random_forest_max_depth=5, random_forest_oob_score=True),
    ]
    validation_calls = []

    class FakeForest:
        def __init__(self, depth):
            self.depth = depth

        def fit(self, x, y):
            return self

    monkeypatch.setattr(training, "_traditional_candidate_configs", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(training, "_grouped_inner_cv_indices", lambda **_kwargs: [(np.asarray([0, 1, 5, 6]), np.asarray([2, 7]))] * 5)
    monkeypatch.setattr(
        training,
        "build_traditional_model",
        lambda config, y, class_count: FakeForest(config.random_forest_max_depth),
    )

    def fake_evaluate(model, x, y, indices, labels):
        validation_calls.append(model.depth)
        return {
            "accuracy": 0.2,
            "balanced_accuracy": 0.8 if model.depth == 3 else 0.2,
            "macro_f1": 0.2,
            "true": [0, 1],
            "pred": [1, 0],
            "probabilities": [[0.1, 0.9], [0.9, 0.1]],
        }

    monkeypatch.setattr(training, "_evaluate_traditional_model", fake_evaluate)
    selection = training._select_traditional_config(
        training.TrainConfig(model_type="random_forest"),
        "random_forest",
        np.zeros((10, 3), dtype=np.float32),
        np.asarray([0] * 5 + [1] * 5),
        np.asarray([str(index) for index in range(10)]),
        {"train": list(range(8)), "valid": [8, 9], "test": []},
        ["A", "B"],
    )

    assert selection.config.random_forest_max_depth == 3
    assert validation_calls.count(3) == 6
    assert validation_calls.count(5) == 5
    selected = [row for row in selection.search_rows if row["is_selected"]]
    assert len(selected) == 1
    assert selected[0]["selection_metric"] == "mean_balanced_accuracy_grouped_5fold"
    assert selected[0]["selection_score"] == pytest.approx(0.8)
    assert all("oob_accuracy" not in row and "oob_balanced_accuracy" not in row for row in selection.search_rows)


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


def test_spls_da_is_sparse_supervised_and_pickle_safe():
    """A fixed signal fixture checks sparse components and probability semantics."""
    import pickle
    from backend.app.models.spls_da import SPLSDAClassifier

    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], 24)
    values = rng.normal(0, 0.05, size=(48, 8))
    values[labels == 1, 0] += 2.5
    values[labels == 0, 1] += 2.5
    model = SPLSDAClassifier(n_components=2, keepX=2).fit(values, labels)
    restored = pickle.loads(pickle.dumps(model))
    probabilities = restored.predict_proba(values)

    assert model.actual_n_components_ >= 1
    assert all(len(component) <= 2 for component in model.selected_feature_indices_by_component)
    assert {0, 1}.intersection(model.selected_feature_indices_by_component[0])
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (restored.predict(values) == labels).mean() >= 0.95


def test_pca_svm_pca_is_fit_only_on_the_caller_training_fold():
    from backend.app.models.pca_svm import build_pca_svm

    train = np.asarray([[-2.0, -1.0, 0.0], [-1.5, -0.9, 0.1], [1.5, 1.0, 0.0], [2.0, 0.9, -0.1]])
    held_out = np.asarray([[1000.0, -1000.0, 500.0]])
    model = build_pca_svm(n_components=2, c=1.0, seed=9)
    assert not hasattr(model.named_steps["pca"], "components_")
    model.fit(train, np.asarray([0, 0, 1, 1]))

    assert np.allclose(model.named_steps["pca"].mean_, train.mean(axis=0))
    assert not np.allclose(model.named_steps["pca"].mean_, np.vstack([train, held_out]).mean(axis=0))
    probabilities = model.predict_proba(held_out)
    assert np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)


def test_grouped_inner_cv_rejects_less_than_five_sample_ids_per_class():
    import backend.app.training as training

    with pytest.raises(ValueError, match="每类至少 5 个 Sample_ID"):
        training._grouped_inner_cv_indices(
            y=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            sample_id=np.asarray(["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"]),
            candidate_indices=list(range(8)),
            split_seed=42,
            label_names=["A", "B"],
        )


def test_split_seed_controls_group_partitions_independently_of_model_seed():
    import backend.app.training as training

    sample_id = np.asarray([str(index) for index in range(1, 15) for _ in range(2)])
    y = np.asarray([0 if index <= 7 else 1 for index in range(1, 15) for _ in range(2)])
    first = training.TrainConfig(split_seed=31, model_seed=31, seed=31)
    second = training.TrainConfig(split_seed=31, model_seed=87, seed=87)
    changed = training.TrainConfig(split_seed=32, model_seed=31, seed=31)

    assert training._split_indices(y, sample_id, first, ["A", "B"]) == training._split_indices(y, sample_id, second, ["A", "B"])
    assert training._leave_one_sample_id_folds(y, sample_id, first) == training._leave_one_sample_id_folds(y, sample_id, second)
    assert training._split_indices(y, sample_id, first, ["A", "B"]) != training._split_indices(y, sample_id, changed, ["A", "B"])


@pytest.mark.parametrize(
    ("model_type", "expected_keys"),
    [
        ("pls_da", {"pls_components"}),
        ("spls_da", {"spls_components", "spls_keepx"}),
        ("pca_lda", {"pca_components"}),
        ("logistic_regression", {"logistic_c", "logistic_l1_ratio"}),
        ("svm", {"svm_kernel", "svm_c", "svm_gamma"}),
        ("pca_svm", {"pca_components", "svm_kernel", "svm_c"}),
        (
            "random_forest",
            {
                "random_forest_n_estimators",
                "random_forest_max_depth",
                "random_forest_min_samples_leaf",
                "random_forest_max_features",
            },
        ),
        (
            "xgboost",
            {
                "xgboost_n_estimators",
                "xgboost_max_depth",
                "xgboost_learning_rate",
                "xgboost_subsample",
                "xgboost_colsample_bytree",
                "xgboost_min_child_weight",
                "xgboost_reg_lambda",
                "xgboost_gamma",
            },
        ),
    ],
)
def test_traditional_audit_contains_only_selected_model_parameters(model_type, expected_keys):
    import backend.app.training as training

    params = training._traditional_params(training.TrainConfig(model_type=model_type), model_type)

    assert set(params) == expected_keys


def test_deep_training_defaults_are_exact():
    from backend.app.classification_policy import DEEP_TRAINING_DEFAULTS
    import backend.app.training as training

    expected = {
        "epochs": 200,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler_factor": 0.5,
        "scheduler_patience": 10,
        "min_learning_rate": 1e-6,
        "early_stopping_patience": 20,
        "seed": 42,
    }
    assert {key: getattr(DEEP_TRAINING_DEFAULTS, key) for key in expected} == expected
    config = training.TrainConfig()
    for key, value in expected.items():
        assert getattr(config, key) == value


def test_deep_training_uses_adamw_scheduler_and_lowest_validation_loss(monkeypatch, tmp_path):
    import torch
    from torch import nn
    import backend.app.training as training

    optimizer_kwargs = {}
    scheduler_kwargs = {}
    scheduler_step_values = []
    validation_losses = [0.4, 0.2, 0.3]

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.train_calls = 0
            self.state_id = 0
            self.loaded_state_id = None

        def train(self, mode=True):
            if mode:
                self.state_id = self.train_calls
                self.train_calls += 1
            return super().train(mode)

        def forward(self, values):
            return self.anchor.expand(values.shape[0], 2)

        def state_dict(self, *args, **kwargs):
            return {"state_id": torch.tensor(self.state_id)}

        def load_state_dict(self, state, *args, **kwargs):
            self.loaded_state_id = int(state["state_id"])
            return nn.modules.module._IncompatibleKeys([], [])

    model = FakeModel()

    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            del parameters
            optimizer_kwargs.update(kwargs)
            self.param_groups = [{"lr": kwargs["lr"]}]

        def zero_grad(self):
            pass

        def step(self):
            pass

    class FakeScheduler:
        def __init__(self, optimizer, **kwargs):
            del optimizer
            scheduler_kwargs.update(kwargs)

        def step(self, value):
            scheduler_step_values.append(float(value))

    monkeypatch.setattr(training, "build_deep_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(training.torch.optim, "AdamW", FakeOptimizer)
    monkeypatch.setattr(training.torch.optim.lr_scheduler, "ReduceLROnPlateau", FakeScheduler)
    monkeypatch.setattr(training, "_loader", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        training,
        "_evaluate_deep_loss",
        lambda *args, **kwargs: validation_losses.pop(0),
    )
    monkeypatch.setattr(
        training,
        "_evaluate",
        lambda *args, **kwargs: {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
        },
    )

    config = training.TrainConfig(epochs=3, early_stopping_patience=0, model_type="cnn1d")
    _, history, _, _ = training._fit_deep_fold(
        config=config,
        model_type="cnn1d",
        x=np.zeros((4, 6), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1]),
        splits={"train": [0, 1], "valid": [2, 3], "test": []},
        label_names=["A", "B"],
        run_dir=tmp_path,
        sample_count=4,
    )

    assert optimizer_kwargs == {"lr": 1e-3, "weight_decay": 1e-4}
    assert scheduler_kwargs == {
        "mode": "min",
        "factor": 0.5,
        "patience": 10,
        "min_lr": 1e-6,
    }
    assert scheduler_step_values == [0.4, 0.2, 0.3]
    assert model.loaded_state_id == 1
    assert [row["valid_loss"] for row in history] == [0.4, 0.2, 0.3]
    assert [row["best_valid_loss"] for row in history] == [0.4, 0.2, 0.2]
    assert all(row["learning_rate"] == 1e-3 for row in history)
