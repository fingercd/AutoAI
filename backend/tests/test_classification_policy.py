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
