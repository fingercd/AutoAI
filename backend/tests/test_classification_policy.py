from backend.app.parsers import load_modeling_csv
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
