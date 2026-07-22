from __future__ import annotations

import numpy as np
import pytest

from backend.app.training import TrainConfig, _split_indices


def _grouped_labels(*, class_count: int, groups_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    sample_ids: list[str] = []
    for label in range(class_count):
        for group_index in range(groups_per_class):
            labels.append(label)
            sample_ids.append(f"class-{label}-sample-{group_index + 1}")
    return np.asarray(labels, dtype=np.int64), np.asarray(sample_ids, dtype=str)


def test_8_1_1_raises_valid_and_test_minimum_to_one_group_per_class() -> None:
    labels, sample_ids = _grouped_labels(class_count=4, groups_per_class=6)

    splits = _split_indices(
        labels,
        sample_ids,
        TrainConfig(split_train=8, split_valid=1, split_test=1, seed=42),
        ["A", "B", "C", "D"],
    )

    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 16,
        "valid": 4,
        "test": 4,
    }
    for split_name in ("train", "valid", "test"):
        assert set(labels[splits[split_name]].tolist()) == {0, 1, 2, 3}
    assert set(splits["train"]).isdisjoint(splits["valid"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["valid"]).isdisjoint(splits["test"])


def test_8_1_1_preserves_ratio_counts_when_they_already_cover_every_class() -> None:
    labels, sample_ids = _grouped_labels(class_count=2, groups_per_class=15)

    splits = _split_indices(
        labels,
        sample_ids,
        TrainConfig(split_train=8, split_valid=1, split_test=1, seed=7),
        ["A", "B"],
    )

    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 24,
        "valid": 3,
        "test": 3,
    }
    for split_name in ("train", "valid", "test"):
        assert set(labels[splits[split_name]].tolist()) == {0, 1}


def test_8_1_1_rejects_class_with_fewer_than_three_sample_ids() -> None:
    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    sample_ids = np.asarray(["A1", "A2", "B1", "B2", "B3"], dtype=str)

    with pytest.raises(
        ValueError,
        match=r"每个类别至少需要 3 个不同的 Sample_ID.*A（2 个 Sample_ID）",
    ):
        _split_indices(
            labels,
            sample_ids,
            TrainConfig(split_train=8, split_valid=1, split_test=1),
            ["A", "B"],
        )
