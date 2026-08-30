from __future__ import annotations

import numpy as np

from backend.app.classification_split import stratified_group_holdout_indices
from backend.app.training import TrainConfig, _split_indices


def test_training_wrapper_and_shared_split_are_identical():
    groups = np.asarray([f'g-{index}' for index in range(30) for _ in range(2)])
    labels = np.asarray([index // 10 for index in range(30) for _ in range(2)])
    config = TrainConfig(seed=73, split_train=8, split_valid=1, split_test=1)
    expected = stratified_group_holdout_indices(
        labels,
        groups,
        seed=73,
        split_train=8,
        split_valid=1,
        split_test=1,
        label_names=['a', 'b', 'c'],
    )
    assert _split_indices(labels, groups, config, ['a', 'b', 'c']) == expected
