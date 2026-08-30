"""Deterministic Sample_ID-grouped stratified holdout planning."""

from __future__ import annotations

import numpy as np

from .parsers import natural_sort_key


SPLITTER_VERSION = 'grouped-stratified-holdout-v1'


def stratified_group_holdout_indices(
    labels: np.ndarray,
    sample_id: np.ndarray,
    *,
    seed: int,
    split_train: int,
    split_valid: int,
    split_test: int,
    label_names: list[str] | None = None,
) -> dict[str, list[int]]:
    """Split immutable Sample_ID groups while keeping every class complete."""
    ratios = (int(split_train), int(split_valid), int(split_test))
    if any(value < 0 for value in ratios):
        raise ValueError('划分比例必须是非负整数')
    ratio_total = sum(ratios)
    if ratios[2] > 0 and ratio_total != 10:
        raise ValueError('划分比例必须是非负整数，且训练、验证、测试三项相加必须等于 10')
    if ratios[2] == 0 and ratios[0] + ratios[1] <= 0:
        raise ValueError('训练/验证比例必须大于 0')
    if ratios[0] <= 0:
        raise ValueError('训练集比例必须大于 0')

    group_values = np.asarray(
        sorted(np.unique(sample_id).tolist(), key=natural_sort_key)
    )
    nonzero_splits = sum(1 for value in ratios if value > 0)
    if len(group_values) < nonzero_splits:
        raise ValueError(
            f'当前只有 {len(group_values)} 个 Sample_ID 分组，'
            f'无法划分为 {nonzero_splits} 个非空集合'
        )

    group_to_label: dict[str, int] = {}
    for group in group_values:
        group_labels = np.unique(labels[sample_id == group])
        if len(group_labels) != 1:
            raise ValueError(f'Sample_ID={group} 内存在多个 Label，无法按组划分')
        group_to_label[str(group)] = int(group_labels[0])

    label_to_groups: dict[int, list[str]] = {}
    for group, label in group_to_label.items():
        label_to_groups.setdefault(label, []).append(group)
    label_values = sorted(label_to_groups)
    label_count = len(label_values)
    insufficient = {
        label: len(groups)
        for label, groups in label_to_groups.items()
        if len(groups) < nonzero_splits
    }
    if insufficient:
        details = '、'.join(
            f"{label_names[label] if label_names and 0 <= label < len(label_names) else f'类别编码 {label}'}"
            f'（{count} 个 Sample_ID）'
            for label, count in sorted(insufficient.items())
        )
        split_names = 'Train/Valid/Test' if ratios[2] > 0 else 'Train/Valid'
        raise ValueError(
            f'要让 {split_names} 都包含全部类别，每个类别至少需要 '
            f'{nonzero_splits} 个不同的 Sample_ID；当前不足：{details}'
        )

    ratio_denominator = ratios[0] + ratios[1] if ratios[2] == 0 else 10
    valid_minimum = label_count if ratios[1] > 0 else 0
    test_minimum = label_count if ratios[2] > 0 else 0
    valid_count = (
        max(
            valid_minimum,
            int(np.floor(len(group_values) * ratios[1] / ratio_denominator)),
        )
        if ratios[1] > 0
        else 0
    )
    test_count = (
        max(
            test_minimum,
            int(np.floor(len(group_values) * ratios[2] / ratio_denominator)),
        )
        if ratios[2] > 0
        else 0
    )
    if valid_count + test_count >= len(group_values):
        train_count = label_count
        overflow = valid_count + test_count + train_count - len(group_values)
        while overflow > 0 and test_count > test_minimum:
            test_count -= 1
            overflow -= 1
        while overflow > 0 and valid_count > valid_minimum:
            valid_count -= 1
            overflow -= 1
        if overflow > 0:
            raise ValueError('Sample_ID 分组数量太少，无法让每个集合都包含全部类别')

    rng = np.random.default_rng(seed)
    for groups in label_to_groups.values():
        rng.shuffle(groups)

    def take_class_complete(count: int, *, reserve_per_label: int) -> set[str]:
        if count == 0:
            return set()
        selected: set[str] = set()
        for label in label_values:
            selected.add(label_to_groups[label].pop())
        while len(selected) < count:
            labels_by_remaining = sorted(
                (
                    label
                    for label in label_values
                    if len(label_to_groups[label]) > reserve_per_label
                ),
                key=lambda label: (-len(label_to_groups[label]), label),
            )
            if not labels_by_remaining:
                raise ValueError('无法在保留各集合类别完整性的同时完成当前比例划分')
            for label in labels_by_remaining:
                if len(selected) >= count:
                    break
                if len(label_to_groups[label]) > reserve_per_label:
                    selected.add(label_to_groups[label].pop())
        return selected

    test_groups = take_class_complete(
        test_count,
        reserve_per_label=1 + (1 if valid_count > 0 else 0),
    )
    valid_groups = take_class_complete(valid_count, reserve_per_label=1)
    train_groups = {
        group for groups in label_to_groups.values() for group in groups
    }
    split_groups = {
        'train': train_groups,
        'valid': valid_groups,
        'test': test_groups,
    }
    return {
        split: np.where(np.isin(sample_id, list(groups)))[0].tolist()
        for split, groups in split_groups.items()
    }
