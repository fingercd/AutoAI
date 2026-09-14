"""Deterministic grouped split plans owned by the training backend."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .parsers import load_modeling_csv, natural_sort_key

SPLIT_ALGORITHM_VERSION = 'group-stratified-test-first-v1'


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class PreparationResourceExhausted(ValueError):
    pass


@dataclass(frozen=True)
class PreparationLimits:
    deadline: float
    max_bytes: int
    block_size: int

    @classmethod
    def configured(cls):
        seconds = float(os.environ.get('AUTOAI_PREPARATION_SECONDS', '25'))
        size = int(os.environ.get('AUTOAI_PREPARATION_MAX_BYTES', str(1024**3)))
        block = int(os.environ.get('AUTOAI_PREPARATION_BLOCK_SIZE', '256'))
        if not np.isfinite(seconds) or seconds <= 0 or size < 1 or block < 1:
            raise ValueError('invalid preparation resource configuration')
        return cls(time.monotonic() + seconds, size, block)

    def check(self, size=0):
        if time.monotonic() > self.deadline or size > self.max_bytes:
            raise PreparationResourceExhausted('preparation_resource_exhausted')


@dataclass(frozen=True)
class DatasetView:
    x: np.ndarray
    labels: tuple[str, ...]
    sample_ids: tuple[str, ...]
    axis: tuple[float, ...]
    dataset_sha256: str

    @classmethod
    def loaded(cls, dataset, sha256):
        x = np.asarray(dataset.intensity, dtype=np.float32)
        if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
            raise ValueError('invalid dataset features')
        return cls(x, tuple(dataset.labels), tuple(dataset.sample_id),
                   tuple(dataset.x_axis[0]), sha256)


def load_dataset_view(path: Path, expected_sha256: str, limits=None) -> DatasetView:
    limits = limits or PreparationLimits.configured()
    limits.check(path.stat().st_size)
    def file_digest():
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    if file_digest() != expected_sha256:
        raise ValueError('dataset_changed')
    data = load_modeling_csv(path)
    limits.check(np.asarray(data.intensity).nbytes * 4)
    if file_digest() != expected_sha256:
        raise ValueError('dataset_changed')
    return DatasetView.loaded(data, expected_sha256)


class EvaluationPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['evaluation-plan-v1'] = 'evaluation-plan-v1'
    split_algorithm_version: Literal['group-stratified-test-first-v1'] = SPLIT_ALGORITHM_VERSION
    dataset_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    axis_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    evaluation_config_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    seed: int
    indices: dict[str, list[int]]
    groups: dict[str, list[str]]
    partition_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    plan_digest: str = Field(pattern=r'^[a-f0-9]{64}$')

    def safe_reference(self):
        return {k:v for k,v in self.model_dump(mode='json').items() if k not in ('indices','groups')}


@dataclass(frozen=True)
class PreparedEvaluation:
    plan_digest: str
    dataset_sha256: str


@dataclass(frozen=True)
class TrainView:
    x: np.ndarray
    labels: tuple[str, ...]
    sample_ids: tuple[str, ...]
    axis: tuple[float, ...]
    dataset_sha256: str
    plan_digest: str


def evaluation_config(config):
    return {k:config[k] for k in ('split_mode','split_train','split_valid','split_test')}


def _partition(indices, groups):
    return digest({'indices':indices, 'groups':groups})


def build_evaluation_plan(view: DatasetView, evaluation: dict, seed: int) -> EvaluationPlan:
    if evaluation.get('split_mode') != 'stratified_holdout':
        raise ValueError('prepared plans currently require stratified_holdout')
    labels = sorted(set(view.labels))
    mapping = {name:i for i,name in enumerate(labels)}
    splits = split_indices(np.asarray([mapping[v] for v in view.labels]),
        np.asarray(view.sample_ids), SimpleNamespace(**evaluation, seed=seed), labels)
    groups = {name:sorted({view.sample_ids[i] for i in values},key=natural_sort_key)
              for name,values in splits.items()}
    body = dict(schema_version='evaluation-plan-v1', split_algorithm_version=SPLIT_ALGORITHM_VERSION,
        dataset_sha256=view.dataset_sha256,axis_digest=digest(view.axis),
        evaluation_config_digest=digest(evaluation_config(evaluation)),seed=seed,
        indices=splits,groups=groups,partition_digest=_partition(splits,groups))
    plan = EvaluationPlan(**body,plan_digest=digest(body))
    validate_plan(plan,view,evaluation,seed)
    return plan


def validate_plan(plan: EvaluationPlan, view: DatasetView, evaluation: dict, seed: int):
    body=plan.model_dump(mode='json')
    bound=body.pop('plan_digest')
    if (digest(body)!=bound or view.dataset_sha256!=plan.dataset_sha256
            or digest(view.axis)!=plan.axis_digest or seed!=plan.seed
            or digest(evaluation_config(evaluation))!=plan.evaluation_config_digest):
        raise ValueError('evaluation_plan_binding_mismatch')
    if set(plan.indices)!= {'train','valid','test'} or set(plan.groups)!=set(plan.indices):
        raise ValueError('invalid evaluation partitions')
    rows=[];seen_groups=set();expected_labels=set(view.labels)
    for name,indices in plan.indices.items():
        if not indices or any(type(i) is not int or i<0 or i>=len(view.x) for i in indices):
            raise ValueError('invalid evaluation indices')
        groups={view.sample_ids[i] for i in indices}
        if groups!=set(plan.groups[name]) or seen_groups & groups:
            raise ValueError('evaluation groups cross partitions')
        if {view.labels[i] for i in indices}!=expected_labels:
            raise ValueError('evaluation partition missing classes')
        seen_groups.update(groups);rows.extend(indices)
    if sorted(rows)!=list(range(len(view.x))):
        raise ValueError('evaluation row coverage mismatch')
    group_labels={}
    for group,label in zip(view.sample_ids,view.labels):
        if group in group_labels and group_labels[group]!=label:
            raise ValueError('invalid group labels')
        group_labels[group]=label
    if _partition(plan.indices,plan.groups)!=plan.partition_digest:
        raise ValueError('evaluation partition digest mismatch')


def select_train(view: DatasetView, plan: EvaluationPlan) -> TrainView:
    indices=plan.indices['train']
    x=view.x[indices].copy();x.setflags(write=False)
    return TrainView(x,tuple(view.labels[i] for i in indices),
        tuple(view.sample_ids[i] for i in indices),view.axis,view.dataset_sha256,plan.plan_digest)


def split_indices(
    labels: np.ndarray,
    sample_id: np.ndarray,
    config: SimpleNamespace,
    label_names: list[str] | None = None,
) -> dict[str, list[int]]:
    """按 Sample_ID 分组划分，并保证每个非空集合至少含每类一个样品组。"""
    ratios = (int(config.split_train), int(config.split_valid), int(config.split_test))
    if any(value < 0 for value in ratios):
        raise ValueError("划分比例必须是非负整数")
    ratio_total = sum(ratios)
    if ratios[2] > 0 and ratio_total != 10:
        raise ValueError("划分比例必须是非负整数，且训练、验证、测试三项相加必须等于 10")
    if ratios[2] == 0 and ratios[0] + ratios[1] <= 0:
        raise ValueError("训练/验证比例必须大于 0")
    if ratios[0] <= 0:
        raise ValueError("训练集比例必须大于 0")

        # 以 Sample_ID 为不可拆分单位；natural_sort_key 让 XJ-2 排在 XJ-10 之前，结果可复现。
    group_values = np.asarray(sorted(np.unique(sample_id).tolist(), key=natural_sort_key))
    nonzero_splits = sum(1 for value in ratios if value > 0)
    if len(group_values) < nonzero_splits:
        raise ValueError(f"当前只有 {len(group_values)} 个 Sample_ID 分组，无法划分为 {nonzero_splits} 个非空集合")

    group_to_label: dict[str, int] = {}
    for group in group_values:
        group_labels = np.unique(labels[sample_id == group])
        if len(group_labels) != 1:
            raise ValueError(f"Sample_ID={group} 内存在多个 Label，无法按组划分")
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
        details = "、".join(
            f"{label_names[label] if label_names and 0 <= label < len(label_names) else f'类别编码 {label}'}"
            f"（{count} 个 Sample_ID）"
            for label, count in sorted(insufficient.items())
        )
        split_names = "Train/Valid/Test" if ratios[2] > 0 else "Train/Valid"
        raise ValueError(
            f"要让 {split_names} 都包含全部类别，每个类别至少需要 {nonzero_splits} 个不同的 "
            f"Sample_ID；当前不足：{details}"
        )

        # 计算各集合应含的“组数”：valid/test 的硬下限是类别数（保证类别完整），
        # 再按比例目标向下取整；总数超界时优先从 test、其次 valid 回退，仍不够则拒绝。
    ratio_denominator = ratios[0] + ratios[1] if ratios[2] == 0 else 10
    valid_minimum = label_count if ratios[1] > 0 else 0
    test_minimum = label_count if ratios[2] > 0 else 0
    valid_count = max(valid_minimum, int(np.floor(len(group_values) * ratios[1] / ratio_denominator))) if ratios[1] > 0 else 0
    test_count = max(test_minimum, int(np.floor(len(group_values) * ratios[2] / ratio_denominator))) if ratios[2] > 0 else 0
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
            raise ValueError("Sample_ID 分组数量太少，无法让每个集合都包含全部类别")
    train_count = len(group_values) - valid_count - test_count

        # 固定种子打乱各类别内部的组顺序，使同一数据 + 同一种子的划分完全可复现。
    rng = np.random.default_rng(config.seed)
    for groups in label_to_groups.values():
        rng.shuffle(groups)

    def take_class_complete(count: int, *, reserve_per_label: int) -> set[str]:
        if count == 0:
            return set()
        selected: set[str] = set()
        # 先为每个类别固定拿 1 个 Sample_ID，形成评估集合的硬下限。
        for label in label_values:
            selected.add(label_to_groups[label].pop())
        # 比例目标大于类别数时继续分层补足，但始终为后续集合保留每类样品。
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
                raise ValueError("无法在保留各集合类别完整性的同时完成当前比例划分")
            for label in labels_by_remaining:
                if len(selected) >= count:
                    break
                if len(label_to_groups[label]) > reserve_per_label:
                    selected.add(label_to_groups[label].pop())
        return selected

        # 先取 test、再取 valid，剩余全部归 train；reserve_per_label 为后续集合
        # 预留每类至少 1 组——这是“每类至少 3 个不同 Sample_ID”硬约束的执行点。
    test_groups = take_class_complete(
        test_count,
        reserve_per_label=1 + (1 if valid_count > 0 else 0),
    )
    valid_groups = take_class_complete(valid_count, reserve_per_label=1)
    train_groups = {group for groups in label_to_groups.values() for group in groups}
    split_groups = {"train": train_groups, "valid": valid_groups, "test": test_groups}
    return {
        split: np.where(np.isin(sample_id, list(groups)))[0].tolist()
        for split, groups in split_groups.items()
    }
