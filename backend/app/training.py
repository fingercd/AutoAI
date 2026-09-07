"""SpecAutoAI 分类训练、评估、模型选择与 artifact 编排主模块。

核心不变量：所有标准化、PCA、AggMap 和超参数选择只能查看当前训练折；传统模型
在 valid/OOB 选定参数后用 train+valid 重训；深度模型以最低 validation loss
保存最佳权重。每折 test 只用于最终指标和解释性分析，不参与拟合或早停。

本模块可以被 worker 调用，也保留直接 ``train_model`` 兼容入口。真正的 Run 状态
迁移和 Manifest 原子发布由 ``runs.execution`` / ``runs.artifacts`` 负责。
"""

# ──────────────────────────────────────────────────────────────────────────
# 模块导览（教学注释）
#
# 职责：本模块是 SpecAutoAI 后端的“训练编排中枢”。一次训练请求在这里完成：
#   读取 wide-feature 宽表 → 解析评估策略（stratified_holdout /
#   leave_one_sample_id_cv / external_test_holdout）→ 以 Sample_ID 整组为单位
#   划分 → 逐折训练传统模型或深度模型 → 汇总指标与可解释性 → 落盘
#   config.json / metrics.json / split.json / model.pkl|model.pt 等 Run 产物。
#
# 协作模块：
#   - parsers.load_modeling_csv：读取 wide-feature-v1/v2 宽表并还原真实 X 轴；
#   - classification_policy：把请求体解析成 EvaluationPolicy（评估口径）；
#   - models / models.profiles：按 model_type 构造目录内模型与容量档位；
#   - dscarnet_mapping：DSCARNet 的 AggMap/PCA SAR/CAR 二维映射；
#   - feature_selection：窗口遮挡 Log-loss、Grad-CAM 等可解释性计算与落盘；
#   - runs.repository / runs.artifacts / runs.status_projection：Run 状态机、
#     Manifest 原子发布与 status.json 投影（本模块不直接拥有状态机）。
#
# 关键设计约束：
#   1. 数据泄漏防线：标准化、PCA、AggMap、超参搜索只拟合当前折的 train；
#      test 永远只参与最终评估与解释性分析，不参与拟合、选参或早停。
#   2. 划分单位是 Sample_ID 整组而非单行曲线，同一 Sample_ID 不会跨集合。
#   3. CV 汇报的 test 主指标固定使用 pooled OOF（所有折 test 预测合并计算），
#      fold mean/std 仅作审计参考，不得冒充主指标。
#   4. 取消/替换检查（check_run_active / cancel_check）贯穿训练全程。
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .classification_policy import DEEP_TRAINING_DEFAULTS, EvaluationPolicy, resolve_evaluation_policy
from .feature_flags import EXPLAINABILITY_ENABLED
from .dscarnet_mapping import DSCARNetMappedInputs, fit_dscarnet_2d_mapping, save_dscarnet_mapping_artifacts
from .feature_selection import (
    aggregate_attribution_sanity,
    aggregate_dscarnet_branch_sanity,
    sample_deep_attribution_importance,
    sample_dscarnet_dual_2d_gradcam_importance,
    sample_dscarnet_single_2d_gradcam_importance,
    sample_occlusion_importance,
    write_sample_feature_importance_artifacts,
)
from .models import ARCHITECTURE_VERSION, build_deep_model, build_dscarnet_model, build_traditional_model, canonical_model_type, model_family
from .models.profiles import build_dscarnet_profile, build_model_profile, model_range_warnings
from .parsers import load_modeling_csv, natural_sort_key
from .paths import RUNS_DIR
from .runs.contracts import RunRecord
from .runs.artifacts import RunArtifactWriter
from .runs.repository import InvalidRunTransition, RunRepository
from .runs.status_projection import project_status
from .training_explainability import explainability_method
from .training_visualization import (
    FEATURE_VISUALIZATION_SCHEMA,
    build_model_feature_visualization,
)


# 训练被“新任务替换”时抛出的内部异常：旧直接调用入口用 status.json 的
# paused + replaced_by 表达替换；worker 正式路径则由 RunRepository 状态机表达。
class TrainingRunReplaced(RuntimeError):
    """Raised when a training run has been paused because a newer run replaced it."""


@dataclass
class TrainConfig:
    """训练请求的完整内部配置，包括兼容字段与运行时解析字段。

    ``resolved_*`` 字段由每个训练折覆盖，用于记录真正参与构造模型的 N/L，
    不能直接相信客户端传入值。
    """
    # ── 深度模型训练超参数：默认值统一来自 classification_policy.DEEP_TRAINING_DEFAULTS ──
    epochs: int = DEEP_TRAINING_DEFAULTS.epochs
    batch_size: int = DEEP_TRAINING_DEFAULTS.batch_size
    learning_rate: float = DEEP_TRAINING_DEFAULTS.learning_rate
    weight_decay: float = DEEP_TRAINING_DEFAULTS.weight_decay
    scheduler_factor: float = DEEP_TRAINING_DEFAULTS.scheduler_factor
    scheduler_patience: int = DEEP_TRAINING_DEFAULTS.scheduler_patience
    min_learning_rate: float = DEEP_TRAINING_DEFAULTS.min_learning_rate
    seed: int = DEEP_TRAINING_DEFAULTS.seed
    # split_seed is batch-wide and controls only data partitioning.  seed/model_seed
    # controls model initialisation and stochastic learners.  Keeping both makes
    # cross-model/repeat comparisons fair and reproducible.
    split_seed: int = DEEP_TRAINING_DEFAULTS.seed
    model_seed: int = DEEP_TRAINING_DEFAULTS.seed
    # ── 预处理与划分：normalization 只用当前折 train 拟合；split_* 为 10 份制比例 ──
    normalization: str = "zscore"
    split_mode: str = "stratified"
    split_train: int = 8
    split_valid: int = 1
    split_test: int = 1
    class_balance: str = "none"
    # ── 模型与结构参数：model_type 会经 canonical_model_type 归一化（如 transformer1d 别名） ──
    model_type: str = "cnn1d"
    experiment_version: str | None = None
    early_stopping_patience: int = DEEP_TRAINING_DEFAULTS.early_stopping_patience
    dropout: float | None = None
    hidden_size: int = 64
    transformer_heads: int = 4
    unet_depth: int = 3
    dscarnet_inception_blocks: int = 1
    dscarnet_pca_components: int = 30
    dscarnet_cluster_channels: int = 9
    dscarnet_input_mode: str = "dual"
    # ── 运行时解析字段：每折真实参与建模的 N/L，由训练循环回填，不信任客户端传值 ──
    resolved_train_sample_count: int = 100
    resolved_feature_count: int = 1000
    # ── 传统模型超参数（knn 为历史兼容字段，当前 15 模型能力目录不含 knn） ──
    knn_n_neighbors: int = 5
    knn_weights: str = "distance"
    knn_metric: str = "minkowski"
    knn_p: int = 2
    random_forest_n_estimators: int = 500
    random_forest_search_iterations: int = 10
    random_forest_max_depth: int | None = 3
    random_forest_min_samples_leaf: int = 2
    svm_c: float = 1.0
    svm_gamma: str | float = 0.03
    xgboost_n_estimators: int = 50
    xgboost_max_depth: int = 2
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.9
    xgboost_colsample_bytree: float = 0.9
    xgboost_reg_lambda: float = 2.0
    pls_components: int | None = None
    pca_components: int | None = None
    logistic_c: float = 1.0
    logistic_l1_ratio: float = 0.5
    spls_components: int | None = None
    spls_keepx: int | None = None
    svm_kernel: str = "rbf"
    random_forest_max_features: str | float = "sqrt"
    random_forest_oob_score: bool = False
    xgboost_min_child_weight: float = 1.0
    xgboost_gamma: float = 0.0
    # ── 可解释性（特征区间识别）配置：窗口数、top_k、重复次数与评估集合 ──
    feature_selection_enabled: bool = False
    feature_window_count: int = 100
    feature_top_k: int = 5
    feature_n_repeats: int = 5
    feature_eval_split: str = "valid"


@dataclass
class TraditionalSelection:
    """传统模型一次候选搜索的胜出配置、模型和审计记录。"""
    config: TrainConfig
    valid_balanced_accuracy: float
    valid_macro_f1: float
    search_rows: list[dict[str, Any]]
    model: Any
    valid_eval: dict[str, Any]


# status.json 兼容辅助：仅用于旧“新任务替换旧任务”入口，不是 Run 状态机来源。
def _read_status_file(status_file: Path) -> dict[str, Any]:
    """读取旧版 status.json 投影；文件缺失或 JSON 损坏时返回空 dict 而不是抛错。

        仅服务于“直接调用/旧入口”的替换检测与进度展示，不是 Run 状态机的权威来源；
        正式 worker 路径以 RunRepository 的 SQLite 记录为准。"""
    if not status_file.exists():
        return {}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_status_file(status_file: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 覆盖写 status.json（先确保父目录存在），供旧入口展示训练进度。"""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_is_replaced(run_id: str, *, runs_dir: Path | None = None) -> bool:
    """兼容判断旧 status 投影是否因新 Run 替换而 paused。"""
    status = _read_status_file((runs_dir or RUNS_DIR) / run_id / "status.json")
    return status.get("status") == "paused" and bool(status.get("replaced_by"))


def _raise_if_run_replaced(status_file: Path) -> None:
    """若 status.json 显示本 Run 已被新任务替换（paused + replaced_by），立即中断。

        旧入口没有 worker 的 claim/cancel 机制，只能在关键检查点读盘判断，
        因此训练循环会在每个折/阶段边界调用它；replaced_by 缺失时用占位名。"""
    status = _read_status_file(status_file)
    if status.get("status") != "paused":
        return
    replaced_by = status.get("replaced_by") or "newer_run"
    raise TrainingRunReplaced(f"训练任务 {status.get('run_id') or status_file.parent.name} 已被新任务 {replaced_by} 替换")


# 数据划分与归一化：所有 group split 都以 Sample_ID 为不可拆分单位。
def _validate_split_ratio_config(policy: EvaluationPolicy) -> None:
    """按评估口径校验 10 份制划分比例（policy 由 classification_policy 解析）。

        - external_test_holdout：内部 test 比例必须为 0，train+valid=10（test 来自独立数据集）；
        - leave_one_sample_id_cv：同样 train+valid=10，每折 test 固定为 1 个 Sample_ID；
        - stratified_holdout：train/valid/test 均 > 0 且相加等于 10。"""
    ratios = (int(policy.split_train), int(policy.split_valid), int(policy.split_test))
    if any(value < 0 for value in ratios):
        raise ValueError("划分比例必须是非负整数")
    if policy.strategy in {"external_test_holdout", "leave_one_sample_id_cv_with_external_test"}:
        if ratios[2] != 0 or ratios[0] <= 0 or ratios[1] <= 0 or ratios[0] + ratios[1] != 10:
            raise ValueError("独立测试集模式要求主数据训练/验证比例相加必须等于 10，且内部测试比例为 0")
        return
    if policy.strategy == "leave_one_sample_id_cv":
        if not policy.cv_allowed or ratios[2] != 0 or ratios[0] <= 0 or ratios[1] <= 0 or ratios[0] + ratios[1] != 10:
            raise ValueError("留一交叉验证要求训练/验证比例相加必须等于 10，且内部测试比例为 0")
        return
    if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] <= 0 or sum(ratios) != 10:
        raise ValueError("训练、验证、测试比例相加必须等于 10，且三项均大于 0")


def _normalize(x: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """按“每条曲线自身”做行内归一化（axis=1），返回归一化结果与模式记录。

        这是早期兼容路径：minmax/zscore 在单条曲线内部计算统计量。正式训练折使用
        _fit_x_normalizer/_transform_x_with_normalizer 的按特征（axis=0）、仅用 train
        拟合的版本，以避免 valid/test 信息泄漏。"""
    if mode == "none":
        return x.astype(np.float32), {"mode": mode}
    if mode == "minmax":
        mins = x.min(axis=1, keepdims=True)
        maxs = x.max(axis=1, keepdims=True)
        return ((x - mins) / np.maximum(maxs - mins, 1e-8)).astype(np.float32), {"mode": mode}
    if mode == "area":
        area = np.trapz(np.abs(x), axis=1, keepdims=True)
        return (x / np.maximum(area, 1e-8)).astype(np.float32), {"mode": mode}
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    return ((x - means) / np.maximum(stds, 1e-8)).astype(np.float32), {"mode": "zscore"}


def _split_indices(
    labels: np.ndarray,
    sample_id: np.ndarray,
    config: TrainConfig,
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
    rng = np.random.default_rng(config.split_seed)
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


def _validate_external_test_dataset(
    train_labels: list[str],
    train_curve_length: int,
    train_axis: list[float],
    test_dataset: Any,
) -> None:
    """独立测试集（external_test_holdout）的硬校验。

        Label 必须被训练集覆盖；真实 XXX 特征轴必须与主数据逐点一致（wide-feature
        宽表契约要求同轴，不做插值迁就），不一致时报出首个差异坐标便于排查。"""
    unknown_labels = sorted(set(test_dataset.labels).difference(train_labels))
    if unknown_labels:
        raise ValueError(f"测试集包含训练集中不存在的 Label: {', '.join(unknown_labels)}")
    test_lengths = {len(values) for values in test_dataset.x_axis}
    if test_lengths != {train_curve_length}:
        raise ValueError(f"测试集曲线长度必须与训练数据一致，训练长度 {train_curve_length}，测试集长度 {sorted(test_lengths)}")
    test_axis = test_dataset.x_axis[0] if test_dataset.x_axis else []
    train_values = np.asarray(train_axis, dtype=np.float64)
    test_values = np.asarray(test_axis, dtype=np.float64)
    if train_values.shape != test_values.shape or not np.array_equal(train_values, test_values):
        if train_values.shape == test_values.shape:
            difference = np.flatnonzero(train_values != test_values)
            position = int(difference[0]) if difference.size else 0
            detail = (
                f"第 {position + 1} 个坐标分别为 {train_values[position]:.17g} 与 "
                f"{test_values[position]:.17g}"
            )
        else:
            detail = f"坐标数分别为 {train_values.size} 与 {test_values.size}"
        raise ValueError(
            f"独立测试集的真实 XXX 特征轴必须与训练数据完全一致（{detail}）"
        )


def _axis_to_float_list(axis: Any, n_features: int) -> list[float]:
    """把任意来源的 X 轴转成 float 列表；缺失、不可转或长度不符时退化为
        0..n-1 序号轴，保证下游绘图/解释性产物永远有轴可用。"""
    try:
        values = np.asarray(axis, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=np.float64)
    if values.size != n_features:
        values = np.arange(n_features, dtype=np.float64)
    return [float(item) for item in values]


def _x_axis_warning(sample_axes: list[Any], n_features: int) -> dict[str, Any]:
    """检查批次内各样品 XXX 坐标是否与首条一致（容差 rtol=1e-6/atol=1e-8）。

        只产出 warning 不拒绝训练：聚合特征图使用首条样品坐标，单样品图使用各自
        坐标——与前端结果页的展示约定一致。"""
    if not sample_axes:
        return {"status": "unavailable", "message": "没有可检查的 x 轴坐标"}
    reference = np.asarray(_axis_to_float_list(sample_axes[0], n_features), dtype=np.float64)
    mismatch_count = 0
    max_abs_delta = 0.0
    for axis in sample_axes:
        values = np.asarray(_axis_to_float_list(axis, n_features), dtype=np.float64)
        if values.shape != reference.shape or not np.allclose(values, reference, rtol=1e-6, atol=1e-8):
            mismatch_count += 1
            if values.shape == reference.shape:
                max_abs_delta = max(max_abs_delta, float(np.max(np.abs(values - reference))))
    if mismatch_count:
        return {
            "status": "inconsistent",
            "message": (
                f"检测到 {mismatch_count}/{len(sample_axes)} 条样品的 XXX 坐标与首条样品不一致；"
                "聚合特征图使用首条样品坐标，单样品图使用各自坐标。"
            ),
            "sample_count": int(len(sample_axes)),
            "mismatch_count": int(mismatch_count),
            "max_abs_delta": max_abs_delta,
        }
    return {
        "status": "consistent",
        "message": "所有样品的 XXX 坐标一致",
        "sample_count": int(len(sample_axes)),
        "mismatch_count": 0,
        "max_abs_delta": 0.0,
    }


def _validate_splits(splits: dict[str, list[int]], y: np.ndarray, label_names: list[str]) -> None:
    """划分后的兜底校验：任一集合为空、或 train 缺少任何类别都直接拒绝训练。"""
    for split_name, indices in splits.items():
        if not indices:
            raise ValueError(f"{split_name} 集为空，请增加样品种类或调整划分比例")
    train_labels = set(np.unique(y[splits["train"]]).tolist())
    missing = [label for idx, label in enumerate(label_names) if idx not in train_labels]
    if missing:
        raise ValueError(f"训练集中缺少类别: {', '.join(missing)}。请增加样品种类或调整划分比例")


def _loader(x: np.ndarray, y: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    """构造 1D 深度模型的 DataLoader：输入张量补 channel 维成 (B, 1, L)。"""
    tx = torch.tensor(x[indices], dtype=torch.float32).unsqueeze(1)
    ty = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def _dual_loader(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """构造 DSCARNet 双通路（SAR + CAR 两张 2D 图）的 DataLoader。"""
    tx1 = torch.tensor(x1[indices], dtype=torch.float32)
    tx2 = torch.tensor(x2[indices], dtype=torch.float32)
    ty = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tx1, tx2, ty), batch_size=batch_size, shuffle=shuffle)


def _single_2d_loader(x: np.ndarray, y: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    """构造 DSCARNet 单通路（仅 SAR 或仅 CAR 2D 图）的 DataLoader，不再补 channel 维。"""
    tx = torch.tensor(x[indices], dtype=torch.float32)
    ty = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


# 模型评估适配：统一把单 logit 二分类和多 logit 多分类转换为概率矩阵。
def _evaluate_deep_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    dual_input: bool,
) -> float:
    """eval 模式下按 batch 计算平均 loss；二分类单 logit 时把标签 reshape 成
        与 logits 同形，以适配 BCEWithLogitsLoss 的目标形状约定。"""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            if dual_input:
                batch_x1, batch_x2, batch_y = batch
                logits = model(batch_x1, batch_x2)
            else:
                batch_x, batch_y = batch
                logits = model(batch_x)
            target = batch_y.float().view_as(logits) if logits.ndim == 2 and logits.shape[1] == 1 else batch_y
            losses.append(float(criterion(logits, target).detach().item()))
    return float(np.mean(losses)) if losses else 0.0


def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
    """对 1D 深度模型做一次前向评估，输出全套分类指标与逐样品概率。

        二分类网络只输出 1 个 logit，这里用 sigmoid 后拼成 [1-p, p] 两列概率；
        多分类直接 softmax。confusion_matrix 显式给定 labels，保证某集合缺少
        类别时矩阵维度仍与 label_names 对齐；zero_division=0 避免 NaN/告警。"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[indices], dtype=torch.float32).unsqueeze(1))
        if logits.ndim == 2 and logits.shape[1] == 1:
            positive = torch.sigmoid(logits)
            probs = torch.cat((1.0 - positive, positive), dim=1).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": probs.tolist(),
        "pred": pred.tolist(),
        "true": true.tolist(),
    }


def _deep_probabilities(model: nn.Module, values: np.ndarray) -> np.ndarray:
    """Return class probabilities for a normalized 1D deep-model batch."""

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
        if logits.ndim == 2 and logits.shape[1] == 1:
            positive = torch.sigmoid(logits)
            probs = torch.cat((1.0 - positive, positive), dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def _evaluate_dual(
    model: nn.Module,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    indices: list[int],
    labels: list[str],
) -> dict[str, Any]:
    """双通路 DSCARNet 的评估版本，指标字段与 _evaluate 完全一致。"""
    model.eval()
    with torch.no_grad():
        tx1 = torch.tensor(x1[indices], dtype=torch.float32)
        tx2 = torch.tensor(x2[indices], dtype=torch.float32)
        logits = model(tx1, tx2)
        if logits.ndim == 2 and logits.shape[1] == 1:
            positive = torch.sigmoid(logits)
            probs = torch.cat((1.0 - positive, positive), dim=1).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": probs.tolist(),
        "pred": pred.tolist(),
        "true": true.tolist(),
    }


def _traditional_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    """统一传统模型的概率出口：优先 predict_proba；无概率接口（如线性 SVM）
        时对 decision_function 做数值稳定版 softmax 近似，保证解释性计算总有概率可用。"""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values), dtype=float)
    decision = model.decision_function(values)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    shifted = decision - decision.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _evaluate_traditional_model(model: Any, x: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
    """传统模型评估：predict 出标签 + _traditional_probabilities 出概率，
        指标字段与深度模型完全对齐，便于上层不区分模型族地汇总。"""
    pred = model.predict(x[indices])
    probs = _traditional_probabilities(model, x[indices])
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": np.asarray(probs, dtype=float).tolist(),
        "pred": np.asarray(pred, dtype=int).tolist(),
        "true": true.tolist(),
    }


def _compute_sample_feature_importance(
    *,
    run_dir: Path,
    config: TrainConfig,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    x_axis: list[float],
    x_axis_warning: dict[str, Any],
    label_names: list[str],
    metadata: list[dict[str, Any]],
    score_fn: Any,
) -> dict[str, Any]:
    """传统模型单样品可解释性入口（窗口遮挡后的真实类别 Log-loss 增量）。

        关闭时落盘 status=disabled；计算异常时降级为 status=failed 并保留训练
        均值基线曲线，绝不让解释性失败拖垮整个训练 Run。"""
    mean_indices = splits.get("train", [])
    if not config.feature_selection_enabled:
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        result = {
            "status": "disabled",
            "reason": "特征区间识别已关闭",
            "method": "sample_occlusion_log_loss",
            "baseline": "train_mean_curve",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "x_axis_warning": x_axis_warning,
            "samples": [],
        }
        return write_sample_feature_importance_artifacts(run_dir, result)
    try:
        result = sample_occlusion_importance(
            x,
            y,
            x_axis=x_axis,
            splits=splits,
            label_names=label_names,
            score_fn=score_fn,
            mean_indices=mean_indices,
            metadata=metadata,
            window_count=config.feature_window_count,
            top_k=config.feature_top_k,
        )
        result["x_axis_warning"] = x_axis_warning
    except Exception as exc:
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        result = {
            "status": "failed",
            "reason": f"单样品重要区间计算失败: {exc}",
            "method": "sample_occlusion_log_loss",
            "baseline": "train_mean_curve",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "x_axis_warning": x_axis_warning,
            "samples": [],
        }
    return write_sample_feature_importance_artifacts(run_dir, result)


def _unsupported_explainability_summary(reason: str, *, method: str = "unsupported") -> dict[str, Any]:
    """生成统一的“不可解释/未启用”占位摘要，字段形状与正常结果一致，前端无需特判。"""
    return {
        "status": "unsupported",
        "reason": reason,
        "method": method,
        "artifact": None,
        "csv_artifact": None,
        "top_segments": [],
        "sample_count": 0,
        "window_count": None,
        "top_k": None,
    }


def _deep_sample_feature_result(
    *,
    config: TrainConfig,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    x_axis: list[float],
    x_axis_warning: dict[str, Any],
    label_names: list[str],
    metadata: list[dict[str, Any]],
    dscarnet_mapped: DSCARNetMappedInputs | None = None,
    dscarnet_mapping_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """深度模型单样品可解释性分发器。

        按 explainability_method 分三路：窗口遮挡 Log-loss（pca_mlp /
        cnn_transformer1d 等无 1D 卷积结构）、DSCARNet 双/单通路 2D Grad-CAM
        回投 1D 特征、其余 1D 卷积网络的 Grad-CAM-like 归因；任何一路失败都
        降级为 status=failed 而不向外抛出。"""
    model_type = canonical_model_type(config.model_type)
    if not config.feature_selection_enabled:
        mean_indices = splits.get("train", [])
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        return {
            "status": "disabled",
            "reason": "特征区间识别已关闭",
            "method": explainability_method(model_type),
            "baseline": "train_mean_curve",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "x_axis_warning": x_axis_warning,
            "samples": [],
        }

    try:
        method = explainability_method(model_type)
        if method == "window_occlusion_log_loss":
            sample_result = sample_occlusion_importance(
                x,
                y,
                x_axis=x_axis,
                splits=splits,
                label_names=label_names,
                score_fn=lambda values: _deep_probabilities(model, values),
                mean_indices=splits.get("train", []),
                metadata=metadata,
                window_count=config.feature_window_count,
                top_k=config.feature_top_k,
            )
        elif model_type == "dscarnet":
            if dscarnet_mapped is None or dscarnet_mapping_metadata is None:
                raise ValueError("DSCARNet 缺少 SAR/CAR 二维映射结果，无法计算双通路解释性")
            mode = dscarnet_mapping_metadata.get("mode", "dual")
            if mode == "dual":
                sample_result = sample_dscarnet_dual_2d_gradcam_importance(
                    model, x, y, x_sar=dscarnet_mapped.x_sar, x_car=dscarnet_mapped.x_car,
                    pca=dscarnet_mapped.pca, sar_mapper=dscarnet_mapped.sar_mapper,
                    car_mapper=dscarnet_mapped.car_mapper, mapping_metadata=dscarnet_mapping_metadata,
                    x_axis=x_axis, splits=splits, label_names=label_names, metadata=metadata, top_k=config.feature_top_k,
                )
            else:
                mapped_values = dscarnet_mapped.x_sar if mode == "sar" else dscarnet_mapped.x_car
                mapper = dscarnet_mapped.sar_mapper if mode == "sar" else dscarnet_mapped.car_mapper
                sample_result = sample_dscarnet_single_2d_gradcam_importance(
                    model, x, y, mapped_values=mapped_values, mapper=mapper, mode=mode, pca=dscarnet_mapped.pca,
                    mapping_metadata=dscarnet_mapping_metadata, x_axis=x_axis, splits=splits,
                    label_names=label_names, metadata=metadata, top_k=config.feature_top_k,
                )
        else:
            sample_result = sample_deep_attribution_importance(
                model,
                x,
                y,
                x_axis=x_axis,
                splits=splits,
                label_names=label_names,
                metadata=metadata,
                model_type=model_type,
                top_k=config.feature_top_k,
            )
        sample_result["x_axis_warning"] = x_axis_warning
        return sample_result
    except Exception as exc:
        mean_indices = splits.get("train", [])
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        return {
            "status": "failed",
            "reason": f"单样品可解释性计算失败: {exc}",
            "method": explainability_method(model_type),
            "baseline": "train_mean_curve",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "x_axis_warning": x_axis_warning,
            "samples": [],
        }


def _tag_fold_sample_result(result: dict[str, Any], fold_index: int) -> dict[str, Any]:
    """给每个样品的解释结果打上 fold_index 并刷新 sample_count，供 CV 多折合并。"""
    for sample in result.get("samples", []):
        sample["fold_index"] = int(fold_index)
    result["sample_count"] = len(result.get("samples", []))
    return result


def _merge_deep_sample_results(results: list[dict[str, Any]], x_axis_warning: dict[str, Any]) -> dict[str, Any]:
    """合并多折单样品解释结果：只保留 status=ready 的折，样品级拼接；
        Grad-CAM 类方法顺带聚合 sanity check（随机权重对照）结果。"""
    ready_results = [result for result in results if result.get("status") == "ready"]
    if not ready_results:
        result = dict(results[-1]) if results else {"status": "unavailable", "reason": "没有可解释的测试集样品", "samples": []}
        result["x_axis_warning"] = x_axis_warning
        result["sample_count"] = len(result.get("samples", []))
        return result
    combined = dict(ready_results[0])
    samples = [sample for result in ready_results for sample in result.get("samples", [])]
    combined["samples"] = samples
    combined["sample_count"] = len(samples)
    combined["x_axis_warning"] = x_axis_warning
    if combined.get("method") == "gradcam_1d":
        combined["sanity_checks"] = aggregate_attribution_sanity(samples)
    elif combined.get("method") == "dscarnet_dual_2d_gradcam":
        combined["sanity_checks"] = aggregate_dscarnet_branch_sanity(samples)
    return combined


def _write_sample_explainability_artifacts(
    *,
    run_dir: Path,
    sample_result: dict[str, Any],
    x_axis_warning: dict[str, Any],
) -> dict[str, Any]:
    """补全 sample_count / x_axis_warning 后，落盘 sample_feature_importance
        的 json/csv 产物并返回摘要。"""
    sample_result["sample_count"] = len(sample_result.get("samples", []))
    sample_result["x_axis_warning"] = x_axis_warning
    return write_sample_feature_importance_artifacts(run_dir, sample_result)


def _compute_deep_explainability(
    *,
    run_dir: Path,
    config: TrainConfig,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    x_axis: list[float],
    x_axis_warning: dict[str, Any],
    label_names: list[str],
    metadata: list[dict[str, Any]],
    dscarnet_mapped: DSCARNetMappedInputs | None = None,
    dscarnet_mapping_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """单次（fold 1）深度解释性的组合入口：计算 → 打 fold 标签 → 落盘。"""
    sample_result = _tag_fold_sample_result(
        _deep_sample_feature_result(
            config=config,
            model=model,
            x=x,
            y=y,
            splits=splits,
            x_axis=x_axis,
            x_axis_warning=x_axis_warning,
            label_names=label_names,
            metadata=metadata,
            dscarnet_mapped=dscarnet_mapped,
            dscarnet_mapping_metadata=dscarnet_mapping_metadata,
        ),
        int(1),
    )
    return _write_sample_explainability_artifacts(
        run_dir=run_dir,
        sample_result=sample_result,
        x_axis_warning=x_axis_warning,
    )


# 训练折公共准备：样品元数据、scaler、Sample_ID 分组和 fold 构造。
def _sample_metadata(frame: pd.DataFrame, sample_axes: list[Any], n_features: int) -> list[dict[str, Any]]:
    """从宽表逐行提取样品元数据（Index/Name/Sample_ID/各自 X 轴）。

        Name 保留原始文件名（wide-feature-v2 契约）；Index 能转 int 则用 int，
        否则保留字符串，兼容没有规范 Index 的 v1 数据。"""
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        raw_index = row.get("Index", "")
        index_value = int(raw_index) if str(raw_index).isdigit() else str(raw_index)
        axis = sample_axes[position] if position < len(sample_axes) else []
        rows.append(
            {
                "index": index_value,
                "name": str(row.get("Name", "") or raw_index),
                "sample_id": str(row.get("Sample_ID", "")),
                "sample_x_axis": _axis_to_float_list(axis, n_features),
            }
        )
    return rows


def _now_iso() -> str:
    """本地时间 ISO 字符串（秒级），用于 status_payload 的 started_at/completed_at。"""
    return datetime.now().isoformat(timespec="seconds")


def _clone_config(config: TrainConfig, **overrides: Any) -> TrainConfig:
    """复制 TrainConfig 并覆盖指定字段；dataclass 没有内建 copy 语义，这里用
        __dict__ 展开重建，候选超参搜索大量复用。"""
    return TrainConfig(**{**config.__dict__, **overrides})


def _fit_x_normalizer(x_train: np.ndarray, mode: str) -> dict[str, Any]:
    """只用当前折 train 拟合按特征（axis=0）的标准化参数。

        zscore/minmax 记录逐特征统计量；area 是逐曲线自身面积归一化，无训练统计量，
        因此只记模式。scale 下限 1e-8 防零方差除零。"""
    mode = str(mode or "zscore").lower()
    if mode == "none":
        return {"mode": "none"}
    if mode == "minmax":
        mins = x_train.min(axis=0)
        maxs = x_train.max(axis=0)
        return {"mode": "minmax", "min": mins, "scale": np.maximum(maxs - mins, 1e-8)}
    if mode == "area":
        return {"mode": "area"}
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    return {"mode": "zscore", "mean": means, "scale": np.maximum(stds, 1e-8)}


def _transform_x_with_normalizer(x: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    """把已拟合的 normalizer 应用到任意矩阵（train/valid/test 共用同一组参数）。"""
    mode = normalizer.get("mode", "zscore")
    values = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return values.astype(np.float32)
    if mode == "minmax":
        return ((values - normalizer["min"]) / normalizer["scale"]).astype(np.float32)
    if mode == "area":
        area = np.trapz(np.abs(values), axis=1, keepdims=True)
        return (values / np.maximum(area, 1e-8)).astype(np.float32)
    return ((values - normalizer["mean"]) / normalizer["scale"]).astype(np.float32)


def _json_normalizer(normalizer: dict[str, Any]) -> dict[str, Any]:
    """把 normalizer 中的 ndarray 转成 JSON 可序列化的 float 列表（写入 split.json）。"""
    return {key: (np.asarray(value).astype(float).tolist() if isinstance(value, np.ndarray) else value) for key, value in normalizer.items()}


def _dimension_band(n_features: int) -> str:
    """按特征数划分维度档位（1000-3000 / 3000-6000 / 6000-10000），写入
        config.json 供结果页与审计使用。"""
    if n_features <= 3000:
        return "1000-3000"
    if n_features <= 6000:
        return "3000-6000"
    return "6000-10000"


def _group_label_map(y: np.ndarray, sample_id: np.ndarray) -> dict[str, int]:
    """构造 Sample_ID → 类别编码映射；同组出现多 Label 直接拒绝（整组划分的前提）。"""
    mapping: dict[str, int] = {}
    for group in sorted(np.unique(sample_id).tolist(), key=natural_sort_key):
        labels = np.unique(y[sample_id == group])
        if len(labels) != 1:
            raise ValueError(f"Sample_ID={group} 内存在多个 Label，无法按组划分")
        mapping[str(group)] = int(labels[0])
    return mapping


def _choose_valid_groups(
    *,
    train_valid_groups: list[str],
    group_to_label: dict[str, int],
    label_count: int,
    valid_ratio: float,
    seed: int,
) -> list[str]:
    """为单个 CV 折挑选 valid 组：按固定种子的随机顺序逐个尝试，只有“挑走它
        之后剩余 train 仍类别完整”才接受，保证每折 train 覆盖全部类别。"""
    valid_count = max(1, int(round(len(train_valid_groups) * valid_ratio)))
    rng = np.random.default_rng(seed)
    candidates = list(train_valid_groups)
    rng.shuffle(candidates)
    selected: list[str] = []
    for group in candidates:
        remaining = [item for item in train_valid_groups if item not in {*selected, group}]
        if len(set(group_to_label[item] for item in remaining)) < label_count:
            continue
        selected.append(group)
        if len(selected) >= valid_count:
            break
    if not selected:
        raise ValueError("Sample_ID 分组数量太少，无法在每个交叉验证折中保留包含全部类别的训练集")
    return sorted(selected, key=lambda item: train_valid_groups.index(item))


def _leave_one_sample_id_folds(y: np.ndarray, sample_id: np.ndarray, config: TrainConfig) -> list[dict[str, Any]]:
    """构造 leave_one_sample_id_cv 的全部折：每折留 1 个 Sample_ID 做 test，
        其余按 split_train:split_valid 比例（默认 8:2）划 train/valid。

        至少需要 3 个组；某组留作 test 后剩余组必须仍覆盖全部类别，否则该折
        无法评估，直接拒绝整个训练。"""
    groups = [str(item) for item in sorted(np.unique(sample_id).tolist(), key=natural_sort_key)]
    if len(groups) < 3:
        raise ValueError("交叉验证至少需要 3 个 Sample_ID 分组")
    group_to_label = _group_label_map(y, sample_id)
    label_count = int(np.unique(y).size)
    train_valid_total = max(1, int(config.split_train) + int(config.split_valid))
    valid_ratio = max(0.0, min(1.0, float(config.split_valid) / train_valid_total))
    folds: list[dict[str, Any]] = []
    for fold_index, test_group in enumerate(groups, start=1):
        train_valid_groups = [group for group in groups if group != test_group]
        if len(set(group_to_label[group] for group in train_valid_groups)) < label_count:
            raise ValueError(f"Sample_ID={test_group} 留作测试后训练集缺少类别，无法完成分类评估")
        valid_groups = _choose_valid_groups(
            train_valid_groups=train_valid_groups,
            group_to_label=group_to_label,
            label_count=label_count,
            valid_ratio=valid_ratio,
            seed=int(config.split_seed) + fold_index,
        )
        train_groups = [group for group in train_valid_groups if group not in set(valid_groups)]
        folds.append(
            {
                "fold_index": fold_index,
                "test_sample_id": test_group,
                "train_sample_ids": train_groups,
                "valid_sample_ids": valid_groups,
                "splits": {
                    "train": np.where(np.isin(sample_id, train_groups))[0].tolist(),
                    "valid": np.where(np.isin(sample_id, valid_groups))[0].tolist(),
                    "test": np.where(sample_id == test_group)[0].tolist(),
                },
            }
        )
    return folds


# 指标汇总与模型候选生成。test 指标只在候选已经锁定后计算。
def _classification_metrics_payload(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    """由 y_true/y_pred 生成统一指标包：标量指标 + 混淆矩阵 + 逐类
        classification_report；labels 显式对齐 label_names，zero_division=0
        避免类别缺失时告警或产生 NaN。"""
    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": report,
    }


METRIC_SCALAR_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")


def _metrics_from_eval(eval_payload: dict[str, Any], label_names: list[str]) -> dict[str, Any]:
    """把 _evaluate* 的 eval payload（含 true/pred 列表）换算成统一指标包。"""
    return _classification_metrics_payload(
        np.asarray(eval_payload["true"], dtype=np.int64),
        np.asarray(eval_payload["pred"], dtype=np.int64),
        label_names,
    )


def _scalar_metric_summary(metric_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """跨折对标量指标取 mean；全部缺失时给 None 而不是 0，避免伪造指标。"""
    summary: dict[str, float | None] = {}
    for key in METRIC_SCALAR_KEYS:
        values = [float(row[key]) for row in metric_rows if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def _scalar_metric_std(metric_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """跨折对标量指标取 std（np.std 总体标准差），与 mean 一样缺失时给 None。"""
    summary: dict[str, float | None] = {}
    for key in METRIC_SCALAR_KEYS:
        values = [float(row[key]) for row in metric_rows if row.get(key) is not None]
        summary[key] = float(np.std(values)) if values else None
    return summary


def _aggregate_split_metrics(
    split_evals: list[dict[str, Any]],
    label_names: list[str],
) -> tuple[dict[str, Any], dict[str, float | None], dict[str, float | None]]:
    """返回 (pooled, fold_mean, fold_std)：pooled 把所有折的 true/pred 拼接后
        一次性计算——这是 CV 主指标口径；mean/std 只是逐折指标的平均与离散度。"""
    metric_rows = [_metrics_from_eval(item, label_names) for item in split_evals]
    all_true = [int(value) for item in split_evals for value in item["true"]]
    all_pred = [int(value) for item in split_evals for value in item["pred"]]
    pooled = _classification_metrics_payload(
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        label_names,
    )
    return pooled, _scalar_metric_summary(metric_rows), _scalar_metric_std(metric_rows)


def _build_metrics_payload(
    fold_split_evals: list[dict[str, dict[str, Any]]],
    label_names: list[str],
    evaluation_strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造 UI 主指标与 CV 审计指标，Test 主值固定使用 pooled OOF。"""
    aggregates: dict[str, dict[str, Any]] = {}
    fold_mean: dict[str, dict[str, float | None]] = {}
    fold_std: dict[str, dict[str, float | None]] = {}
    for split_name in ("train", "valid", "test"):
        pooled, mean, std = _aggregate_split_metrics(
            [fold_eval[split_name] for fold_eval in fold_split_evals],
            label_names,
        )
        aggregates[split_name] = pooled
        fold_mean[split_name] = mean
        fold_std[split_name] = std

    is_cv = evaluation_strategy == "leave_one_sample_id_cv"
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid"):
        primary = dict(aggregates[split_name])
        primary.update(fold_mean[split_name])
        primary["aggregation"] = "fold_mean" if is_cv else "direct_holdout"
        for key in METRIC_SCALAR_KEYS:
            primary[f"pooled_{key}"] = aggregates[split_name][key]
        split_metrics[split_name] = primary

        # 契约关键点：test 主指标固定是 pooled OOF（CV）或 holdout 直算，
        # 绝不用 fold mean 充数；train/valid 则同时给出 fold_mean 与 pooled_* 供对照。
    test_metrics = dict(aggregates["test"])
    test_metrics["aggregation"] = "pooled_out_of_fold" if is_cv else "direct_holdout"
    split_metrics["test"] = test_metrics

    metrics = {**test_metrics, **split_metrics}
    cv_summary = {
        "strategy": evaluation_strategy,
        "fold_count": len(fold_split_evals),
        "primary_test_aggregation": "pooled_out_of_fold" if is_cv else "direct_holdout",
        "pooled_test": aggregates["test"],
        "fold_mean": fold_mean,
        "fold_std": fold_std,
    }
    return metrics, cv_summary


def _traditional_candidate_configs(
    config: TrainConfig,
    model_type: str,
    n_features: int,
    min_inner_train_size: int | list[int] | np.ndarray,
) -> list[TrainConfig]:
    """为各传统模型生成超参候选列表（候选上限只依赖 train 数据统计量）。

        pls_da/pca_lda 是成分数网格；logistic/SVM 是 C 网格；random_forest 与
        xgboost 使用小型笛卡尔积网格。所有候选均由分组内层 CV 的 Balanced
        Accuracy 选择，不能退回 OOB 或外层验证集选参。"""
    # Kept tolerant of the historical fourth-argument shape (the caller used
    # to pass the training labels themselves).  The grouped-search path passes
    # an integer minimum fold size; compatibility callers are reduced to their
    # sample count without inspecting labels or test data.
    if isinstance(min_inner_train_size, (list, tuple, np.ndarray)):
        inner_train_size = len(min_inner_train_size)
    else:
        inner_train_size = int(min_inner_train_size)
    if inner_train_size < 1:
        return []

    if model_type == "pls_da":
        raw = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        cap = max(1, min(inner_train_size, n_features))
        return [_clone_config(config, pls_components=value) for value in raw if value <= cap]
    if model_type == "spls_da":
        component_cap = max(1, min(inner_train_size - 1, n_features))
        # Documented grid is 25/50/100/300 for normal spectra.  Small unit
        # fixtures still need a real sparse estimator, so cap (not drop) the
        # requested keepX values and keep their stable order.
        keepx_values = list(dict.fromkeys(min(value, n_features) for value in (25, 50, 100, 300)))
        return [
            _clone_config(config, spls_components=components, spls_keepx=keepx)
            for components in (1, 2, 3, 5, 8, 10)
            if components <= component_cap
            for keepx in keepx_values
        ]
    if model_type == "pca_lda":
        raw = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
        cap = max(1, min(inner_train_size, n_features))
        return [_clone_config(config, pca_components=value) for value in raw if value <= cap]
    if model_type == "logistic_regression":
        return [
            _clone_config(config, logistic_c=c, logistic_l1_ratio=l1_ratio)
            for c in (0.01, 0.1, 1.0, 10.0, 100.0)
            for l1_ratio in (0.1, 0.5, 0.9)
        ]
    if model_type == "svm":
        return [_clone_config(config, svm_kernel="linear", svm_gamma="scale", svm_c=value) for value in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if model_type == "pca_svm":
        component_cap = max(1, min(inner_train_size, n_features))
        return [
            _clone_config(config, pca_components=components, svm_c=c, svm_kernel="linear", svm_gamma="scale")
            for components in (2, 3, 5, 8, 10, 15, 20, 30, 40, 50)
            if components <= component_cap
            for c in (0.01, 0.1, 1.0, 10.0, 100.0)
        ]
    if model_type == "random_forest":
        return [
            _clone_config(
                config,
                random_forest_n_estimators=500,
                random_forest_max_depth=max_depth,
                random_forest_min_samples_leaf=min_samples_leaf,
                random_forest_max_features=max_features,
                random_forest_oob_score=False,
            )
            for max_depth in (None, 5, 10)
            for min_samples_leaf in (1, 5)
            for max_features in ("sqrt", 0.1)
        ]
    if model_type == "xgboost":
        candidates = [
            (n_estimators, depth, min_child)
            for n_estimators in (100, 300)
            for depth in (2, 3, 5)
            for min_child in (3, 5)
        ]
        return [
            _clone_config(
                config,
                xgboost_n_estimators=n_estimators,
                xgboost_max_depth=depth,
                xgboost_learning_rate=0.1,
                xgboost_subsample=0.8,
                xgboost_colsample_bytree=0.3,
                xgboost_reg_lambda=10.0,
                xgboost_min_child_weight=min_child,
            )
            for n_estimators, depth, min_child in candidates
        ]
    return [config]


def _traditional_params(config: TrainConfig, model_type: str) -> dict[str, Any]:
    """Return only the parameters that actually configure the selected model."""

    if model_type == "pls_da":
        return {"pls_components": config.pls_components}
    if model_type == "spls_da":
        return {"spls_components": config.spls_components, "spls_keepx": config.spls_keepx}
    if model_type == "pca_lda":
        return {"pca_components": config.pca_components}
    if model_type == "logistic_regression":
        return {"logistic_c": config.logistic_c, "logistic_l1_ratio": config.logistic_l1_ratio}
    if model_type == "svm":
        return {
            "svm_kernel": config.svm_kernel,
            "svm_c": config.svm_c,
            "svm_gamma": config.svm_gamma,
        }
    if model_type == "pca_svm":
        return {"pca_components": config.pca_components, "svm_c": config.svm_c, "svm_kernel": "linear"}
    if model_type == "random_forest":
        return {
            "random_forest_n_estimators": config.random_forest_n_estimators,
            "random_forest_max_depth": config.random_forest_max_depth,
            "random_forest_min_samples_leaf": config.random_forest_min_samples_leaf,
            "random_forest_max_features": config.random_forest_max_features,
        }
    if model_type == "xgboost":
        return {
            "xgboost_n_estimators": config.xgboost_n_estimators,
            "xgboost_max_depth": config.xgboost_max_depth,
            "xgboost_learning_rate": config.xgboost_learning_rate,
            "xgboost_subsample": config.xgboost_subsample,
            "xgboost_colsample_bytree": config.xgboost_colsample_bytree,
            "xgboost_min_child_weight": config.xgboost_min_child_weight,
            "xgboost_reg_lambda": config.xgboost_reg_lambda,
            "xgboost_gamma": config.xgboost_gamma,
        }
    return {}


def _evaluate_single_2d(model: nn.Module, values: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
    """DSCARNet 单通路（2D 输入）的评估版本，指标口径与 _evaluate 一致。"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(values[indices], dtype=torch.float32))
        if logits.ndim == 2 and logits.shape[1] == 1:
            positive = torch.sigmoid(logits)
            probs = torch.cat((1.0 - positive, positive), dim=1).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": probs.tolist(), "pred": pred.tolist(), "true": true.tolist(),
    }


def _grouped_inner_cv_indices(
    *,
    y: np.ndarray,
    sample_id: np.ndarray,
    candidate_indices: list[int],
    split_seed: int,
    label_names: list[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create the required five stratified, Sample_ID-grouped inner folds."""
    pool = np.asarray(candidate_indices, dtype=np.int64)
    group_labels = _group_label_map(y[pool], sample_id[pool])
    counts = {label: sum(item == label for item in group_labels.values()) for label in range(len(label_names))}
    insufficient = {label_names[label]: count for label, count in counts.items() if count < 5}
    if insufficient:
        detail = "、".join(f"{name}（{count} 个 Sample_ID）" for name, count in insufficient.items())
        raise ValueError(f"传统模型 5 折分组搜索要求每类至少 5 个 Sample_ID；当前不足：{detail}")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(split_seed))
    return [
        (pool[train_local], pool[valid_local])
        for train_local, valid_local in splitter.split(np.zeros(len(pool)), y[pool], groups=sample_id[pool])
    ]


def _select_traditional_config(
    config: TrainConfig,
    model_type: str,
    x_raw: np.ndarray,
    y: np.ndarray,
    sample_id: np.ndarray,
    splits: dict[str, list[int]],
    label_names: list[str],
) -> TraditionalSelection:
    """Select on grouped 5-fold mean balanced accuracy with fold-local preprocessing."""
    pool = sorted({*splits["train"], *splits["valid"]})
    inner_folds = _grouped_inner_cv_indices(
        y=y, sample_id=sample_id, candidate_indices=pool,
        split_seed=config.split_seed, label_names=label_names,
    )
    candidates = _traditional_candidate_configs(
        config, model_type, x_raw.shape[1], min(len(train) for train, _ in inner_folds)
    )
    if not candidates:
        raise ValueError("当前数据维度下没有可用的传统模型候选参数")
    search_rows: list[dict[str, Any]] = []
    best_index = 0
    best_score = -np.inf
    for index, candidate in enumerate(candidates):
        inner_scores: list[float] = []
        for inner_train, inner_valid in inner_folds:
            normalizer = _fit_x_normalizer(x_raw[inner_train], config.normalization)
            train_values = _transform_x_with_normalizer(x_raw[inner_train], normalizer)
            valid_values = _transform_x_with_normalizer(x_raw[inner_valid], normalizer)
            model = build_traditional_model(candidate, y[inner_train], len(label_names))
            model.fit(train_values, y[inner_train])
            inner_scores.append(float(_evaluate_traditional_model(
                model, valid_values, y[inner_valid], list(range(len(inner_valid))), label_names
            )["balanced_accuracy"]))
        mean_score = float(np.mean(inner_scores))
        params = _traditional_params(candidate, model_type)
        search_rows.append({
            "model_type": model_type,
            "is_selected": False,
            "selection_metric": "mean_balanced_accuracy_grouped_5fold",
            "selection_score": mean_score,
            "inner_fold_scores": inner_scores,
            "inner_fold_score_std": float(np.std(inner_scores)),
            "valid_balanced_accuracy": mean_score,
            "valid_macro_f1": None,
            "valid_accuracy": None,
            "params": params,
            "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
        })
        # Candidate order is stable and intentionally acts as the documented
        # lower-complexity/fixed-order tie break.
        if mean_score > best_score + 1e-12:
            best_score = mean_score
            best_index = index
    selected = candidates[best_index]
    search_rows[best_index]["is_selected"] = True
    outer_normalizer = _fit_x_normalizer(x_raw[splits["train"]], config.normalization)
    outer_x = _transform_x_with_normalizer(x_raw, outer_normalizer)
    outer_model = build_traditional_model(selected, y[splits["train"]], len(label_names))
    outer_model.fit(outer_x[splits["train"]], y[splits["train"]])
    valid_eval = _evaluate_traditional_model(outer_model, outer_x, y, splits["valid"], label_names)
    return TraditionalSelection(
        config=selected,
        valid_balanced_accuracy=float(best_score),
        valid_macro_f1=float(valid_eval["macro_f1"]),
        search_rows=search_rows,
        model=outer_model,
        valid_eval=valid_eval,
    )


# 单折拟合：传统模型先选参再重训；深度模型按 validation loss 早停。
def _fit_traditional_fold(
    config: TrainConfig,
    model_type: str,
    x_raw: np.ndarray,
    y: np.ndarray,
    sample_id: np.ndarray,
    splits: dict[str, list[int]],
    label_names: list[str],
) -> tuple[Any, TrainConfig, dict[str, Any], list[dict[str, Any]]]:
    """单个训练折的传统模型拟合：只在 train/valid 上完成候选搜索，
        返回胜出模型、胜出配置、valid 评估与全部搜索审计行。"""
    selection = _select_traditional_config(
        config,
        model_type,
        x_raw,
        y,
        sample_id,
        splits,
        label_names,
    )
    return selection.model, selection.config, selection.valid_eval, selection.search_rows


def _fit_final_traditional_model(
    *,
    selected_config: TrainConfig | TraditionalSelection,
    model_type: str,
    x_raw: np.ndarray,
    y: np.ndarray,
    train_valid_indices: list[int],
    normalization: str,
    label_names: list[str],
) -> tuple[Any, dict[str, Any], np.ndarray]:
    """选参结束后用 train+valid 重训最终模型（标准化也在 train+valid 上重拟合）。

        这是传统模型的既定口径：候选比较用 train-only 保证公平，交付模型用更大的
        train+valid 训练池提升拟合质量；test 始终不参与。random_forest 重训时
        关闭 oob_score 以节省计算。"""
    if isinstance(selected_config, TraditionalSelection):
        selected_config = selected_config.config
    selected_config = _clone_config(selected_config, model_type=model_type)
    if model_type == "random_forest":
        selected_config = _clone_config(selected_config, random_forest_oob_score=False)
    final_indices = np.asarray(sorted({int(index) for index in train_valid_indices}), dtype=np.int64)
    if final_indices.size == 0:
        raise ValueError("传统模型最终训练池不能为空")
    normalizer = _fit_x_normalizer(x_raw[final_indices.tolist()], normalization)
    x_final = _transform_x_with_normalizer(x_raw, normalizer)
    model = build_traditional_model(selected_config, y[final_indices], len(label_names))
    model.fit(x_final[final_indices], y[final_indices])
    return model, normalizer, final_indices


def _fit_deep_fold(
    *,
    config: TrainConfig,
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    label_names: list[str],
    run_dir: Path,
    sample_count: int,
    cancel_check: Any | None = None,
    progress_callback: Any | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], DSCARNetMappedInputs | None, dict[str, Any] | None]:
    """单个训练折的深度模型训练（含 DSCARNet 二维映射与早停）。

        流程：统计 train 类别频次（可选 class_weight）→ DSCARNet 先用 train 折
        fit SAR/CAR 映射并落盘 joblib → 构造模型与损失（二分类单 logit BCE /
        多分类 CE）→ AdamW + ReduceLROnPlateau 逐 epoch 训练 → 以最低 valid_loss
        保存并回灌最佳权重；early_stopping_patience 控制早停，cancel_check 每个
        epoch 检查取消/替换。test 在整个循环中完全不出现。"""
    counts = np.bincount(y[splits["train"]], minlength=len(label_names)).astype(np.float32)
    class_weights = None
    if config.class_balance == "class_weight":
        class_weights = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / len(label_names), dtype=torch.float32)
    dscarnet_mapped: DSCARNetMappedInputs | None = None
    dscarnet_mapping_metadata: dict[str, Any] | None = None
    if model_type == "dscarnet":
        mode = str(config.dscarnet_input_mode or "dual").lower()
        dscarnet_profile = build_dscarnet_profile(train_sample_count=len(splits["train"]), feature_count=x.shape[1])
        config.resolved_train_sample_count = len(splits["train"])
        config.resolved_feature_count = x.shape[1]
        dscarnet_mapped = fit_dscarnet_2d_mapping(
            x,
            splits["train"],
            pca_components=int(dscarnet_profile["pca_components"]),
            cluster_channels=int(dscarnet_profile["cluster_channels"]),
            seed=config.seed,
            mode=mode,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        if cancel_check is not None:
            cancel_check()
        dscarnet_mapped.metadata["resolved_profile"] = dscarnet_profile
        dscarnet_mapping_metadata = save_dscarnet_mapping_artifacts(run_dir, dscarnet_mapped)
        if cancel_check is not None:
            cancel_check()
        model = build_dscarnet_model(
            config,
            None if dscarnet_mapped.x_sar is None else dscarnet_mapped.model_input_shape_sar,
            None if dscarnet_mapped.x_car is None else dscarnet_mapped.model_input_shape_car,
            len(label_names),
        )
    else:
        model = build_deep_model(
            config,
            input_length=x.shape[1],
            class_count=len(label_names),
            sample_count=sample_count,
            x_train=x[splits["train"]] if model_type == "pca_mlp" else None,
        )
        if model_type == "pca_mlp" and getattr(model, "pca_model", None) is not None:
            try:
                import joblib

                joblib.dump(model.pca_model, run_dir / "pca_mlp_pca.joblib")
            except Exception:
                pass
        # 二分类用单 logit + BCEWithLogitsLoss（数值上比 sigmoid+BCE 稳定），
        # pos_weight 来自 train 类别频次比；多分类用 CrossEntropyLoss + 反频次权重。
    if len(label_names) == 2:
        pos_weight = None
        if config.class_balance == "class_weight":
            pos_weight = torch.tensor([float(counts[0] / max(counts[1], 1.0))], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_valid_loss = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        if progress_callback is not None:
            progress_callback("epoch_training", f"正在训练 Epoch {epoch}/{config.epochs}", epoch)
        if cancel_check is not None:
            cancel_check()
        model.train()
        losses = []
        if dscarnet_mapped is not None and dscarnet_mapped.metadata["mode"] == "dual":
            for bx1, bx2, by in _dual_loader(dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y, splits["train"], config.batch_size, True):
                optimizer.zero_grad()
                logits = model(bx1, bx2)
                target = by.float().view_as(logits) if logits.ndim == 2 and logits.shape[1] == 1 else by
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            valid_loss = _evaluate_deep_loss(
                model,
                _dual_loader(dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y, splits["valid"], config.batch_size, False),
                criterion,
                dual_input=True,
            )
            valid_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y, splits["valid"], label_names)
        elif dscarnet_mapped is not None:
            branch_values = dscarnet_mapped.x_sar if dscarnet_mapped.metadata["mode"] == "sar" else dscarnet_mapped.x_car
            assert branch_values is not None
            for bx, by in _single_2d_loader(branch_values, y, splits["train"], config.batch_size, True):
                optimizer.zero_grad()
                logits = model(bx)
                target = by.float().view_as(logits) if logits.ndim == 2 and logits.shape[1] == 1 else by
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            valid_loss = _evaluate_deep_loss(model, _single_2d_loader(branch_values, y, splits["valid"], config.batch_size, False), criterion, dual_input=False)
            valid_eval = _evaluate_single_2d(model, branch_values, y, splits["valid"], label_names)
        else:
            for bx, by in _loader(x, y, splits["train"], config.batch_size, True):
                optimizer.zero_grad()
                logits = model(bx)
                target = by.float().view_as(logits) if logits.ndim == 2 and logits.shape[1] == 1 else by
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            valid_loss = _evaluate_deep_loss(
                model,
                _loader(x, y, splits["valid"], config.batch_size, False),
                criterion,
                dual_input=False,
            )
            valid_eval = _evaluate(model, x, y, splits["valid"], label_names)
                # 学习率调度、最佳权重与早停都只盯 valid_loss；valid 指标仅记录，不进决策。
        scheduler.step(valid_loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = valid_loss < best_valid_loss - 1e-8
        if improved:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        best_score = max(best_score, float(valid_eval["macro_f1"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "valid_loss": valid_loss,
                "best_valid_loss": best_valid_loss,
                "learning_rate": current_learning_rate,
                "valid_accuracy": valid_eval["accuracy"],
                "valid_balanced_accuracy": valid_eval["balanced_accuracy"],
                "valid_macro_f1": valid_eval["macro_f1"],
                "best_valid_macro_f1": best_score,
                "bad_epochs": bad_epochs,
            }
        )
        if cancel_check is not None:
            cancel_check()
        if config.early_stopping_patience > 0 and bad_epochs >= config.early_stopping_patience:
            history[-1]["early_stopped"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, dscarnet_mapped, dscarnet_mapping_metadata


def _traditional_fold_log_loss_importance(
    *,
    config: TrainConfig,
    model: Any,
    normalizer: Any,
    x_raw: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    mean_indices: list[int],
    x_axis: list[float],
    label_names: list[str],
    metadata: list[dict[str, Any]],
    x_axis_warning: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-sample true-class log-loss attribution for one traditional-model fold."""

    result = sample_occlusion_importance(
        x_raw,
        y,
        x_axis=x_axis,
        splits=splits,
        label_names=label_names,
        score_fn=lambda values: _traditional_probabilities(
            model,
            _transform_x_with_normalizer(values, normalizer),
        ),
        mean_indices=mean_indices,
        metadata=metadata,
        window_count=config.feature_window_count,
        top_k=config.feature_top_k,
    )
    result["x_axis_warning"] = x_axis_warning
    return result


# 把三种评估口径转换成统一 fold 描述，供主训练循环顺序执行。
def _canonical_evaluation_strategy(config: TrainConfig, has_external_test: bool) -> str:
    """把历史/别名形式的 split_mode 归一到三种正式口径；有独立测试集时
        external_test_holdout 优先级最高；无法识别时直接报错。"""
    if has_external_test:
        return "external_test_holdout"
    mode = str(config.split_mode or "stratified_holdout").strip().lower()
    if mode in {
        "leave_one_sample_id_cv",
        "leave_one_repeat_index_cv",
        "outer_leave_one_repeat_index_cv",
        "loocv",
        "loo",
    }:
        return "leave_one_sample_id_cv"
    if mode in {"stratified", "stratified_holdout", "holdout"}:
        return "stratified_holdout"
    raise ValueError("split_mode 必须是 stratified_holdout 或 leave_one_sample_id_cv")


def _validate_required_splits(splits: dict[str, list[int]], y: np.ndarray, label_names: list[str], required: tuple[str, ...]) -> None:
    """检查指定集合非空且类别完整；CV 只要求 train/valid，holdout 还要求 test。"""
    expected_labels = set(range(len(label_names)))
    for split_name in required:
        if not splits.get(split_name):
            raise ValueError(f"{split_name} 集为空，请增加样品种类或调整划分比例")
        present_labels = set(np.unique(y[splits[split_name]]).tolist())
        missing = [label for idx, label in enumerate(label_names) if idx not in present_labels]
        if missing:
            raise ValueError(
                f"{split_name} 集中缺少类别: {', '.join(missing)}。"
                "请保证每个类别有足够的不同 Sample_ID"
            )


def _sample_ids_for_split(sample_id: np.ndarray, indices: list[int]) -> list[str]:
    """把行索引集合转成去重、按自然序排序的 Sample_ID 列表（审计用）。"""
    if not indices:
        return []
    return [str(item) for item in sorted(np.unique(sample_id[indices]).tolist(), key=natural_sort_key)]


def _holdout_fold(splits: dict[str, list[int]], sample_id: np.ndarray, *, strategy: str) -> dict[str, Any]:
    """把一次 holdout 划分包装成与 CV 折同构的 fold 描述（fold_index=1）。"""
    return {
        "fold_index": 1,
        "test_sample_id": "holdout",
        "train_sample_ids": _sample_ids_for_split(sample_id, splits["train"]),
        "valid_sample_ids": _sample_ids_for_split(sample_id, splits["valid"]),
        "test_sample_ids": _sample_ids_for_split(sample_id, splits.get("test", [])),
        "splits": splits,
        "strategy": strategy,
    }


def _external_test_fold(
    *,
    splits: dict[str, list[int]],
    sample_id: np.ndarray,
    external_test_indices: list[int],
    external_sample_id: np.ndarray,
) -> dict[str, Any]:
    """构造独立测试集折：internal_splits 记录主数据内部划分；splits.test
        指向拼接矩阵中外部样本的行号（排在主数据行号之后连续编号）。"""
    return {
        "fold_index": 1,
        "test_sample_id": "external_test",
        "train_sample_ids": _sample_ids_for_split(sample_id, splits["train"]),
        "valid_sample_ids": _sample_ids_for_split(sample_id, splits["valid"]),
        "test_sample_ids": _sample_ids_for_split(external_sample_id, list(range(len(external_sample_id)))),
        "internal_splits": {**splits, "test": []},
        "external_test_indices": external_test_indices,
        "splits": {**splits, "test": external_test_indices},
        "strategy": "external_test_holdout",
    }


def _run_legacy_training(
    data_path: str | Path,
    config_data: dict[str, Any] | None = None,
    run_id: str | None = None,
    *,
    repository: RunRepository | None = None,
    record: RunRecord | None = None,
) -> dict[str, Any]:
    """执行完整训练闭环并写出一组可由 Manifest 发布的 Run 产物。

    名称保留 ``legacy`` 是为了兼容既有调用者；函数内部执行的是当前
    classification-v2 模型、三种评估策略和现行解释性契约。
    """
        # ── 阶段 0：解析请求 → EvaluationPolicy；只有 TrainConfig 认识的键才进入配置 ──
    raw_config = dict(config_data or {})
    test_data_path = raw_config.get("test_data_path")
    policy = resolve_evaluation_policy(raw_config, has_external_test=bool(test_data_path))
    config_input = {key: value for key, value in raw_config.items() if key in TrainConfig().__dict__}
    legacy_seed = raw_config.get("seed", DEEP_TRAINING_DEFAULTS.seed)
    model_seed = int(raw_config.get("model_seed", legacy_seed))
    split_seed = int(raw_config.get("split_seed", legacy_seed))
    config = TrainConfig(
        **{
            **TrainConfig().__dict__,
            **config_input,
            "split_mode": policy.strategy,
            "split_train": policy.split_train,
            "split_valid": policy.split_valid,
            "split_test": policy.split_test,
            # Preserve the legacy seed surface while keeping the two new seed
            # roles explicit in all emitted config/split audit data.
            "seed": model_seed,
            "model_seed": model_seed,
            "split_seed": split_seed,
            # TEMPORARILY_HIDDEN: this source constant is authoritative; an
            # incoming request must never be able to turn explanation on.
            "feature_selection_enabled": False,
        }
    )
    _validate_split_ratio_config(policy)
    model_type = canonical_model_type(config.model_type)
    modern = config.experiment_version == 'word-0904'
    if modern:
        from . import feature_policy
        if not feature_policy.FEATURE_ENGINEERING_ENABLED:
            raise ValueError('特征工程暂时停用，请使用普通训练配置')
        from .feature_engineering import MODELS
        from .training_experiments import fit_experiment_fold, summarize_experiments
        if model_type not in MODELS:
            raise ValueError('0904 方案仅支持六类公开模型')
    experiment_folds = []
    external_experiment = None
    evaluation_strategy = policy.strategy
    run_id = run_id or (record.run_id if record is not None else uuid.uuid4().hex[:12])
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir / "status.json"
    previous_status: dict[str, Any] = {}
    if status_file.exists():
        try:
            previous_status = json.loads(status_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_status = {}
    if repository is None or record is None:
        _raise_if_run_replaced(status_file)

    def check_run_active() -> None:
        if repository is not None and record is not None:
            repository.assert_active(
                record.run_id,
                claim_token=record.claim_token or "",
                now=datetime.now(timezone.utc),
            )
            return
        _raise_if_run_replaced(status_file)

        # ── 阶段 1：固定随机种子并读取宽表；Label 按字典序编成类别 id（即使为
        # 数字也按类别名处理），Sample_ID 作为分组单位取出 ──
    torch.manual_seed(config.model_seed)
    np.random.seed(config.model_seed)
    dataset = load_modeling_csv(data_path)
    sample_count = int(len(dataset.labels))
    x_raw = np.asarray(dataset.intensity, dtype=np.float32)
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels], dtype=np.int64)
    sample_id = dataset.frame["Sample_ID"].astype(str).to_numpy()
    test_dataset = load_modeling_csv(test_data_path) if test_data_path else None
    test_sample_count = 0
    external_test_x_raw: np.ndarray | None = None
    external_test_y: np.ndarray | None = None
    external_test_sample_id: np.ndarray | None = None
    external_test_metadata: list[dict[str, Any]] = []
        # 有独立测试集：先校验同轴，再把主数据与测试数据纵向拼接成一个矩阵，
        # 外部样本行号排在主数据之后，fold 用行号区间引用它们。
    if test_dataset is not None:
        _validate_external_test_dataset(
            label_names,
            x_raw.shape[1],
            dataset.x_axis[0] if dataset.x_axis else [],
            test_dataset,
        )
        test_x_raw = np.asarray(test_dataset.intensity, dtype=np.float32)
        test_y = np.asarray([label_to_id[label] for label in test_dataset.labels], dtype=np.int64)
        test_sample_id = test_dataset.frame["Sample_ID"].astype(str).to_numpy()
        if modern and set(sample_id).intersection(test_sample_id):
            raise ValueError('独立测试集与主数据包含相同 Sample_ID，不能跨集合使用同一样品')
        test_sample_count = int(len(test_y))
        external_test_x_raw = test_x_raw
        external_test_y = test_y
        external_test_sample_id = test_sample_id
        external_test_metadata = _sample_metadata(test_dataset.frame, list(test_dataset.x_axis), x_raw.shape[1])
        if evaluation_strategy == "leave_one_sample_id_cv_with_external_test":
            # The outer audit sees *only* the primary dataset.  The external
            # dataset is retained for one final, full-primary-data fit below.
            x_model_raw = x_raw
            y_model = y
            folds = _leave_one_sample_id_folds(y, sample_id, config)
            metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1])
        else:
            x_model_raw = np.vstack([x_raw, test_x_raw])
            y_model = np.concatenate([y, test_y])
            internal_splits = _split_indices(y, sample_id, config, label_names)
            _validate_required_splits(internal_splits, y, label_names, ("train", "valid"))
            external_indices = list(range(len(y), len(y_model)))
            folds = [
                _external_test_fold(
                    splits=internal_splits,
                    sample_id=sample_id,
                    external_test_indices=external_indices,
                    external_sample_id=test_sample_id,
                )
            ]
            metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1]) + external_test_metadata
    else:
        x_model_raw = x_raw
        y_model = y
        if evaluation_strategy == "leave_one_sample_id_cv":
            folds = _leave_one_sample_id_folds(y, sample_id, config)
        else:
            splits = _split_indices(y, sample_id, config, label_names)
            _validate_required_splits(splits, y, label_names, ("train", "valid", "test"))
            folds = [_holdout_fold(splits, sample_id, strategy=evaluation_strategy)]
        metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1])
    feature_x_axis = dataset.x_axis[0] if dataset.x_axis else list(range(x_raw.shape[1]))
    combined_axes = list(dataset.x_axis) + (list(test_dataset.x_axis) if test_dataset is not None else [])
    x_axis_warning = _x_axis_warning(combined_axes, x_raw.shape[1])

        # ── 阶段 2：初始化跨折累加器（预测明细、折指标、训练历史、OOF 汇总等） ──
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    cv_fold_payloads: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    fold_split_evals: list[dict[str, dict[str, Any]]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    last_model: Any | None = None
    last_model_family = model_family(model_type)
    last_model_artifact = "model.pkl" if last_model_family == "traditional_ml" else "model.pt"
    final_deep_context: dict[str, Any] | None = None
    deep_sample_results: list[dict[str, Any]] = []
    traditional_sample_results: list[dict[str, Any]] = []
    best_search_rows: list[dict[str, Any]] = []
    model_visualization_context: dict[str, Any] | None = None
    fold_count = len(folds)
    started_at = previous_status.get("started_at") or _now_iso()

    def write_progress(
        fold_index: int,
        completed_folds: int,
        fold: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        check_run_active()
        progress = {
            "current_fold": fold_index,
            "completed_folds": completed_folds,
            "fold_progress_text": f"{fold_index}/{fold_count}",
            "target_epochs": config.epochs,
            **(extra or {}),
        }
        if repository is not None and record is not None:
            updated = repository.update_progress(
                record.run_id,
                claim_token=record.claim_token or "",
                now=datetime.now(timezone.utc),
                progress=progress,
            )
            project_status(
                run_dir,
                updated,
                **progress,
                current_fold_sample_id=fold.get("test_sample_id"),
                model_type=model_type,
                evaluation_strategy=evaluation_strategy,
            )
            return
        current_status = _read_status_file(status_file)
        _write_status_file(
            status_file,
            {
                **previous_status,
                **current_status,
                "run_id": run_id,
                "status": "running",
                "data_path": str(Path(data_path).resolve()),
                "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
                "config": {**config.__dict__, "model_type": model_type},
                "model_type": model_type,
                "sample_count": sample_count,
                "target_epochs": config.epochs,
                "fold_count": fold_count,
                "current_fold": fold_index,
                "completed_folds": completed_folds,
                "fold_progress_text": f"{fold_index}/{fold_count}",
                "current_fold_sample_id": fold.get("test_sample_id"),
                "total_target_epochs": int(fold_count * config.epochs),
                "evaluation_strategy": evaluation_strategy,
                "started_at": started_at,
                **(extra or {}),
            },
        )

        # ── 阶段 3：逐折执行。每折开头先用当前折 train 拟合 normalizer 并
        # 变换全矩阵——这是“标准化只看 train”的核心防线 ──
    for fold in folds:
        check_run_active()
        splits = fold["splits"]
        normalizer = _fit_x_normalizer(x_model_raw[splits["train"]], config.normalization)
        x = _transform_x_with_normalizer(x_model_raw, normalizer)
        fold_index = int(fold["fold_index"])
        write_progress(fold_index, max(0, fold_index - 1), fold)
        fold_final_fit_indices: np.ndarray | None = None
        fold_best_params: dict[str, Any] = {}
        fold_selection_metric: str | None = None
        fold_selection_score: float | None = None
                # 传统模型分支：搜索选参（train/valid）→ train+valid 重训 → 评估三个集合
                # → 可选窗口遮挡解释性；深度分支：训练整折 → 评估 → 收集解释性上下文。
        if modern:
            winner = fit_experiment_fold(
                config, model_type, x_model_raw, y_model,
                np.concatenate([sample_id, test_sample_id]) if test_dataset is not None and evaluation_strategy != 'leave_one_sample_id_cv_with_external_test' else sample_id,
                splits, label_names, fold_index, check_run_active,
                lambda extra: write_progress(fold_index, fold_index - 1, fold, extra=extra),
            )
            model = winner['model']
            normalizer = winner['transform']
            train_eval, valid_eval, test_eval = (winner['evals'][name] for name in ('train', 'valid', 'test'))
            fold_best_params = {'feature_scheme': winner['scheme_id'], **winner['params']}
            fold_selection_metric, fold_selection_score = winner['selection_metric'], winner['selection_score']
            fold_final_fit_indices = np.asarray(sorted(set(splits['train']) | set(splits['valid']))) if model_type != 'cnn1d' else np.asarray(splits['train'])
            best_search_rows.extend(winner['search_rows'])
            history_rows.extend({**row, 'fold_index': fold_index, 'feature_scheme': winner['scheme_id']} for row in winner['history'])
            winner['test_sample_ids'] = fold.get('test_sample_ids') or ([fold['test_sample_id']] if fold.get('test_sample_id') is not None else [])
            winner['split_summary'] = {
                key: {'sample_count': len(set((np.concatenate([sample_id, test_sample_id]) if test_dataset is not None and evaluation_strategy != 'leave_one_sample_id_cv_with_external_test' else sample_id)[indices])),
                      'measurement_count': len(indices)} for key, indices in splits.items()
            }
            winner['requested_ratio'] = {'train': config.split_train, 'valid': config.split_valid, 'test': config.split_test}
            experiment_folds.append({k: v for k, v in winner.items() if k not in {'model', 'transformer', 'history', 'search_rows', 'evals'}})
            if model_type == 'cnn1d':
                import joblib
                joblib.dump(winner['transformer'], run_dir / 'feature_transform.joblib')
            dscarnet_mapping_metadata = None
        elif model_family(model_type) == "traditional_ml":
            model, selected_config, valid_eval, search_rows = _fit_traditional_fold(
                config,
                model_type,
                x_model_raw,
                y_model,
                np.concatenate([sample_id, test_sample_id]) if test_dataset is not None else sample_id,
                splits,
                label_names,
            )
            selection_x = x
            check_run_active()
            best_search_rows.extend({**row, "fold_index": fold_index} for row in search_rows)
            fold_best_params = _traditional_params(selected_config, model_type)
            selected_search_row = next((row for row in search_rows if row.get("is_selected")), {})
            fold_selection_metric = str(selected_search_row.get("selection_metric") or "balanced_accuracy")
            fold_selection_score = float(
                selected_search_row.get("selection_score", valid_eval["balanced_accuracy"])
            )
            train_eval = _evaluate_traditional_model(model, x, y_model, splits["train"], label_names)
            train_valid_indices = sorted({*splits["train"], *splits["valid"]})
            final_model, final_normalizer, fold_final_fit_indices = _fit_final_traditional_model(
                selected_config=selected_config,
                model_type=model_type,
                x_raw=x_model_raw,
                y=y_model,
                train_valid_indices=train_valid_indices,
                normalization=config.normalization,
                label_names=label_names,
            )
            final_x = _transform_x_with_normalizer(x_model_raw, final_normalizer)
            test_eval = _evaluate_traditional_model(final_model, final_x, y_model, splits["test"], label_names)
            model = final_model
            normalizer = final_normalizer
            model_visualization_context = {
                "model": model,
                "x": final_x,
                "selection_x": selection_x,
                "splits": splits,
            }
            if EXPLAINABILITY_ENABLED and config.feature_selection_enabled:
                check_run_active()
                traditional_sample_results.append(
                    _tag_fold_sample_result(
                        _traditional_fold_log_loss_importance(
                            config=config,
                            model=model,
                            normalizer=normalizer,
                            x_raw=x_model_raw,
                            y=y_model,
                            splits=splits,
                            mean_indices=fold_final_fit_indices.tolist(),
                            x_axis=feature_x_axis,
                            label_names=label_names,
                            metadata=metadata,
                            x_axis_warning=x_axis_warning,
                        ),
                        fold_index,
                    )
                )
            dscarnet_mapping_metadata = None
        else:
            def report_deep_stage(stage: str, label: str, epoch: int | None = None) -> None:
                stage_progress: dict[str, Any] = {
                    "training_stage": stage,
                    "training_stage_label": label,
                }
                if epoch is not None:
                    stage_progress["current_epoch"] = int(epoch)
                write_progress(
                    fold_index,
                    max(0, fold_index - 1),
                    fold,
                    extra=stage_progress,
                )

            model, history, dscarnet_mapped, dscarnet_mapping_metadata = _fit_deep_fold(
                config=config,
                model_type=model_type,
                x=x,
                y=y_model,
                splits=splits,
                label_names=label_names,
                run_dir=run_dir,
                sample_count=int(len(splits["train"])),
                cancel_check=check_run_active,
                progress_callback=report_deep_stage,
            )
            check_run_active()
            for row in history:
                history_rows.append({**row, "fold_index": fold_index})
            if dscarnet_mapped is not None:
                mode = dscarnet_mapped.metadata.get("mode", "dual")
                if mode == "dual":
                    train_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["train"], label_names)
                    valid_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["valid"], label_names)
                    test_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["test"], label_names)
                else:
                    branch_values = dscarnet_mapped.x_sar if mode == "sar" else dscarnet_mapped.x_car
                    train_eval = _evaluate_single_2d(model, branch_values, y_model, splits["train"], label_names)
                    valid_eval = _evaluate_single_2d(model, branch_values, y_model, splits["valid"], label_names)
                    test_eval = _evaluate_single_2d(model, branch_values, y_model, splits["test"], label_names)
            else:
                train_eval = _evaluate(model, x, y_model, splits["train"], label_names)
                valid_eval = _evaluate(model, x, y_model, splits["valid"], label_names)
                test_eval = _evaluate(model, x, y_model, splits["test"], label_names)
            final_deep_context = {
                "model": model,
                "x": x,
                "splits": splits,
                "dscarnet_mapped": dscarnet_mapped,
                "dscarnet_mapping_metadata": dscarnet_mapping_metadata,
            }
            model_visualization_context = {
                "model": model,
                "x": x,
                "selection_x": None,
                "splits": splits,
            }
            if EXPLAINABILITY_ENABLED:
                check_run_active()
                deep_sample_results.append(
                    _tag_fold_sample_result(
                        _deep_sample_feature_result(
                            config=config,
                            model=model,
                            x=x,
                            y=y_model,
                            splits=splits,
                            x_axis=feature_x_axis,
                            x_axis_warning=x_axis_warning,
                            label_names=label_names,
                            metadata=metadata,
                            dscarnet_mapped=dscarnet_mapped,
                            dscarnet_mapping_metadata=dscarnet_mapping_metadata,
                        ),
                        fold_index,
                    )
                )
        last_model = model
        split_evals = {"train": train_eval, "valid": valid_eval, "test": test_eval}
        split_metrics = {name: _metrics_from_eval(eval_payload, label_names) for name, eval_payload in split_evals.items()}
        fold_split_evals.append(split_evals)
        all_true.extend(int(item) for item in test_eval["true"])
        all_pred.extend(int(item) for item in test_eval["pred"])
        fold_metrics = split_metrics["test"]
        fold_metric_rows.append(
            {
                "fold_index": fold_index,
                "test_sample_id": fold["test_sample_id"],
                "accuracy": fold_metrics["accuracy"],
                "balanced_accuracy": fold_metrics["balanced_accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "weighted_f1": fold_metrics["weighted_f1"],
                "train_accuracy": split_metrics["train"]["accuracy"],
                "train_balanced_accuracy": split_metrics["train"]["balanced_accuracy"],
                "train_macro_precision": split_metrics["train"]["macro_precision"],
                "train_macro_recall": split_metrics["train"]["macro_recall"],
                "train_macro_f1": split_metrics["train"]["macro_f1"],
                "valid_accuracy": split_metrics["valid"]["accuracy"],
                "valid_balanced_accuracy": split_metrics["valid"]["balanced_accuracy"],
                "valid_macro_precision": split_metrics["valid"]["macro_precision"],
                "valid_macro_recall": split_metrics["valid"]["macro_recall"],
                "valid_macro_f1": split_metrics["valid"]["macro_f1"],
                "test_accuracy": fold_metrics["accuracy"],
                "test_balanced_accuracy": fold_metrics["balanced_accuracy"],
                "test_macro_precision": fold_metrics["macro_precision"],
                "test_macro_recall": fold_metrics["macro_recall"],
                "test_macro_f1": fold_metrics["macro_f1"],
            }
        )
        cv_fold_payloads.append(
            {
                "fold_index": fold_index,
                "test_sample_id": fold["test_sample_id"],
                "train_sample_ids": fold["train_sample_ids"],
                "valid_sample_ids": fold["valid_sample_ids"],
                "test_sample_ids": fold.get("test_sample_ids", []),
                "final_fit_indices": fold_final_fit_indices.tolist() if fold_final_fit_indices is not None else [],
                "best_params": fold_best_params,
                "selection_metric": fold_selection_metric,
                "selection_score": fold_selection_score,
                "metrics": fold_metrics,
                "split_metrics": split_metrics,
                "preprocess": _json_normalizer(normalizer),
                "dscarnet_mapping": dscarnet_mapping_metadata,
                "splits": fold.get("internal_splits", splits),
                "external_test_indices": fold.get("external_test_indices", []),
            }
        )
                # 逐测试样品写预测明细：每类一列 prob_<label>，前端结果页直接消费。
        for local_idx, source_idx in enumerate(splits["test"]):
            source_metadata = metadata[source_idx]
            row = {
                "dataset": "external_test" if evaluation_strategy == "external_test_holdout" else "test",
                "fold_index": fold_index,
                "index": source_metadata["index"],
                "Sample_ID": source_metadata["sample_id"],
                "true_label": label_names[test_eval["true"][local_idx]],
                "pred_label": label_names[test_eval["pred"][local_idx]],
            }
            for label, prob in zip(label_names, test_eval["probabilities"][local_idx]):
                row[f"prob_{label}"] = float(prob)
            prediction_rows.append(row)
        check_run_active()
        write_progress(fold_index, fold_index, fold)

        # ── 阶段 4：汇总指标（test 主值 = pooled OOF）、合并解释性并落盘全部产物 ──
    external_final_split_payload: dict[str, Any] | None = None
    if evaluation_strategy == "leave_one_sample_id_cv_with_external_test":
        # Keep the LOO folds above as an auditable, primary-data-only OOF
        # experiment.  The ranking metric below is deliberately a *separate*
        # final fit on all primary samples followed by exactly one external
        # test evaluation.  External rows never enter the CV splitter, inner
        # search, normalizer fit, PCA fit, or model training.
        if external_test_x_raw is None or external_test_y is None or external_test_sample_id is None:
            raise RuntimeError("独立测试集+留一模式缺少已校验的独立测试数据")
        audit_metrics, audit_cv_summary = _build_metrics_payload(
            fold_split_evals, label_names, "leave_one_sample_id_cv"
        )
        final_train_indices = list(range(len(y)))
        external_indices = list(range(len(y), len(y) + len(external_test_y)))
        final_x_raw = np.vstack([x_raw, external_test_x_raw])
        final_y = np.concatenate([y, external_test_y])
        final_sample_id = np.concatenate([sample_id, external_test_sample_id])
        final_metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1]) + external_test_metadata
        final_splits = {"train": final_train_indices, "valid": final_train_indices, "test": external_indices}
        final_fold_index = len(folds) + 1

        if modern:
            selection_config = _clone_config(config, split_mode='external_test_holdout', split_test=0)
            selection_splits = _split_indices(y, sample_id, selection_config, label_names)
            selection_splits['test'] = external_indices
            external_experiment = fit_experiment_fold(
                config, model_type, final_x_raw, final_y, final_sample_id,
                selection_splits, label_names, 'external_final', check_run_active,
                lambda extra: write_progress(len(folds), len(folds), folds[-1], extra=extra), external_final=True,
            )
            last_model = external_experiment['model']
            external_experiment['test_sample_ids'] = sorted(set(external_test_sample_id))
            external_experiment['requested_ratio'] = {'train': config.split_train, 'valid': config.split_valid, 'test': 0}
            external_experiment['split_summary'] = {
                key: {'sample_count': len(set(final_sample_id[indices])), 'measurement_count': len(indices)}
                for key, indices in selection_splits.items()
            }
            final_normalizer = external_experiment['transform']
            final_train_eval, external_test_eval = (external_experiment['evals'][name] for name in ('train', 'test'))
            external_best_params = {'feature_scheme': external_experiment['scheme_id'], **external_experiment['params']}
            external_selection_score = external_experiment['selection_score']
            final_fit_indices = np.asarray(final_train_indices)
            external_mapping_metadata = None
            best_search_rows.extend(external_experiment['search_rows'])
            history_rows.extend({**row, 'fold_index': 'external_final'} for row in external_experiment['history'])
            if model_type == 'cnn1d':
                import joblib
                joblib.dump(external_experiment['transformer'], run_dir / 'feature_transform.joblib')
        elif model_family(model_type) == "traditional_ml":
            selection_splits = {"train": final_train_indices, "valid": final_train_indices, "test": []}
            selection = _select_traditional_config(
                config, model_type, x_raw, y, sample_id, selection_splits, label_names
            )
            best_search_rows.extend({**row, "fold_index": "external_final"} for row in selection.search_rows)
            final_model, final_normalizer, final_fit_indices = _fit_final_traditional_model(
                selected_config=selection.config,
                model_type=model_type,
                x_raw=final_x_raw,
                y=final_y,
                train_valid_indices=final_train_indices,
                normalization=config.normalization,
                label_names=label_names,
            )
            final_x = _transform_x_with_normalizer(final_x_raw, final_normalizer)
            final_train_eval = _evaluate_traditional_model(final_model, final_x, final_y, final_train_indices, label_names)
            external_test_eval = _evaluate_traditional_model(final_model, final_x, final_y, external_indices, label_names)
            last_model = final_model
            last_model_family = "traditional_ml"
            model_visualization_context = {"model": final_model, "x": final_x, "selection_x": None, "splits": final_splits}
            external_best_params = _traditional_params(selection.config, model_type)
            external_selection_score = selection.valid_balanced_accuracy
            external_mapping_metadata = None
            final_deep_context = None
        else:
            final_normalizer = _fit_x_normalizer(x_raw, config.normalization)
            final_x = _transform_x_with_normalizer(final_x_raw, final_normalizer)
            model, history, mapped, external_mapping_metadata = _fit_deep_fold(
                config=config,
                model_type=model_type,
                x=final_x,
                y=final_y,
                splits=final_splits,
                label_names=label_names,
                run_dir=run_dir,
                sample_count=len(final_train_indices),
                cancel_check=check_run_active,
                progress_callback=lambda stage, label, epoch=None: _write_status_file(
                    status_file,
                    {**_read_status_file(status_file), "training_stage": stage, "training_stage_label": label, "current_epoch": epoch, "current_fold": final_fold_index, "fold_progress_text": f"{final_fold_index}/{final_fold_index}"},
                ),
            )
            for row in history:
                history_rows.append({**row, "fold_index": "external_final"})
            if mapped is not None:
                mode = mapped.metadata.get("mode", "dual")
                if mode == "dual":
                    final_train_eval = _evaluate_dual(model, mapped.x_sar, mapped.x_car, final_y, final_train_indices, label_names)
                    external_test_eval = _evaluate_dual(model, mapped.x_sar, mapped.x_car, final_y, external_indices, label_names)
                else:
                    branch_values = mapped.x_sar if mode == "sar" else mapped.x_car
                    final_train_eval = _evaluate_single_2d(model, branch_values, final_y, final_train_indices, label_names)
                    external_test_eval = _evaluate_single_2d(model, branch_values, final_y, external_indices, label_names)
            else:
                final_train_eval = _evaluate(model, final_x, final_y, final_train_indices, label_names)
                external_test_eval = _evaluate(model, final_x, final_y, external_indices, label_names)
            last_model = model
            last_model_family = "deep_learning"
            final_deep_context = {"model": model, "x": final_x, "splits": final_splits, "dscarnet_mapped": mapped, "dscarnet_mapping_metadata": external_mapping_metadata}
            model_visualization_context = {"model": model, "x": final_x, "selection_x": None, "splits": final_splits}
            external_best_params = {}
            external_selection_score = None
            final_fit_indices = np.asarray(final_train_indices, dtype=np.int64)

        external_metrics = _metrics_from_eval(external_test_eval, label_names)
        external_metrics["aggregation"] = "direct_external_test"
        final_train_metrics = _metrics_from_eval(final_train_eval, label_names)
        metrics = {**external_metrics, "train": final_train_metrics, "valid": final_train_metrics, "test": external_metrics,
                   "audit": {"pooled_oof": audit_metrics["test"], "fold_mean": audit_cv_summary["fold_mean"]["test"], "fold_std": audit_cv_summary["fold_std"]["test"]}}
        cv_summary = {
            **audit_cv_summary,
            "strategy": evaluation_strategy,
            "primary_test_aggregation": "direct_external_test",
            "external_test": external_metrics,
            "audit": {"strategy": "leave_one_sample_id_cv", "pooled_oof": audit_metrics["test"], "fold_mean": audit_cv_summary["fold_mean"]["test"], "fold_std": audit_cv_summary["fold_std"]["test"]},
        }
        all_true.extend(int(item) for item in external_test_eval["true"])
        all_pred.extend(int(item) for item in external_test_eval["pred"])
        for local_idx, source_idx in enumerate(external_indices):
            source_metadata = final_metadata[source_idx]
            row = {"dataset": "external_test", "fold_index": "external_final", "index": source_metadata["index"], "Sample_ID": source_metadata["sample_id"],
                   "true_label": label_names[external_test_eval["true"][local_idx]], "pred_label": label_names[external_test_eval["pred"][local_idx]]}
            for label, probability in zip(label_names, external_test_eval["probabilities"][local_idx]):
                row[f"prob_{label}"] = float(probability)
            prediction_rows.append(row)
        external_final_split_payload = {
            "fold_index": "external_final", "test_sample_id": "external_test", "train_sample_ids": _sample_ids_for_split(sample_id, final_train_indices),
            "valid_sample_ids": _sample_ids_for_split(sample_id, final_train_indices), "test_sample_ids": _sample_ids_for_split(external_test_sample_id, list(range(len(external_test_sample_id)))),
            "final_fit_indices": final_fit_indices.tolist(), "best_params": external_best_params, "selection_metric": "mean_balanced_accuracy_grouped_5fold" if model_family(model_type) == "traditional_ml" else None,
            "selection_score": external_selection_score, "metrics": external_metrics, "split_metrics": {"train": final_train_metrics, "valid": final_train_metrics, "test": external_metrics},
            "preprocess": _json_normalizer(final_normalizer), "dscarnet_mapping": external_mapping_metadata,
            "splits": {"train": final_train_indices, "valid": final_train_indices, "test": external_indices}, "external_test_indices": external_indices,
        }
    else:
        metrics, cv_summary = _build_metrics_payload(fold_split_evals, label_names, evaluation_strategy)
    cv_metrics = {
        "strategy": evaluation_strategy,
        "fold_count": len(folds),
        "metrics": metrics,
        "cv_summary": cv_summary,
        "folds": cv_fold_payloads,
    }
    if external_final_split_payload is not None:
        cv_metrics["external_final"] = external_final_split_payload

    # TEMPORARILY_HIDDEN: do not re-enable without an explicit product requirement
    # and synchronized backend/frontend contract tests.  Keep the implementation
    # modules intact, but never calculate or write their artifacts in new Runs.
    sample_feature_summary = {"status": "temporarily_hidden"}
    model_feature_visualization_summary = {"status": "temporarily_hidden"}

    metadata_train_count = len(folds[0]["splits"].get("train", [])) if folds else 0
    if model_type == "dscarnet":
        profile_payload = build_dscarnet_profile(train_sample_count=metadata_train_count, feature_count=x_raw.shape[1])
    else:
        profile_payload = asdict(build_model_profile(model_type, train_sample_count=metadata_train_count, feature_count=x_raw.shape[1]))
    range_warnings = model_range_warnings(
        train_sample_count=metadata_train_count,
        feature_count=x_raw.shape[1],
    )
    model_metadata = {
        "model_type": model_type,
        "model_family": last_model_family,
        "architecture_version": ARCHITECTURE_VERSION,
        "N_train": metadata_train_count,
        "L": int(x_raw.shape[1]),
        "train_sample_count": metadata_train_count,
        "feature_count": int(x_raw.shape[1]),
        "profile": profile_payload,
        "model_profile": profile_payload,
        "resolved_profile": profile_payload,
        "model_range_warnings": range_warnings,
        "explainability_status": "temporarily_hidden",
    }
    experiment = summarize_experiments(experiment_folds, label_names, external_final=external_experiment) if modern else None
    if modern:
        selected = external_experiment or experiment_folds[-1]
        model_metadata.update(architecture_version='docx-classification-v4-0904', experiment_version='word-0904', feature_transform=selected['transform'])
        if model_type == 'cnn1d':
            model_metadata.update(profile=selected['profile'], model_profile=selected['profile'], resolved_profile=selected['profile'], N_train=selected['profile']['N'], L=selected['profile']['L'])
            profile_payload = selected['profile']
            metadata_train_count = selected['profile']['N']
    if last_model_family == "deep_learning":
        model_metadata.update(
            {
                "classification_head": "binary_single_logit" if len(label_names) == 2 else "multiclass_logits",
                "loss_function": "BCEWithLogitsLoss" if len(label_names) == 2 else "CrossEntropyLoss",
                "output_dim": 1 if len(label_names) == 2 else len(label_names),
            }
        )
    if getattr(last_model, "pca_metadata", None) is not None:
        model_metadata["pca"] = last_model.pca_metadata
    if modern and model_type == 'cnn1d':
        model_metadata.update(classification_head='multiclass_logits', loss_function='CrossEntropyLoss', output_dim=len(label_names))
    if model_type == "spls_da" and getattr(last_model, "selected_feature_indices_by_component", None) is not None:
        model_metadata["spls_da"] = {
            "selected_feature_indices_by_component": [
                [int(index) for index in component]
                for component in last_model.selected_feature_indices_by_component
            ],
            "actual_n_components": int(getattr(last_model, "actual_n_components_", 0)),
            "actual_keepx": int(getattr(last_model, "actual_keepx_", 0)),
        }
    config_out = {
        **config.__dict__,
        "model_type": model_type,
        "architecture_version": ARCHITECTURE_VERSION,
        "preprocess": {"mode": config.normalization, "fit_scope": "train_fold"},
        "evaluation_strategy": evaluation_strategy,
        "dimension_band": _dimension_band(x_raw.shape[1]),
        "fold_count": len(folds),
        "N_train": metadata_train_count,
        "L": int(x_raw.shape[1]),
        "model_profile": profile_payload,
        "model_range_warnings": range_warnings,
        "explainability_status": "temporarily_hidden",
    }
    if modern:
        config_out['architecture_version'] = 'docx-classification-v4-0904'
        config_out['experiment_version'] = 'word-0904'
        config_out['model_profile'] = profile_payload
        config_out['L'] = selected['transform']['output_features']
        (run_dir / 'feature_experiments.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
        pd.DataFrame([{'scheme_id': item['scheme_id'], 'scheme_name': item['scheme_name'], 'status': item['status'], 'reason': item['reason'], **{key: (item['metrics'] or {}).get(key) for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1')}} for item in experiment['schemes']]).to_csv(run_dir / 'feature_experiments.csv', index=False, encoding='utf-8-sig')
        for row in best_search_rows:
            row['params_json'] = json.dumps(row.get('params', {}), ensure_ascii=False)
    (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "model_metadata.json").write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    split_audit_payloads = list(cv_fold_payloads)
    if external_final_split_payload is not None:
        split_audit_payloads.append(external_final_split_payload)
    (run_dir / "split.json").write_text(json.dumps(split_audit_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(fold_metric_rows).to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    if last_model_family == "deep_learning":
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    search_columns = [
        "scheme_id", "scheme_name", "status", "reason", "inner_fold_scores",
        "fold_index",
        "model_type",
        "is_selected",
        "selection_metric",
        "selection_score",
        "valid_accuracy",
        "valid_balanced_accuracy",
        "valid_macro_f1",
        "params_json",
    ]
    search_frame = pd.DataFrame(best_search_rows)
    for column in search_columns:
        if column not in search_frame.columns:
            search_frame[column] = None
    search_frame.reindex(columns=search_columns).to_csv(
        run_dir / "hyperparameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if last_model is not None and last_model_family == "traditional_ml":
        with (run_dir / "model.pkl").open("wb") as fh:
            pickle.dump(last_model, fh)
    elif last_model is not None:
        torch.save(last_model.state_dict(), run_dir / "model.pt")

        # status_payload 是结果页 run-result-v1 的主要数据来源；旧入口直接写
        # status.json，worker 路径则由状态机投影生成等价内容。
    status_payload = {
        **previous_status,
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "cv_summary": cv_summary,
        "history": history_rows,
        "model_type": model_type,
        "model_family": last_model_family,
        "architecture_version": config_out['architecture_version'],
        "model_metadata": model_metadata,
        "model_profile": profile_payload,
        "model_artifact": last_model_artifact,
        "model_artifact_note": (
            "最后一个交叉验证折模型，仅作下载参考，不用于汇报的交叉验证指标"
            if evaluation_strategy == "leave_one_sample_id_cv"
            else "在全部主数据重训后仅用于独立测试集最终评估的模型；留一 OOF 仅作审计"
            if evaluation_strategy == "leave_one_sample_id_cv_with_external_test"
            else "本次 holdout 训练得到的模型，用于对应测试指标"
        ),
        "not_used_for_reported_cv_metrics": evaluation_strategy == "leave_one_sample_id_cv",
        "explainability_status": "temporarily_hidden",
        "x_axis_warning": x_axis_warning,
        "sample_count": sample_count,
        "curve_count": sample_count,
        "sample_id_count": int(len(set(dataset.sample_id))),
        "class_count": int(len(label_names)),
        "feature_count": int(x_raw.shape[1]),
        "test_sample_count": int(len(external_test_y)) if evaluation_strategy == "leave_one_sample_id_cv_with_external_test" and external_test_y is not None else int(len(all_true)),
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history_rows),
        "total_target_epochs": int(len(folds) * config.epochs),
        "current_fold": len(folds),
        "completed_folds": len(folds),
        "fold_progress_text": f"{len(folds)}/{len(folds)}",
        "current_fold_sample_id": folds[-1].get("test_sample_id") if folds else None,
        "best_valid_macro_f1": max((row.get("best_valid_macro_f1") or 0.0 for row in history_rows), default=None),
        "best_valid_balanced_accuracy": max((row.get("valid_balanced_accuracy") or 0.0 for row in history_rows), default=None),
        "evaluation_strategy": evaluation_strategy,
        "fold_count": len(folds),
        "external_final_completed": external_final_split_payload is not None,
        "config": config_out,
        "data_path": str(Path(data_path).resolve()),
        "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
        "completed_at": _now_iso(),
    }
    if repository is None or record is None:
        (run_dir / "status.json").write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**status_payload, "run_dir": str(run_dir.resolve())}


# 直接调用兼容 API：worker 正式路径会额外使用 TrainingExecution 做 claim/cancel 校验。
def train_model(data_path: str | Path, config_data: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    """直接执行训练并发布 Manifest；正式 HTTP 路径应由 worker 调用。"""
    return run_legacy_training_compatibility(data_path=data_path, config_data=config_data, run_id=run_id)


def run_legacy_training_compatibility(
    *,
    data_path: str | Path,
    config_data: dict[str, Any] | None,
    run_id: str | None,
) -> dict[str, Any]:
    """从旧参数形态执行当前训练器，并允许注入 RunRepository 取消检查。"""
    result = _run_legacy_training(data_path=Path(data_path), config_data=config_data or {}, run_id=run_id)
        # 训练结束后确保 manifest.json 存在（SHA-256/大小校验的显式 catalog），
        # 缺失时用 RunArtifactWriter 原子发布——结果页下载白名单依赖它。
    run_dir = Path(result["run_dir"])
    if not (run_dir / "manifest.json").is_file():
        RunArtifactWriter(run_dir).finalize(
            run_id=str(result["run_id"]),
            metadata={
                key: result[key]
                for key in ("model_type", "model_family", "evaluation_strategy", "fold_count")
                if key in result
            },
        )
    return result


def list_runs() -> list[dict[str, Any]]:
    """列出本地 Run 目录中可解析的 status.json，供旧调用方使用。"""
    runs = []
    for path in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        status_file = path / "status.json"
        if status_file.exists():
            try:
                runs.append(json.loads(status_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                runs.append({"run_id": path.name, "status": "unknown"})
    return runs
