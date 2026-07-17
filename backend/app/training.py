"""SpecAutoAI 分类训练、评估、模型选择与 artifact 编排主模块。

核心不变量：所有标准化、PCA、AggMap 和超参数选择只能查看当前训练折；传统模型
在 valid/OOB 选定参数后用 train+valid 重训；深度模型以最低 validation loss
保存最佳权重。每折 test 只用于最终指标和解释性分析，不参与拟合或早停。

本模块可以被 worker 调用，也保留直接 ``train_model`` 兼容入口。真正的 Run 状态
迁移和 Manifest 原子发布由 ``runs.execution`` / ``runs.artifacts`` 负责。
"""

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
from sklearn.model_selection import ParameterSampler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .classification_policy import DEEP_TRAINING_DEFAULTS, EvaluationPolicy, resolve_evaluation_policy
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


class TrainingRunReplaced(RuntimeError):
    """Raised when a training run has been paused because a newer run replaced it."""


@dataclass
class TrainConfig:
    """训练请求的完整内部配置，包括兼容字段与运行时解析字段。

    ``resolved_*`` 字段由每个训练折覆盖，用于记录真正参与构造模型的 N/L，
    不能直接相信客户端传入值。
    """
    epochs: int = DEEP_TRAINING_DEFAULTS.epochs
    batch_size: int = DEEP_TRAINING_DEFAULTS.batch_size
    learning_rate: float = DEEP_TRAINING_DEFAULTS.learning_rate
    weight_decay: float = DEEP_TRAINING_DEFAULTS.weight_decay
    scheduler_factor: float = DEEP_TRAINING_DEFAULTS.scheduler_factor
    scheduler_patience: int = DEEP_TRAINING_DEFAULTS.scheduler_patience
    min_learning_rate: float = DEEP_TRAINING_DEFAULTS.min_learning_rate
    seed: int = DEEP_TRAINING_DEFAULTS.seed
    normalization: str = "zscore"
    split_mode: str = "stratified"
    split_train: int = 8
    split_valid: int = 1
    split_test: int = 1
    class_balance: str = "none"
    model_type: str = "cnn1d"
    early_stopping_patience: int = DEEP_TRAINING_DEFAULTS.early_stopping_patience
    dropout: float | None = None
    hidden_size: int = 64
    transformer_heads: int = 4
    unet_depth: int = 3
    dscarnet_inception_blocks: int = 1
    dscarnet_pca_components: int = 30
    dscarnet_cluster_channels: int = 9
    dscarnet_input_mode: str = "dual"
    resolved_train_sample_count: int = 100
    resolved_feature_count: int = 1000
    knn_n_neighbors: int = 5
    knn_weights: str = "distance"
    knn_metric: str = "minkowski"
    knn_p: int = 2
    random_forest_n_estimators: int = 200
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
    svm_kernel: str = "rbf"
    random_forest_max_features: str | float = "sqrt"
    random_forest_oob_score: bool = False
    xgboost_min_child_weight: float = 1.0
    xgboost_gamma: float = 0.0
    feature_selection_enabled: bool = True
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
    if not status_file.exists():
        return {}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_status_file(status_file: Path, payload: dict[str, Any]) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_is_replaced(run_id: str, *, runs_dir: Path | None = None) -> bool:
    """兼容判断旧 status 投影是否因新 Run 替换而 paused。"""
    status = _read_status_file((runs_dir or RUNS_DIR) / run_id / "status.json")
    return status.get("status") == "paused" and bool(status.get("replaced_by"))


def _raise_if_run_replaced(status_file: Path) -> None:
    status = _read_status_file(status_file)
    if status.get("status") != "paused":
        return
    replaced_by = status.get("replaced_by") or "newer_run"
    raise TrainingRunReplaced(f"训练任务 {status.get('run_id') or status_file.parent.name} 已被新任务 {replaced_by} 替换")


# 数据划分与归一化：所有 group split 都以 Sample_ID 为不可拆分单位。
def _validate_split_ratio_config(policy: EvaluationPolicy) -> None:
    ratios = (int(policy.split_train), int(policy.split_valid), int(policy.split_test))
    if any(value < 0 for value in ratios):
        raise ValueError("划分比例必须是非负整数")
    if policy.strategy == "external_test_holdout":
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


def _split_indices(labels: np.ndarray, sample_id: np.ndarray, config: TrainConfig) -> dict[str, list[int]]:
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

    ratio_denominator = ratios[0] + ratios[1] if ratios[2] == 0 else 10
    valid_count = max(1, int(np.floor(len(group_values) * ratios[1] / ratio_denominator))) if ratios[1] > 0 else 0
    test_count = max(1, int(np.floor(len(group_values) * ratios[2] / ratio_denominator))) if ratios[2] > 0 else 0
    if valid_count + test_count >= len(group_values):
        train_count = 1
        overflow = valid_count + test_count + train_count - len(group_values)
        while overflow > 0 and test_count > (1 if ratios[2] > 0 else 0):
            test_count -= 1
            overflow -= 1
        while overflow > 0 and valid_count > (1 if ratios[1] > 0 else 0):
            valid_count -= 1
            overflow -= 1
        if overflow > 0:
            raise ValueError("Sample_ID 分组数量太少，无法完成当前比例划分")
    train_count = len(group_values) - valid_count - test_count

    rng = np.random.default_rng(config.seed)
    label_to_groups: dict[int, list[str]] = {}
    for group, label in group_to_label.items():
        label_to_groups.setdefault(label, []).append(group)
    for groups in label_to_groups.values():
        rng.shuffle(groups)

    def take_stratified(count: int) -> set[str]:
        selected: set[str] = set()
        while len(selected) < count and any(label_to_groups.values()):
            labels_by_remaining = sorted(label_to_groups, key=lambda label: len(label_to_groups[label]), reverse=True)
            for label in labels_by_remaining:
                if len(selected) >= count:
                    break
                if label_to_groups[label]:
                    selected.add(label_to_groups[label].pop())
        return selected

    test_groups = take_stratified(test_count)
    valid_groups = take_stratified(valid_count)
    train_groups = {group for groups in label_to_groups.values() for group in groups}
    split_groups = {"train": train_groups, "valid": valid_groups, "test": test_groups}
    return {
        split: np.where(np.isin(sample_id, list(groups)))[0].tolist()
        for split, groups in split_groups.items()
    }


def _validate_external_test_dataset(train_labels: list[str], train_curve_length: int, test_dataset: Any) -> None:
    unknown_labels = sorted(set(test_dataset.labels).difference(train_labels))
    if unknown_labels:
        raise ValueError(f"测试集包含训练集中不存在的 Label: {', '.join(unknown_labels)}")
    test_lengths = {len(values) for values in test_dataset.x_axis}
    if test_lengths != {train_curve_length}:
        raise ValueError(f"测试集曲线长度必须与训练数据一致，训练长度 {train_curve_length}，测试集长度 {sorted(test_lengths)}")


def _axis_to_float_list(axis: Any, n_features: int) -> list[float]:
    try:
        values = np.asarray(axis, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=np.float64)
    if values.size != n_features:
        values = np.arange(n_features, dtype=np.float64)
    return [float(item) for item in values]


def _x_axis_warning(sample_axes: list[Any], n_features: int) -> dict[str, Any]:
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
    for split_name, indices in splits.items():
        if not indices:
            raise ValueError(f"{split_name} 集为空，请增加样品种类或调整划分比例")
    train_labels = set(np.unique(y[splits["train"]]).tolist())
    missing = [label for idx, label in enumerate(label_names) if idx not in train_labels]
    if missing:
        raise ValueError(f"训练集中缺少类别: {', '.join(missing)}。请增加样品种类或调整划分比例")


def _loader(x: np.ndarray, y: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
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
    tx1 = torch.tensor(x1[indices], dtype=torch.float32)
    tx2 = torch.tensor(x2[indices], dtype=torch.float32)
    ty = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tx1, tx2, ty), batch_size=batch_size, shuffle=shuffle)


def _single_2d_loader(x: np.ndarray, y: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
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
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values), dtype=float)
    decision = model.decision_function(values)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    shifted = decision - decision.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _evaluate_traditional_model(model: Any, x: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
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
    for sample in result.get("samples", []):
        sample["fold_index"] = int(fold_index)
    result["sample_count"] = len(result.get("samples", []))
    return result


def _merge_deep_sample_results(results: list[dict[str, Any]], x_axis_warning: dict[str, Any]) -> dict[str, Any]:
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
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        raw_index = row.get("Index", "")
        index_value = int(raw_index) if str(raw_index).isdigit() else str(raw_index)
        axis = sample_axes[position] if position < len(sample_axes) else []
        rows.append(
            {
                "index": index_value,
                "name": str(row.get("Name", "")),
                "sample_id": str(row.get("Sample_ID", "")),
                "sample_x_axis": _axis_to_float_list(axis, n_features),
            }
        )
    return rows


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clone_config(config: TrainConfig, **overrides: Any) -> TrainConfig:
    return TrainConfig(**{**config.__dict__, **overrides})


def _fit_x_normalizer(x_train: np.ndarray, mode: str) -> dict[str, Any]:
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
    return {key: (np.asarray(value).astype(float).tolist() if isinstance(value, np.ndarray) else value) for key, value in normalizer.items()}


def _dimension_band(n_features: int) -> str:
    if n_features <= 3000:
        return "1000-3000"
    if n_features <= 6000:
        return "3000-6000"
    return "6000-10000"


def _group_label_map(y: np.ndarray, sample_id: np.ndarray) -> dict[str, int]:
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
            seed=int(config.seed) + fold_index,
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
    return _classification_metrics_payload(
        np.asarray(eval_payload["true"], dtype=np.int64),
        np.asarray(eval_payload["pred"], dtype=np.int64),
        label_names,
    )


def _scalar_metric_summary(metric_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in METRIC_SCALAR_KEYS:
        values = [float(row[key]) for row in metric_rows if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def _scalar_metric_std(metric_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in METRIC_SCALAR_KEYS:
        values = [float(row[key]) for row in metric_rows if row.get(key) is not None]
        summary[key] = float(np.std(values)) if values else None
    return summary


def _aggregate_split_metrics(
    split_evals: list[dict[str, Any]],
    label_names: list[str],
) -> tuple[dict[str, Any], dict[str, float | None], dict[str, float | None]]:
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


def _traditional_candidate_configs(config: TrainConfig, model_type: str, n_features: int, y_train: np.ndarray) -> list[TrainConfig]:
    if model_type == "pls_da":
        raw = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        cap = max(1, min(len(y_train), n_features))
        return [_clone_config(config, pls_components=value) for value in raw if value <= cap]
    if model_type == "pca_lda":
        raw = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
        cap = max(1, min(len(y_train), n_features))
        return [_clone_config(config, pca_components=value) for value in raw if value <= cap]
    if model_type == "logistic_regression":
        return [_clone_config(config, logistic_c=value) for value in (0.1, 1.0, 10.0)]
    if model_type == "svm":
        return [_clone_config(config, svm_kernel="linear", svm_gamma="scale", svm_c=value) for value in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if model_type == "random_forest":
        n_estimators = int(config.random_forest_n_estimators)
        search_iterations = int(config.random_forest_search_iterations)
        if not 50 <= n_estimators <= 1000:
            raise ValueError("随机森林每组树数必须在 50 到 1000 之间")
        if not 1 <= search_iterations <= 18:
            raise ValueError("随机森林搜索候选数必须在 1 到 18 之间")
        candidates = ParameterSampler(
            {
                "random_forest_max_depth": [3, 5, 10],
                "random_forest_min_samples_leaf": [2, 5],
                "random_forest_max_features": ["sqrt", "log2", 0.1],
            },
            n_iter=search_iterations,
            random_state=config.seed,
        )
        return [
            _clone_config(
                config,
                random_forest_n_estimators=n_estimators,
                random_forest_oob_score=True,
                **params,
            )
            for params in candidates
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
    if model_type == "pca_lda":
        return {"pca_components": config.pca_components}
    if model_type == "logistic_regression":
        return {"logistic_c": config.logistic_c}
    if model_type == "svm":
        return {
            "svm_kernel": config.svm_kernel,
            "svm_c": config.svm_c,
            "svm_gamma": config.svm_gamma,
        }
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


def _select_traditional_config(
    config: TrainConfig,
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    label_names: list[str],
) -> TraditionalSelection:
    if model_type == "random_forest":
        return _select_random_forest_config(
            config,
            x_train,
            y_train,
            x_valid,
            y_valid,
            label_names,
        )
    best_model: Any | None = None
    best_config = config
    best_eval: dict[str, Any] | None = None
    best_balanced_accuracy: float | None = None
    search_rows: list[dict[str, Any]] = []
    candidates = _traditional_candidate_configs(config, model_type, x_train.shape[1], y_train)
    for candidate in candidates:
        model = build_traditional_model(candidate, y_train, len(label_names))
        model.fit(x_train, y_train)
        valid_eval = _evaluate_traditional_model(
            model,
            x_valid,
            y_valid,
            list(range(len(y_valid))),
            label_names,
        )
        params = _traditional_params(candidate, model_type)
        balanced_accuracy = float(valid_eval["balanced_accuracy"])
        macro_f1 = float(valid_eval["macro_f1"])
        search_rows.append(
            {
                "model_type": model_type,
                "is_selected": False,
                "valid_balanced_accuracy": balanced_accuracy,
                "valid_macro_f1": macro_f1,
                "valid_accuracy": float(valid_eval["accuracy"]),
                "selection_metric": "balanced_accuracy",
                "selection_score": balanced_accuracy,
                "oob_accuracy": None,
                "oob_balanced_accuracy": None,
                "params": params,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
            }
        )
        if best_balanced_accuracy is None or balanced_accuracy > best_balanced_accuracy + 1e-12:
            best_model = model
            best_config = candidate
            best_eval = valid_eval
            best_balanced_accuracy = balanced_accuracy
    if best_model is None or best_eval is None or best_balanced_accuracy is None:
        raise ValueError("传统模型验证集搜索未产生可用模型")
    selected_index = max(
        range(len(search_rows)),
        key=lambda index: (
            float(search_rows[index]["valid_balanced_accuracy"]),
            -index,
        ),
    )
    search_rows[selected_index]["is_selected"] = True
    return TraditionalSelection(
        config=best_config,
        valid_balanced_accuracy=best_balanced_accuracy,
        valid_macro_f1=float(best_eval["macro_f1"]),
        search_rows=search_rows,
        model=best_model,
        valid_eval=best_eval,
    )


def _random_forest_oob_metrics(model: Any, y_train: np.ndarray) -> tuple[float, float]:
    probabilities = np.asarray(model.oob_decision_function_, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(y_train):
        raise ValueError("随机森林未产生有效的袋外预测")
    covered = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
    if not np.any(covered):
        raise ValueError("随机森林袋外预测没有覆盖任何训练样本")
    classes = np.asarray(model.classes_)
    predicted = classes[np.argmax(probabilities[covered], axis=1)]
    true = np.asarray(y_train)[covered]
    return float(accuracy_score(true, predicted)), float(balanced_accuracy_score(true, predicted))


def _select_random_forest_config(
    config: TrainConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    label_names: list[str],
) -> TraditionalSelection:
    best_model: Any | None = None
    best_config: TrainConfig | None = None
    best_key: tuple[float, float, int] | None = None
    best_index: int | None = None
    search_rows: list[dict[str, Any]] = []
    candidates = _traditional_candidate_configs(config, "random_forest", x_train.shape[1], y_train)
    for index, candidate in enumerate(candidates):
        model = build_traditional_model(candidate, y_train, len(label_names))
        model.fit(x_train, y_train)
        oob_accuracy, oob_balanced_accuracy = _random_forest_oob_metrics(model, y_train)
        params = _traditional_params(candidate, "random_forest")
        search_rows.append(
            {
                "model_type": "random_forest",
                "is_selected": False,
                "valid_balanced_accuracy": None,
                "valid_macro_f1": None,
                "valid_accuracy": None,
                "selection_metric": "oob_balanced_accuracy",
                "selection_score": oob_balanced_accuracy,
                "oob_accuracy": oob_accuracy,
                "oob_balanced_accuracy": oob_balanced_accuracy,
                "params": params,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
            }
        )
        candidate_key = (oob_balanced_accuracy, oob_accuracy, -index)
        if best_key is None or candidate_key > best_key:
            best_model = model
            best_config = candidate
            best_key = candidate_key
            best_index = index
    if best_model is None or best_config is None or best_key is None or best_index is None:
        raise ValueError("随机森林袋外随机搜索未产生可用模型")
    valid_eval = _evaluate_traditional_model(
        best_model,
        x_valid,
        y_valid,
        list(range(len(y_valid))),
        label_names,
    )
    selected_row = search_rows[best_index]
    selected_row.update(
        {
            "is_selected": True,
            "valid_balanced_accuracy": float(valid_eval["balanced_accuracy"]),
            "valid_macro_f1": float(valid_eval["macro_f1"]),
            "valid_accuracy": float(valid_eval["accuracy"]),
        }
    )
    return TraditionalSelection(
        config=best_config,
        valid_balanced_accuracy=float(valid_eval["balanced_accuracy"]),
        valid_macro_f1=float(valid_eval["macro_f1"]),
        search_rows=search_rows,
        model=best_model,
        valid_eval=valid_eval,
    )


# 单折拟合：传统模型先选参再重训；深度模型按 validation loss 早停。
def _fit_traditional_fold(
    config: TrainConfig,
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    label_names: list[str],
) -> tuple[Any, TrainConfig, dict[str, Any], list[dict[str, Any]]]:
    selection = _select_traditional_config(
        config,
        model_type,
        x[splits["train"]],
        y[splits["train"]],
        x[splits["valid"]],
        y[splits["valid"]],
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
) -> tuple[nn.Module, list[dict[str, Any]], DSCARNetMappedInputs | None, dict[str, Any] | None]:
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
        )
        dscarnet_mapped.metadata["resolved_profile"] = dscarnet_profile
        dscarnet_mapping_metadata = save_dscarnet_mapping_artifacts(run_dir, dscarnet_mapped)
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
    for split_name in required:
        if not splits.get(split_name):
            raise ValueError(f"{split_name} 集为空，请增加样品种类或调整划分比例")
    train_labels = set(np.unique(y[splits["train"]]).tolist())
    missing = [label for idx, label in enumerate(label_names) if idx not in train_labels]
    if missing:
        raise ValueError(f"训练集中缺少类别: {', '.join(missing)}。请增加样品种类或调整划分比例")


def _sample_ids_for_split(sample_id: np.ndarray, indices: list[int]) -> list[str]:
    if not indices:
        return []
    return [str(item) for item in sorted(np.unique(sample_id[indices]).tolist(), key=natural_sort_key)]


def _holdout_fold(splits: dict[str, list[int]], sample_id: np.ndarray, *, strategy: str) -> dict[str, Any]:
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
    raw_config = dict(config_data or {})
    test_data_path = raw_config.get("test_data_path")
    policy = resolve_evaluation_policy(raw_config, has_external_test=bool(test_data_path))
    config_input = {key: value for key, value in raw_config.items() if key in TrainConfig().__dict__}
    config = TrainConfig(
        **{
            **TrainConfig().__dict__,
            **config_input,
            "split_mode": policy.strategy,
            "split_train": policy.split_train,
            "split_valid": policy.split_valid,
            "split_test": policy.split_test,
        }
    )
    _validate_split_ratio_config(policy)
    model_type = canonical_model_type(config.model_type)
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

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dataset = load_modeling_csv(data_path)
    sample_count = int(len(dataset.labels))
    x_raw = np.asarray(dataset.intensity, dtype=np.float32)
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels], dtype=np.int64)
    sample_id = dataset.frame["Sample_ID"].astype(str).to_numpy()
    test_dataset = load_modeling_csv(test_data_path) if test_data_path else None
    test_sample_count = 0
    if test_dataset is not None:
        _validate_external_test_dataset(label_names, x_raw.shape[1], test_dataset)
        test_x_raw = np.asarray(test_dataset.intensity, dtype=np.float32)
        test_y = np.asarray([label_to_id[label] for label in test_dataset.labels], dtype=np.int64)
        test_sample_id = test_dataset.frame["Sample_ID"].astype(str).to_numpy()
        test_sample_count = int(len(test_y))
        x_model_raw = np.vstack([x_raw, test_x_raw])
        y_model = np.concatenate([y, test_y])
        internal_splits = _split_indices(y, sample_id, config)
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
        metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1]) + _sample_metadata(test_dataset.frame, list(test_dataset.x_axis), x_raw.shape[1])
    else:
        x_model_raw = x_raw
        y_model = y
        if evaluation_strategy == "leave_one_sample_id_cv":
            folds = _leave_one_sample_id_folds(y, sample_id, config)
        else:
            splits = _split_indices(y, sample_id, config)
            _validate_required_splits(splits, y, label_names, ("train", "valid", "test"))
            folds = [_holdout_fold(splits, sample_id, strategy=evaluation_strategy)]
        metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1])
    feature_x_axis = dataset.x_axis[0] if dataset.x_axis else list(range(x_raw.shape[1]))
    combined_axes = list(dataset.x_axis) + (list(test_dataset.x_axis) if test_dataset is not None else [])
    x_axis_warning = _x_axis_warning(combined_axes, x_raw.shape[1])

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
    fold_count = len(folds)
    started_at = previous_status.get("started_at") or _now_iso()

    def write_progress(fold_index: int, completed_folds: int, fold: dict[str, Any]) -> None:
        check_run_active()
        if repository is not None and record is not None:
            progress = {
                "current_fold": fold_index,
                "completed_folds": completed_folds,
                "fold_progress_text": f"{fold_index}/{fold_count}",
                "target_epochs": config.epochs,
            }
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
            },
        )

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
        if model_family(model_type) == "traditional_ml":
            model, selected_config, valid_eval, search_rows = _fit_traditional_fold(config, model_type, x, y_model, splits, label_names)
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
            if config.feature_selection_enabled:
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

    metrics, cv_summary = _build_metrics_payload(fold_split_evals, label_names, evaluation_strategy)
    cv_metrics = {
        "strategy": evaluation_strategy,
        "fold_count": len(folds),
        "metrics": metrics,
        "cv_summary": cv_summary,
        "folds": cv_fold_payloads,
    }

    if model_family(model_type) == "traditional_ml":
        if config.feature_selection_enabled:
            sample_result = _merge_deep_sample_results(traditional_sample_results, x_axis_warning)
            sample_feature_summary = _write_sample_explainability_artifacts(
                run_dir=run_dir,
                sample_result=sample_result,
                x_axis_warning=x_axis_warning,
            )
        else:
            sample_feature_summary = _unsupported_explainability_summary(
                "特征区间识别已关闭",
                method="sample_occlusion_log_loss",
            )
    else:
        if final_deep_context is None:
            raise ValueError("深度模型训练未产生可解释性上下文")
        check_run_active()
        sample_result = _merge_deep_sample_results(deep_sample_results, x_axis_warning)
        sample_feature_summary = _write_sample_explainability_artifacts(
            run_dir=run_dir,
            sample_result=sample_result,
            x_axis_warning=x_axis_warning,
        )

    metadata_train_count = len(folds[0]["splits"].get("train", [])) if folds else 0
    if model_type == "dscarnet":
        profile_payload = build_dscarnet_profile(train_sample_count=metadata_train_count, feature_count=x_raw.shape[1])
    else:
        profile_payload = asdict(build_model_profile(model_type, train_sample_count=metadata_train_count, feature_count=x_raw.shape[1]))
    dscarnet_mode = str(config.dscarnet_input_mode or "dual").lower()
    explainability = explainability_method(model_type, dscarnet_mode=dscarnet_mode)
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
        "explainability_method": explainability,
        "artifact_explainability_method": (
            sample_feature_summary.get("method")
            or explainability
        ),
    }
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
        "explainability_method": explainability,
        "artifact_explainability_method": model_metadata["artifact_explainability_method"],
    }
    (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "model_metadata.json").write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(cv_fold_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(fold_metric_rows).to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    if last_model_family == "deep_learning":
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    search_columns = [
        "fold_index",
        "model_type",
        "is_selected",
        "selection_metric",
        "selection_score",
        "oob_accuracy",
        "oob_balanced_accuracy",
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

    status_payload = {
        **previous_status,
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "cv_summary": cv_summary,
        "history": history_rows,
        "model_type": model_type,
        "model_family": last_model_family,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_metadata": model_metadata,
        "model_profile": profile_payload,
        "explainability_method": explainability,
        "model_artifact": last_model_artifact,
        "model_artifact_note": (
            "最后一个交叉验证折模型，仅作下载参考，不用于汇报的交叉验证指标"
            if evaluation_strategy == "leave_one_sample_id_cv"
            else "本次 holdout 训练得到的模型，用于对应测试指标"
        ),
        "not_used_for_reported_cv_metrics": evaluation_strategy == "leave_one_sample_id_cv",
        "sample_feature_importance": sample_feature_summary,
        "x_axis_warning": x_axis_warning,
        "sample_count": sample_count,
        "curve_count": sample_count,
        "sample_id_count": int(len(set(dataset.sample_id))),
        "class_count": int(len(label_names)),
        "feature_count": int(x_raw.shape[1]),
        "test_sample_count": int(len(all_true)),
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
