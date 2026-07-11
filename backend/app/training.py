from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dscarnet_mapping import DSCARNetMappedInputs, fit_dscarnet_2d_mapping, save_dscarnet_mapping_artifacts
from .feature_selection import (
    aggregate_attribution_sanity,
    aggregate_dscarnet_branch_sanity,
    aggregate_sample_feature_importance,
    build_feature_windows,
    interval_permutation_importance,
    merge_ranked_windows,
    primary_feature_segment,
    sample_deep_attribution_importance,
    sample_dscarnet_dual_2d_gradcam_importance,
    sample_occlusion_importance,
    unavailable_feature_importance,
    write_feature_importance_artifacts,
    write_sample_feature_importance_artifacts,
)
from .models import build_deep_model, build_dscarnet_model, build_traditional_model, canonical_model_type, model_family
from .parsers import load_modeling_csv
from .paths import RUNS_DIR
from .runs.contracts import RunRecord
from .runs.artifacts import RunArtifactWriter
from .runs.repository import InvalidRunTransition, RunRepository
from .runs.status_projection import project_status


class TrainingRunReplaced(RuntimeError):
    """Raised when a training run has been paused because a newer run replaced it."""


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    normalization: str = "zscore"
    split_mode: str = "stratified"
    split_train: int = 8
    split_valid: int = 1
    split_test: int = 1
    class_balance: str = "none"
    model_type: str = "cnn1d"
    early_stopping_patience: int = 20
    dropout: float | None = None
    hidden_size: int = 64
    transformer_heads: int = 4
    unet_depth: int = 3
    dscarnet_inception_blocks: int = 1
    dscarnet_pca_components: int = 30
    dscarnet_cluster_channels: int = 9
    knn_n_neighbors: int = 5
    knn_weights: str = "distance"
    knn_metric: str = "minkowski"
    knn_p: int = 2
    random_forest_n_estimators: int = 100
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
    svm_kernel: str = "rbf"
    random_forest_max_features: str | float = "sqrt"
    xgboost_min_child_weight: float = 1.0
    xgboost_gamma: float = 0.0
    feature_selection_enabled: bool = True
    feature_window_count: int = 100
    feature_top_k: int = 5
    feature_n_repeats: int = 5
    feature_eval_split: str = "valid"


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
    status = _read_status_file((runs_dir or RUNS_DIR) / run_id / "status.json")
    return status.get("status") == "paused" and bool(status.get("replaced_by"))


def _raise_if_run_replaced(status_file: Path) -> None:
    status = _read_status_file(status_file)
    if status.get("status") != "paused":
        return
    replaced_by = status.get("replaced_by") or "newer_run"
    raise TrainingRunReplaced(f"训练任务 {status.get('run_id') or status_file.parent.name} 已被新任务 {replaced_by} 替换")


def _validate_split_ratio_config(config: TrainConfig) -> None:
    ratios = (int(config.split_train), int(config.split_valid), int(config.split_test))
    if any(value < 0 for value in ratios):
        raise ValueError("划分比例必须是非负整数")
    if sum(ratios) != 10:
        raise ValueError("训练、验证、测试比例相加必须等于 10")
    if ratios[0] <= 0:
        raise ValueError("训练集比例必须大于 0")


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


def _split_indices(labels: np.ndarray, repeat_index: np.ndarray, config: TrainConfig) -> dict[str, list[int]]:
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

    group_values = np.asarray(sorted(np.unique(repeat_index).tolist()))
    nonzero_splits = sum(1 for value in ratios if value > 0)
    if len(group_values) < nonzero_splits:
        raise ValueError(f"当前只有 {len(group_values)} 个 Repeat_index 分组，无法划分为 {nonzero_splits} 个非空集合")

    group_to_label: dict[str, int] = {}
    for group in group_values:
        group_labels = np.unique(labels[repeat_index == group])
        if len(group_labels) != 1:
            raise ValueError(f"Repeat_index={group} 内存在多个 Label，无法按组划分")
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
            raise ValueError("Repeat_index 分组数量太少，无法完成当前比例划分")
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
        split: np.where(np.isin(repeat_index, list(groups)))[0].tolist()
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


def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[indices], dtype=torch.float32).unsqueeze(1))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": probs.tolist(),
        "pred": pred.tolist(),
        "true": true.tolist(),
    }


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
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
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
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": np.asarray(probs, dtype=float).tolist(),
        "pred": np.asarray(pred, dtype=int).tolist(),
        "true": true.tolist(),
    }


def _compute_feature_importance(
    *,
    run_dir: Path,
    config: TrainConfig,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    x_axis: list[float],
    x_axis_warning: dict[str, Any],
    predict_fn: Any,
) -> dict[str, Any]:
    eval_split = config.feature_eval_split if config.feature_eval_split in splits else "valid"
    mean_indices = splits.get("train", [])
    if not config.feature_selection_enabled:
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        result = unavailable_feature_importance(
            "特征区间识别已关闭",
            x_axis=x_axis,
            mean_curve=mean_curve,
        )
        result["status"] = "disabled"
        result["eval_split"] = eval_split
        result["x_axis_warning"] = x_axis_warning
        return write_feature_importance_artifacts(run_dir, result)
    try:
        result = interval_permutation_importance(
            x,
            y,
            x_axis=x_axis,
            eval_indices=splits.get(eval_split, []),
            mean_indices=mean_indices,
            predict_fn=predict_fn,
            window_count=config.feature_window_count,
            top_k=config.feature_top_k,
            n_repeats=config.feature_n_repeats,
            seed=config.seed,
            eval_split=eval_split,
        )
    except Exception as exc:
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        result = unavailable_feature_importance(
            f"重要区间计算失败: {exc}",
            x_axis=x_axis,
            mean_curve=mean_curve,
        )
        result["status"] = "failed"
        result["eval_split"] = eval_split
    result["x_axis_warning"] = x_axis_warning
    return write_feature_importance_artifacts(run_dir, result)


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
            "method": "sample_occlusion_importance",
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
            "method": "sample_occlusion_importance",
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
            "method": "deep_attribution",
            "baseline": "deep_attribution",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "x_axis_warning": x_axis_warning,
            "samples": [],
        }

    try:
        if model_type == "dscarnet":
            if dscarnet_mapped is None or dscarnet_mapping_metadata is None:
                raise ValueError("DSCARNet 缺少 SAR/CAR 二维映射结果，无法计算双通路解释性")
            sample_result = sample_dscarnet_dual_2d_gradcam_importance(
                model,
                x,
                y,
                x_sar=dscarnet_mapped.x_sar,
                x_car=dscarnet_mapped.x_car,
                pca=dscarnet_mapped.pca,
                sar_mapper=dscarnet_mapped.sar_mapper,
                car_mapper=dscarnet_mapped.car_mapper,
                mapping_metadata=dscarnet_mapping_metadata,
                x_axis=x_axis,
                splits=splits,
                label_names=label_names,
                metadata=metadata,
                top_k=config.feature_top_k,
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
            "method": "deep_attribution",
            "baseline": "deep_attribution",
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


def _write_deep_explainability_artifacts(
    *,
    run_dir: Path,
    sample_result: dict[str, Any],
    x_axis_warning: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_result["sample_count"] = len(sample_result.get("samples", []))
    sample_summary = write_sample_feature_importance_artifacts(run_dir, sample_result)
    feature_result = aggregate_sample_feature_importance(sample_result)
    feature_result["x_axis_warning"] = x_axis_warning
    if sample_result.get("status") in {"disabled", "failed"}:
        feature_result["status"] = sample_result.get("status")
        feature_result["reason"] = sample_result.get("reason")
    feature_summary = write_feature_importance_artifacts(run_dir, feature_result)
    return feature_summary, sample_summary


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
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return _write_deep_explainability_artifacts(
        run_dir=run_dir,
        sample_result=sample_result,
        x_axis_warning=x_axis_warning,
    )


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
                "repeat_index": str(row.get("Repeat_index", "")),
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


def _group_label_map(y: np.ndarray, repeat_index: np.ndarray) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for group in sorted(np.unique(repeat_index).tolist()):
        labels = np.unique(y[repeat_index == group])
        if len(labels) != 1:
            raise ValueError(f"Repeat_index={group} 内存在多个 Label，无法按组划分")
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
        raise ValueError("Repeat_index 分组数量太少，无法在每个交叉验证折中保留包含全部类别的训练集")
    return sorted(selected, key=lambda item: train_valid_groups.index(item))


def _leave_one_repeat_index_folds(y: np.ndarray, repeat_index: np.ndarray, config: TrainConfig) -> list[dict[str, Any]]:
    groups = [str(item) for item in sorted(np.unique(repeat_index).tolist())]
    if len(groups) < 3:
        raise ValueError("交叉验证至少需要 3 个 Repeat_index 分组")
    group_to_label = _group_label_map(y, repeat_index)
    label_count = int(np.unique(y).size)
    train_valid_total = max(1, int(config.split_train) + int(config.split_valid))
    valid_ratio = max(0.0, min(1.0, float(config.split_valid) / train_valid_total))
    folds: list[dict[str, Any]] = []
    for fold_index, test_group in enumerate(groups, start=1):
        train_valid_groups = [group for group in groups if group != test_group]
        if len(set(group_to_label[group] for group in train_valid_groups)) < label_count:
            raise ValueError(f"Repeat_index={test_group} 留作测试后训练集缺少类别，无法完成分类评估")
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
                "test_repeat_index": test_group,
                "train_repeat_indices": train_groups,
                "valid_repeat_indices": valid_groups,
                "splits": {
                    "train": np.where(np.isin(repeat_index, train_groups))[0].tolist(),
                    "valid": np.where(np.isin(repeat_index, valid_groups))[0].tolist(),
                    "test": np.where(repeat_index == test_group)[0].tolist(),
                },
            }
        )
    return folds


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
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": report,
    }


METRIC_SCALAR_KEYS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")


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
) -> tuple[dict[str, Any], dict[str, float | None]]:
    metric_rows = [_metrics_from_eval(item, label_names) for item in split_evals]
    all_true = [int(value) for item in split_evals for value in item["true"]]
    all_pred = [int(value) for item in split_evals for value in item["pred"]]
    pooled = _classification_metrics_payload(
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        label_names,
    )
    payload = {**pooled, **_scalar_metric_summary(metric_rows)}
    for key in METRIC_SCALAR_KEYS:
        payload[f"pooled_{key}"] = pooled[key]
    return payload, _scalar_metric_std(metric_rows)


def _traditional_candidate_configs(config: TrainConfig, model_type: str, n_features: int, y_train: np.ndarray) -> list[TrainConfig]:
    band = _dimension_band(n_features)
    if model_type == "pls_da":
        raw = [1, 2, 3, 5] if band == "1000-3000" else ([1, 2, 3, 5, 8] if band == "3000-6000" else [1, 2, 3, 5, 8, 10])
        cap = max(1, min(max(raw), len(y_train) - 2, n_features))
        return [_clone_config(config, pls_components=value) for value in raw if value <= cap]
    if model_type == "svm":
        scale_gamma = max(1e-6, 1.0 / max(1, n_features))
        if band == "6000-10000":
            candidates = [("linear", "scale", 0.1), ("linear", "scale", 1.0), ("rbf", scale_gamma, 1.0)]
        elif band == "3000-6000":
            candidates = [("linear", "scale", 0.1), ("linear", "scale", 1.0), ("linear", "scale", 10.0), ("rbf", scale_gamma, 1.0)]
        else:
            candidates = [("linear", "scale", 0.1), ("linear", "scale", 1.0), ("linear", "scale", 10.0), ("rbf", scale_gamma, 1.0), ("rbf", scale_gamma * 10, 1.0)]
        return [_clone_config(config, svm_kernel=kernel, svm_gamma=gamma, svm_c=c) for kernel, gamma, c in candidates]
    if model_type == "random_forest":
        if band == "1000-3000":
            candidates = [(300, 3, "sqrt"), (300, 5, "log2"), (300, None, 0.2)]
        elif band == "3000-6000":
            candidates = [(500, 3, "sqrt"), (500, 5, "log2"), (500, 8, 0.1)]
        else:
            candidates = [(600, 3, "sqrt"), (600, 5, "log2"), (600, 5, 0.05)]
        return [
            _clone_config(config, random_forest_n_estimators=n, random_forest_max_depth=depth, random_forest_max_features=max_features)
            for n, depth, max_features in candidates
        ]
    if model_type == "xgboost":
        if band == "1000-3000":
            candidates = [(2, 0.6, 1, 1), (3, 1.0, 1, 5)]
        elif band == "3000-6000":
            candidates = [(2, 0.3, 3, 5), (3, 0.6, 5, 5)]
        else:
            candidates = [(2, 0.2, 5, 10), (2, 0.3, 10, 10)]
        return [
            _clone_config(config, xgboost_max_depth=depth, xgboost_colsample_bytree=colsample, xgboost_min_child_weight=child, xgboost_reg_lambda=reg_lambda)
            for depth, colsample, child, reg_lambda in candidates
        ]
    return [config]


def _fit_traditional_fold(
    config: TrainConfig,
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    label_names: list[str],
) -> tuple[Any, TrainConfig, dict[str, Any], list[dict[str, Any]]]:
    best_model: Any | None = None
    best_config = config
    best_eval: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []
    for candidate in _traditional_candidate_configs(config, model_type, x.shape[1], y[splits["train"]]):
        model = build_traditional_model(candidate, y[splits["train"]], len(label_names))
        model.fit(x[splits["train"]], y[splits["train"]])
        valid_eval = _evaluate_traditional_model(model, x, y, splits["valid"], label_names)
        row = {
            "model_type": model_type,
            "valid_macro_f1": valid_eval["macro_f1"],
            "valid_accuracy": valid_eval["accuracy"],
            "params": {
                "pls_components": candidate.pls_components,
                "svm_kernel": candidate.svm_kernel,
                "svm_c": candidate.svm_c,
                "svm_gamma": candidate.svm_gamma,
                "random_forest_n_estimators": candidate.random_forest_n_estimators,
                "random_forest_max_depth": candidate.random_forest_max_depth,
                "random_forest_max_features": candidate.random_forest_max_features,
                "xgboost_max_depth": candidate.xgboost_max_depth,
                "xgboost_colsample_bytree": candidate.xgboost_colsample_bytree,
                "xgboost_min_child_weight": candidate.xgboost_min_child_weight,
                "xgboost_reg_lambda": candidate.xgboost_reg_lambda,
            },
        }
        search_rows.append(row)
        if best_eval is None or valid_eval["macro_f1"] > best_eval["macro_f1"] + 1e-12:
            best_model = model
            best_config = candidate
            best_eval = valid_eval
    if best_model is None or best_eval is None:
        raise ValueError("传统模型验证集搜索未产生可用模型")
    return best_model, best_config, best_eval, search_rows


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
        dscarnet_mapped = fit_dscarnet_2d_mapping(
            x,
            splits["train"],
            pca_components=config.dscarnet_pca_components,
            cluster_channels=config.dscarnet_cluster_channels,
            seed=config.seed,
        )
        dscarnet_mapping_metadata = save_dscarnet_mapping_artifacts(run_dir, dscarnet_mapped)
        model = build_dscarnet_model(
            config,
            dscarnet_mapped.model_input_shape_sar,
            dscarnet_mapped.model_input_shape_car,
            len(label_names),
        )
    else:
        model = build_deep_model(config, input_length=x.shape[1], class_count=len(label_names), sample_count=sample_count)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        if dscarnet_mapped is not None:
            for bx1, bx2, by in _dual_loader(dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y, splits["train"], config.batch_size, True):
                optimizer.zero_grad()
                loss = criterion(model(bx1, bx2), by)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            valid_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y, splits["valid"], label_names)
        else:
            for bx, by in _loader(x, y, splits["train"], config.batch_size, True):
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            valid_eval = _evaluate(model, x, y, splits["valid"], label_names)
        improved = valid_eval["macro_f1"] > best_score + 1e-8
        if improved:
            best_score = valid_eval["macro_f1"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "valid_accuracy": valid_eval["accuracy"],
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


def _cv_f1_window_importance(
    *,
    fold_artifacts: list[dict[str, Any]],
    x_raw: np.ndarray,
    y: np.ndarray,
    x_axis: list[float],
    label_names: list[str],
    window_count: int,
    top_k: int,
    x_axis_warning: dict[str, Any],
) -> dict[str, Any]:
    x_axis_array = np.asarray(x_axis, dtype=np.float32).reshape(-1)
    if x_axis_array.size != x_raw.shape[1]:
        x_axis_array = np.arange(x_raw.shape[1], dtype=np.float32)
    baseline_true: list[int] = []
    baseline_pred: list[int] = []
    for artifact in fold_artifacts:
        baseline_true.extend(int(item) for item in artifact["test_true"])
        baseline_pred.extend(int(item) for item in artifact["test_pred"])
    baseline_score = float(f1_score(baseline_true, baseline_pred, average="macro", zero_division=0))
    windows = build_feature_windows(x_raw.shape[1], window_count)
    rows: list[dict[str, Any]] = []
    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index"])
        perturbed_true: list[int] = []
        perturbed_pred: list[int] = []
        for artifact in fold_artifacts:
            idxs = artifact["splits"]["test"]
            x_test = x_raw[idxs].copy()
            baseline_curve = artifact["train_mean_curve"]
            x_test[:, start : end + 1] = baseline_curve[start : end + 1]
            x_test_norm = _transform_x_with_normalizer(x_test, artifact["normalizer"])
            pred = np.asarray(artifact["model"].predict(x_test_norm), dtype=np.int64)
            perturbed_true.extend(int(item) for item in y[idxs])
            perturbed_pred.extend(int(item) for item in pred)
        perturbed_score = float(f1_score(perturbed_true, perturbed_pred, average="macro", zero_division=0))
        importance = baseline_score - perturbed_score
        rows.append(
            {
                "window_index": int(window["window_index"]),
                "start_index": start,
                "end_index": end,
                "start_x": float(x_axis_array[start]),
                "end_x": float(x_axis_array[end]),
                "importance": float(importance),
                "importance_std": 0.0,
                "baseline_macro_f1": baseline_score,
                "permuted_macro_f1": perturbed_score,
            }
        )
    ranked = sorted(rows, key=lambda item: (-item["importance"], item["start_index"]))
    values = np.asarray([row["importance"] for row in ranked], dtype=np.float64)
    min_value = float(values.min()) if values.size else 0.0
    max_value = float(values.max()) if values.size else 0.0
    span = max_value - min_value
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = int(rank)
        row["normalized_importance"] = float((row["importance"] - min_value) / span) if span > 1e-12 else 0.0
    top_windows = [row for row in ranked if row["importance"] > 0][: max(1, int(top_k))]
    ranked_by_index = sorted(ranked, key=lambda item: item["window_index"])
    top_segments = merge_ranked_windows(top_windows)
    return {
        "status": "ready",
        "method": "interval_permutation_importance",
        "importance_metric": "baseline_macro_f1_minus_perturbed_macro_f1",
        "eval_split": "outer_cv_test",
        "baseline_macro_f1": baseline_score,
        "window_count": len(windows),
        "top_k": int(top_k),
        "n_repeats": 1,
        "x_axis": [float(item) for item in x_axis_array],
        "mean_curve": [float(item) for item in np.mean(x_raw, axis=0)],
        "windows": ranked_by_index,
        "top_segments": top_segments,
        "primary_segment": primary_feature_segment(top_segments, ranked_by_index),
        "x_axis_warning": x_axis_warning,
    }


def _canonical_evaluation_strategy(config: TrainConfig, has_external_test: bool) -> str:
    if has_external_test:
        return "external_test_holdout"
    mode = str(config.split_mode or "stratified_holdout").strip().lower()
    if mode in {"leave_one_repeat_index_cv", "outer_leave_one_repeat_index_cv", "loocv", "loo"}:
        return "leave_one_repeat_index_cv"
    if mode in {"stratified", "stratified_holdout", "holdout"}:
        return "stratified_holdout"
    raise ValueError("split_mode 必须是 stratified_holdout、leave_one_repeat_index_cv 或 outer_leave_one_repeat_index_cv")


def _validate_required_splits(splits: dict[str, list[int]], y: np.ndarray, label_names: list[str], required: tuple[str, ...]) -> None:
    for split_name in required:
        if not splits.get(split_name):
            raise ValueError(f"{split_name} 集为空，请增加样品种类或调整划分比例")
    train_labels = set(np.unique(y[splits["train"]]).tolist())
    missing = [label for idx, label in enumerate(label_names) if idx not in train_labels]
    if missing:
        raise ValueError(f"训练集中缺少类别: {', '.join(missing)}。请增加样品种类或调整划分比例")


def _repeat_indices_for_split(repeat_index: np.ndarray, indices: list[int]) -> list[str]:
    if not indices:
        return []
    return [str(item) for item in sorted(np.unique(repeat_index[indices]).tolist())]


def _holdout_fold(splits: dict[str, list[int]], repeat_index: np.ndarray, *, strategy: str) -> dict[str, Any]:
    return {
        "fold_index": 1,
        "test_repeat_index": "holdout",
        "train_repeat_indices": _repeat_indices_for_split(repeat_index, splits["train"]),
        "valid_repeat_indices": _repeat_indices_for_split(repeat_index, splits["valid"]),
        "test_repeat_indices": _repeat_indices_for_split(repeat_index, splits.get("test", [])),
        "splits": splits,
        "strategy": strategy,
    }


def _external_test_fold(
    *,
    splits: dict[str, list[int]],
    repeat_index: np.ndarray,
    external_test_indices: list[int],
    external_repeat_index: np.ndarray,
) -> dict[str, Any]:
    return {
        "fold_index": 1,
        "test_repeat_index": "external_test",
        "train_repeat_indices": _repeat_indices_for_split(repeat_index, splits["train"]),
        "valid_repeat_indices": _repeat_indices_for_split(repeat_index, splits["valid"]),
        "test_repeat_indices": _repeat_indices_for_split(external_repeat_index, list(range(len(external_repeat_index)))),
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
    test_data_path = (config_data or {}).get("test_data_path")
    config_input = {key: value for key, value in (config_data or {}).items() if key in TrainConfig().__dict__}
    config = TrainConfig(**{**TrainConfig().__dict__, **config_input})
    _validate_split_ratio_config(config)
    model_type = canonical_model_type(config.model_type)
    evaluation_strategy = _canonical_evaluation_strategy(config, bool(test_data_path))
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
    repeat_index = dataset.frame["Repeat_index"].astype(str).to_numpy()
    test_dataset = load_modeling_csv(test_data_path) if test_data_path else None
    test_sample_count = 0
    if test_dataset is not None:
        _validate_external_test_dataset(label_names, x_raw.shape[1], test_dataset)
        test_x_raw = np.asarray(test_dataset.intensity, dtype=np.float32)
        test_y = np.asarray([label_to_id[label] for label in test_dataset.labels], dtype=np.int64)
        test_repeat_index = test_dataset.frame["Repeat_index"].astype(str).to_numpy()
        test_sample_count = int(len(test_y))
        x_model_raw = np.vstack([x_raw, test_x_raw])
        y_model = np.concatenate([y, test_y])
        internal_splits = _split_indices(y, repeat_index, _clone_config(config, split_test=0))
        _validate_required_splits(internal_splits, y, label_names, ("train", "valid"))
        external_indices = list(range(len(y), len(y_model)))
        folds = [
            _external_test_fold(
                splits=internal_splits,
                repeat_index=repeat_index,
                external_test_indices=external_indices,
                external_repeat_index=test_repeat_index,
            )
        ]
        metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1]) + _sample_metadata(test_dataset.frame, list(test_dataset.x_axis), x_raw.shape[1])
    else:
        x_model_raw = x_raw
        y_model = y
        if evaluation_strategy == "leave_one_repeat_index_cv":
            folds = _leave_one_repeat_index_folds(y, repeat_index, config)
        else:
            splits = _split_indices(y, repeat_index, config)
            _validate_required_splits(splits, y, label_names, ("train", "valid", "test"))
            folds = [_holdout_fold(splits, repeat_index, strategy=evaluation_strategy)]
        metadata = _sample_metadata(dataset.frame, list(dataset.x_axis), x_raw.shape[1])
    feature_x_axis = dataset.x_axis[0] if dataset.x_axis else list(range(x_raw.shape[1]))
    combined_axes = list(dataset.x_axis) + (list(test_dataset.x_axis) if test_dataset is not None else [])
    x_axis_warning = _x_axis_warning(combined_axes, x_raw.shape[1])

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    cv_fold_payloads: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    fold_artifacts: list[dict[str, Any]] = []
    fold_split_evals: list[dict[str, dict[str, Any]]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    last_model: Any | None = None
    last_model_family = model_family(model_type)
    last_model_artifact = "model.pkl" if last_model_family == "traditional_ml" else "model.pt"
    final_deep_context: dict[str, Any] | None = None
    deep_sample_results: list[dict[str, Any]] = []
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
                current_fold_repeat_index=fold.get("test_repeat_index"),
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
                "current_fold_repeat_index": fold.get("test_repeat_index"),
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
        if model_family(model_type) == "traditional_ml":
            model, selected_config, valid_eval, search_rows = _fit_traditional_fold(config, model_type, x, y_model, splits, label_names)
            check_run_active()
            best_search_rows.extend({**row, "fold_index": fold_index} for row in search_rows)
            train_eval = _evaluate_traditional_model(model, x, y_model, splits["train"], label_names)
            valid_eval = _evaluate_traditional_model(model, x, y_model, splits["valid"], label_names)
            test_eval = _evaluate_traditional_model(model, x, y_model, splits["test"], label_names)
            history_rows.append(
                {
                    "fold_index": fold_index,
                    "epoch": 1,
                    "train_loss": None,
                    "valid_accuracy": valid_eval["accuracy"],
                    "valid_macro_f1": valid_eval["macro_f1"],
                    "best_valid_macro_f1": valid_eval["macro_f1"],
                    "bad_epochs": 0,
                }
            )
            fold_artifacts.append(
                {
                    "model": model,
                    "normalizer": normalizer,
                    "splits": splits,
                    "test_true": test_eval["true"],
                    "test_pred": test_eval["pred"],
                    "train_mean_curve": np.mean(x_model_raw[splits["train"]], axis=0),
                    "selected_config": selected_config.__dict__,
                }
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
                sample_count=int(len(y_model)),
                cancel_check=check_run_active,
            )
            check_run_active()
            for row in history:
                history_rows.append({**row, "fold_index": fold_index})
            if dscarnet_mapped is not None:
                train_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["train"], label_names)
                valid_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["valid"], label_names)
                test_eval = _evaluate_dual(model, dscarnet_mapped.x_sar, dscarnet_mapped.x_car, y_model, splits["test"], label_names)
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
                "test_repeat_index": fold["test_repeat_index"],
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "weighted_f1": fold_metrics["weighted_f1"],
                "train_accuracy": split_metrics["train"]["accuracy"],
                "valid_accuracy": split_metrics["valid"]["accuracy"],
                "test_accuracy": fold_metrics["accuracy"],
            }
        )
        cv_fold_payloads.append(
            {
                "fold_index": fold_index,
                "test_repeat_index": fold["test_repeat_index"],
                "train_repeat_indices": fold["train_repeat_indices"],
                "valid_repeat_indices": fold["valid_repeat_indices"],
                "test_repeat_indices": fold.get("test_repeat_indices", []),
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
                "Repeat_index": source_metadata["repeat_index"],
                "true_label": label_names[test_eval["true"][local_idx]],
                "pred_label": label_names[test_eval["pred"][local_idx]],
            }
            for label, prob in zip(label_names, test_eval["probabilities"][local_idx]):
                row[f"prob_{label}"] = float(prob)
            prediction_rows.append(row)
        check_run_active()
        write_progress(fold_index, fold_index, fold)

    pooled_test_metrics = _classification_metrics_payload(np.asarray(all_true, dtype=np.int64), np.asarray(all_pred, dtype=np.int64), label_names)
    split_metrics_payload: dict[str, dict[str, Any]] = {}
    split_metric_std: dict[str, dict[str, float | None]] = {}
    for split_name in ("train", "valid", "test"):
        split_metrics_payload[split_name], split_metric_std[split_name] = _aggregate_split_metrics(
            [fold_eval[split_name] for fold_eval in fold_split_evals],
            label_names,
        )
    metrics = {**pooled_test_metrics, **split_metrics_payload}
    cv_summary = {
        "strategy": evaluation_strategy,
        "fold_count": len(folds),
        "pooled_test": pooled_test_metrics,
        "fold_mean": {
            split_name: _scalar_metric_summary([fold["split_metrics"][split_name] for fold in cv_fold_payloads])
            for split_name in ("train", "valid", "test")
        },
        "fold_std": split_metric_std,
    }
    cv_metrics = {
        "strategy": evaluation_strategy,
        "fold_count": len(folds),
        "metrics": metrics,
        "cv_summary": cv_summary,
        "folds": cv_fold_payloads,
    }

    if model_family(model_type) == "traditional_ml":
        if config.feature_selection_enabled:
            feature_result = _cv_f1_window_importance(
                fold_artifacts=fold_artifacts,
                x_raw=x_model_raw,
                y=y_model,
                x_axis=feature_x_axis,
                label_names=label_names,
                window_count=config.feature_window_count,
                top_k=config.feature_top_k,
                x_axis_warning=x_axis_warning,
            )
            feature_summary = write_feature_importance_artifacts(run_dir, feature_result)
        else:
            feature_summary = _unsupported_explainability_summary("特征区间识别已关闭", method="interval_permutation_importance")
        sample_feature_summary = _unsupported_explainability_summary(
            "无卷积模型和机器学习模型使用交叉验证的全局 macro-F1 下降解释，不生成单样本 F1 重要性",
            method="cv_macro_f1_drop_global_importance",
        )
    else:
        if final_deep_context is None:
            raise ValueError("深度模型训练未产生可解释性上下文")
        check_run_active()
        sample_result = _merge_deep_sample_results(deep_sample_results, x_axis_warning)
        feature_summary, sample_feature_summary = _write_deep_explainability_artifacts(
            run_dir=run_dir,
            sample_result=sample_result,
            x_axis_warning=x_axis_warning,
        )

    config_out = {
        **config.__dict__,
        "model_type": model_type,
        "data_path": str(Path(data_path).resolve()),
        "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
        "preprocess": {"mode": config.normalization, "fit_scope": "train_fold"},
        "evaluation_strategy": evaluation_strategy,
        "dimension_band": _dimension_band(x_raw.shape[1]),
        "fold_count": len(folds),
    }
    (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(cv_fold_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(fold_metric_rows).to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_search_rows).to_csv(run_dir / "hyperparameter_search.csv", index=False, encoding="utf-8-sig")
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
        "model_artifact": last_model_artifact,
        "model_artifact_note": (
            "最后一个交叉验证折模型，仅作下载参考，不用于汇报的交叉验证指标"
            if evaluation_strategy == "leave_one_repeat_index_cv"
            else "本次 holdout 训练得到的模型，用于对应测试指标"
        ),
        "not_used_for_reported_cv_metrics": evaluation_strategy == "leave_one_repeat_index_cv",
        "feature_importance": feature_summary,
        "sample_feature_importance": sample_feature_summary,
        "x_axis_warning": x_axis_warning,
        "sample_count": sample_count,
        "test_sample_count": int(len(all_true)),
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history_rows),
        "total_target_epochs": int(len(folds) * config.epochs),
        "current_fold": len(folds),
        "completed_folds": len(folds),
        "fold_progress_text": f"{len(folds)}/{len(folds)}",
        "current_fold_repeat_index": folds[-1].get("test_repeat_index") if folds else None,
        "best_valid_macro_f1": max((row.get("best_valid_macro_f1") or 0.0 for row in history_rows), default=None),
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


def train_model(data_path: str | Path, config_data: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    return run_legacy_training_compatibility(data_path=data_path, config_data=config_data, run_id=run_id)


def run_legacy_training_compatibility(
    *,
    data_path: str | Path,
    config_data: dict[str, Any] | None,
    run_id: str | None,
) -> dict[str, Any]:
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
    runs = []
    for path in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        status_file = path / "status.json"
        if status_file.exists():
            try:
                runs.append(json.loads(status_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                runs.append({"run_id": path.name, "status": "unknown"})
    return runs
