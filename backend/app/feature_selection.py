from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch import nn


PredictFn = Callable[[np.ndarray], np.ndarray]
ScoreFn = Callable[[np.ndarray], np.ndarray]


FEATURE_COLUMNS = [
    "importance_metric",
    "rank",
    "window_index",
    "start_index",
    "end_index",
    "start_x",
    "end_x",
    "importance",
    "importance_std",
    "baseline_macro_f1",
    "permuted_macro_f1",
]

SAMPLE_FEATURE_COLUMNS = [
    "importance_metric",
    "sample_id",
    "fold_index",
    "dataset",
    "source_index",
    "index",
    "name",
    "repeat_index",
    "true_label",
    "pred_label",
    "correct",
    "true_probability",
    "pred_probability",
    "rank",
    "window_index",
    "start_index",
    "end_index",
    "start_x",
    "end_x",
    "importance",
    "normalized_importance",
    "original_loss",
    "masked_loss",
]

SAMPLE_FEATURE_LOSS_COLUMNS = ["original_loss", "masked_loss"]


def build_feature_windows(n_features: int, window_count: int) -> list[dict[str, int]]:
    """Split a spectrum into contiguous feature-importance windows."""
    if n_features <= 0:
        return []
    count = max(1, min(int(window_count), int(n_features)))
    windows = []
    for window_index, chunk in enumerate(np.array_split(np.arange(n_features), count)):
        if len(chunk) == 0:
            continue
        windows.append(
            {
                "window_index": int(window_index),
                "start_index": int(chunk[0]),
                "end_index": int(chunk[-1]),
            }
        )
    return windows


def merge_ranked_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent selected windows into larger feature segments."""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: (item["start_index"], item["end_index"]))
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for window in ordered:
        if current is None or window["start_index"] > current["end_index"] + 1:
            current = {
                "rank": int(window["rank"]),
                "start_index": int(window["start_index"]),
                "end_index": int(window["end_index"]),
                "start_x": float(window["start_x"]),
                "end_x": float(window["end_x"]),
                "importance": float(window["importance"]),
                **(
                    {"normalized_importance": float(window["normalized_importance"])}
                    if "normalized_importance" in window
                    else {}
                ),
                "window_count": 1,
                "window_ranks": [int(window["rank"])],
            }
            segments.append(current)
            continue
        current["end_index"] = int(window["end_index"])
        current["end_x"] = float(window["end_x"])
        current["importance"] = float(max(current["importance"], window["importance"]))
        if "normalized_importance" in current and "normalized_importance" in window:
            current["normalized_importance"] = float(
                max(current["normalized_importance"], window["normalized_importance"])
            )
        current["rank"] = int(min(current["rank"], window["rank"]))
        current["window_count"] = int(current["window_count"] + 1)
        current["window_ranks"].append(int(window["rank"]))
    return sorted(segments, key=lambda item: item["rank"])


def primary_feature_segment(
    top_segments: list[dict[str, Any]] | None,
    windows: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the single segment used by the UI highlight."""
    segments = list(top_segments or [])
    if segments:
        return dict(sorted(segments, key=lambda item: int(item.get("rank", 10**9)))[0])

    ranked_windows = [window for window in (windows or []) if "start_index" in window and "end_index" in window]
    if not ranked_windows:
        return None
    window = dict(sorted(ranked_windows, key=lambda item: int(item.get("rank", 10**9)))[0])
    rank = int(window.get("rank", 1))
    segment = {
        "rank": rank,
        "start_index": int(window["start_index"]),
        "end_index": int(window["end_index"]),
        "start_x": float(window.get("start_x", window["start_index"])),
        "end_x": float(window.get("end_x", window["end_index"])),
        "importance": float(window.get("importance", 0.0)),
        "window_count": 1,
        "window_ranks": [rank],
    }
    if "normalized_importance" in window:
        segment["normalized_importance"] = float(window["normalized_importance"])
    return segment


def unavailable_feature_importance(
    reason: str,
    *,
    x_axis: np.ndarray | list[float] | None = None,
    mean_curve: np.ndarray | list[float] | None = None,
    method: str = "interval_permutation_importance",
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "method": method,
        "x_axis": _float_list(x_axis),
        "mean_curve": _float_list(mean_curve),
        "windows": [],
        "top_segments": [],
    }


def interval_permutation_importance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_axis: np.ndarray | list[float],
    eval_indices: list[int],
    mean_indices: list[int],
    predict_fn: PredictFn,
    window_count: int = 50,
    top_k: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    eval_split: str = "valid",
) -> dict[str, Any]:
    """Compute supervised importance for contiguous spectral intervals."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    mean_curve = _mean_curve(x, mean_indices)

    if x.ndim != 2 or x.shape[1] == 0:
        return unavailable_feature_importance(
            "特征矩阵为空，无法计算重要区间",
            x_axis=x_axis_array,
            mean_curve=mean_curve,
        )
    if len(eval_indices) < 2:
        return unavailable_feature_importance(
            "评估集样本少于 2 个，无法稳定计算重要区间",
            x_axis=x_axis_array,
            mean_curve=mean_curve,
        )

    eval_indices_array = np.asarray(eval_indices, dtype=np.int64)
    y_eval = y[eval_indices_array]
    if np.unique(y_eval).size < 2:
        return unavailable_feature_importance(
            "评估集只有一个类别，无法计算判别性重要区间",
            x_axis=x_axis_array,
            mean_curve=mean_curve,
        )

    x_eval = x[eval_indices_array]
    baseline_pred = np.asarray(predict_fn(x_eval.copy()), dtype=np.int64)
    baseline_score = float(f1_score(y_eval, baseline_pred, average="macro", zero_division=0))
    windows = build_feature_windows(x.shape[1], window_count)
    repeats = max(1, int(n_repeats))
    rng = np.random.default_rng(seed)
    window_rows: list[dict[str, Any]] = []

    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index"])
        drops = []
        permuted_scores = []
        for _ in range(repeats):
            order = rng.permutation(len(eval_indices_array))
            if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
                order = np.roll(order, 1)
            x_permuted = x_eval.copy()
            x_permuted[:, start : end + 1] = x_permuted[order, start : end + 1]
            permuted_pred = np.asarray(predict_fn(x_permuted), dtype=np.int64)
            permuted_score = float(f1_score(y_eval, permuted_pred, average="macro", zero_division=0))
            permuted_scores.append(permuted_score)
            drops.append(baseline_score - permuted_score)

        importance = float(np.mean(drops))
        window_rows.append(
            {
                "window_index": int(window["window_index"]),
                "start_index": start,
                "end_index": end,
                "start_x": float(x_axis_array[start]),
                "end_x": float(x_axis_array[end]),
                "importance": importance,
                "importance_std": float(np.std(drops)),
                "baseline_macro_f1": baseline_score,
                "permuted_macro_f1": float(np.mean(permuted_scores)),
            }
        )

    ranked = sorted(window_rows, key=lambda item: (-item["importance"], item["start_index"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    ranked_by_index = sorted(ranked, key=lambda item: item["window_index"])
    top_windows = [row for row in ranked if row["importance"] > 0][: max(1, int(top_k))]
    top_segments = merge_ranked_windows(top_windows)
    return {
        "status": "ready",
        "method": "interval_permutation_importance",
        "eval_split": eval_split,
        "baseline_macro_f1": baseline_score,
        "window_count": len(windows),
        "top_k": int(top_k),
        "n_repeats": repeats,
        "x_axis": _float_list(x_axis_array),
        "mean_curve": _float_list(mean_curve),
        "windows": ranked_by_index,
        "top_segments": top_segments,
        "primary_segment": primary_feature_segment(top_segments, ranked_by_index),
    }


def sample_occlusion_importance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_axis: np.ndarray | list[float],
    splits: dict[str, list[int]],
    label_names: list[str],
    score_fn: ScoreFn,
    mean_indices: list[int],
    metadata: list[dict[str, Any]] | None = None,
    window_count: int = 50,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute per-sample interval attribution against each sample's true label."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    baseline_curve = _mean_curve(x, mean_indices)
    metadata = metadata or []

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {
            "status": "unavailable",
            "reason": "特征矩阵为空，无法计算单样品重要区间",
            "method": "sample_occlusion_importance",
            "baseline": "train_mean_curve",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "samples": [],
        }

    ordered_items = _ordered_test_indices(splits)
    if not ordered_items:
        return {
            "status": "unavailable",
            "reason": "没有可解释的测试集样品",
            "method": "sample_occlusion_importance",
            "baseline": "train_mean_curve",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "samples": [],
        }

    sample_indices = [idx for _split, idx in ordered_items]
    x_samples = x[sample_indices]
    y_samples = y[sample_indices]
    base_scores = _as_score_matrix(score_fn(x_samples.copy()))
    original_losses = _true_label_losses(base_scores, y_samples)
    windows = build_feature_windows(x.shape[1], window_count)
    top_limit = max(1, int(top_k))
    window_scores: list[np.ndarray] = []
    window_losses: list[np.ndarray] = []

    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index"])
        occluded = x_samples.copy()
        occluded[:, start : end + 1] = baseline_curve[start : end + 1]
        occluded_scores = _as_score_matrix(score_fn(occluded))
        masked_losses = _true_label_losses(occluded_scores, y_samples)
        window_scores.append(masked_losses - original_losses)
        window_losses.append(masked_losses)

    samples = []
    for local_idx, ((split_name, source_idx), true_class) in enumerate(zip(ordered_items, y_samples)):
        true_class_id = int(true_class)
        pred_class_id = int(np.argmax(base_scores[local_idx]))
        sample_meta = metadata[source_idx] if 0 <= source_idx < len(metadata) else {}
        sample_x_axis_array = _sample_x_axis_array(sample_meta, x_axis_array)
        rows = []
        for window, importances, masked_losses in zip(windows, window_scores, window_losses):
            start = int(window["start_index"])
            end = int(window["end_index"])
            true_prob = float(base_scores[local_idx, true_class_id])
            importance = float(importances[local_idx])
            rows.append(
                {
                    "window_index": int(window["window_index"]),
                    "start_index": start,
                    "end_index": end,
                    "start_x": float(sample_x_axis_array[start]),
                    "end_x": float(sample_x_axis_array[end]),
                    "importance": importance,
                    "original_loss": float(original_losses[local_idx]),
                    "masked_loss": float(masked_losses[local_idx]),
                }
            )
        importance_values = np.asarray([row["importance"] for row in rows], dtype=np.float64)
        min_importance = float(np.min(importance_values)) if importance_values.size else 0.0
        max_importance = float(np.max(importance_values)) if importance_values.size else 0.0
        importance_span = max_importance - min_importance
        for row in rows:
            row["normalized_importance"] = (
                float((row["importance"] - min_importance) / importance_span)
                if importance_span > 1e-12
                else 0.0
            )
        ranked = sorted(rows, key=lambda item: (-item["importance"], item["start_index"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = int(rank)
        top_windows = [row for row in ranked if row["importance"] > 0][:top_limit]
        ranked_by_index = sorted(ranked, key=lambda item: item["window_index"])
        top_segments = merge_ranked_windows(top_windows)
        samples.append(
            {
                "sample_id": f"{split_name}:{source_idx}",
                "dataset": split_name,
                "source_index": int(source_idx),
                "index": sample_meta.get("index", int(source_idx)),
                "name": str(sample_meta.get("name", f"sample_{source_idx}")),
                "repeat_index": str(sample_meta.get("repeat_index", "")),
                "true_class_id": true_class_id,
                "pred_class_id": pred_class_id,
                "true_label": _label_at(label_names, true_class_id),
                "pred_label": _label_at(label_names, pred_class_id),
                "correct": bool(pred_class_id == true_class_id),
                "true_probability": float(base_scores[local_idx, true_class_id]),
                "pred_probability": float(base_scores[local_idx, pred_class_id]),
                "curve": _float_list(x[source_idx]),
                "sample_x_axis": _float_list(sample_x_axis_array),
                "windows": ranked_by_index,
                "top_segments": top_segments,
                "primary_segment": primary_feature_segment(top_segments, ranked_by_index),
            }
        )

    return {
        "status": "ready",
        "method": "sample_occlusion_importance",
        "baseline": "train_mean_curve",
        "importance_metric": "masked_loss_minus_original_loss",
        "window_count": len(windows),
        "top_k": top_limit,
        "x_axis": _float_list(x_axis_array),
        "baseline_curve": _float_list(baseline_curve),
        "samples": samples,
    }


def sample_deep_attribution_importance(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_axis: np.ndarray | list[float],
    splits: dict[str, list[int]],
    label_names: list[str],
    metadata: list[dict[str, Any]] | None = None,
    model_type: str = "cnn1d",
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute per-sample deep-model attributions for the true class."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    baseline_curve = _mean_curve(x, splits.get("train", []))
    metadata = metadata or []
    method = "input_gradient_attribution" if model_type in {"transformer", "transformer1d"} else "gradcam_1d"

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {
            "status": "unavailable",
            "reason": "特征矩阵为空，无法计算单样品重要区间",
            "method": method,
            "baseline": "deep_attribution",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "samples": [],
        }

    ordered_items = _ordered_test_indices(splits)
    if not ordered_items:
        return {
            "status": "unavailable",
            "reason": "没有可解释的测试集样品",
            "method": method,
            "baseline": "deep_attribution",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "samples": [],
        }

    sample_indices = [idx for _split, idx in ordered_items]
    x_samples = x[sample_indices]
    y_samples = y[sample_indices]
    if method == "gradcam_1d":
        attributions, scores, actual_method = _gradcam_1d_attributions(model, x_samples, y_samples, x.shape[1])
        method = actual_method
    else:
        attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
    auxiliary_attributions = None
    if method == "gradcam_1d":
        auxiliary_attributions, _auxiliary_scores = _input_gradient_attributions(model, x_samples, y_samples)
    attributions = np.asarray(attributions, dtype=np.float64)
    if auxiliary_attributions is not None:
        auxiliary_attributions = np.asarray(auxiliary_attributions, dtype=np.float64)
    scores = _as_score_matrix(scores)
    top_limit = max(1, int(top_k))
    samples = []

    for local_idx, ((split_name, source_idx), true_class) in enumerate(zip(ordered_items, y_samples)):
        true_class_id = int(true_class)
        pred_class_id = int(np.argmax(scores[local_idx]))
        sample_meta = metadata[source_idx] if 0 <= source_idx < len(metadata) else {}
        sample_x_axis_array = _sample_x_axis_array(sample_meta, x_axis_array)
        rows, top_segments = _attribution_windows(
            attributions[local_idx],
            x_axis_array=sample_x_axis_array,
            top_k=top_limit,
        )
        sample_payload = {
            "sample_id": f"{split_name}:{source_idx}",
            "dataset": split_name,
            "source_index": int(source_idx),
            "index": sample_meta.get("index", int(source_idx)),
            "name": str(sample_meta.get("name", f"sample_{source_idx}")),
            "repeat_index": str(sample_meta.get("repeat_index", "")),
            "true_class_id": true_class_id,
            "pred_class_id": pred_class_id,
            "true_label": _label_at(label_names, true_class_id),
            "pred_label": _label_at(label_names, pred_class_id),
            "correct": bool(pred_class_id == true_class_id),
            "true_probability": float(scores[local_idx, true_class_id]),
            "pred_probability": float(scores[local_idx, pred_class_id]),
            "curve": _float_list(x[source_idx]),
            "sample_x_axis": _float_list(sample_x_axis_array),
            "windows": rows,
            "top_segments": top_segments,
            "primary_segment": primary_feature_segment(top_segments, rows),
        }
        if auxiliary_attributions is not None:
            _auxiliary_rows, auxiliary_top_segments = _attribution_windows(
                auxiliary_attributions[local_idx],
                x_axis_array=sample_x_axis_array,
                top_k=top_limit,
            )
            sample_payload["auxiliary_top_segments"] = auxiliary_top_segments
            sample_payload["sanity_checks"] = _sample_attribution_sanity(
                top_segments,
                auxiliary_top_segments,
                feature_count=x.shape[1],
            )
        samples.append(sample_payload)

    result = {
        "status": "ready",
        "method": method,
        "baseline": "deep_attribution",
        "importance_metric": "gradcam_activation" if method == "gradcam_1d" else "absolute_gradient_x_input",
        "window_count": int(x.shape[1]),
        "top_k": top_limit,
        "x_axis": _float_list(x_axis_array),
        "baseline_curve": _float_list(baseline_curve),
        "samples": samples,
    }
    if auxiliary_attributions is not None:
        result["sanity_checks"] = _aggregate_attribution_sanity(samples)
    return result


def sample_dscarnet_dual_2d_gradcam_importance(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_sar: np.ndarray,
    x_car: np.ndarray,
    pca: Any,
    sar_mapper: Any,
    car_mapper: Any,
    mapping_metadata: dict[str, Any],
    x_axis: np.ndarray | list[float],
    splits: dict[str, list[int]],
    label_names: list[str],
    metadata: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute DSCARNet SAR/CAR dual-branch 2D Grad-CAM and project it to 1D features."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_sar = np.asarray(x_sar, dtype=np.float32)
    x_car = np.asarray(x_car, dtype=np.float32)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    baseline_curve = _mean_curve(x, splits.get("train", []))
    metadata = metadata or []
    method = "dscarnet_dual_2d_gradcam"

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {
            "status": "unavailable",
            "reason": "特征矩阵为空，无法计算 DSCARNet 双通路重要区间",
            "method": method,
            "baseline": "deep_attribution",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "dscarnet_mapping": mapping_metadata,
            "samples": [],
        }

    ordered_items = _ordered_test_indices(splits)
    if not ordered_items:
        return {
            "status": "unavailable",
            "reason": "没有可解释的测试集样品",
            "method": method,
            "baseline": "deep_attribution",
            "x_axis": _float_list(x_axis_array),
            "baseline_curve": _float_list(baseline_curve),
            "dscarnet_mapping": mapping_metadata,
            "samples": [],
        }

    sample_indices = [idx for _split, idx in ordered_items]
    y_samples = y[sample_indices]
    sar_cams, car_cams, scores = _dscarnet_dual_2d_gradcam_attributions(
        model,
        x_sar[sample_indices],
        x_car[sample_indices],
        y_samples,
    )
    sar_feature_attr = _aggmap_cam_to_feature_matrix(sar_cams, sar_mapper, x.shape[1], "f_")
    car_component_attr = _aggmap_cam_to_feature_matrix(car_cams, car_mapper, pca.components_.shape[0], "pc_")
    component_count = min(car_component_attr.shape[1], pca.components_.shape[0])
    car_feature_attr = np.zeros_like(sar_feature_attr)
    if component_count > 0:
        car_feature_attr = car_component_attr[:, :component_count] @ np.abs(pca.components_[:component_count, :])
    sar_norm = _normalize_rows(sar_feature_attr)
    car_norm = _normalize_rows(car_feature_attr)
    combined_attributions = _normalize_rows((sar_norm + car_norm) / 2.0)
    scores = _as_score_matrix(scores)
    top_limit = max(1, int(top_k))
    samples = []

    for local_idx, ((split_name, source_idx), true_class) in enumerate(zip(ordered_items, y_samples)):
        true_class_id = int(true_class)
        pred_class_id = int(np.argmax(scores[local_idx]))
        sample_meta = metadata[source_idx] if 0 <= source_idx < len(metadata) else {}
        sample_x_axis_array = _sample_x_axis_array(sample_meta, x_axis_array)
        rows, top_segments = _attribution_windows(
            combined_attributions[local_idx],
            x_axis_array=sample_x_axis_array,
            top_k=top_limit,
        )
        _sar_rows, sar_top_segments = _attribution_windows(
            sar_norm[local_idx],
            x_axis_array=sample_x_axis_array,
            top_k=top_limit,
        )
        _car_rows, car_top_segments = _attribution_windows(
            car_norm[local_idx],
            x_axis_array=sample_x_axis_array,
            top_k=top_limit,
        )
        samples.append(
            {
                "sample_id": f"{split_name}:{source_idx}",
                "dataset": split_name,
                "source_index": int(source_idx),
                "index": sample_meta.get("index", int(source_idx)),
                "name": str(sample_meta.get("name", f"sample_{source_idx}")),
                "repeat_index": str(sample_meta.get("repeat_index", "")),
                "true_class_id": true_class_id,
                "pred_class_id": pred_class_id,
                "true_label": _label_at(label_names, true_class_id),
                "pred_label": _label_at(label_names, pred_class_id),
                "correct": bool(pred_class_id == true_class_id),
                "true_probability": float(scores[local_idx, true_class_id]),
                "pred_probability": float(scores[local_idx, pred_class_id]),
                "curve": _float_list(x[source_idx]),
                "sample_x_axis": _float_list(sample_x_axis_array),
                "windows": rows,
                "top_segments": top_segments,
                "primary_segment": primary_feature_segment(top_segments, rows),
                "sar_top_segments": sar_top_segments,
                "car_top_segments": car_top_segments,
                "sanity_checks": _dscarnet_branch_sanity(top_segments, sar_top_segments, car_top_segments),
            }
        )

    return {
        "status": "ready",
        "method": method,
        "baseline": "deep_attribution",
        "importance_metric": "sar_gradcam_plus_car_pca_backprojection",
        "window_count": int(x.shape[1]),
        "top_k": top_limit,
        "x_axis": _float_list(x_axis_array),
        "baseline_curve": _float_list(baseline_curve),
        "dscarnet_mapping": mapping_metadata,
        "samples": samples,
        "sanity_checks": _aggregate_dscarnet_branch_sanity(samples),
    }


def aggregate_sample_feature_importance(result: dict[str, Any]) -> dict[str, Any]:
    samples = result.get("samples", [])
    if result.get("status") != "ready" or not samples:
        unavailable = unavailable_feature_importance(
            result.get("reason") or "没有可聚合的单样品重要区间",
            x_axis=result.get("x_axis", []),
            mean_curve=result.get("baseline_curve", []),
            method=f"mean_{result.get('method') or 'sample_feature_importance'}",
        )
        if result.get("x_axis_warning"):
            unavailable["x_axis_warning"] = result.get("x_axis_warning")
        if result.get("sanity_checks"):
            unavailable["sanity_checks"] = result.get("sanity_checks")
        if result.get("importance_metric"):
            unavailable["importance_metric"] = result.get("importance_metric")
        if result.get("dscarnet_mapping"):
            unavailable["dscarnet_mapping"] = result.get("dscarnet_mapping")
        return unavailable

    first_windows = samples[0].get("windows", [])
    rows = []
    for window_idx, window in enumerate(first_windows):
        values = [
            float(sample.get("windows", [])[window_idx].get("importance", 0.0))
            for sample in samples
            if window_idx < len(sample.get("windows", []))
        ]
        importance = float(np.mean(values)) if values else 0.0
        rows.append(
            {
                "window_index": int(window.get("window_index", window_idx)),
                "start_index": int(window.get("start_index", window_idx)),
                "end_index": int(window.get("end_index", window_idx)),
                "start_x": float(window.get("start_x", window_idx)),
                "end_x": float(window.get("end_x", window_idx)),
                "importance": importance,
                "importance_std": float(np.std(values)) if values else 0.0,
                "baseline_macro_f1": None,
                "permuted_macro_f1": None,
            }
        )

    ranked = sorted(rows, key=lambda item: (-item["importance"], item["start_index"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    top_limit = max(1, int(result.get("top_k") or 5))
    top_windows = [row for row in ranked if row["importance"] > 0][:top_limit]
    ranked_by_index = sorted(ranked, key=lambda item: item["window_index"])
    top_segments = merge_ranked_windows(top_windows)
    payload = {
        "status": "ready",
        "method": f"mean_{result.get('method') or 'sample_feature_importance'}",
        "importance_metric": result.get("importance_metric"),
        "eval_split": "test",
        "window_count": len(rows),
        "top_k": top_limit,
        "n_repeats": None,
        "x_axis": result.get("x_axis", []),
        "mean_curve": result.get("baseline_curve", []),
        "windows": ranked_by_index,
        "top_segments": top_segments,
        "primary_segment": primary_feature_segment(top_segments, ranked_by_index),
    }
    if result.get("x_axis_warning"):
        payload["x_axis_warning"] = result.get("x_axis_warning")
    if result.get("sanity_checks"):
        payload["sanity_checks"] = result.get("sanity_checks")
    if result.get("dscarnet_mapping"):
        payload["dscarnet_mapping"] = result.get("dscarnet_mapping")
    return payload


def write_feature_importance_artifacts(run_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    json_path = run_path / "feature_importance.json"
    csv_path = run_path / "feature_importance.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        {"importance_metric": result.get("importance_metric"), **window}
        for window in result.get("windows", [])
    ]
    pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason"),
        "method": result.get("method"),
        "importance_metric": result.get("importance_metric"),
        "eval_split": result.get("eval_split"),
        "baseline_macro_f1": result.get("baseline_macro_f1"),
        "artifact": "feature_importance.json",
        "csv_artifact": "feature_importance.csv",
        "top_segments": result.get("top_segments", []),
        "primary_segment": result.get("primary_segment"),
        "x_axis_warning": result.get("x_axis_warning"),
        "sanity_checks": result.get("sanity_checks"),
        "dscarnet_mapping": result.get("dscarnet_mapping"),
    }


def write_sample_feature_importance_artifacts(run_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    json_path = run_path / "sample_feature_importance.json"
    csv_path = run_path / "sample_feature_importance.csv"
    result["sample_count"] = len(result.get("samples", []))
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    importance_metric = result.get("importance_metric")
    for sample in result.get("samples", []):
        base = {
            "importance_metric": importance_metric,
            "sample_id": sample.get("sample_id"),
            "fold_index": sample.get("fold_index"),
            "dataset": sample.get("dataset"),
            "source_index": sample.get("source_index"),
            "index": sample.get("index"),
            "name": sample.get("name"),
            "repeat_index": sample.get("repeat_index"),
            "true_label": sample.get("true_label"),
            "pred_label": sample.get("pred_label"),
            "correct": sample.get("correct"),
            "true_probability": sample.get("true_probability"),
            "pred_probability": sample.get("pred_probability"),
        }
        for window in sample.get("windows", []):
            rows.append({**base, **window})
    columns = list(SAMPLE_FEATURE_COLUMNS)
    if result.get("method") != "sample_occlusion_importance":
        columns = [column for column in columns if column not in SAMPLE_FEATURE_LOSS_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason"),
        "method": result.get("method"),
        "importance_metric": result.get("importance_metric"),
        "baseline": result.get("baseline"),
        "artifact": "sample_feature_importance.json",
        "csv_artifact": "sample_feature_importance.csv",
        "sample_count": len(result.get("samples", [])),
        "window_count": result.get("window_count"),
        "top_k": result.get("top_k"),
        "x_axis_warning": result.get("x_axis_warning"),
        "sanity_checks": result.get("sanity_checks"),
        "dscarnet_mapping": result.get("dscarnet_mapping"),
    }


def _safe_x_axis(x_axis: np.ndarray | list[float], n_features: int) -> np.ndarray:
    values = np.asarray(x_axis, dtype=np.float32).reshape(-1)
    if len(values) == n_features:
        return values
    return np.arange(n_features, dtype=np.float32)


def _mean_curve(x: np.ndarray, indices: list[int]) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    valid_indices = [idx for idx in indices if 0 <= idx < x.shape[0]]
    if not valid_indices:
        valid_indices = list(range(x.shape[0]))
    return np.mean(x[valid_indices], axis=0).astype(np.float32)


def _float_list(values: np.ndarray | list[float] | None) -> list[float]:
    if values is None:
        return []
    return [float(item) for item in np.asarray(values, dtype=np.float32).reshape(-1)]


def _sample_x_axis_array(sample_meta: dict[str, Any], fallback_axis: np.ndarray) -> np.ndarray:
    raw_axis = sample_meta.get("sample_x_axis")
    if raw_axis is None:
        return fallback_axis
    try:
        axis = np.asarray(raw_axis, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return fallback_axis
    return axis if axis.size == fallback_axis.size else fallback_axis


def _ordered_test_indices(splits: dict[str, list[int]]) -> list[tuple[str, int]]:
    return [("test", int(idx)) for idx in splits.get("test", [])]


def _as_score_matrix(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.ndim != 2:
        raise ValueError("score_fn 必须返回二维类别分数矩阵")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_fn 返回了非有限数值")
    return scores


def _true_label_losses(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    row_indices = np.arange(scores.shape[0])
    true_probs = np.clip(scores[row_indices, labels], 1e-12, 1.0)
    return -np.log(true_probs)


def _label_at(label_names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(label_names):
        return str(label_names[class_id])
    return str(class_id)


def _segment_bounds(segment: dict[str, Any] | None) -> tuple[int, int] | None:
    if not segment:
        return None
    try:
        return int(segment["start_index"]), int(segment["end_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _segments_overlap(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_bounds = _segment_bounds(left)
    right_bounds = _segment_bounds(right)
    if left_bounds is None or right_bounds is None:
        return False
    return left_bounds[0] <= right_bounds[1] and right_bounds[0] <= left_bounds[1]


def _is_edge_segment(segment: dict[str, Any] | None, feature_count: int) -> bool:
    bounds = _segment_bounds(segment)
    if bounds is None or feature_count <= 0:
        return False
    margin = max(1, int(np.ceil(feature_count * 0.03)))
    return bounds[0] <= margin or bounds[1] >= feature_count - 1 - margin


def _sample_attribution_sanity(
    primary_top_segments: list[dict[str, Any]],
    auxiliary_top_segments: list[dict[str, Any]],
    *,
    feature_count: int,
) -> dict[str, Any]:
    primary = primary_top_segments[0] if primary_top_segments else None
    auxiliary = auxiliary_top_segments[0] if auxiliary_top_segments else None
    primary_bounds = _segment_bounds(primary)
    auxiliary_bounds = _segment_bounds(auxiliary)
    return {
        "auxiliary_method": "input_gradient_attribution",
        "edge_top_segment": _is_edge_segment(primary, feature_count),
        "method_disagreement": bool(primary is not None and auxiliary is not None and not _segments_overlap(primary, auxiliary)),
        "primary_top_start_index": primary_bounds[0] if primary_bounds else None,
        "primary_top_end_index": primary_bounds[1] if primary_bounds else None,
        "auxiliary_top_start_index": auxiliary_bounds[0] if auxiliary_bounds else None,
        "auxiliary_top_end_index": auxiliary_bounds[1] if auxiliary_bounds else None,
    }


def _aggregate_attribution_sanity(samples: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [sample.get("sanity_checks") for sample in samples if sample.get("sanity_checks")]
    edge_count = sum(1 for item in checks if item.get("edge_top_segment"))
    disagreement_count = sum(1 for item in checks if item.get("method_disagreement"))
    warnings = []
    if edge_count:
        warnings.append(f"{edge_count} 个测试样品的 Grad-CAM 第一重要位置靠近谱线边界")
    if disagreement_count:
        warnings.append(f"{disagreement_count} 个测试样品的 Grad-CAM 与输入梯度第一重要位置不重叠")
    return {
        "status": "checked" if checks else "unavailable",
        "auxiliary_method": "input_gradient_attribution",
        "sample_count": len(checks),
        "edge_top_segment_sample_count": edge_count,
        "disagreement_sample_count": disagreement_count,
        "warnings": warnings,
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    spans = np.maximum(maxs - mins, 1e-12)
    normalized = (arr - mins) / spans
    flat_rows = (maxs <= mins + 1e-12).reshape(-1)
    normalized[flat_rows, :] = 0.0
    return normalized


def _parse_prefixed_index(value: Any, prefix: str) -> int | None:
    text = str(value)
    if not text.startswith(prefix):
        return None
    try:
        return int(text[len(prefix) :])
    except ValueError:
        return None


def _aggmap_positions(mapper: Any, feature_count: int, prefix: str) -> list[list[tuple[int, int]]]:
    positions: list[list[tuple[int, int]]] = [[] for _ in range(int(feature_count))]
    grid = getattr(mapper, "df_grid", None)
    if grid is not None and all(column in grid for column in ["x", "y", "v"]):
        for _row_idx, row in grid.iterrows():
            feature_idx = _parse_prefixed_index(row.get("v"), prefix)
            if feature_idx is None or not (0 <= feature_idx < feature_count):
                continue
            positions[feature_idx].append((int(row.get("y", 0)), int(row.get("x", 0))))
        return positions

    names = list(getattr(mapper, "feature_names_reshape", []) or [])
    fmap_shape = getattr(mapper, "fmap_shape", None)
    width = int(fmap_shape[1]) if fmap_shape and len(fmap_shape) >= 2 else int(np.ceil(np.sqrt(max(1, len(names)))))
    for flat_idx, name in enumerate(names):
        feature_idx = _parse_prefixed_index(name, prefix)
        if feature_idx is None or not (0 <= feature_idx < feature_count):
            continue
        positions[feature_idx].append((flat_idx // max(1, width), flat_idx % max(1, width)))
    return positions


def _aggmap_cam_to_feature_matrix(
    cams: np.ndarray,
    mapper: Any,
    feature_count: int,
    prefix: str,
) -> np.ndarray:
    cam_values = np.asarray(cams, dtype=np.float64)
    if cam_values.ndim != 3:
        raise ValueError(f"2D Grad-CAM 应为 (N,H,W)，实际 shape={cam_values.shape}")
    output = np.zeros((cam_values.shape[0], int(feature_count)), dtype=np.float64)
    positions = _aggmap_positions(mapper, int(feature_count), prefix)
    height, width = cam_values.shape[1], cam_values.shape[2]
    for feature_idx, coords in enumerate(positions):
        valid = [(y, x) for y, x in coords if 0 <= y < height and 0 <= x < width]
        if not valid:
            continue
        stacked = np.stack([cam_values[:, y, x] for y, x in valid], axis=1)
        output[:, feature_idx] = stacked.mean(axis=1)
    return output


def _dscarnet_dual_2d_gradcam_attributions(
    model: nn.Module,
    x_sar_samples: np.ndarray,
    x_car_samples: np.ndarray,
    y_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sar_cams, scores = _gradcam_2d_branch_attributions(
        model,
        x_sar_samples,
        x_car_samples,
        y_samples,
        branch=1,
    )
    car_cams, _car_scores = _gradcam_2d_branch_attributions(
        model,
        x_sar_samples,
        x_car_samples,
        y_samples,
        branch=2,
    )
    return sar_cams, car_cams, scores


def _gradcam_2d_branch_attributions(
    model: nn.Module,
    x_sar_samples: np.ndarray,
    x_car_samples: np.ndarray,
    y_samples: np.ndarray,
    *,
    branch: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = getattr(model, f"inception{branch}", None)
    if target is None or not isinstance(target, nn.Module):
        raise ValueError("DSCARNet 双通路模型缺少可用于 2D Grad-CAM 的 inception 分支")

    device = _model_device(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor) and output.requires_grad:
            captured["activation"] = output
            output.retain_grad()

    handle = target.register_forward_hook(capture_activation)
    try:
        sar_inputs = torch.tensor(x_sar_samples, dtype=torch.float32, device=device)
        car_inputs = torch.tensor(x_car_samples, dtype=torch.float32, device=device)
        labels = torch.tensor(y_samples, dtype=torch.long, device=device)
        logits = model(sar_inputs, car_inputs)
        activation = captured.get("activation")
        if activation is None or activation.ndim != 4:
            raise ValueError("DSCARNet 2D Grad-CAM 未捕获到四维激活")
        selected = logits[torch.arange(labels.shape[0], device=device), labels].sum()
        selected.backward()
        gradients = activation.grad
        if gradients is None:
            raise ValueError("DSCARNet 2D Grad-CAM 未捕获到梯度")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
        target_size = sar_inputs.shape[-2:] if branch == 1 else car_inputs.shape[-2:]
        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
        scores = torch.softmax(logits.detach(), dim=1)
        return cam.detach().cpu().numpy(), scores.cpu().numpy()
    finally:
        handle.remove()


def _dscarnet_branch_sanity(
    combined_top_segments: list[dict[str, Any]],
    sar_top_segments: list[dict[str, Any]],
    car_top_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    sar_top = sar_top_segments[0] if sar_top_segments else None
    car_top = car_top_segments[0] if car_top_segments else None
    combined_top = combined_top_segments[0] if combined_top_segments else None
    return {
        "auxiliary_method": "sar_car_branch_gradcam",
        "sar_car_top_disagreement": bool(sar_top and car_top and not _segments_overlap(sar_top, car_top)),
        "combined_overlaps_sar": _segments_overlap(combined_top, sar_top),
        "combined_overlaps_car": _segments_overlap(combined_top, car_top),
    }


def _aggregate_dscarnet_branch_sanity(samples: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [sample.get("sanity_checks") for sample in samples if sample.get("sanity_checks")]
    disagreement_count = sum(1 for item in checks if item.get("sar_car_top_disagreement"))
    warnings = []
    if disagreement_count:
        warnings.append(f"{disagreement_count} 个测试样品的 SAR 与 CAR 第一重要位置不重叠")
    return {
        "status": "checked" if checks else "unavailable",
        "auxiliary_method": "sar_car_branch_gradcam",
        "sample_count": len(checks),
        "sar_car_disagreement_sample_count": disagreement_count,
        "warnings": warnings,
    }


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _input_gradient_attributions(
    model: nn.Module,
    x_samples: np.ndarray,
    y_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = _model_device(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    inputs = torch.tensor(x_samples, dtype=torch.float32, device=device).unsqueeze(1)
    inputs.requires_grad_(True)
    labels = torch.tensor(y_samples, dtype=torch.long, device=device)
    logits = model(inputs)
    selected = logits[torch.arange(labels.shape[0], device=device), labels].sum()
    selected.backward()
    gradients = inputs.grad.detach()
    attributions = torch.abs(gradients * inputs.detach()).squeeze(1)
    scores = torch.softmax(logits.detach(), dim=1)
    return attributions.cpu().numpy(), scores.cpu().numpy()


def _gradcam_1d_attributions(
    model: nn.Module,
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    output_length: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    target = _find_gradcam_target(model)
    if target is None:
        attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
        return attributions, scores, "input_gradient_attribution"

    device = _model_device(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor) and output.requires_grad:
            captured["activation"] = output
            output.retain_grad()

    handle = target.register_forward_hook(capture_activation)
    try:
        inputs = torch.tensor(x_samples, dtype=torch.float32, device=device).unsqueeze(1)
        labels = torch.tensor(y_samples, dtype=torch.long, device=device)
        logits = model(inputs)
        activation = captured.get("activation")
        if activation is None or activation.ndim != 3:
            attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
            return attributions, scores, "input_gradient_attribution"
        selected = logits[torch.arange(labels.shape[0], device=device), labels].sum()
        selected.backward()
        gradients = activation.grad
        if gradients is None:
            attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
            return attributions, scores, "input_gradient_attribution"
        weights = gradients.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_length, mode="linear", align_corners=False).squeeze(1)
        scores = torch.softmax(logits.detach(), dim=1)
        return cam.detach().cpu().numpy(), scores.cpu().numpy(), "gradcam_1d"
    finally:
        handle.remove()


def _find_gradcam_target(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "inception") and isinstance(getattr(model, "inception"), nn.Module):
        return getattr(model, "inception")
    up_blocks = getattr(model, "up_blocks", None)
    if up_blocks is not None and len(up_blocks) > 0:
        return up_blocks[-1]
    downs = getattr(model, "downs", None)
    if downs is not None and len(downs) > 0:
        return downs[-1]
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    return conv_layers[-1] if conv_layers else None


def _attribution_windows(
    attribution: np.ndarray,
    *,
    x_axis_array: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = np.asarray(attribution, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return [], []
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max_value - min_value
    rows = []
    for index, importance in enumerate(values):
        normalized = float((importance - min_value) / span) if span > 1e-12 else 0.0
        rows.append(
            {
                "window_index": int(index),
                "start_index": int(index),
                "end_index": int(index),
                "start_x": float(x_axis_array[index]) if index < len(x_axis_array) else float(index),
                "end_x": float(x_axis_array[index]) if index < len(x_axis_array) else float(index),
                "importance": float(importance),
                "normalized_importance": normalized,
            }
        )
    ranked = sorted(rows, key=lambda item: (-item["importance"], item["start_index"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = int(rank)
    top_windows = [row for row in ranked if row["importance"] > 0][: max(1, int(top_k))]
    return sorted(ranked, key=lambda item: item["window_index"]), merge_ranked_windows(top_windows)
