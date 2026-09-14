"""生成单样品光谱特征重要性 artifact。

【模块职责与系统位置】
本模块是训练流水线的“可解释性”层，在模型训练完成后对测试集每个样品计算
“哪些谱段对预测最重要”，产出 sample_feature_importance.json/csv，
供结果页的单样品高亮展示。按模型类型分两条技术路线：
- 传统模型与无卷积深度模型（pca_mlp、cnn_transformer1d 等）：
  sample_occlusion_importance——窗口遮挡 + 真实类别 Log-loss 增量；
- 1D 卷积模型（cnn1d/resnet1d/inception1d/tcn1d 等）：
  sample_deep_attribution_importance——1D Grad-CAM，失败时回退输入梯度；

【关键算法约定】
窗口遮挡把特征轴解析成最接近请求数量且能整除长度的等宽窗口，并用训练集均值
替换窗口。重要性定义为真实类别 ``masked_loss - original_loss``，也就是
``log(p_before / p_after)``。卷积模型走 1D Grad-CAM。正值表示遮挡后真实类别置信度受损（该谱段重要）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from .training_explainability import explainability_method


ScoreFn = Callable[[np.ndarray], np.ndarray]


# sample_feature_importance.csv 的固定列顺序契约：结果页与测试都按此解析。
# 前段是样品级字段（result_id/标签/概率），后段是窗口级字段（区间/重要性/loss）。
SAMPLE_FEATURE_COLUMNS = [
    "importance_metric",
    "result_id",
    "sample_id",
    "fold_index",
    "dataset",
    "source_index",
    "index",
    "name",
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
    "masked_true_probability",
    "true_probability_drop",
]

# 只有窗口遮挡法（sample_occlusion_log_loss）才会产出这些 loss 明细列；
# 深度归因方法没有遮挡 loss，写 CSV 时要把这些列剔除（见 artifact 写出函数）。
SAMPLE_FEATURE_LOSS_COLUMNS = [
    "original_loss",
    "masked_loss",
    "masked_true_probability",
    "true_probability_drop",
]


# 窗口几何与结果分段：所有窗口必须等宽且完整覆盖特征轴。
def resolve_equal_width_window_count(n_features: int, window_count: int) -> int:
    """Choose the closest requested window count that divides the feature axis.

    窗口必须等宽且完整覆盖特征轴，因此窗口数必须是 n_features 的约数。
    本函数在全部约数中找与请求值 window_count 最接近的一个；距离相同时
    偏向更多窗口（保留更高的归因分辨率）。n_features<=0 时返回 0。
    """
    if n_features <= 0:
        return 0
    target = max(1, min(int(window_count), int(n_features)))
    divisors: set[int] = set()
    for candidate in range(1, int(np.sqrt(n_features)) + 1):
        if n_features % candidate != 0:
            continue
        divisors.add(candidate)
        divisors.add(n_features // candidate)
    # Prefer more windows when two divisors are equally close, preserving
    # attribution resolution without reintroducing unequal window widths.
    return min(divisors, key=lambda value: (abs(value - target), -value))


def build_feature_windows(n_features: int, window_count: int) -> list[dict[str, int]]:
    """Split a spectrum into equal-width contiguous importance windows.

    返回 [{window_index, start_index, end_index}, ...]，下标为闭区间、0 基，
    与 wide-feature 宽表的特征列顺序一一对应。
    """
    if n_features <= 0:
        return []
    count = resolve_equal_width_window_count(n_features, window_count)
    width = n_features // count
    windows = []
    for window_index in range(count):
        start = window_index * width
        windows.append(
            {
                "window_index": int(window_index),
                "start_index": int(start),
                "end_index": int(start + width - 1),
            }
        )
    return windows


def merge_ranked_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent selected windows into larger feature segments.

    把 top_k 中相邻（下标连续）的窗口合并成更大的“重要区段”，方便前端
    用连续区间高亮。合并策略：importance 取各窗口最大值（区段的代表性
    强度由最强窗口决定），rank 取最小值（排名由最强窗口决定），
    关联的 loss 明细跟随最强窗口。返回按 rank 升序的区段列表。
    """
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
            for key in (
                "original_loss",
                "masked_loss",
                "masked_true_probability",
                "true_probability_drop",
                "mean_true_probability_drop",
            ):
                if key in window:
                    current[key] = float(window[key])
            segments.append(current)
            continue
        current["end_index"] = int(window["end_index"])
        current["end_x"] = float(window["end_x"])
        window_is_stronger = float(window["importance"]) > float(current["importance"])
        # 相邻窗口并入当前区段：区段边界扩展，importance/rank 跟随更强窗口，
        # 保证区段的语义是“这一片里最强的信号”。
        current["importance"] = float(max(current["importance"], window["importance"]))
        if window_is_stronger:
            for key in (
                "original_loss",
                "masked_loss",
                "masked_true_probability",
                "true_probability_drop",
                "mean_true_probability_drop",
            ):
                if key in window:
                    current[key] = float(window[key])
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
    """Return the single segment used by the UI highlight.

    优先取 top_segments 中 rank 最小者；没有区段时从窗口列表中找
    importance>0 的最优窗口兜底；都没有则返回 None（前端不高亮）。
    """
    segments = list(top_segments or [])
    if segments:
        return dict(sorted(segments, key=lambda item: int(item.get("rank", 10**9)))[0])

    ranked_windows = [
        window
        for window in (windows or [])
        if "start_index" in window
        and "end_index" in window
        and float(window.get("importance", 0.0)) > 1e-12
    ]
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
    max_perturbed_rows: int = 256,
) -> dict[str, Any]:
    """Compute per-sample interval attribution against each sample's true label.

    算法（窗口遮挡 Log-loss 增量）：
    1) 取测试集样品的原始预测，计算真实类别 loss（-log p_true）；
    2) 逐窗口用训练集均值曲线替换该窗口的强度（遮挡）；
    3) 重新预测，importance = masked_loss - original_loss
       = log(p_before/p_after)，正值表示遮挡损害了真实类别置信度。

    参数：x/y 为全体样品的特征矩阵与编码标签；splits 提供 test 下标；
    score_fn 是“特征矩阵→类别概率矩阵”的已训练预测函数；mean_indices
    是构造遮挡基线所用的 train 下标；window_count/top_k 控制窗口数与
    返回的高亮区段数；max_perturbed_rows 限制单次前向的行数（内存保护）。
    返回 dict：status=ready/unavailable，含每个样品的 windows、
    top_segments、primary_segment。score_fn 返回形状不符时抛 ValueError。
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    # 遮挡基线用 train 均值曲线而不是全零：避免遮挡本身引入分布外输入，
    # 让 loss 变化只反映“该窗口信息被抹平”的影响。
    baseline_curve = _mean_curve(x, mean_indices)
    metadata = metadata or []

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {
            "status": "unavailable",
            "reason": "特征矩阵为空，无法计算单样品重要区间",
            "method": "sample_occlusion_log_loss",
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
            "method": "sample_occlusion_log_loss",
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
    # 实际窗口数可能小于请求值（见 resolve_equal_width_window_count 的约数约束）。
    windows = build_feature_windows(x.shape[1], window_count)
    top_limit = max(1, int(top_k))
    window_scores: list[np.ndarray] = []
    window_losses: list[np.ndarray] = []
    window_true_probabilities: list[np.ndarray] = []
    sample_count = max(1, len(x_samples))
    perturbed_row_limit = max(1, int(max_perturbed_rows))
    # 内存保护：一次遮挡批次的总行数（窗口数×样品数）不超过 max_perturbed_rows，
    # 据此反推每批能放多少个窗口，防止大测试集×多窗口时显存/内存爆掉。
    windows_per_batch = max(1, perturbed_row_limit // sample_count)
    true_row_indices = np.arange(len(y_samples))

    def score_perturbed(values: np.ndarray) -> np.ndarray:
        parts = [
            _as_score_matrix(score_fn(values[start : start + perturbed_row_limit]))
            for start in range(0, len(values), perturbed_row_limit)
        ]
        return np.concatenate(parts, axis=0)

    # 分批遮挡：每批把若干窗口的遮挡结果堆成一个大矩阵一次前向，
    # 再 reshape 回 (窗口, 样品, 类别) 逐窗口计算 loss 增量。
    for chunk_start in range(0, len(windows), windows_per_batch):
        chunk = windows[chunk_start : chunk_start + windows_per_batch]
        perturbed_blocks = []
        for window in chunk:
            start = int(window["start_index"])
            end = int(window["end_index"])
            occluded = x_samples.copy()
            # 核心遮挡操作：该窗口的强度整体替换为 train 均值。
            occluded[:, start : end + 1] = baseline_curve[start : end + 1]
            perturbed_blocks.append(occluded)
        stacked = np.concatenate(perturbed_blocks, axis=0)
        stacked_scores = score_perturbed(stacked)
        expected_rows = len(chunk) * len(x_samples)
        # 形状防御：score_fn 必须由调用方保证对任意行数返回 (N, n_classes) 概率，
        # 形状不符说明模型包装层有 bug，直接报错而不是静默错算重要性。
        if stacked_scores.shape[0] != expected_rows or stacked_scores.shape[1] != base_scores.shape[1]:
            raise ValueError("score_fn 返回的类别概率形状与遮挡批次不一致")
        chunk_scores = stacked_scores.reshape(len(chunk), len(x_samples), base_scores.shape[1])
        for occluded_scores in chunk_scores:
            masked_losses = _true_label_losses(occluded_scores, y_samples)
            # 重要性定义：masked_loss - original_loss = log(p_before/p_after)，
            # 正值 = 遮挡后真实类别更不可信 = 该窗口对识别该样品重要。
            window_scores.append(masked_losses - original_losses)
            window_losses.append(masked_losses)
            window_true_probabilities.append(
                np.clip(occluded_scores[true_row_indices, y_samples], 1e-12, 1.0)
            )

    samples = []
    for local_idx, ((split_name, source_idx), true_class) in enumerate(zip(ordered_items, y_samples)):
        true_class_id = int(true_class)
        pred_class_id = int(np.argmax(base_scores[local_idx]))
        sample_meta = metadata[source_idx] if 0 <= source_idx < len(metadata) else {}
        sample_x_axis_array = _sample_x_axis_array(sample_meta, x_axis_array)
        rows = []
        for window, importances, masked_losses, masked_true_probs in zip(
            windows,
            window_scores,
            window_losses,
            window_true_probabilities,
        ):
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
                    "masked_true_probability": float(masked_true_probs[local_idx]),
                    "true_probability_drop": float(true_prob - masked_true_probs[local_idx]),
                }
            )
        # normalized_importance：按本样品最大正值归一化到 [0,1]，仅用于前端着色，
        # 不改变 importance 的原始 log-loss 语义。
        importance_values = np.asarray([row["importance"] for row in rows], dtype=np.float64)
        max_importance = float(np.max(importance_values)) if importance_values.size else 0.0
        for row in rows:
            row["normalized_importance"] = (
                float(max(0.0, row["importance"]) / max_importance)
                if max_importance > 1e-12
                else 0.0
            )
        ranked = sorted(rows, key=lambda item: (-item["importance"], item["start_index"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = int(rank)
        top_windows = [row for row in ranked if row["importance"] > 0][:top_limit]
        ranked_by_index = sorted(ranked, key=lambda item: item["window_index"])
        # 相邻的高重要性窗口合并为连续区段，供前端区段级高亮。
        top_segments = merge_ranked_windows(top_windows)
        samples.append(
            {
                "result_id": f"{split_name}:{source_idx}",
                "sample_id": str(sample_meta.get("sample_id", "")),
                "dataset": split_name,
                "source_index": int(source_idx),
                "index": sample_meta.get("index", int(source_idx)),
                "name": str(sample_meta.get("name", f"sample_{source_idx}")),
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
        "method": "sample_occlusion_log_loss",
        "baseline": "train_mean_curve",
        "importance_metric": "masked_true_class_log_loss_minus_original_true_class_log_loss",
        "window_policy": "nearest_divisor_equal_width",
        "requested_window_count": int(window_count),
        "window_count": len(windows),
        "window_width": int(windows[0]["end_index"] - windows[0]["start_index"] + 1),
        "top_k": top_limit,
        "x_axis": _float_list(x_axis_array),
        "baseline_curve": _float_list(baseline_curve),
        "samples": samples,
    }


# 深度模型归因：1D 网络直接归因。
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
    """Compute per-sample deep-model attributions for the true class.

    按 model_type 经 explainability_method 选择归因方法：
    gradcam_1d（1D 卷积模型）或 input_gradient_attribution（无卷积深度模型）。
    Grad-CAM 计算失败时返回 status=failed 而不是中断训练结果写出；
    同时额外计算输入梯度归因作为 sanity check 对照（method_disagreement、
    edge_top_segment），用于发现“归因全落在谱线边缘”等不可信模式。
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    x_axis_array = _safe_x_axis(x_axis, x.shape[1] if x.ndim == 2 else 0)
    baseline_curve = _mean_curve(x, splits.get("train", []))
    metadata = metadata or []
    method = explainability_method(model_type)

    # 路由防御：无卷积模型（pca_mlp、cnn_transformer1d 等）的 Log-loss 遮挡法
    # 应走 sample_occlusion_importance；传错入口属于调用方 bug，直接报错。
    if method == "window_occlusion_log_loss":
        raise ValueError("该模型应使用 sample_occlusion_importance 计算 Log-loss 窗口重要性")

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
        # Grad-CAM 内部带降级链：无目标层/激活异常时回退输入梯度归因，
        # 只有显式声明了 gradcam_target_layer 却失效才抛错。
        try:
            attributions, scores, actual_method = _gradcam_1d_attributions(model, x_samples, y_samples, x.shape[1])
        except Exception as exc:
            return {
                "status": "failed",
                "reason": f"Grad-CAM 计算失败: {exc}",
                "method": "gradcam_1d",
                "baseline": "deep_attribution",
                "importance_metric": "gradcam_activation",
                "window_count": int(x.shape[1]),
                "top_k": max(1, int(top_k)),
                "x_axis": _float_list(x_axis_array),
                "baseline_curve": _float_list(baseline_curve),
                "samples": [],
            }
        method = actual_method
    else:
        attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
    auxiliary_attributions = None
    if method == "gradcam_1d":
        # 同时计算输入梯度归因作为对照：两种方法的第一重要区段若不重叠，
        # sanity_checks 会给出 method_disagreement 警告。
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
            "result_id": f"{split_name}:{source_idx}",
            "sample_id": str(sample_meta.get("sample_id", "")),
            "dataset": split_name,
            "source_index": int(source_idx),
            "index": sample_meta.get("index", int(source_idx)),
            "name": str(sample_meta.get("name", f"sample_{source_idx}")),
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






def write_sample_feature_importance_artifacts(run_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """写出逐样品、逐窗口的解释性 JSON/CSV，并返回状态摘要。

    JSON 保留完整嵌套结构（曲线、窗口、区段、sanity），是结果页的数据源；
    CSV 是“样品×窗口”的扁平长表，列序固定为 SAMPLE_FEATURE_COLUMNS，
    供 Excel 打开核对。返回的摘要 dict 会并入训练 result，不重复携带大数组。
    """
    run_path = Path(run_dir)
    json_path = run_path / "sample_feature_importance.json"
    csv_path = run_path / "sample_feature_importance.csv"
    # 把样品数写回 result 再落盘，让 JSON 自带计数，前端无需遍历统计。
    result["sample_count"] = len(result.get("samples", []))
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    importance_metric = result.get("importance_metric")
    for sample in result.get("samples", []):
        base = {
            "importance_metric": importance_metric,
            "result_id": sample.get("result_id"),
            "sample_id": sample.get("sample_id"),
            "fold_index": sample.get("fold_index"),
            "dataset": sample.get("dataset"),
            "source_index": sample.get("source_index"),
            "index": sample.get("index"),
            "name": sample.get("name"),
            "true_label": sample.get("true_label"),
            "pred_label": sample.get("pred_label"),
            "correct": sample.get("correct"),
            "true_probability": sample.get("true_probability"),
            "pred_probability": sample.get("pred_probability"),
        }
        for window in sample.get("windows", []):
            rows.append({**base, **window})
    columns = list(SAMPLE_FEATURE_COLUMNS)
    # 非遮挡法没有 original_loss/masked_loss 等明细，CSV 中剔除这些列，
    # 避免整列空值误导阅读。
    if result.get("method") not in {"sample_occlusion_importance", "sample_occlusion_log_loss"}:
        columns = [column for column in columns if column not in SAMPLE_FEATURE_LOSS_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(
        csv_path,
        index=False,
        # utf-8-sig 带 BOM，保证 Excel 直接打开 CSV 时中文列名/标签不乱码。
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


# 以下辅助函数负责坐标防御、sanity check和 Grad-CAM hook。
# X 轴防御：外部传入的 x_axis 长度与特征数不符时回退为 0..n-1 的整数轴，
# 保证后续 start_x/end_x 永远有值可取，不让脏轴数据中断解释性输出。
def _safe_x_axis(x_axis: np.ndarray | list[float], n_features: int) -> np.ndarray:
    values = np.asarray(x_axis, dtype=np.float32).reshape(-1)
    if len(values) == n_features:
        return values
    return np.arange(n_features, dtype=np.float32)


# 计算指定下标（通常是 train）的均值曲线作为遮挡基线；下标全无效时
# 退化为全体均值，保证基线永远存在。
def _mean_curve(x: np.ndarray, indices: list[int]) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    valid_indices = [idx for idx in indices if 0 <= idx < x.shape[0]]
    if not valid_indices:
        valid_indices = list(range(x.shape[0]))
    return np.mean(x[valid_indices], axis=0).astype(np.float32)


# 统一把数组转成 float list，便于 JSON 序列化（numpy 标量不能直接 json.dumps）。
def _float_list(values: np.ndarray | list[float] | None) -> list[float]:
    if values is None:
        return []
    return [float(item) for item in np.asarray(values, dtype=np.float32).reshape(-1)]


# 优先使用样品元数据里携带的 sample_x_axis（逐样品真实坐标）；
# 缺失、解析失败或长度不符都回退到批次公共轴。
def _sample_x_axis_array(sample_meta: dict[str, Any], fallback_axis: np.ndarray) -> np.ndarray:
    raw_axis = sample_meta.get("sample_x_axis")
    if raw_axis is None:
        return fallback_axis
    try:
        axis = np.asarray(raw_axis, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return fallback_axis
    return axis if axis.size == fallback_axis.size else fallback_axis


# 可解释性只针对 test 划分：返回 (split名, 源下标) 的有序列表，
# 后续所有样品循环都以它为准，保证 result_id 与数据行严格对齐。
def _ordered_test_indices(splits: dict[str, list[int]]) -> list[tuple[str, int]]:
    return [("test", int(idx)) for idx in splits.get("test", [])]


# 预测分数统一为 (N, n_classes) 二维 float64，并拒绝 NaN/Inf——
# 重要性建立在 log(p) 之上，非有限概率会让整个 artifact 失去意义。
def _as_score_matrix(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.ndim != 2:
        raise ValueError("score_fn 必须返回二维类别分数矩阵")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_fn 返回了非有限数值")
    return scores


# 逐样品真实类别的 -log(p)；clip 到 [1e-12, 1] 防止 log(0)=inf。
def _true_label_losses(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    row_indices = np.arange(scores.shape[0])
    true_probs = np.clip(scores[row_indices, labels], 1e-12, 1.0)
    return -np.log(true_probs)


# class_id 越界时退化为数字字符串，不让异常标签名中断结果写出。
def _label_at(label_names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(label_names):
        return str(label_names[class_id])
    return str(class_id)


# 安全提取区段下标边界；字段缺失/类型错误返回 None，由调用方按“无区段”处理。
def _segment_bounds(segment: dict[str, Any] | None) -> tuple[int, int] | None:
    if not segment:
        return None
    try:
        return int(segment["start_index"]), int(segment["end_index"])
    except (KeyError, TypeError, ValueError):
        return None


# 闭区间重叠判定：用于比较两种归因方法/两个分支的第一重要区段是否一致。
def _segments_overlap(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_bounds = _segment_bounds(left)
    right_bounds = _segment_bounds(right)
    if left_bounds is None or right_bounds is None:
        return False
    return left_bounds[0] <= right_bounds[1] and right_bounds[0] <= left_bounds[1]


# 判断区段是否贴近谱线两端（3% 边缘带内）。Grad-CAM 热点若总在边缘，
# 通常是基线/截断伪影而非真实化学信号，sanity check 会据此告警。
def _is_edge_segment(segment: dict[str, Any] | None, feature_count: int) -> bool:
    bounds = _segment_bounds(segment)
    if bounds is None or feature_count <= 0:
        return False
    margin = max(1, int(np.ceil(feature_count * 0.03)))
    return bounds[0] <= margin or bounds[1] >= feature_count - 1 - margin


# 单样品级 sanity：比较 Grad-CAM（primary）与输入梯度（auxiliary）的
# 第一重要区段——是否贴边缘、是否互不重叠（method_disagreement）。
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


# 把全部测试样品的 sanity 聚合成计数与中文警告，随结果一起展示。
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


def aggregate_attribution_sanity(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总普通深度模型归因的非零、边缘与重叠 sanity 指标。

    对外公开入口（训练流水线调用），内部委托 _aggregate_attribution_sanity。
    """
    return _aggregate_attribution_sanity(samples)




# 取模型所在设备；无参数模型（极端情况）回退 CPU。
def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# 统一两种输出形式的“真实类别得分 + 概率矩阵”：
# 单 logit 二分类按标签取 ±logit（sigmoid 概率），多分类取对应 logit
#（softmax 概率）；返回值前者带梯度用于 backward，后者已 detach。
def _selected_logits_and_scores(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 2 and logits.shape[1] == 1:
        signed = torch.where(labels == 1, logits[:, 0], -logits[:, 0])
        positive = torch.sigmoid(logits.detach())
        return signed.sum(), torch.cat((1.0 - positive, positive), dim=1)
    selected = logits[torch.arange(labels.shape[0], device=logits.device), labels].sum()
    return selected, torch.softmax(logits.detach(), dim=1)


# 输入梯度归因：对真实类别得分 backward，取 |grad × input| 作为特征强度；
# 同时充当 Grad-CAM 的 sanity 对照与无卷积模型的主归因方法。
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
    selected, scores = _selected_logits_and_scores(logits, labels)
    selected.backward()
    gradients = inputs.grad.detach()
    attributions = torch.abs(gradients * inputs.detach()).squeeze(1)
    return attributions.cpu().numpy(), scores.cpu().numpy()


# 1D Grad-CAM：目标层由 _find_gradcam_target 决定。模型显式声明了
# gradcam_target_layer 却拿不到目标/激活/梯度时抛错（声明即承诺）；
# 未声明时逐级回退到输入梯度归因，并把实际方法名一并返回。
def _gradcam_1d_attributions(
    model: nn.Module,
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    output_length: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    explicit_target = callable(getattr(model, "gradcam_target_layer", None))
    target = _find_gradcam_target(model)
    if target is None:
        if explicit_target:
            raise RuntimeError("模型已声明 gradcam_target_layer，但未返回可用目标层")
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
            if explicit_target:
                raise RuntimeError("Grad-CAM 目标层未产生 Batch×Channel×Length 激活")
            attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
            return attributions, scores, "input_gradient_attribution"
        selected, scores = _selected_logits_and_scores(logits, labels)
        selected.backward()
        gradients = activation.grad
        if gradients is None:
            if explicit_target:
                raise RuntimeError("Grad-CAM 目标层未捕获梯度")
            attributions, scores = _input_gradient_attributions(model, x_samples, y_samples)
            return attributions, scores, "input_gradient_attribution"
        # 通道权重 = 梯度在长度维上的全局平均池化（1D Grad-CAM）。
        weights = gradients.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_length, mode="linear", align_corners=False).squeeze(1)
        return cam.detach().cpu().numpy(), scores.cpu().numpy(), "gradcam_1d"
    finally:
        handle.remove()


# 目标层解析顺序：模型自定义 gradcam_target_layer() → inception 属性 →
# up_blocks/downs 末块 → 最后一个 Conv1d；都找不到返回 None（走梯度回退）。
def _find_gradcam_target(model: nn.Module) -> nn.Module | None:
    resolver = getattr(model, "gradcam_target_layer", None)
    if callable(resolver):
        target = resolver()
        if target is None:
            return None
        if not isinstance(target, nn.Module):
            raise TypeError("gradcam_target_layer() 必须返回 torch.nn.Module 或 None")
        return target
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


# 把逐点归因向量包装成“每点一个窗口”的窗口列表（与遮挡法输出同构），
# 做 min-max 归一、排序赋 rank、取 top_k 合并区段，供统一的结果契约消费。
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
