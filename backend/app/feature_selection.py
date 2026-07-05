from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PredictFn = Callable[[np.ndarray], np.ndarray]
ScoreFn = Callable[[np.ndarray], np.ndarray]


FEATURE_COLUMNS = [
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
    "sample_id",
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
                    "start_x": float(x_axis_array[start]),
                    "end_x": float(x_axis_array[end]),
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
        sample_meta = metadata[source_idx] if 0 <= source_idx < len(metadata) else {}
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
                "windows": sorted(ranked, key=lambda item: item["window_index"]),
                "top_segments": merge_ranked_windows(top_windows),
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


def write_feature_importance_artifacts(run_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    json_path = run_path / "feature_importance.json"
    csv_path = run_path / "feature_importance.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result.get("windows", []), columns=FEATURE_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason"),
        "method": result.get("method"),
        "eval_split": result.get("eval_split"),
        "baseline_macro_f1": result.get("baseline_macro_f1"),
        "artifact": "feature_importance.json",
        "csv_artifact": "feature_importance.csv",
        "top_segments": result.get("top_segments", []),
    }


def write_sample_feature_importance_artifacts(run_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    json_path = run_path / "sample_feature_importance.json"
    csv_path = run_path / "sample_feature_importance.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for sample in result.get("samples", []):
        base = {
            "sample_id": sample.get("sample_id"),
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
    pd.DataFrame(rows, columns=SAMPLE_FEATURE_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason"),
        "method": result.get("method"),
        "baseline": result.get("baseline"),
        "artifact": "sample_feature_importance.json",
        "csv_artifact": "sample_feature_importance.csv",
        "sample_count": len(result.get("samples", [])),
        "window_count": result.get("window_count"),
        "top_k": result.get("top_k"),
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
