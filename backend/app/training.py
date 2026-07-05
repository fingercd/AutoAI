from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .feature_selection import (
    interval_permutation_importance,
    sample_occlusion_importance,
    unavailable_feature_importance,
    write_feature_importance_artifacts,
    write_sample_feature_importance_artifacts,
)
from .models import build_deep_model, build_traditional_model, model_family
from .parsers import load_modeling_csv
from .paths import RUNS_DIR


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
    feature_selection_enabled: bool = True
    feature_window_count: int = 100
    feature_top_k: int = 5
    feature_n_repeats: int = 5
    feature_eval_split: str = "valid"


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
    if any(value < 0 for value in ratios) or sum(ratios) != 10:
        raise ValueError("划分比例必须是非负整数，且训练、验证、测试三项相加必须等于 10")
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

    valid_count = max(1, int(np.floor(len(group_values) * ratios[1] / 10))) if ratios[1] > 0 else 0
    test_count = max(1, int(np.floor(len(group_values) * ratios[2] / 10))) if ratios[2] > 0 else 0
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
    return write_feature_importance_artifacts(run_dir, result)


def _compute_sample_feature_importance(
    *,
    run_dir: Path,
    config: TrainConfig,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    x_axis: list[float],
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
    except Exception as exc:
        mean_curve = np.mean(x[mean_indices], axis=0) if mean_indices else np.mean(x, axis=0)
        result = {
            "status": "failed",
            "reason": f"单样品重要区间计算失败: {exc}",
            "method": "sample_occlusion_importance",
            "baseline": "train_mean_curve",
            "x_axis": [float(item) for item in np.asarray(x_axis, dtype=np.float32).reshape(-1)],
            "baseline_curve": [float(item) for item in np.asarray(mean_curve, dtype=np.float32).reshape(-1)],
            "samples": [],
        }
    return write_sample_feature_importance_artifacts(run_dir, result)


def _sample_metadata(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.iterrows():
        raw_index = row.get("Index", "")
        index_value = int(raw_index) if str(raw_index).isdigit() else str(raw_index)
        rows.append(
            {
                "index": index_value,
                "name": str(row.get("Name", "")),
                "repeat_index": str(row.get("Repeat_index", "")),
            }
        )
    return rows


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def train_model(data_path: str | Path, config_data: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    test_data_path = (config_data or {}).get("test_data_path")
    config_input = {key: value for key, value in (config_data or {}).items() if key in TrainConfig().__dict__}
    config = TrainConfig(**{**TrainConfig().__dict__, **config_input})
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir / "status.json"
    previous_status: dict[str, Any] = {}
    if status_file.exists():
        try:
            previous_status = json.loads(status_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_status = {}

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = load_modeling_csv(data_path)
    combined_x_axis = list(dataset.x_axis)
    sample_count = int(len(dataset.labels))
    x, norm_config = _normalize(dataset.intensity, config.normalization)
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels], dtype=np.int64)
    repeat_index = dataset.frame["Repeat_index"].astype(str).to_numpy()
    test_dataset = load_modeling_csv(test_data_path) if test_data_path else None
    test_sample_count = int(len(test_dataset.labels)) if test_dataset is not None else 0
    if test_dataset is not None:
        _validate_external_test_dataset(label_names, x.shape[1], test_dataset)
        split_config = TrainConfig(**{**config.__dict__, "split_valid": config.split_valid + config.split_test, "split_test": 0})
        splits = _split_indices(y, repeat_index, split_config)
        test_x, _test_norm_config = _normalize(test_dataset.intensity, config.normalization)
        test_y = np.asarray([label_to_id[label] for label in test_dataset.labels], dtype=np.int64)
        test_offset = len(y)
        x = np.concatenate([x, test_x], axis=0)
        y = np.concatenate([y, test_y], axis=0)
        combined_frame = pd.concat([dataset.frame, test_dataset.frame], ignore_index=True)
        combined_x_axis = list(dataset.x_axis) + list(test_dataset.x_axis)
        splits["test"] = list(range(test_offset, test_offset + len(test_y)))
    else:
        combined_frame = dataset.frame
        splits = _split_indices(y, repeat_index, config)
    _validate_splits(splits, y, label_names)
    feature_x_axis = combined_x_axis[0] if combined_x_axis else list(range(x.shape[1]))
    metadata = _sample_metadata(combined_frame)

    if model_family(config.model_type) == "traditional_ml":
        model = build_traditional_model(config, y[splits["train"]], len(label_names))
        model.fit(x[splits["train"]], y[splits["train"]])
        evaluations = {name: _evaluate_traditional_model(model, x, y, idx, label_names) for name, idx in splits.items()}
        metrics = {
            split: {key: value for key, value in result.items() if key not in {"probabilities", "pred", "true"}}
            for split, result in evaluations.items()
        }
        history = [
            {
                "epoch": 1,
                "train_loss": None,
                "train_accuracy": metrics["train"]["accuracy"],
                "valid_accuracy": metrics["valid"]["accuracy"],
                "valid_macro_f1": metrics["valid"]["macro_f1"],
                "best_valid_macro_f1": metrics["valid"]["macro_f1"],
                "bad_epochs": 0,
            }
        ]

        prediction_rows = []
        for split, idxs in splits.items():
            result = evaluations[split]
            for local_idx, source_idx in enumerate(idxs):
                row = {
                    "dataset": split,
                    "index": combined_frame.iloc[source_idx]["Index"],
                    "Repeat_index": combined_frame.iloc[source_idx]["Repeat_index"],
                    "true_label": label_names[result["true"][local_idx]],
                    "pred_label": label_names[result["pred"][local_idx]],
                }
                for label, prob in zip(label_names, result["probabilities"][local_idx]):
                    row[f"prob_{label}"] = float(prob)
                prediction_rows.append(row)

        feature_summary = _compute_feature_importance(
            run_dir=run_dir,
            config=config,
            x=x,
            y=y,
            splits=splits,
            x_axis=feature_x_axis,
            predict_fn=lambda values: np.asarray(model.predict(values), dtype=np.int64),
        )
        sample_feature_summary = _compute_sample_feature_importance(
            run_dir=run_dir,
            config=config,
            x=x,
            y=y,
            splits=splits,
            x_axis=feature_x_axis,
            label_names=label_names,
            metadata=metadata,
            score_fn=lambda values: _traditional_probabilities(model, values),
        )

        config_out = {
            **config.__dict__,
            "data_path": str(Path(data_path).resolve()),
            "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
            "preprocess": norm_config,
        }
        (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "split.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        with (run_dir / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh)

        status_payload = {
            **previous_status,
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "history": history,
            "model_type": config.model_type,
            "model_family": "traditional_ml",
            "model_artifact": "model.pkl",
            "feature_importance": feature_summary,
            "sample_feature_importance": sample_feature_summary,
            "sample_count": sample_count,
            "test_sample_count": test_sample_count,
            "label_names": label_names,
            "target_epochs": 1,
            "actual_epochs": 1,
            "best_valid_macro_f1": metrics["valid"]["macro_f1"],
            "config": config_out,
            "data_path": str(Path(data_path).resolve()),
            "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
            "completed_at": _now_iso(),
        }
        (run_dir / "status.json").write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "run_id": run_id,
            "status": "success",
            "metrics": metrics,
            "history": history,
            "model_type": config.model_type,
            "model_family": "traditional_ml",
            "model_artifact": "model.pkl",
            "feature_importance": feature_summary,
            "sample_feature_importance": sample_feature_summary,
            "sample_count": sample_count,
            "test_sample_count": test_sample_count,
            "label_names": label_names,
            "target_epochs": 1,
            "actual_epochs": 1,
            "best_valid_macro_f1": metrics["valid"]["macro_f1"],
            "run_dir": str(run_dir.resolve()),
        }

    class_weights = None
    if config.class_balance == "class_weight":
        counts = np.bincount(y[splits["train"]], minlength=len(label_names)).astype(np.float32)
        class_weights = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / len(label_names), dtype=torch.float32)

    model = build_deep_model(config, input_length=x.shape[1], class_count=len(label_names), sample_count=sample_count)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []
    best_score = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
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
        if config.early_stopping_patience > 0 and bad_epochs >= config.early_stopping_patience:
            history[-1]["early_stopped"] = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    evaluations = {name: _evaluate(model, x, y, idx, label_names) for name, idx in splits.items()}
    metrics = {
        split: {key: value for key, value in result.items() if key not in {"probabilities", "pred", "true"}}
        for split, result in evaluations.items()
    }

    prediction_rows = []
    for split, idxs in splits.items():
        result = evaluations[split]
        for local_idx, source_idx in enumerate(idxs):
            row = {
                "dataset": split,
                "index": combined_frame.iloc[source_idx]["Index"],
                "Repeat_index": combined_frame.iloc[source_idx]["Repeat_index"],
                "true_label": label_names[result["true"][local_idx]],
                "pred_label": label_names[result["pred"][local_idx]],
            }
            for label, prob in zip(label_names, result["probabilities"][local_idx]):
                row[f"prob_{label}"] = float(prob)
            prediction_rows.append(row)

    def score_deep(values: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_deep(values: np.ndarray) -> np.ndarray:
        return score_deep(values).argmax(axis=1)

    feature_summary = _compute_feature_importance(
        run_dir=run_dir,
        config=config,
        x=x,
        y=y,
        splits=splits,
        x_axis=feature_x_axis,
        predict_fn=predict_deep,
    )
    sample_feature_summary = _compute_sample_feature_importance(
        run_dir=run_dir,
        config=config,
        x=x,
        y=y,
        splits=splits,
        x_axis=feature_x_axis,
        label_names=label_names,
        metadata=metadata,
        score_fn=score_deep,
    )

    config_out = {
        **config.__dict__,
        "data_path": str(Path(data_path).resolve()),
        "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
        "preprocess": norm_config,
    }
    (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), run_dir / "model.pt")
    status_payload = {
        **previous_status,
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "history": history,
        "model_type": config.model_type,
        "model_family": "deep_learning",
        "model_artifact": "model.pt",
        "feature_importance": feature_summary,
        "sample_feature_importance": sample_feature_summary,
        "sample_count": sample_count,
        "test_sample_count": test_sample_count,
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history),
        "best_valid_macro_f1": best_score,
        "config": config_out,
        "data_path": str(Path(data_path).resolve()),
        "test_data_path": str(Path(test_data_path).resolve()) if test_data_path else None,
        "completed_at": _now_iso(),
    }
    (run_dir / "status.json").write_text(
        json.dumps(
            status_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "history": history,
        "model_type": config.model_type,
        "model_family": "deep_learning",
        "model_artifact": "model.pt",
        "feature_importance": feature_summary,
        "sample_feature_importance": sample_feature_summary,
        "sample_count": sample_count,
        "test_sample_count": test_sample_count,
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history),
        "best_valid_macro_f1": best_score,
        "run_dir": str(run_dir.resolve()),
    }


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
