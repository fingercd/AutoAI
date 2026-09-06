"""Safe batch state and ``model-comparison-v1`` projections."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .artifacts import ManifestCorruptError, RunArtifactWriter
from .contracts import RunRecord

COMPARISON_SCHEMA_VERSION = "model-comparison-v1"
_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def aggregate_batch_state(records: list[RunRecord]) -> str:
    states = [record.state for record in records]
    if not states or all(state == "queued" for state in states):
        return "queued"
    if any(state == "running" for state in states):
        return "running"
    if all(state == "succeeded" for state in states):
        return "succeeded"
    if all(state == "cancelled" for state in states):
        return "cancelled"
    if all(state in {"succeeded", "failed", "cancelled"} for state in states):
        return "partial" if any(state == "succeeded" for state in states) else "failed"
    return "queued"


def batch_status_payload(batch_id: str, records: list[RunRecord]) -> dict[str, Any]:
    counts = Counter(record.state for record in records)
    return {
        "batch_id": batch_id,
        "state": aggregate_batch_state(records),
        "run_count": len(records),
        "counts": {state: int(counts.get(state, 0)) for state in ("queued", "running", "succeeded", "failed", "cancelled")},
        "runs": [
            {
                "run_id": record.run_id,
                "model_type": record.config.get("model_type"),
                "repeat_index": record.batch_repeat_index,
                "state": record.state,
                "error": record.error if record.state == "failed" else None,
                "progress": {key: record.progress[key] for key in ('feature_scheme', 'current_fold', 'completed_folds', 'search_completed', 'search_total', 'current_epoch', 'training_stage_label') if key in record.progress},
            }
            for record in records
        ],
    }


def _json_artifact(run_dir: Path, run_id: str, name: str) -> dict[str, Any] | None:
    writer = RunArtifactWriter(run_dir)
    try:
        _manifest, descriptors = writer.descriptors(run_id=run_id)
    except (FileNotFoundError, ManifestCorruptError):
        return None
    descriptor = next((item for item in descriptors if item.get("name") == name), None)
    if not isinstance(descriptor, dict) or descriptor.get("integrity") != "ok":
        return None
    try:
        payload = json.loads((run_dir / name).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _csv_artifact(run_dir: Path, run_id: str, name: str) -> list[dict[str, str]] | None:
    writer = RunArtifactWriter(run_dir)
    try:
        _manifest, descriptors = writer.descriptors(run_id=run_id)
    except (FileNotFoundError, ManifestCorruptError):
        return None
    descriptor = next((item for item in descriptors if item.get("name") == name), None)
    if not isinstance(descriptor, dict) or descriptor.get("integrity") != "ok":
        return None
    try:
        with (run_dir / name).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _split_fingerprint(run_dir: Path, run_id: str) -> str | None:
    split = _json_artifact(run_dir, run_id, "split.json")
    if not isinstance(split, list):
        # split.json is an array, so use a separately verified light read here.
        writer = RunArtifactWriter(run_dir)
        try:
            _manifest, descriptors = writer.descriptors(run_id=run_id)
            descriptor = next(item for item in descriptors if item.get("name") == "split.json")
            if descriptor.get("integrity") != "ok":
                return None
            split = json.loads((run_dir / "split.json").read_text(encoding="utf-8-sig"))
        except (StopIteration, FileNotFoundError, ManifestCorruptError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    if not isinstance(split, list):
        return None
    canonical = [
        {
            "fold_index": item.get("fold_index"),
            "train_sample_ids": item.get("train_sample_ids"),
            "valid_sample_ids": item.get("valid_sample_ids"),
            "test_sample_ids": item.get("test_sample_ids"),
        }
        for item in split if isinstance(item, dict)
    ]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _run_payload(record: RunRecord, run_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]] | None, str | None, tuple[str, ...] | None]:
    metrics = _json_artifact(run_dir, record.run_id, "metrics.json")
    predictions = _csv_artifact(run_dir, record.run_id, "predictions.csv")
    label_map = _json_artifact(run_dir, record.run_id, "label_map.json")
    labels = None
    if isinstance(label_map, dict):
        labels = tuple(str(label_map[key]) for key in sorted(label_map, key=lambda key: int(key)))
    return metrics, predictions, _split_fingerprint(run_dir, record.run_id), labels


def _sample_predictions(rows: list[dict[str, str]]) -> tuple[dict[str, tuple[str, str]], str | None]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("Sample_ID") or "").strip()
        true_label = str(row.get("true_label") or "").strip()
        if not sample_id or not true_label:
            return {}, "预测文件缺少 Sample_ID 或 true_label"
        grouped[sample_id].append(row)
    result: dict[str, tuple[str, str]] = {}
    for sample_id, values in grouped.items():
        true_labels = {str(row.get("true_label")) for row in values}
        if len(true_labels) != 1:
            return {}, f"Sample_ID={sample_id} 包含多个真实类别"
        probability_keys = sorted(key for key in values[0] if key.startswith("prob_"))
        if probability_keys and all(all(key in row for key in probability_keys) for row in values):
            averages = {
                key: sum(float(row[key]) for row in values) / len(values)
                for key in probability_keys
            }
            predicted = max(averages, key=averages.get)[5:]
        else:
            predicted = Counter(str(row.get("pred_label") or "") for row in values).most_common(1)[0][0]
        result[sample_id] = (next(iter(true_labels)), predicted)
    return result, None


def project_model_comparison(
    *,
    batch_id: str,
    records: list[RunRecord],
    run_dir_for: Callable[[str], Path],
    include_history: bool = True,
) -> dict[str, Any]:
    """Aggregate only verified success artifacts; missing values are explicit."""
    successful = [record for record in records if record.state == "succeeded"]
    base = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "batch_id": batch_id,
        "state": aggregate_batch_state(records),
        "comparable": False,
        "reason": None,
        "models": [],
        "overall_metrics": {"metrics": list(_METRICS), "rows": []},
        "sample_correctness": {"status": "missing", "sample_ids": [], "models": [], "values": []},
        "repeat_stability": {"status": "missing", "reason": "没有可比较的成功训练"},
        "class_recall": {"status": "missing", "labels": [], "rows": []},
        "confusion_matrices": [],
    }
    if not successful:
        base["reason"] = "批次中尚无成功且结果完整的模型"
        return base
    payloads: list[tuple[RunRecord, dict[str, Any], list[dict[str, str]], str]] = []
    fingerprints: set[str] = set()
    strategies: set[str] = set()
    label_sets: set[tuple[str, ...]] = set()
    dataset_signatures: set[tuple[str | None, str | None]] = set()
    test_sample_sets: set[tuple[tuple[str, str], ...]] = set()
    for record in successful:
        metrics, predictions, fingerprint, labels = _run_payload(record, run_dir_for(record.run_id))
        test = metrics.get("test") if isinstance(metrics, dict) else None
        if not isinstance(test, dict) or predictions is None or fingerprint is None or labels is None:
            continue
        scalar_values = {key: _finite_metric(test.get(key)) for key in _METRICS}
        if any(value is None for value in scalar_values.values()):
            continue
        payloads.append((record, test, predictions, fingerprint))
        fingerprints.add(fingerprint)
        strategy = str(record.config.get("evaluation_strategy") or record.config.get("split_mode") or "")
        strategies.add(strategy)
        label_sets.add(labels)
        dataset_signatures.add((str(record.dataset_snapshot.get("sha256") or "") or None, str(record.config.get("test_dataset_sha256") or "") or None))
        primary_rows = [row for row in predictions if row.get("dataset") == "external_test"] if strategy == "leave_one_sample_id_cv_with_external_test" else predictions
        sample_map, sample_error = _sample_predictions(primary_rows)
        if sample_error is None:
            test_sample_sets.add(tuple(sorted((sample_id, true_label) for sample_id, (true_label, _predicted) in sample_map.items())))
    if not payloads:
        base["reason"] = "成功子 Run 缺少经过完整性校验的指标、预测或划分产物"
        return base
    if len(fingerprints) != 1 or len(strategies) != 1 or len(label_sets) != 1 or len(dataset_signatures) != 1 or len(test_sample_sets) != 1:
        base["reason"] = "子 Run 的数据快照、划分、测试样品/标签映射或评估口径不一致，不能进行公平比较"
        return base
    if include_history:
        grouped_records: dict[str, list[RunRecord]] = defaultdict(list)
        for record, _test, _rows, _fingerprint in payloads:
            grouped_records[str(record.config.get('model_type'))].append(record)
        if any(len(items) > 1 for items in grouped_records.values()):
            histories = {model: [project_model_comparison(batch_id=batch_id, records=[item], run_dir_for=run_dir_for, include_history=False) for item in sorted(items, key=lambda r: (r.batch_repeat_index or 0, r.run_id))] for model, items in grouped_records.items()}
            selected = [sorted(items, key=lambda r: (r.batch_repeat_index or 0, r.run_id))[0] for items in grouped_records.values()]
            result = project_model_comparison(batch_id=batch_id, records=selected, run_dir_for=run_dir_for, include_history=False)
            result['run_comparisons'] = histories
            result['state'] = aggregate_batch_state(records)
            return result
    by_model: dict[str, list[tuple[RunRecord, dict[str, Any], list[dict[str, str]]]]] = defaultdict(list)
    for record, test, predictions, _ in payloads:
        by_model[str(record.config.get("model_type"))].append((record, test, predictions))
    model_rows: list[dict[str, Any]] = []
    all_sample_predictions: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    recall_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    confusion_payloads: list[dict[str, Any]] = []
    for model_type, entries in by_model.items():
        metric_summary: dict[str, Any] = {}
        repeat_values: list[dict[str, Any]] = []
        for key in _METRICS:
            values = [float(test[key]) for _record, test, _rows in entries]
            metric_summary[key] = {
                "mean": float(sum(values) / len(values)), "std": float(math.sqrt(sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values))),
                "min": float(min(values)), "max": float(max(values)), "values": values,
            }
        matrix: list[list[int]] | None = None
        labels: list[str] = list(next(iter(label_sets)))
        for record, test, rows in entries:
            repeat_values.append({"run_id": record.run_id, "repeat_index": record.batch_repeat_index, **{key: test[key] for key in _METRICS}})
            # The external+LOSO strategy has both audit OOF rows and final
            # external-test rows.  Only the latter is its ranking/test object.
            strategy = str(record.config.get("evaluation_strategy") or record.config.get("split_mode") or "")
            primary_rows = [row for row in rows if row.get("dataset") == "external_test"] if strategy == "leave_one_sample_id_cv_with_external_test" else rows
            sample_map, error = _sample_predictions(primary_rows)
            if error is None:
                for sample_id, pair in sample_map.items():
                    all_sample_predictions[model_type][sample_id].append(pair)
            report = test.get("classification_report")
            if isinstance(report, dict):
                for label, item in report.items():
                    if isinstance(item, dict) and "recall" in item and label not in {"macro avg", "weighted avg"}:
                        recall = _finite_metric(item.get("recall"))
                        if recall is not None:
                            recall_values[model_type][label].append(recall)
            raw_matrix = test.get("confusion_matrix")
            if isinstance(raw_matrix, list):
                if matrix is None:
                    matrix = [[0 for _ in row] for row in raw_matrix]
                for row_index, row in enumerate(raw_matrix):
                    for column_index, value in enumerate(row if isinstance(row, list) else []):
                        matrix[row_index][column_index] += int(value)
        model_rows.append({
            "model_type": model_type,
            "successful_repeats": len(entries),
            "requested_repeats": max((record.batch_repeat_index or 0 for record in records if record.config.get("model_type") == model_type), default=0),
            "metrics": metric_summary,
            "repeats": repeat_values,
            "run_ids": [record.run_id for record, _test, _rows in entries],
            "experiment": _json_artifact(run_dir_for(entries[0][0].run_id), entries[0][0].run_id, 'feature_experiments.json'),
        })
        confusion_payloads.append({"model_type": model_type, "labels": labels, "confusion_matrix": matrix, "run_ids": [record.run_id for record, _test, _rows in entries]})
    model_rows.sort(key=lambda item: (-item["metrics"]["balanced_accuracy"]["mean"], -item["metrics"]["accuracy"]["mean"], -item["metrics"]["macro_f1"]["mean"], item["model_type"]))
    sample_ids = sorted({sample_id for model in all_sample_predictions.values() for sample_id in model})
    correctness_rows = []
    stability_rows = []
    for model in model_rows:
        model_type = model["model_type"]
        values: list[float | None] = []
        consistency: list[float] = []
        for sample_id in sample_ids:
            predictions = all_sample_predictions[model_type].get(sample_id, [])
            if not predictions:
                values.append(None)
                continue
            correct = sum(true == predicted for true, predicted in predictions)
            values.append(correct / len(predictions))
            consistency.append(Counter(predicted for _true, predicted in predictions).most_common(1)[0][1] / len(predictions))
        source_record, _test, source_rows = by_model[model_type][0]
        if strategies == {'leave_one_sample_id_cv_with_external_test'}:
            source_rows = [row for row in source_rows if row.get('dataset') == 'external_test']
        sample_map, _error = _sample_predictions(source_rows)
        details = []
        for sid in sample_ids:
            pair = sample_map.get(sid)
            measurements = [row for row in source_rows if str(row.get('Sample_ID')) == sid]
            details.append({'true_label': pair[0], 'pred_label': pair[1], 'measurement_count': len(measurements), 'aggregation': '各类别概率均值后取最大值' if measurements and any(key.startswith('prob_') for key in measurements[0]) else '重复测量预测投票'} if pair else None)
        correctness_rows.append({"model_type": model_type, "values": values, "details": details})
        if model["successful_repeats"] >= 2:
            stability_rows.append({"model_type": model_type, "mean": float(sum(consistency) / len(consistency)) if consistency else None, "min": float(min(consistency)) if consistency else None, "max": float(max(consistency)) if consistency else None})
    labels = list(next(iter(label_sets)))
    recall_rows = [{
        "model_type": model["model_type"],
        "values": [{"label": label, "mean": float(sum(recall_values[model["model_type"]][label]) / len(recall_values[model["model_type"]][label])), "std": float(np_std(recall_values[model["model_type"]][label]))} if recall_values[model["model_type"]].get(label) else None for label in labels],
    } for model in model_rows]
    for row in recall_rows:
        entries = by_model[row['model_type']]
        for item in row['values']:
            if item is not None:
                item['support'] = sum(test.get('classification_report', {}).get(item['label'], {}).get('support', 0) for _record, test, _rows in entries)
    base.update({
        "comparable": True,
        "models": model_rows,
        "overall_metrics": {"metrics": list(_METRICS), "rows": model_rows},
        "sample_correctness": {"status": "ready", "sample_ids": sample_ids, "models": [row["model_type"] for row in model_rows], "values": correctness_rows},
        "repeat_stability": {"status": "ready" if stability_rows else "not_applicable", "reason": None if stability_rows else "需要至少 2 次成功重复实验", "rows": stability_rows},
        "class_recall": {"status": "ready" if labels else "missing", "labels": labels, "rows": recall_rows},
        "confusion_matrices": confusion_payloads,
        "split_digest": next(iter(fingerprints)),
        "evaluation": {"strategy": next(iter(strategies)), "primary_aggregation": "pooled_oof" if next(iter(strategies)) == "leave_one_sample_id_cv" else "direct_external_test" if next(iter(strategies)) == "leave_one_sample_id_cv_with_external_test" else "direct"},
    })
    return base


def np_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
