"""Freeze the formal benchmark only after offline, source-bound readiness checks.

The command reads prepared manifests, record facts, telemetry, task tracking receipts,
and approval files. It never starts the benchmark driver, training, or an LLM call.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.benchmark_driver import (
    BenchmarkDriver,
    DEFAULT_ROOT,
    MODELS,
    TRADITIONAL,
    identity,
    tracking_id,
)


PROTOCOL_FILE = "approval/protocol.json"
DATA_FILE = "data/manifest.json"
SOURCE_FILE = "approval/source-freeze.json"
LLM_ASSET_FILE = "approval/llm-asset.json"
BUDGETS = (1, 3)
PROMPT_SOURCE_PATHS = (
    "agent_poc/orchestration/llm.py",
    "agent_poc/orchestration/projection.py",
)


class FormalFreezeBlocked(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = sorted(set(missing))
        super().__init__("formal benchmark freeze blocked")


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str, missing: list[str]) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        missing.append(f"missing {label}: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        missing.append(f"unreadable {label}: {path} ({type(exc).__name__})")
    return None


def _inside(root: Path, path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _tracking_id(suite: str, task_id: str, source_digest: str) -> str:
    return f"{suite}:{task_id}::{source_digest[:12]}"


def _verify_tracking(root: Path, task: dict, source_digest: str, missing: list[str]) -> dict | None:
    online_id = _tracking_id(task["suite"], task["task_id"], source_digest)
    directory = root / "tracking" / "tasks" / hashlib.sha256(online_id.encode("utf-8")).hexdigest()
    task_path, verified_path = directory / "task.json", directory / "verified.json"
    receipt = _read_json(task_path, f"tracking task {task['task_id']}", missing)
    from backend.app.benchmark_tracking import cloud_required
    if not cloud_required(root):
        if receipt is None:
            return None
        if (receipt.get("task_id") != online_id or receipt.get("finished") is not True
                or receipt.get("config", {}).get("source_digest") != source_digest):
            missing.append(f"local task journal incomplete or binding differs: {task['task_id']}")
        llm = receipt.get("config", {}).get("llm_config")
        return dict(task_id=task["task_id"], suite=task["suite"], tracking_id=online_id,
            verified=receipt.get("finished") is True, verification_scope="local_only",
            cloud_verified=False, cloud_requirement_waived_by_user=True,
            finish_event_id=receipt.get("finish_event_id"), cloud_ref=None, verified_at=None,
            tracking_task_sha256=_sha256(task_path), verified_receipt_sha256=None, llm_config=llm)
    verified = _read_json(verified_path, f"cloud verification {task['task_id']}", missing)
    if receipt is None or verified is None:
        return None
    if receipt.get("task_id") != online_id:
        missing.append(f"tracking identity differs for task {task['task_id']}")
    if receipt.get("finished") is not True:
        missing.append(f"tracking task is not finished: {task['task_id']}")
    config = receipt.get("config") or {}
    if config.get("source_digest") != source_digest:
        missing.append(f"tracking source digest differs: {task['task_id']}")
    llm_config = config.get("llm_config")
    if not isinstance(llm_config, Mapping):
        missing.append(f"tracking LLM config summary is missing: {task['task_id']}")
        llm_config = None
    if verified.get("verified") is not True:
        missing.append(f"cloud completion not verified: {task['task_id']}")
    if not verified.get("finish_event_id") or verified.get("finish_event_id") != receipt.get("finish_event_id"):
        missing.append(f"cloud finish event does not match task receipt: {task['task_id']}")
    if not verified.get("cloud_ref") or not isinstance(verified.get("cloud_summary"), Mapping):
        missing.append(f"cloud run evidence is incomplete: {task['task_id']}")
    if verified.get("summary") != receipt.get("summary"):
        missing.append(f"cloud final summary differs from task receipt: {task['task_id']}")
    if not isinstance(verified.get("verified_numeric_summary"), Mapping):
        missing.append(f"cloud numeric summary is absent: {task['task_id']}")
    return {
        "task_id": task["task_id"],
        "suite": task["suite"],
        "tracking_id": online_id,
        "verified": verified.get("verified") is True,
        "finish_event_id": verified.get("finish_event_id"),
        "cloud_ref": verified.get("cloud_ref"),
        "verified_at": verified.get("verified_at"),
        "tracking_task_sha256": _sha256(task_path),
        "verified_receipt_sha256": _sha256(verified_path),
        "llm_config": dict(llm_config) if isinstance(llm_config, Mapping) else None,
    }


def _protocol_summary(protocol: dict) -> dict:
    fields = (
        "protocol_id", "date", "status", "task", "primary_metric", "secondary_metrics",
        "split_seed", "training_seed", "hpo_seed", "group_unit", "official_test",
        "without_official_test", "selection_boundary", "model_eligibility",
        "primary_comparisons", "primary_agent_modules", "hpo", "predictions_and_aggregation",
        "development_and_locking", "logging", "scope_limit", "execution", "weights",
    )
    return {key: protocol[key] for key in fields if key in protocol}


def _source_summary(source: dict, prompt_hashes: dict[str, str]) -> dict:
    return {
        "source_digest": source["source_digest"],
        "snapshot_path": source["snapshot_path"],
        "head": source.get("head"),
        "tag": source.get("tag"),
        "dirty_patch_sha256": source.get("dirty_patch_sha256"),
        "data_manifest_sha256": source.get("data_manifest_sha256"),
        "source_file_count": len(source["source_files"]),
        "source_files_digest": _json_digest(source["source_files"]),
        "prompt_sources": prompt_hashes,
    }


def _verify_source(root: Path, source: dict, missing: list[str]) -> dict[str, str]:
    digest = source.get("source_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        missing.append("source-freeze.json has no valid source_digest")
        return {}
    source_files = source.get("source_files")
    if not isinstance(source_files, Mapping):
        missing.append("source-freeze.json has no source_files hash map")
        return {}
    if _json_digest(dict(source_files)) != digest:
        missing.append("source digest does not match the frozen source_files map")
    snapshot = Path(source.get("snapshot_path") or "")
    if not snapshot.is_dir():
        missing.append(f"frozen source snapshot is missing: {snapshot}")
        return {}
    for relative, expected in source_files.items():
        path = (snapshot / relative).resolve()
        if not _inside(snapshot, path):
            missing.append(f"frozen source path escapes snapshot: {relative}")
        elif not path.is_file() or _sha256(path) != expected:
            missing.append(f"frozen source file hash differs: {relative}")
    prompt_hashes = {}
    prompt_paths = list(PROMPT_SOURCE_PATHS)
    if "backend/app/benchmark_llm.py" in source_files:
        prompt_paths.append("backend/app/benchmark_llm.py")
    for relative in prompt_paths:
        expected = source_files.get(relative)
        if not isinstance(expected, str):
            missing.append(f"prompt source is absent from source freeze: {relative}")
            continue
        if not (snapshot / relative).is_file() or _sha256(snapshot / relative) != expected:
            missing.append(f"prompt source snapshot hash differs: {relative}")
            continue
        prompt_hashes[relative] = expected
    return prompt_hashes


def _verify_asset(asset: dict | None, missing: list[str]) -> dict | None:
    if not isinstance(asset, Mapping):
        missing.append("LLM asset verification file is missing or invalid")
        return None
    if asset.get("status") != "ok":
        missing.append("LLM asset status is not ok")
    if asset.get("model") != "qwen35_9b":
        missing.append("approved qwen35_9b asset is not registered")
    files, hashes = asset.get("weight_files"), asset.get("weight_sha256")
    if not isinstance(files, Mapping) or not files:
        missing.append("LLM weight_files are missing")
    if not isinstance(hashes, Mapping) or set(hashes) != set(files or {}):
        missing.append("LLM weight SHA256 map does not match weight_files")
    elif any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        missing.append("LLM weight SHA256 entry is invalid")
    fields = (
        "status", "model", "asset", "runtime", "dtype", "device", "gpu_physical_index",
        "service_context_window", "model_context_window", "enable_thinking", "protocols",
        "asset_metadata", "weight_files", "weight_sha256",
    )
    return {key: asset[key] for key in fields if key in asset}


def _verify_environment(root: Path, source_digest: str, missing: list[str]) -> dict | None:
    path = root / "approval" / f"environment-{source_digest}.json"
    environment = _read_json(path, "frozen execution environment", missing)
    if not isinstance(environment, Mapping):
        return None
    if environment.get("cuda_available") is not True or not environment.get("cuda_runtime"):
        missing.append("frozen environment does not report CUDA availability and runtime")
    fields = (
        "python", "executable", "packages", "torch", "cuda_available", "cuda_runtime",
        "cpu_affinity", "cpu_tokens", "classifier_gpu_ids", "llm_gpu_id",
    )
    return {key: environment[key] for key in fields if key in environment}


def _validate_manifests(root: Path, source: dict, protocol: dict, data: dict,
                        missing: list[str]) -> tuple[Path | None, dict | None]:
    source_digest = source.get("source_digest")
    protocol_digest = protocol.get("approval_digest")
    if source.get("protocol_digest") != protocol_digest:
        missing.append("source-freeze protocol digest differs from approval/protocol.json")
    if source.get("data_manifest_sha256") != _sha256(root / DATA_FILE):
        missing.append("source-freeze data manifest SHA256 differs from data/manifest.json")
    if data.get("status") != "complete":
        missing.append("data manifest is not complete")
    version_directory = root / "driver" / "versions" / str(source_digest)[:12]
    path = version_directory / "manifest.json"
    manifest = _read_json(path, "source-version driver manifest", missing)
    if not isinstance(manifest, Mapping):
        return version_directory, None
    if not isinstance(manifest.get("tasks"), list):
        missing.append("driver manifest tasks list is missing")
    if not isinstance(manifest.get("jobs"), list):
        missing.append("driver manifest jobs list is missing")
    if manifest.get("schema_version") != "benchmark-job-manifest-v1":
        missing.append("driver manifest schema is not benchmark-job-manifest-v1")
    if manifest.get("preparation_status") != "sessions_frozen":
        missing.append("driver manifest is not fully prepared with frozen Sessions")
    if manifest.get("source_digest") != source_digest:
        missing.append("driver manifest source digest differs")
    if manifest.get("protocol_digest") != protocol_digest:
        missing.append("driver manifest protocol digest differs")
    if manifest.get("data_manifest_digest") != _json_digest(data):
        missing.append("driver manifest data digest differs")
    data_tasks = {
        task["task_id"]: task["role"] for task in data.get("tasks", [])
        if task.get("role") in {"development", "locked"}
    }
    driver_tasks = {
        task["task_id"]: task["role"] for task in manifest.get("tasks", [])
        if task.get("role") in {"development", "locked"}
    }
    if data_tasks != driver_tasks:
        missing.append("driver task roles or task IDs differ from the frozen data manifest")
    if not any(role == "development" for role in driver_tasks.values()):
        missing.append("driver manifest has no development tasks")
    if not any(role == "locked" for role in driver_tasks.values()):
        missing.append("driver manifest has no locked tasks")
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        missing.append("driver manifest has no task rounds")
        return version_directory, manifest
    seen_rounds = set()
    for row in rounds:
        key = (row.get("task_id"), row.get("round_id"))
        if key in seen_rounds:
            missing.append(f"duplicate driver task round: {key[0]}:{key[1]}")
        seen_rounds.add(key)
        if data_tasks.get(row.get("task_id")) != row.get("role"):
            missing.append(f"driver round role differs from data manifest: {key[0]}:{key[1]}")
        eligible_by_budget = row.get("eligible_models_by_budget")
        if not isinstance(eligible_by_budget, Mapping):
            missing.append(f"round eligibility is missing: {key[0]}:{key[1]}")
            continue
        for budget in BUDGETS:
            eligible = eligible_by_budget.get(str(budget), eligible_by_budget.get(budget))
            if not isinstance(eligible, list) or not eligible:
                missing.append(f"no eligible models for budget {budget}: {key[0]}:{key[1]}")
            elif any(model not in MODELS for model in eligible):
                missing.append(f"unknown eligible model for budget {budget}: {key[0]}:{key[1]}")
    for task in manifest.get("tasks", []):
        task_id = task.get("task_id")
        declared_rounds = {
            item.get("round_id"): item for item in task.get("rounds", [])
            if isinstance(item, Mapping)
        }
        actual_rounds = {
            row.get("round_id"): row for row in rounds if row.get("task_id") == task_id
        }
        if set(declared_rounds) != set(actual_rounds):
            missing.append(f"task-round list differs between manifest sections: {task_id}")
        for round_id in set(declared_rounds) & set(actual_rounds):
            for budget in BUDGETS:
                declared_models = declared_rounds[round_id].get("eligible_models_by_budget", {}).get(str(budget), [])
                actual_models = actual_rounds[round_id].get("eligible_models_by_budget", {}).get(str(budget), [])
                if sorted(declared_models) != sorted(actual_models):
                    missing.append(f"task-round eligibility differs between manifest sections: {task_id}/{round_id}/b{budget}")
    for task_id, role in data_tasks.items():
        if not any(row.get("task_id") == task_id for row in rounds):
            missing.append(f"task has no prepared rounds: {role} {task_id}")
    return version_directory, dict(manifest)


def _dev_matrix(root: Path, version: Path, manifest: dict, missing: list[str]) -> tuple[list[dict], dict]:
    tasks = {task["task_id"]: task for task in manifest["tasks"] if task["role"] == "development"}
    round_rows = [row for row in manifest["rounds"] if row["role"] == "development"]
    records: list[dict] = []
    record_hashes: dict[str, str] = {}
    signatures: dict[tuple[str, str], tuple] = {}
    sample_round: dict[tuple[str, str], str] = {}
    true_labels: dict[tuple[str, str, str], str] = {}
    pooled: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    counts = {str(budget): 0 for budget in BUDGETS}
    per_task = {}

    for row in round_rows:
        task_id, round_id = row["task_id"], row["round_id"]
        task = tasks[task_id]
        task_summary = per_task.setdefault(task_id, {
            "suite": task["suite"], "family_id": task["family_id"], "round_ids": [],
            "expected_records_by_budget": {str(b): 0 for b in BUDGETS},
            "completed_records_by_budget": {str(b): 0 for b in BUDGETS},
        })
        task_summary["round_ids"].append(round_id)
        for budget in BUDGETS:
            eligible = row["eligible_models_by_budget"][str(budget)]
            signature_key = (task_id, round_id)
            for model_id in eligible:
                task_summary["expected_records_by_budget"][str(budget)] += 1
                path = version / "records" / (
                    identity("record", task_id, round_id, model_id, budget) + ".json"
                )
                record = _read_json(path, f"development record {task_id}/{round_id}/{model_id}/b{budget}", missing)
                if not isinstance(record, Mapping):
                    continue
                if (
                    record.get("task_id") != task_id or record.get("round_id") != round_id
                    or record.get("model_id") != model_id or record.get("budget") != budget
                    or record.get("role") != "development"
                ):
                    missing.append(f"development record identity differs: {path}")
                    continue
                if record.get("status") != "completed":
                    missing.append(
                        f"development record is not completed: {task_id}/{round_id}/{model_id}/b{budget}"
                    )
                    continue
                rows = record.get("prediction_rows")
                if not isinstance(rows, list) or not rows:
                    missing.append(f"development prediction rows are empty: {task_id}/{round_id}/{model_id}/b{budget}")
                    continue
                labels = set(task.get("labels") or [])
                signature_rows = []
                for prediction in rows:
                    sample_id = str(prediction.get("sample_id", ""))
                    truth = prediction.get("true_label")
                    predicted = prediction.get("predicted_label")
                    if not sample_id or truth not in labels or predicted not in labels:
                        missing.append(f"development prediction label/sample is invalid: {task_id}/{round_id}/{model_id}/b{budget}")
                        break
                    sample_key = (task_id, sample_id)
                    if sample_key in sample_round and sample_round[sample_key] != round_id:
                        missing.append(f"development Test sample repeats across rounds: {task_id}/{sample_id}")
                    sample_round[sample_key] = round_id
                    label_key = (task_id, round_id, sample_id)
                    if label_key in true_labels and true_labels[label_key] != truth:
                        missing.append(f"development Test true label differs across models: {task_id}/{round_id}/{sample_id}/b{budget}")
                    true_labels[label_key] = truth
                    signature_rows.append((sample_id, truth))
                if not signature_rows:
                    continue
                signature = tuple(sorted(signature_rows))
                if signature_key in signatures and signature != signatures[signature_key]:
                    missing.append(f"development models or budgets do not share Test groups: {task_id}/{round_id}")
                signatures[signature_key] = signature
                validation_score = record.get("validation_score")
                if (isinstance(validation_score, bool) or not isinstance(validation_score, (int, float))
                        or not math.isfinite(validation_score)):
                    missing.append(f"development validation score missing or invalid: {task_id}/{round_id}/{model_id}/b{budget}")
                records.append(dict(record))
                record_hashes[f"{task_id}/{round_id}/{model_id}/b{budget}"] = _sha256(path)
                pooled[(task_id, model_id, budget)].extend(rows)
                counts[str(budget)] += 1
                task_summary["completed_records_by_budget"][str(budget)] += 1

    common_candidates = {str(budget): set(MODELS) for budget in BUDGETS}
    for row in manifest["rounds"]:
        for budget in BUDGETS:
            common_candidates[str(budget)] &= set(row["eligible_models_by_budget"][str(budget)])
    score_summary = {}
    for budget in BUDGETS:
        for task_id, task in tasks.items():
            candidates = sorted(common_candidates[str(budget)])
            for model_id in candidates:
                rows = pooled.get((task_id, model_id, budget), [])
                if not rows:
                    continue
                labels = task.get("labels") or []
                support = {label: 0 for label in labels}
                correct = {label: 0 for label in labels}
                for row in rows:
                    support[row["true_label"]] += 1
                    correct[row["true_label"]] += int(row["true_label"] == row["predicted_label"])
                if any(value == 0 for value in support.values()):
                    missing.append(f"development pooled Test lacks a full label for {task_id}/{model_id}/b{budget}")
                    continue
                score = sum(correct[label] / support[label] for label in labels) / len(labels)
                score_summary.setdefault(str(budget), {}).setdefault(task_id, {})[model_id] = score

    expected_counts = {
        str(budget): sum(
            len(row["eligible_models_by_budget"][str(budget)])
            for row in round_rows
        )
        for budget in BUDGETS
    }
    matrix_digest = _json_digest(record_hashes)
    return records, {
        "task_count": len(tasks),
        "round_count": len(round_rows),
        "expected_records_by_budget": expected_counts,
        "completed_records_by_budget": counts,
        "records_digest": matrix_digest,
        "record_hash_count": len(record_hashes),
        "task_matrix": per_task,
        "pooled_balanced_accuracy_by_task_model": score_summary,
    }


def _verify_acceptance(root: Path, version: Path, source: dict, manifest: dict,
                       missing: list[str]) -> tuple[list[dict], dict | None, dict | None]:
    driver = object.__new__(BenchmarkDriver)
    driver.root, driver.directory = root, version
    acceptance_path = version / "acceptance-manifest.json"
    acceptance = _read_json(acceptance_path, "Coffee functional acceptance manifest", missing)
    acceptance_records = []
    coffee_evidence = None
    if isinstance(acceptance, Mapping):
        if acceptance.get("schema_version") != "benchmark-functional-acceptance-v1":
            missing.append("Coffee functional acceptance schema differs")
        if acceptance.get("source_digest") != source["source_digest"]:
            missing.append("Coffee functional acceptance source digest differs")
        if acceptance.get("protocol_digest") != manifest["protocol_digest"]:
            missing.append("Coffee functional acceptance protocol digest differs")
        if acceptance.get("preparation_status") != "sessions_frozen":
            missing.append("Coffee functional acceptance Sessions are not frozen")
        jobs = acceptance.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != 2:
            missing.append("Coffee functional acceptance must contain exactly two planned runs")
            jobs = []
        for job in jobs:
            if job.get("task_id") != "Coffee_functional_acceptance" or job.get("budgets") != [1]:
                missing.append("Coffee functional acceptance job binding differs")
            path = version / "records" / (
                identity("record", job.get("task_id"), job.get("round_id"),
                         job.get("model_id"), 1) + ".json"
            )
            record = _read_json(path, f"Coffee functional record {job.get('model_id')}", missing)
            if not isinstance(record, Mapping):
                continue
            if record.get("task_id") != "Coffee_functional_acceptance" or record.get("status") != "completed":
                missing.append(f"Coffee functional record is not completed: {job.get('model_id')}")
            if record.get("budget") != 1 or record.get("model_id") != job.get("model_id"):
                missing.append(f"Coffee functional record identity differs: {job.get('model_id')}")
            acceptance_records.append(dict(record))
        if len(acceptance_records) == 2 and all(row.get("status") == "completed" for row in acceptance_records):
            coffee_evidence = {
                "task_id": "Coffee_functional_acceptance",
                "record_count": 2,
                "records": [
                    {
                        "model_id": record["model_id"], "round_id": record["round_id"],
                        "budget": record["budget"], "status": record["status"],
                        "run_id": record.get("run_id"),
                        "record_sha256": _sha256(
                            version / "records" / (
                                identity("record", record["task_id"], record["round_id"],
                                         record["model_id"], record["budget"]) + ".json"
                            )
                        ),
                    }
                    for record in acceptance_records
                ],
                "paper_evidence": False,
                "independence_status": acceptance.get("independence_status"),
            }

    coffee_task = {"suite": "ucr85", "task_id": "Coffee_functional_acceptance"}
    coffee_tracking = _verify_tracking(root, coffee_task, source["source_digest"], missing)
    return acceptance_records, coffee_evidence, coffee_tracking


def _cuda_evidence(root: Path, source_digest: str, records: list[dict],
                   missing: list[str]) -> dict | None:
    driver = object.__new__(BenchmarkDriver)
    driver.root = root
    try:
        driver._require_cuda_acceptance(records, source_digest)
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        missing.append(f"CUDA fit/checkpoint/explanation acceptance incomplete: {exc}")
        return None

    deep_models = sorted(set(MODELS) - set(TRADITIONAL))
    indexed = {
        (row["task_id"], row["round_id"], row["model_id"], row["budget"]): row
        for row in records
    }
    models = {}
    for model in deep_models:
        evidence = None
        for key, first in sorted(indexed.items()):
            task_id, round_id, model_id, budget = key
            if model_id != model or budget != 1 or first.get("status") != "completed":
                continue
            second = indexed.get((task_id, round_id, model, 3))
            if second is None or second.get("status") != "completed":
                continue
            first_path = Path(first["artifact_directory"]).resolve()
            second_path = Path(second["artifact_directory"]).resolve()
            explanation_path = first_path / "telemetry" / "cuda_explanation_acceptance.json"
            reload_path = second_path / "telemetry" / "checkpoint_reload.json"
            summary_path = second_path / "search_summary.json"
            trials_path = second_path / "search_trials.json"
            explanation = _read_json(explanation_path, "CUDA explanation acceptance", missing)
            reload = _read_json(reload_path, "CUDA checkpoint reload telemetry", missing)
            search = _read_json(summary_path, "cached search summary", missing)
            trials = _read_json(trials_path, "cached search trials", missing)
            if not all(isinstance(v, Mapping) for v in (explanation, reload, search)):
                continue
            trial_rows = trials.get("trials", []) if isinstance(trials, Mapping) else trials
            default = next((row for row in trial_rows if row.get("trial_index") == 0), None)
            if (
                first.get("cost", {}).get("actual_fits", 0) >= 1
                and first.get("prediction_rows")
                and explanation.get("status") == "ready"
                and explanation.get("sample_count") == 1
                and str(explanation.get("actual_device", "")).startswith("cuda")
                and explanation.get("dtype") == "torch.float32"
                and explanation.get("additional_fits") == 0
                and explanation.get("source_digest") == source_digest
                and explanation.get("model_id") == model
                and explanation.get("source_run_id") == first.get("run_id")
                and reload.get("status") == "succeeded"
                and str(reload.get("loaded_device", "")).startswith("cuda")
                and reload.get("dtype") == "torch.float32"
                and reload.get("additional_fits") == 0
                and len(reload.get("source_checkpoint", {}).get("sha256", "")) == 64
                and search.get("cache_hit_count", 0) >= 1
                and default is not None
                and default.get("cached") is True
                and default.get("executed_epochs") == 0
            ):
                evidence = {
                    "task_id": task_id,
                    "round_id": round_id,
                    "model_id": model,
                    "budget_1_run_id": first.get("run_id"),
                    "budget_3_run_id": second.get("run_id"),
                    "actual_fits": first.get("cost", {}).get("actual_fits"),
                    "prediction_group_count": len(first.get("prediction_rows", [])),
                    "actual_device": explanation.get("actual_device"),
                    "dtype": explanation.get("dtype"),
                    "explanation_model_id": explanation.get("model_id"),
                    "explanation_source_run_id": explanation.get("source_run_id"),
                    "checkpoint_reload_status": reload.get("status"),
                    "loaded_device": reload.get("loaded_device"),
                    "loaded_dtype": reload.get("dtype"),
                    "cache_hit_count": search.get("cache_hit_count"),
                    "telemetry_sha256": {
                        "cuda_explanation_acceptance": _sha256(explanation_path),
                        "checkpoint_reload": _sha256(reload_path),
                        "search_summary": _sha256(summary_path),
                        "search_trials": _sha256(trials_path),
                    },
                }
                break
        if evidence is None:
            missing.append(f"no qualifying current-source CUDA FP32 fit/prediction telemetry for {model}")
        else:
            models[model] = evidence
    if len(models) != len(deep_models):
        return None
    return {"required_models": deep_models, "accepted_models": models}


def _freeze_payload(root: Path, source: dict, protocol: dict, data: dict,
                    manifest: dict, environment: dict, llm_asset: dict,
                    prompt_hashes: dict, fixed_models: dict,
                    development: dict, development_records: list[dict],
                    tracking: list[dict], coffee: dict, coffee_tracking: dict,
                    cuda: dict, driver_state: dict) -> dict:
    model_id = llm_asset["model"]
    prompt_version = "agent-decision-budget-v1"
    llm_public_config = {
        "version": "agent-llm-http-v1",
        "model_id_sha256": hashlib.sha256(model_id.encode("utf-8")).hexdigest(),
        "protocol": "json_action",
        "timeout": 120,
        "max_tokens": 1024,
        "max_response_bytes": 32768,
        "temperature": 0,
        "top_p": 1,
        "prompt_version": prompt_version,
        "context_version": "agent-context-budget-v1",
    }
    driver_llm_config = next(
        (item.get("llm_config") for item in tracking if item.get("llm_config")),
        coffee_tracking.get("llm_config") if coffee_tracking else None,
    )
    driver_llm_config = dict(driver_llm_config or {})
    task_counts = {}
    for role in ("development", "locked", "excluded"):
        role_tasks = [task for task in data.get("tasks", []) if task.get("role") == role]
        task_counts[role] = {
            "count": len(role_tasks),
            "task_ids": sorted(task["task_id"] for task in role_tasks),
            "suites": {
                suite: sum(task["suite"] == suite for task in role_tasks)
                for suite in sorted({task["suite"] for task in role_tasks})
            },
        }
    clocks = driver_state.get("task_clocks", [])
    clocks_by_id = {row.get("task_id"): row for row in clocks if isinstance(row, Mapping)}
    state_wall = driver_state.get("task_wall_seconds") or {}
    task_clock_summary = []
    for task in manifest["tasks"]:
        if task["role"] != "development":
            continue
        clock = clocks_by_id.get(task["task_id"], {})
        wall = clock.get("task_wall_seconds", state_wall.get(task["task_id"]))
        task_clock_summary.append({
            "task_id": task["task_id"], "role": task["role"],
            "task_wall_seconds": wall,
            "cloud_verified_at": clock.get("cloud_verified_at"),
        })
    execution = {
        "protocol_execution": protocol.get("execution"),
        "frozen_environment": environment,
        "development_task_wall_clocks": task_clock_summary,
    }
    return {
        "schema_version": "benchmark-formal-freeze-v1",
        "status": "frozen",
        "source_digest": source["source_digest"],
        "protocol_digest": protocol["approval_digest"],
        "data_manifest_sha256": source["data_manifest_sha256"],
        "data_manifest_digest": manifest["data_manifest_digest"],
        "source": _source_summary(source, prompt_hashes),
        "protocol": _protocol_summary(protocol),
        "data": {
            "status": data.get("status"),
            "data_manifest_sha256": source["data_manifest_sha256"],
            "data_manifest_digest": manifest["data_manifest_digest"],
            "task_counts": task_counts,
        },
        "prompt": {
            "prompt_version": llm_public_config["prompt_version"],
            "context_version": llm_public_config["context_version"],
            "llm_public_config": llm_public_config,
            "driver_llm_config": driver_llm_config,
            "source_hashes": prompt_hashes,
            "source_hashes_digest": _json_digest(prompt_hashes),
        },
        "llm_asset": llm_asset,
        "execution": execution,
        "fixed_models": fixed_models,
        "development_matrix": development,
        "verification_evidence": {
            "development_records_count": len(development_records),
            "development_tracking": tracking,
            "coffee_functional_acceptance": coffee,
            "coffee_cloud_tracking": coffee_tracking,
            "cuda_acceptance": cuda,
        },
    }


def _write_once(path: Path, payload: dict) -> bool:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("formal-freeze target must not be a symlink")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("formal-freeze.json exists with different contents; refusing to rewrite")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("formal-freeze.json exists with different contents; refusing to rewrite")
            return False
    finally:
        temporary.unlink(missing_ok=True)


def freeze_formal(root: Path = DEFAULT_ROOT) -> dict:
    """Validate prerequisites and create approval/formal-freeze.json once."""
    root = Path(root).resolve()
    missing: list[str] = []
    source = _read_json(root / SOURCE_FILE, "source-freeze.json", missing)
    protocol = _read_json(root / PROTOCOL_FILE, "approved protocol", missing)
    data_path = root / DATA_FILE
    data = _read_json(data_path, "data manifest", missing)
    llm_asset = _read_json(root / LLM_ASSET_FILE, "LLM asset receipt", missing)
    if missing:
        raise FormalFreezeBlocked(missing)
    if not all(isinstance(item, Mapping) for item in (source, protocol, data)):
        raise FormalFreezeBlocked(["source/protocol/data JSON must be objects"])

    protocol_digest = protocol.get("approval_digest")
    protocol_body = {key: value for key, value in protocol.items() if key != "approval_digest"}
    if not isinstance(protocol_digest, str) or _json_digest(protocol_body) != protocol_digest:
        missing.append("approved protocol digest is invalid")
    if protocol.get("governance", {}).get("user_approved") is not True:
        missing.append("protocol has no explicit user approval")
    if protocol.get("split_seed") != 42 or protocol.get("hpo", {}).get("candidate_budgets") != [1, 3]:
        missing.append("protocol split seed or candidate budgets differ from the approved v3 contract")

    source_digest = source.get("source_digest")
    prompt_hashes = _verify_source(root, source, missing)
    llm_summary = _verify_asset(llm_asset, missing)
    environment = _verify_environment(root, source_digest, missing) if isinstance(source_digest, str) else None
    version_directory, manifest = _validate_manifests(root, source, protocol, data, missing)
    if not isinstance(manifest, Mapping) or version_directory is None:
        raise FormalFreezeBlocked(missing or ["source-version benchmark manifest is unavailable"])
    if llm_summary is None or environment is None:
        raise FormalFreezeBlocked(missing or ["LLM asset or frozen execution environment is unavailable"])

    # The immutable model list comes from BenchmarkDriver.MODELS. Locked rounds
    # provide legal model IDs only; their prediction records are never loaded here.
    locked_tasks = [task for task in manifest["tasks"] if task["role"] == "locked"]
    for task in locked_tasks:
        locked_rounds = [row for row in manifest["rounds"] if row["task_id"] == task["task_id"]]
        if not locked_rounds:
            missing.append(f"locked task has no declared rounds: {task['task_id']}")
        for round_info in locked_rounds:
            eligibility = round_info.get("eligible_models_by_budget", {})
            for budget in BUDGETS:
                models = eligibility.get(str(budget), [])
                if not models:
                    missing.append(f"locked legal model list is empty: {task['task_id']}/{round_info.get('round_id')}/b{budget}")
                elif any(model not in MODELS for model in models):
                    missing.append(f"locked legal model list contains an unknown model: {task['task_id']}/{round_info.get('round_id')}/b{budget}")

    development_records, development_summary = _dev_matrix(
        root, version_directory, manifest, missing
    )

    tracking_evidence = []
    for task in manifest["tasks"]:
        if task["role"] == "development":
            evidence = _verify_tracking(root, task, source_digest, missing)
            if evidence:
                tracking_evidence.append(evidence)

    acceptance_records, coffee_evidence, coffee_tracking = _verify_acceptance(
        root, version_directory, source, manifest, missing
    )

    driver = object.__new__(BenchmarkDriver)
    driver.root, driver.directory = root, version_directory
    cuda_evidence = _cuda_evidence(root, source_digest, development_records, missing)

    expected_llm_config = {
        "model": llm_asset.get("model"),
        "protocol": "json_action",
        "prompt_version": "agent-decision-budget-v1",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1024,
        "timeout_seconds": 120,
    }
    observed_llm_configs = [
        item.get("llm_config") for item in tracking_evidence
        if isinstance(item.get("llm_config"), Mapping)
    ]
    if isinstance(coffee_tracking, Mapping) and isinstance(coffee_tracking.get("llm_config"), Mapping):
        observed_llm_configs.append(coffee_tracking["llm_config"])
    if any(config != expected_llm_config for config in observed_llm_configs):
        missing.append("frozen driver LLM configuration differs from the registered public configuration")
    if not observed_llm_configs:
        missing.append("no verified tracking task contains the frozen LLM configuration")
    if missing:
        raise FormalFreezeBlocked(missing)

    # This is the established development-only selector. It writes only its
    # deterministic support file under the source-version directory.
    try:
        fixed_models = driver.development_fixed_models()
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise FormalFreezeBlocked([f"development fixed-model selection failed: {exc}"]) from None
    if any(not fixed_models[str(budget)].get("available") for budget in BUDGETS):
        raise FormalFreezeBlocked(["fixed model is unavailable for one or both budgets"])

    driver_state = _read_json(version_directory / "state.json", "driver state", missing) or {}
    driver_state["task_clocks"] = [
        _read_json(path, f"task clock {path.stem}", missing)
        for path in sorted((version_directory / "task-clocks").glob("*.json"))
    ]
    driver_state["task_clocks"] = [clock for clock in driver_state["task_clocks"] if isinstance(clock, Mapping)]
    if missing:
        raise FormalFreezeBlocked(missing)
    payload = _freeze_payload(
        root, source, protocol, data, manifest, environment, llm_summary,
        prompt_hashes, fixed_models, development_summary, development_records,
        tracking_evidence, coffee_evidence, coffee_tracking, cuda_evidence,
        driver_state,
    )
    output = root / "approval" / "formal-freeze.json"
    try:
        created = _write_once(output, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FormalFreezeBlocked([str(exc)]) from None
    return {"path": str(output), "created": created, "formal_freeze": payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = freeze_formal(args.root)
    except FormalFreezeBlocked as exc:
        print(json.dumps({"status": "not_frozen", "missing": exc.missing},
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "status": "frozen",
        "created": result["created"],
        "path": result["path"],
        "source_digest": result["formal_freeze"]["source_digest"],
        "protocol_digest": result["formal_freeze"]["protocol_digest"],
        "fixed_models": result["formal_freeze"]["fixed_models"],
        "development_records_count": result["formal_freeze"]["verification_evidence"]["development_records_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
