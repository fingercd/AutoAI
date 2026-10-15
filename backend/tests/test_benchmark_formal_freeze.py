import hashlib
import json
from pathlib import Path

import pytest

from backend.app.benchmark_driver import MODELS, TRADITIONAL, identity
from scripts.benchmark_freeze_formal import (
    FormalFreezeBlocked,
    freeze_formal,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _tracking(root, suite, task_id, source_digest):
    tracking_id = f"{suite}:{task_id}::{source_digest[:12]}"
    directory = root / "tracking" / "tasks" / hashlib.sha256(tracking_id.encode()).hexdigest()
    summary = {"task_wall_seconds": 10.0}
    _write(directory / "task.json", {
        "task_id": tracking_id,
        "finished": True,
        "finish_event_id": f"finish-{task_id}",
        "summary": summary,
        "config": {
            "source_digest": source_digest,
            "llm_config": {
                "model": "qwen35_9b", "protocol": "json_action",
                "prompt_version": "agent-decision-budget-v1", "temperature": 0,
                "top_p": 1, "max_tokens": 1024, "timeout_seconds": 120,
            },
        },
    })
    _write(directory / "verified.json", {
        "verified": True,
        "finish_event_id": f"finish-{task_id}",
        "verified_at": 1000.0,
        "cloud_ref": f"https://swanlab.test/runs/{task_id}",
        "summary": summary,
        "verified_numeric_summary": {"summary/task_wall_seconds": 10.0},
        "cloud_summary": {"summary/task_wall_seconds": {"value": 10.0}},
    })


def _record_path(version, task_id, round_id, model_id, budget):
    return version / "records" / (
        identity("record", task_id, round_id, model_id, budget) + ".json"
    )


def _predictions(task_id, round_id, model_id, budget, perfect):
    return [
        {
            "sample_id": f"{task_id}-{round_id}-a",
            "true_label": "a",
            "predicted_label": "a" if perfect else "a",
            "probabilities": {"a": 0.9, "b": 0.1},
        },
        {
            "sample_id": f"{task_id}-{round_id}-b",
            "true_label": "b",
            "predicted_label": "b" if perfect else "a",
            "probabilities": {"a": 0.9 if not perfect else 0.1,
                              "b": 0.1 if not perfect else 0.9},
        },
    ]


def _freeze_fixture(tmp_path):
    root = tmp_path / "benchmark-20261010"
    approval = root / "approval"
    approval.mkdir(parents=True)
    source_digest_seed = "initial"
    prompt_files = {
        "agent_poc/orchestration/llm.py": b"prompt adapter frozen source",
        "agent_poc/orchestration/projection.py": b"prompt projection frozen source",
    }
    source_files = {name: hashlib.sha256(value).hexdigest() for name, value in prompt_files.items()}
    source_digest = _digest(source_files)
    snapshot = root / "source-snapshots" / source_digest
    for name, value in prompt_files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)

    protocol = {
        "protocol_id": "autoai-classification-benchmark-v3",
        "date": "2026-10-10",
        "status": "approved_implementation_pending_freeze",
        "task": "classification",
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": ["macro_f1", "accuracy"],
        "split_seed": 42,
        "training_seed": 42,
        "hpo_seed": 42,
        "governance": {"user_approved": True},
        "hpo": {"candidate_budgets": [1, 3]},
        "execution": {"cpu_tokens": 32, "classifier_gpu_ids": [1, 2, 3, 5]},
        "primary_agent_modules": {"knowledge": "off", "memory": "off"},
    }
    protocol["approval_digest"] = _digest(protocol)
    protocol_digest = protocol["approval_digest"]
    _write(approval / "protocol.json", protocol)

    task_rows = [
        {
            "task_id": "dev-1", "suite": "tabmini25", "family_id": "family-dev",
            "role": "development", "labels": ["a", "b"],
            "main_path": "/not-read/dev.csv", "main_sha256": "d" * 64,
        },
        {
            "task_id": "locked-1", "suite": "ucr85", "family_id": "family-locked",
            "role": "locked", "labels": ["a", "b"],
            "main_path": "/not-read/locked.csv", "main_sha256": "e" * 64,
        },
    ]
    data = {"status": "complete", "tasks": task_rows}
    data_path = _write(root / "data" / "manifest.json", data)
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()

    _write(approval / "source-freeze.json", {
        "source_digest": source_digest,
        "snapshot_path": str(snapshot),
        "head": "a" * 40,
        "tag": f"benchmark-20261010-development-{source_digest[:12]}",
        "protocol_digest": protocol_digest,
        "data_manifest_sha256": data_sha256,
        "source_files": source_files,
        "dirty_patch_sha256": "f" * 64,
        "approval_basis": "approved benchmark plan",
        "phase": "development",
    })
    _write(approval / "llm-asset.json", {
        "status": "ok",
        "model": "qwen35_9b",
        "asset": "/models/qwen35_9b",
        "runtime": "transformers",
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "gpu_physical_index": 4,
        "service_context_window": 32768,
        "model_context_window": 262144,
        "enable_thinking": False,
        "protocols": ["json_action"],
        "asset_metadata": {"config.json": "a" * 64},
        "weight_files": {"model.safetensors-00001-of-00004.safetensors": 123},
        "weight_sha256": {"model.safetensors-00001-of-00004.safetensors": "b" * 64},
    })
    _write(approval / f"environment-{source_digest}.json", {
        "python": "3.12.12",
        "executable": "/venv/bin/python",
        "packages": {"torch": "2.7"},
        "torch": "2.7",
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "cpu_affinity": [0, 1, 2, 3],
        "cpu_tokens": 32,
        "classifier_gpu_ids": [1, 2, 3, 5],
        "llm_gpu_id": 4,
    })

    budgets = {"1": list(MODELS), "3": list(MODELS)}
    locked_budgets = {"1": [MODELS[0], MODELS[1]], "3": [MODELS[0], MODELS[1]]}
    manifest_tasks = []
    manifest_rounds = []
    jobs = []
    for base in task_rows:
        eligible = budgets if base["role"] == "development" else locked_budgets
        task = {
            "task_id": base["task_id"], "suite": base["suite"], "family_id": base["family_id"],
            "role": base["role"], "labels": base["labels"],
            "rounds": [{
                "round_id": "01",
                "eligible_models_by_budget": eligible,
            }],
        }
        manifest_tasks.append(task)
        row = {
            "task_id": base["task_id"], "suite": base["suite"], "family_id": base["family_id"],
            "role": base["role"], "round_id": "01", "labels": base["labels"],
            "eligible_models_by_budget": eligible,
            "source_digest": source_digest,
        }
        manifest_rounds.append(row)
        models = sorted(set(eligible["1"]) | set(eligible["3"]))
        for model in models:
            jobs.append({
                "job_id": f"job-{base['task_id']}-{model}",
                "task_id": base["task_id"], "suite": base["suite"], "family_id": base["family_id"],
                "role": base["role"], "round_id": "01", "model_id": model,
                "round_path": "/not-read/round.json", "budgets": [1, 3],
            })

    version = root / "driver" / "versions" / source_digest[:12]
    _write(version / "manifest.json", {
        "schema_version": "benchmark-job-manifest-v1",
        "protocol_digest": protocol_digest,
        "data_manifest_digest": _digest(data),
        "source_digest": source_digest,
        "preparation_status": "sessions_frozen",
        "tasks": manifest_tasks,
        "rounds": manifest_rounds,
        "jobs": jobs,
    })
    for task in manifest_tasks:
        if task["role"] == "development":
            _tracking(root, task["suite"], task["task_id"], source_digest)

    for model in MODELS:
        for budget in (1, 3):
            perfect = (budget == 1 and model == MODELS[0]) or (budget == 3 and model == MODELS[1])
            artifact = root / "storage" / model / f"b{budget}"
            rows = _predictions("dev-1", "01", model, budget, perfect)
            cost = {"actual_fits": 1 if model not in TRADITIONAL else 1,
                    "wall_seconds": 2.0, "driver_wall_seconds": 3.0}
            if model not in TRADITIONAL and budget == 1:
                cost["actual_fits"] = 1
            record = {
                "task_id": "dev-1", "suite": "tabmini25", "family_id": "family-dev",
                "role": "development", "round_id": "01", "model_id": model,
                "budget": budget, "status": "completed", "run_id": f"run-{model}-{budget}",
                "artifact_directory": str(artifact), "validation_score": 0.7,
                "prediction_rows": rows, "cost": cost,
            }
            _write(_record_path(version, "dev-1", "01", model, budget), record)
            if model in set(MODELS) - set(TRADITIONAL):
                telemetry = artifact / "telemetry"
                if budget == 1:
                    _write(telemetry / "cuda_explanation_acceptance.json", {
                        "status": "ready", "sample_count": 1, "actual_device": "cuda:0",
                        "dtype": "torch.float32", "additional_fits": 0,
                        "source_digest": source_digest, "model_id": model,
                        "source_run_id": f"run-{model}-{budget}",
                    })
                else:
                    _write(telemetry / "checkpoint_reload.json", {
                        "status": "succeeded", "loaded_device": "cuda:0",
                        "dtype": "torch.float32", "additional_fits": 0,
                        "source_checkpoint": {"sha256": "c" * 64},
                    })
                    _write(artifact / "search_summary.json", {"cache_hit_count": 1})
                    _write(artifact / "search_trials.json", {
                        "trials": [{"trial_index": 0, "cached": True, "executed_epochs": 0}],
                    })

    acceptance = {
        "schema_version": "benchmark-functional-acceptance-v1",
        "source_digest": source_digest,
        "protocol_digest": protocol_digest,
        "preparation_status": "sessions_frozen",
        "acceptance_only": True,
        "paper_evidence": False,
        "independence_status": "group_archive_case",
        "jobs": [],
    }
    for index, model in enumerate(("logistic_regression", "cnn1d")):
        acceptance["jobs"].append({
            "job_id": f"coffee-{index}", "task_id": "Coffee_functional_acceptance",
            "suite": "ucr85", "family_id": "group_archive_case", "role": "acceptance_only",
            "round_id": "01", "model_id": model, "budgets": [1],
        })
        record = {
            "task_id": "Coffee_functional_acceptance", "suite": "ucr85",
            "family_id": "group_archive_case", "role": "acceptance_only",
            "round_id": "01", "model_id": model, "budget": 1, "status": "completed",
            "run_id": f"coffee-run-{index}", "prediction_rows": [],
            "cost": {"actual_fits": 1},
        }
        _write(_record_path(version, "Coffee_functional_acceptance", "01", model, 1), record)
    _write(version / "acceptance-manifest.json", acceptance)
    _write(version / "state.json", {
        "phase": "reporting", "job_counts": {"completed": len(jobs)},
        "task_wall_seconds": {"dev-1": 10.0},
    })
    _tracking(root, "ucr85", "Coffee_functional_acceptance", source_digest)
    return root, version, source_digest


def test_formal_freeze_requires_evidence_and_is_idempotent(tmp_path):
    root, _, _ = _freeze_fixture(tmp_path)
    result = freeze_formal(root)
    output = root / "approval" / "formal-freeze.json"
    original = output.read_bytes()
    formal = json.loads(original)
    assert result["created"] is True
    assert formal["fixed_models"]["1"]["model_id"] == MODELS[0]
    assert formal["fixed_models"]["3"]["model_id"] == MODELS[1]
    assert formal["fixed_models"]["1"]["candidate_models"] == sorted([MODELS[0], MODELS[1]])
    assert formal["verification_evidence"]["cuda_acceptance"]["required_models"] == sorted(set(MODELS) - set(TRADITIONAL))
    assert formal["verification_evidence"]["development_records_count"] == len(MODELS) * 2
    assert "test_balanced_accuracy" not in output.read_text(encoding="utf-8")
    again = freeze_formal(root)
    assert again["created"] is False
    assert output.read_bytes() == original


def test_formal_freeze_needs_every_development_model_budget_record(tmp_path):
    root, version, _ = _freeze_fixture(tmp_path)
    missing_path = _record_path(version, "dev-1", "01", MODELS[0], 3)
    missing_path.unlink()
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("development record" in message and MODELS[0] in message for message in exc.value.missing)
    assert not (root / "approval" / "formal-freeze.json").exists()


def test_formal_freeze_needs_development_cloud_verification(tmp_path):
    root, _, source_digest = _freeze_fixture(tmp_path)
    task_id = f"tabmini25:dev-1::{source_digest[:12]}"
    directory = root / "tracking" / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()
    (directory / "verified.json").unlink()
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("cloud verification dev-1" in message for message in exc.value.missing)
    assert not (root / "approval" / "formal-freeze.json").exists()


def test_formal_freeze_requires_both_coffee_runs_and_cloud_verification(tmp_path):
    root, version, source_digest = _freeze_fixture(tmp_path)
    path = _record_path(version, "Coffee_functional_acceptance", "01", "cnn1d", 1)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    _write(path, record)
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("Coffee functional record is not completed" in message for message in exc.value.missing)
    assert not (root / "approval" / "formal-freeze.json").exists()


def test_formal_freeze_requires_current_deep_cuda_telemetry(tmp_path):
    root, _, _ = _freeze_fixture(tmp_path)
    telemetry = root / "storage" / "cnn1d" / "b3" / "telemetry" / "checkpoint_reload.json"
    telemetry.unlink()
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("CUDA fit/checkpoint/explanation acceptance incomplete" in message for message in exc.value.missing)
    assert not (root / "approval" / "formal-freeze.json").exists()


def test_existing_formal_freeze_is_never_rewritten(tmp_path):
    root, _, _ = _freeze_fixture(tmp_path)
    output = root / "approval" / "formal-freeze.json"
    original = b'{"existing":"different"}\n'
    output.write_bytes(original)
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("refusing to rewrite" in message for message in exc.value.missing)
    assert output.read_bytes() == original



def test_formal_freeze_requires_coffee_cloud_verification(tmp_path):
    root, _, source_digest = _freeze_fixture(tmp_path)
    tracking_id = f"ucr85:Coffee_functional_acceptance::{source_digest[:12]}"
    directory = root / "tracking" / "tasks" / hashlib.sha256(tracking_id.encode()).hexdigest()
    (directory / "verified.json").unlink()
    with pytest.raises(FormalFreezeBlocked) as exc:
        freeze_formal(root)
    assert any("cloud verification Coffee_functional_acceptance" in message for message in exc.value.missing)
    assert not (root / "approval" / "formal-freeze.json").exists()
