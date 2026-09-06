"""Batch queue, reproducibility and model-comparison contract coverage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.contracts import TrainingBatchRequest
from backend.app.models import canonical_model_type
from backend.app.runs.artifacts import RunArtifactWriter
from backend.app.runs.batch_projection import aggregate_batch_state, project_model_comparison
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository


def _repository(tmp_path: Path) -> RunRepository:
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.initialize()
    return repository


def _batch(repository: RunRepository, *, principal: Principal = Principal()):
    return repository.create_batch_queued(
        dataset_id="dataset-1",
        test_dataset_id=None,
        legacy_data_path=None,
        config={"normalization": "zscore", "split_mode": "stratified_holdout"},
        model_types=["pls_da", "pca_svm", "cnn1d"],
        repeat_count=2,
        base_seed=73,
        dataset_snapshot={"name": "fixture.csv", "sha256": "a" * 64},
        principal=principal,
    )


def test_catalog_has_exactly_ten_ui_models_and_keeps_hidden_models_callable() -> None:
    from backend.app.routers.catalog import _MODEL_CATALOG
    from backend.app.main import app
    from fastapi.testclient import TestClient

    visible = [item[0] for item in _MODEL_CATALOG if item[-1] is True]
    assert visible == [
        "pls_da", "logistic_regression", "svm", "random_forest", "xgboost", "cnn1d",
    ]
    assert canonical_model_type("resnet1d") == "resnet1d"
    assert canonical_model_type("dscarnet") == "dscarnet"
    assert canonical_model_type("cnn_transformer1d") == "cnn_transformer1d"
    response = TestClient(app).get("/api/models")
    assert response.status_code == 200
    catalog = response.json()["models"]
    assert [item["id"] for item in catalog if item["ui_visible"]] == visible
    hidden = {item["id"]: item for item in catalog if not item["ui_visible"]}
    assert set(hidden) == {"spls_da", "pca_lda", "pca_svm", "pca_mlp", "cnn1d_se", "resnet1d", "inception1d", "tcn1d", "cnn_transformer1d", "cnn_mamba1d", "dscarnet"}
    assert all(item["visibility_reason"] == "temporarily_hidden_from_ui" for item in hidden.values())
    assert all("explainability_method" not in item for item in catalog)


def test_batch_creation_assigns_fair_split_seed_and_repeat_model_seeds(tmp_path: Path) -> None:
    batch, records = _batch(_repository(tmp_path))

    assert batch.repeat_count == 2
    assert len(records) == 6
    assert [record.config["model_type"] for record in records] == [
        "pls_da", "pls_da", "pca_svm", "pca_svm", "cnn1d", "cnn1d",
    ]
    by_repeat = {index: [record for record in records if record.batch_repeat_index == index] for index in (1, 2)}
    assert {record.config["split_seed"] for record in records} == {73}
    assert {record.config["model_seed"] for record in by_repeat[1]} == {73}
    assert {record.config["model_seed"] for record in by_repeat[2]} == {74}
    assert all(record.batch_id == batch.batch_id and record.state == "queued" for record in records)


def test_new_batch_request_defaults_to_one_run_per_model() -> None:
    request = TrainingBatchRequest(dataset_id="dataset-1", model_types=["pls_da", "pca_svm"])

    assert request.repeat_count == 1


def test_batch_http_creates_one_queued_run_per_model_and_rejects_repeats_or_per_run_seed(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.routers import batches
    from backend.app.routers.deps import TrainingDataReference

    repository = _repository(tmp_path)
    monkeypatch.setattr(batches, "get_run_repository", lambda: repository)
    monkeypatch.setattr(batches, "get_run_dir", lambda run_id: tmp_path / "run-artifacts" / run_id)
    monkeypatch.setattr(
        batches,
        "resolve_training_data_reference",
        lambda _payload, *, principal: TrainingDataReference(
            dataset_id="dataset-1", legacy_path=None, dataset_name="fixture.csv",
            test_dataset_id=None, test_legacy_path=None, test_dataset_name=None,
        ),
    )
    monkeypatch.setattr(batches, "_dataset_snapshot_for_reference", lambda **_kwargs: {"name": "fixture.csv", "sha256": "b" * 64})

    client = TestClient(app)
    response = client.post("/api/training/batches", json={
        "dataset_id": "dataset-1", "model_types": ["pls_da", "pca_svm"],
        "base_seed": 19,
        "config": {"split_mode": "stratified_holdout", "normalization": "zscore"},
    })
    assert response.status_code == 202
    created = response.json()
    assert created["repeat_count"] == 1
    assert created["run_count"] == 2
    assert {item["state"] for item in created["runs"]} == {"queued"}
    assert all(record.state == "queued" for record in repository.list_batch_runs_scoped(created["batch_id"], principal=Principal()))

    default_response = client.post("/api/training/batches", json={
        "dataset_id": "dataset-1", "model_types": ["pls_da", "pca_svm", "svm"],
        "base_seed": 21, "config": {"split_mode": "stratified_holdout", "normalization": "zscore"},
    })
    assert default_response.status_code == 202
    assert default_response.json()["repeat_count"] == 1
    assert default_response.json()["run_count"] == 3

    rejected = client.post("/api/training/batches", json={
        "dataset_id": "dataset-1", "model_types": ["pls_da"],
        "config": {"seed": 123},
    })
    assert rejected.status_code == 422
    assert len(repository.list_batch_runs_scoped(created["batch_id"], principal=Principal())) == 2

    repeated = client.post("/api/training/batches", json={
        "dataset_id": "dataset-1", "model_types": ["pls_da"], "repeat_count": 2,
        "config": {"split_mode": "stratified_holdout", "normalization": "zscore"},
    })
    assert repeated.status_code == 422


def test_batch_status_partial_failure_and_cancel_are_aggregated(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    batch, records = _batch(repository)
    now = datetime.now(timezone.utc)
    first = repository.claim_next(worker_id="test", now=now)
    assert first is not None
    repository.finish_success(first.run_id, claim_token=first.claim_token or "", now=now)
    second = repository.claim_next(worker_id="test", now=now)
    assert second is not None
    repository.finish_failure(second.run_id, claim_token=second.claim_token or "", now=now, error="intentional")

    before_stop = repository.list_batch_runs_scoped(batch.batch_id, principal=Principal())
    assert aggregate_batch_state(before_stop) == "queued"
    cancelled = repository.cancel_batch_scoped(batch.batch_id, now=now, principal=Principal())
    assert aggregate_batch_state(cancelled) == "cancelled"
    assert sum(record.state == "cancelled" for record in cancelled) == len(records)
    assert sum(record.state == "succeeded" for record in cancelled) == 0
    assert sum(record.state == "failed" for record in cancelled) == 0


def _write_comparison_artifacts(run_dir: Path, run_id: str, *, score: float, sample_suffix: str, second_label: str = "B") -> None:
    writer = RunArtifactWriter(run_dir)
    report = {
        "A": {"recall": score}, second_label: {"recall": score},
        "macro avg": {"recall": score}, "weighted avg": {"recall": score},
    }
    writer.write_json("metrics.json", {"test": {
        "accuracy": score, "balanced_accuracy": score, "macro_f1": score,
        "weighted_f1": score, "classification_report": report,
        "confusion_matrix": [[2, 0], [0, 2]],
    }})
    writer.write_bytes("predictions.csv", (
        "Sample_ID,true_label,pred_label,prob_A,prob_B\n"
        f"1,A,A,{score},0.1\n2,{second_label},{second_label},0.1,{score}\n"
    ).encode("utf-8"))
    writer.write_json("split.json", [{
        "fold_index": 1, "train_sample_ids": ["3", "4"],
        "valid_sample_ids": ["5", "6"], "test_sample_ids": ["1", "2"],
        "sample_suffix": sample_suffix,
    }])
    writer.write_json("label_map.json", {"0": "A", "1": second_label})
    writer.finalize(run_id=run_id)


def test_historical_comparison_uses_single_runs_instead_of_summed_matrix(tmp_path: Path) -> None:
    from dataclasses import replace
    repository = _repository(tmp_path)
    batch, records = _batch(repository)
    successful = []
    for record in records:
        _write_comparison_artifacts(tmp_path / record.run_id, record.run_id, score=.8, sample_suffix='same')
        successful.append(replace(record, state='succeeded'))
    result = project_model_comparison(batch_id=batch.batch_id, records=successful, run_dir_for=lambda rid: tmp_path / rid)
    assert result['comparable']
    assert all(sum(map(sum, m['confusion_matrix'])) == 4 for m in result['confusion_matrices'])
    assert all(len(runs) == 2 for runs in result['run_comparisons'].values())
    assert all(len(m['run_ids']) == 1 for m in result['models'])
    assert all(m['experiment'] is None for m in result['models'])


def test_comparison_is_adaptive_and_does_not_fabricate_missing_values(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    batch, records = _batch(repository)
    now = datetime.now(timezone.utc)
    # SQLite queues runs by creation time/run_id, so do not assume the UUID
    # insertion order.  Mark the first occurrence of two distinct models
    # successful and their second occurrences failed.
    successful_models: set[str] = set()
    score_for = {"pls_da": 0.9, "pca_svm": 0.7, "cnn1d": 0.6}
    while len(successful_models) < 2:
        claimed = repository.claim_next(worker_id="test", now=now)
        assert claimed is not None
        model_type = str(claimed.config["model_type"])
        if model_type not in successful_models:
            run_dir = tmp_path / "runs" / claimed.run_id
            _write_comparison_artifacts(run_dir, claimed.run_id, score=score_for[model_type], sample_suffix="same")
            repository.finish_success(claimed.run_id, claim_token=claimed.claim_token or "", now=now)
            successful_models.add(model_type)
        else:
            repository.finish_failure(claimed.run_id, claim_token=claimed.claim_token or "", now=now, error="failed")

    records_now = repository.list_batch_runs_scoped(batch.batch_id, principal=Principal())
    comparison = project_model_comparison(
        batch_id=batch.batch_id,
        records=records_now,
        run_dir_for=lambda run_id: tmp_path / "runs" / run_id,
    )
    assert comparison["schema_version"] == "model-comparison-v1"
    assert comparison["comparable"] is True
    assert len(comparison["models"]) == 2
    assert comparison["overall_metrics"]["metrics"] == ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    assert comparison["sample_correctness"]["status"] == "ready"
    assert comparison["repeat_stability"]["status"] == "not_applicable"
    assert comparison["class_recall"]["status"] == "ready"
    assert len(comparison["confusion_matrices"]) == 2


def test_comparison_rejects_different_label_maps_even_when_metrics_exist(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    batch, _records = _batch(repository)
    now = datetime.now(timezone.utc)
    claimed_models: set[str] = set()
    while len(claimed_models) < 2:
        claimed = repository.claim_next(worker_id="test", now=now)
        assert claimed is not None
        model_type = str(claimed.config["model_type"])
        if model_type in claimed_models:
            repository.finish_failure(claimed.run_id, claim_token=claimed.claim_token or "", now=now, error="skip repeat")
            continue
        _write_comparison_artifacts(
            tmp_path / "runs" / claimed.run_id,
            claimed.run_id,
            score=0.8,
            sample_suffix="same",
            second_label="C" if claimed_models else "B",
        )
        repository.finish_success(claimed.run_id, claim_token=claimed.claim_token or "", now=now)
        claimed_models.add(model_type)
    comparison = project_model_comparison(
        batch_id=batch.batch_id,
        records=repository.list_batch_runs_scoped(batch.batch_id, principal=Principal()),
        run_dir_for=lambda run_id: tmp_path / "runs" / run_id,
    )
    assert comparison["comparable"] is False
    assert "标签映射" in comparison["reason"]


def test_server_scoped_batch_cannot_be_read_or_cancelled_by_other_principal(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    owner = Principal(owner_id="owner-a", tenant_id="tenant-a")
    other = Principal(owner_id="owner-b", tenant_id="tenant-a")
    batch, _records = _batch(repository, principal=owner)

    with pytest.raises(Exception):
        repository.get_batch_scoped(batch.batch_id, principal=other)
    with pytest.raises(Exception):
        repository.cancel_batch_scoped(batch.batch_id, now=datetime.now(timezone.utc), principal=other)


def test_batch_child_run_cannot_be_deleted_directly(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.routers import runs as runs_router

    repository = _repository(tmp_path)
    batch, records = repository.create_batch_queued(
        dataset_id="dataset-1",
        test_dataset_id=None,
        legacy_data_path=None,
        config={"normalization": "zscore", "split_mode": "stratified_holdout"},
        model_types=["pls_da", "pca_svm"],
        repeat_count=1,
        base_seed=42,
        dataset_snapshot={"name": "fixture.csv", "sha256": "c" * 64},
        principal=Principal(),
    )
    claimed = repository.claim_next(worker_id="test", now=datetime.now(timezone.utc))
    assert claimed is not None
    repository.finish_failure(
        claimed.run_id,
        claim_token=claimed.claim_token or "",
        now=datetime.now(timezone.utc),
        error="intentional",
    )
    monkeypatch.setattr(runs_router, "get_run_repository", lambda: repository)
    monkeypatch.setattr(runs_router, "get_run_dir", lambda run_id: tmp_path / "runs" / run_id)
    monkeypatch.setattr(runs_router, "_reconcile_interrupted_runs", lambda _repository: None)

    response = TestClient(app).delete(f"/api/training/runs/{claimed.run_id}")

    assert response.status_code == 409
    assert "Batch 子 Run" in response.json()["detail"]
    assert repository.get_batch_scoped(batch.batch_id, principal=Principal())
    assert repository.get(claimed.run_id).state == "failed"
