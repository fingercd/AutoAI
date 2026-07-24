from __future__ import annotations

from fastapi.testclient import TestClient
from datetime import datetime, timezone

from backend.app.runs.repository import RunRepository


def _repository(tmp_path) -> RunRepository:
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.initialize()
    return repository


def _patch_run_storage(monkeypatch, tmp_path, repository) -> None:
    from backend.app.routers import runs

    monkeypatch.setattr(runs, "get_run_repository", lambda: repository)
    monkeypatch.setattr(runs, "get_run_dir", lambda run_id: tmp_path / "runs" / run_id)


def test_delete_terminal_run_removes_record_directory_and_artifact_access(tmp_path, monkeypatch):
    from backend.app.main import app

    repository = _repository(tmp_path)
    record = repository.import_legacy(
        run_id="finished-run",
        state="succeeded",
        config={"model_type": "pls_da"},
        dataset_id=None,
        legacy_data_path=None,
    )
    run_dir = tmp_path / "runs" / record.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"success"}', encoding="utf-8")
    _patch_run_storage(monkeypatch, tmp_path, repository)
    client = TestClient(app)

    response = client.delete(f"/api/training/runs/{record.run_id}")

    assert response.status_code == 200
    assert response.json() == {"run_id": record.run_id, "deleted": True}
    assert not repository.exists(record.run_id)
    assert not run_dir.exists()
    assert client.get(f"/api/training/runs/{record.run_id}/artifact/status.json").status_code == 404


def test_stop_run_keeps_record_and_discards_all_artifacts(tmp_path, monkeypatch):
    from backend.app.main import app

    repository = _repository(tmp_path)
    created = repository.create_queued(dataset_id=None, config={"model_type": "dscarnet"})
    claimed = repository.claim_next(worker_id="worker", now=datetime.now(timezone.utc))
    assert claimed is not None
    run_dir = tmp_path / "runs" / created.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"running"}', encoding="utf-8")
    (run_dir / "partial.joblib").write_bytes(b"partial")
    _patch_run_storage(monkeypatch, tmp_path, repository)
    client = TestClient(app)

    response = client.post(f"/api/training/runs/{created.run_id}/stop")

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    assert response.json()["stop_status"] == "stopped"
    assert response.json()["stop_reason"] == "user_requested"
    assert repository.exists(created.run_id)
    assert repository.get(created.run_id).state == "cancelled"
    assert not run_dir.exists()


def test_delete_active_or_unknown_run_is_rejected_without_removing_files(tmp_path, monkeypatch):
    from backend.app.main import app

    repository = _repository(tmp_path)
    record = repository.create_queued(dataset_id=None, config={"model_type": "pls_da"})
    run_dir = tmp_path / "runs" / record.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"pending"}', encoding="utf-8")
    _patch_run_storage(monkeypatch, tmp_path, repository)
    client = TestClient(app)

    active_response = client.delete(f"/api/training/runs/{record.run_id}")
    missing_response = client.delete("/api/training/runs/missing-run")

    assert active_response.status_code == 409
    assert missing_response.status_code == 404
    assert repository.exists(record.run_id)
    assert run_dir.exists()


def test_delete_filesystem_failure_keeps_repository_record(tmp_path, monkeypatch):
    from backend.app.main import app
    from backend.app.routers import runs

    repository = _repository(tmp_path)
    record = repository.import_legacy(
        run_id="locked-run",
        state="failed",
        config={"model_type": "svm"},
        dataset_id=None,
        legacy_data_path=None,
    )
    run_dir = tmp_path / "runs" / record.run_id
    run_dir.mkdir(parents=True)
    _patch_run_storage(monkeypatch, tmp_path, repository)
    monkeypatch.setattr(runs.shutil, "rmtree", lambda path: (_ for _ in ()).throw(PermissionError("locked")))
    client = TestClient(app)

    response = client.delete(f"/api/training/runs/{record.run_id}")

    assert response.status_code == 500
    assert repository.exists(record.run_id)
    assert run_dir.exists()
