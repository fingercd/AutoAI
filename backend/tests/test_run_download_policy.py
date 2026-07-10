from fastapi.testclient import TestClient

from backend.app.main import app


def test_generic_files_route_rejects_a_run_artifact(tmp_path, monkeypatch):
    import backend.app.main as main

    runs = tmp_path / 'runs'
    run_file = runs / 'run-1' / 'dscarnet_pca.joblib'
    run_file.parent.mkdir(parents=True)
    run_file.write_bytes(b'private')
    monkeypatch.setattr(main, 'RUNS_DIR', runs)

    response = TestClient(app).get('/api/files', params={'path': str(run_file)})
    assert response.status_code == 403
