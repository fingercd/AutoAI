from __future__ import annotations

from datetime import datetime, timezone
import json

from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import pytest

from backend.app.contracts import TrainingRunRequest
from backend.app.http.security import (
    ServerAuthMiddleware,
    cors_origins,
    is_loopback_host,
    load_security_settings,
)
from backend.app.routers import auth, catalog
from backend.app.routers import deps as router_deps
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunNotFound, RunRepository


TOKEN = "a" * 32


def _server_app(*, allowed_origins: str = "https://autoai.example.edu") -> FastAPI:
    settings = load_security_settings(
        {
            "AUTOAI_DEPLOYMENT_MODE": "server",
            "AUTOAI_API_TOKEN": TOKEN,
            "AUTOAI_PRINCIPAL_ID": "research-team",
            "AUTOAI_TENANT_ID": "spectroscopy",
            "AUTOAI_ALLOWED_ORIGINS": allowed_origins,
        }
    )
    app = FastAPI()
    app.state.security_settings = settings
    app.add_middleware(ServerAuthMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins(settings)),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )
    app.include_router(auth.router)

    @app.get("/")
    def home() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/private")
    def private(request: Request) -> dict[str, object]:
        principal = request.state.principal
        return {
            "owner_id": principal.owner_id,
            "tenant_id": principal.tenant_id,
        }

    return app


def test_server_mode_requires_a_long_token_and_rejects_wildcard_cors() -> None:
    with pytest.raises(RuntimeError, match="AUTOAI_API_TOKEN"):
        load_security_settings({"AUTOAI_DEPLOYMENT_MODE": "server"})
    with pytest.raises(RuntimeError, match="至少 32"):
        load_security_settings(
            {
                "AUTOAI_DEPLOYMENT_MODE": "server",
                "AUTOAI_API_TOKEN": "too-short",
            }
        )
    with pytest.raises(RuntimeError, match=r"不允许使用 \*"):
        load_security_settings(
            {
                "AUTOAI_DEPLOYMENT_MODE": "server",
                "AUTOAI_API_TOKEN": TOKEN,
                "AUTOAI_ALLOWED_ORIGINS": "*",
            }
        )


def test_server_mode_exposes_only_non_secret_bootstrap_without_authentication() -> None:
    client = TestClient(_server_app())

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    response = client.get("/api/auth/config")

    assert response.status_code == 200
    assert response.json() == {
        "mode": "server",
        "auth_required": True,
        "token_storage": "session",
    }
    assert TOKEN not in response.text


def test_server_mode_protects_api_and_injects_principal_from_bearer_token() -> None:
    client = TestClient(_server_app())

    missing = client.get("/api/private")
    invalid = client.get("/api/private", headers={"Authorization": "Bearer invalid"})
    valid = client.get("/api/private", headers={"Authorization": f"Bearer {TOKEN}"})
    session = client.get("/api/auth/session", headers={"Authorization": f"Bearer {TOKEN}"})

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["error"]["code"] == "authentication_required"
    assert invalid.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == {
        "owner_id": "research-team",
        "tenant_id": "spectroscopy",
    }
    assert session.json() == {
        "authenticated": True,
        "principal": {
            "owner_id": "research-team",
            "tenant_id": "spectroscopy",
        },
    }


def test_server_cors_allows_only_explicit_origins() -> None:
    client = TestClient(_server_app())
    headers = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Authorization",
    }

    allowed = client.options(
        "/api/private",
        headers={"Origin": "https://autoai.example.edu", **headers},
    )
    denied = client.options(
        "/api/private",
        headers={"Origin": "https://evil.example", **headers},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://autoai.example.edu"
    assert "authorization" in allowed.headers["access-control-allow-headers"].lower()
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_local_bind_detection_does_not_treat_all_interfaces_as_loopback() -> None:
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("0.0.0.0") is False
    assert is_loopback_host("::") is False


def test_run_repository_scopes_records_to_the_server_derived_principal(tmp_path) -> None:
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.initialize()
    owner_a = Principal(owner_id="owner-a", tenant_id="tenant")
    owner_b = Principal(owner_id="owner-b", tenant_id="tenant")
    local = Principal()
    run_a = repository.create_queued(
        dataset_id=None,
        config={"model_type": "pls_da"},
        principal=owner_a,
    )
    run_b = repository.create_queued(
        dataset_id=None,
        config={"model_type": "svm"},
        principal=owner_b,
    )
    legacy_local = repository.create_queued(
        dataset_id=None,
        config={"model_type": "pca_lda"},
        principal=local,
    )

    assert repository.get_scoped(run_a.run_id, principal=owner_a).run_id == run_a.run_id
    assert [record.run_id for record in repository.list_scoped(principal=owner_a)] == [run_a.run_id]
    assert [record.run_id for record in repository.list_scoped(principal=owner_b)] == [run_b.run_id]
    assert [record.run_id for record in repository.list_scoped(principal=local)] == [legacy_local.run_id]

    with pytest.raises(RunNotFound):
        repository.get_scoped(run_a.run_id, principal=owner_b)
    with pytest.raises(RunNotFound):
        repository.get_scoped(legacy_local.run_id, principal=owner_a)


@pytest.mark.parametrize(
    'payload',
    [
        TrainingRunRequest(data_path='D:/server/data.csv'),
        TrainingRunRequest(),
        TrainingRunRequest(dataset_id='dataset-1', test_data_path='D:/server/test.csv'),
    ],
)
def test_server_principal_requires_stable_dataset_ids(payload) -> None:
    principal = Principal(owner_id='server-admin', tenant_id='default')

    with pytest.raises(HTTPException) as caught:
        router_deps.resolve_training_data_reference(payload, principal=principal)

    assert caught.value.status_code == 422
    assert 'dataset_id' in str(caught.value.detail)


def test_server_principal_can_resolve_owned_dataset_ids(monkeypatch) -> None:
    principal = Principal(owner_id='server-admin', tenant_id='default')

    class Dataset:
        original_name = 'owned.csv'

    class Repository:
        def resolve(self, dataset_id, *, principal):
            assert dataset_id == 'dataset-1'
            assert principal == Principal(owner_id='server-admin', tenant_id='default')
            return Dataset()

    monkeypatch.setattr(router_deps, '_dataset_repository', lambda: Repository())

    reference = router_deps.resolve_training_data_reference(
        TrainingRunRequest(dataset_id='dataset-1'),
        principal=principal,
    )

    assert reference.dataset_id == 'dataset-1'
    assert reference.legacy_path is None
    assert reference.dataset_name == 'owned.csv'


def test_local_principal_keeps_controlled_legacy_path_compatibility(tmp_path, monkeypatch) -> None:
    source = tmp_path / 'legacy.csv'
    source.write_text('Index,Label,Sample_ID,0\n', encoding='utf-8')
    monkeypatch.setattr(router_deps, '_dataset_repository', lambda: object())
    monkeypatch.setattr(router_deps, 'UPLOADS_DIR', tmp_path)
    monkeypatch.setattr(router_deps, 'PREPROCESSED_DIR', tmp_path / 'preprocessed')
    monkeypatch.setattr(router_deps, 'DEFAULT_DATA', tmp_path / 'default.csv')

    reference = router_deps.resolve_training_data_reference(
        TrainingRunRequest(data_path=str(source)),
        principal=Principal(),
    )

    assert reference.dataset_id is None
    assert reference.legacy_path == str(source.resolve())
    assert reference.dataset_name == 'legacy.csv'


def test_health_remains_compatible_and_reports_no_worker(tmp_path, monkeypatch) -> None:
    from backend.app import paths
    from backend.app.main import app as main_app
    from backend.app.version import WEB_CONTRACTS

    monkeypatch.setattr(paths, 'RUNS_DATABASE', tmp_path / 'runs.sqlite3')

    response = TestClient(main_app).get('/health')

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['deployment_mode'] == 'local'
    assert response.json()['contracts'] == WEB_CONTRACTS
    assert response.json()['worker'] == {
        'available': False,
        'compatible': False,
        'contract_version': None,
        'live_count': 0,
        'last_seen_at': None,
        'active_run_count': 0,
    }


def test_health_reports_live_worker_summary_without_exposing_worker_id(tmp_path, monkeypatch) -> None:
    from backend.app import paths
    from backend.app.main import app as main_app
    from backend.app.version import WORKER_CONTRACT_VERSION

    database = tmp_path / 'runs.sqlite3'
    repository = RunRepository(database)
    repository.initialize()
    repository.record_worker_heartbeat(
        worker_id='host-secret-worker-id',
        now=datetime.now(timezone.utc),
        active_run_id='active-run',
        contract_version=WORKER_CONTRACT_VERSION,
    )
    monkeypatch.setattr(paths, 'RUNS_DATABASE', database)

    payload = TestClient(main_app).get('/health').json()

    assert payload['status'] == 'ok'
    assert payload['worker']['available'] is True
    assert payload['worker']['compatible'] is True
    assert payload['worker']['contract_version'] == WORKER_CONTRACT_VERSION
    assert payload['worker']['live_count'] == 1
    assert payload['worker']['active_run_count'] == 1
    assert payload['worker']['last_seen_at']
    assert 'host-secret-worker-id' not in json.dumps(payload)


def test_server_health_is_public_and_contains_no_secret_identity_data(tmp_path, monkeypatch) -> None:
    from backend.app import paths
    from backend.app.version import WORKER_CONTRACT_VERSION

    settings = load_security_settings(
        {
            'AUTOAI_DEPLOYMENT_MODE': 'server',
            'AUTOAI_API_TOKEN': TOKEN,
            'AUTOAI_PRINCIPAL_ID': 'private-owner',
            'AUTOAI_TENANT_ID': 'private-tenant',
        }
    )
    database = tmp_path / 'runs.sqlite3'
    repository = RunRepository(database)
    repository.initialize()
    repository.record_worker_heartbeat(
        worker_id='private-worker-id',
        now=datetime.now(timezone.utc),
        contract_version=WORKER_CONTRACT_VERSION,
    )
    monkeypatch.setattr(paths, 'RUNS_DATABASE', database)
    server_app = FastAPI()
    server_app.state.security_settings = settings
    server_app.add_middleware(ServerAuthMiddleware, settings=settings)
    server_app.include_router(catalog.router)

    response = TestClient(server_app).get('/health')

    assert response.status_code == 200
    payload_text = response.text
    assert response.json()['status'] == 'ok'
    assert response.json()['deployment_mode'] == 'server'
    assert response.json()['worker']['available'] is True
    assert response.json()['worker']['compatible'] is True
    assert TOKEN not in payload_text
    assert 'private-owner' not in payload_text
    assert 'private-tenant' not in payload_text
    assert 'private-worker-id' not in payload_text


def test_health_marks_legacy_worker_without_contract_version_incompatible(tmp_path, monkeypatch) -> None:
    from backend.app import paths
    from backend.app.main import app as main_app

    database = tmp_path / 'runs.sqlite3'
    repository = RunRepository(database)
    repository.initialize()
    repository.record_worker_heartbeat(
        worker_id='legacy-worker',
        now=datetime.now(timezone.utc),
    )
    monkeypatch.setattr(paths, 'RUNS_DATABASE', database)

    worker = TestClient(main_app).get('/health').json()['worker']

    assert worker['available'] is True
    assert worker['compatible'] is False
    assert worker['contract_version'] is None
