import json
import subprocess
import sys

from backend.app.runs.migration import import_legacy_runs
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository


def test_import_legacy_status_json_is_idempotent(tmp_path):
    run_dir = tmp_path / 'runs' / 'legacy-1'
    run_dir.mkdir(parents=True)
    (run_dir / 'status.json').write_text(json.dumps({'run_id': 'legacy-1', 'status': 'success', 'config': {'model_type': 'pls_da'}}), encoding='utf-8')
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()

    assert import_legacy_runs(run_root=tmp_path / 'runs', repository=repo) == 1
    assert import_legacy_runs(run_root=tmp_path / 'runs', repository=repo) == 0
    assert repo.get('legacy-1').state == 'succeeded'


def test_dry_run_reports_import_without_modifying_database_or_files(tmp_path):
    run_dir = tmp_path / 'runs' / 'legacy-dry-run'
    run_dir.mkdir(parents=True)
    status = run_dir / 'status.json'
    status.write_text(
        json.dumps({'run_id': 'legacy-dry-run', 'status': 'success'}),
        encoding='utf-8',
    )
    original = status.read_bytes()
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()

    count = import_legacy_runs(
        run_root=tmp_path / 'runs',
        repository=repo,
        dry_run=True,
    )

    assert count == 1
    assert not repo.exists('legacy-dry-run')
    assert status.read_bytes() == original


def test_rebind_unowned_run_is_explicit_and_idempotent(tmp_path):
    run_dir = tmp_path / 'runs' / 'legacy-owned'
    run_dir.mkdir(parents=True)
    status = run_dir / 'status.json'
    status.write_text(
        json.dumps({'run_id': 'legacy-owned', 'status': 'success'}),
        encoding='utf-8',
    )
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    repo.import_legacy(
        run_id='legacy-owned',
        state='succeeded',
        config={},
        dataset_id=None,
        legacy_data_path=None,
    )
    principal = Principal(owner_id='server-admin', tenant_id='default')

    assert import_legacy_runs(
        run_root=tmp_path / 'runs',
        repository=repo,
        principal=principal,
        dry_run=True,
        rebind_unowned=True,
    ) == 1
    assert repo.get('legacy-owned').owner_id is None

    assert import_legacy_runs(
        run_root=tmp_path / 'runs',
        repository=repo,
        principal=principal,
        rebind_unowned=True,
    ) == 1
    rebound = repo.get('legacy-owned')
    assert rebound.owner_id == 'server-admin'
    assert rebound.tenant_id == 'default'
    assert import_legacy_runs(
        run_root=tmp_path / 'runs',
        repository=repo,
        principal=principal,
        rebind_unowned=True,
    ) == 0


def test_migration_cli_supports_audited_server_rebind(tmp_path):
    run_root = tmp_path / 'runs'
    run_dir = run_root / 'legacy-cli'
    run_dir.mkdir(parents=True)
    (run_dir / 'status.json').write_text(
        json.dumps({'run_id': 'legacy-cli', 'status': 'success'}),
        encoding='utf-8',
    )
    database = tmp_path / 'runs.sqlite3'
    repo = RunRepository(database)
    repo.initialize()
    repo.import_legacy(
        run_id='legacy-cli',
        state='succeeded',
        config={},
        dataset_id=None,
        legacy_data_path=None,
    )
    command = [
        sys.executable,
        '-m',
        'backend.app.runs.migration',
        '--run-root',
        str(run_root),
        '--database',
        str(database),
        '--owner-id',
        'server-admin',
        '--tenant-id',
        'default',
        '--rebind-unowned',
    ]

    dry_run = subprocess.run(
        [*command, '--dry-run'],
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )
    assert dry_run.returncode == 0
    assert 'would import/bind: 1' in dry_run.stdout
    assert RunRepository(database).get('legacy-cli').owner_id is None

    applied = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )
    assert applied.returncode == 0
    assert 'imported/bound: 1' in applied.stdout
    assert RunRepository(database).get('legacy-cli').owner_id == 'server-admin'
