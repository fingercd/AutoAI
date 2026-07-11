import json

from backend.app.runs.migration import import_legacy_runs
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
