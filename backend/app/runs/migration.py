from __future__ import annotations

import json
from pathlib import Path

from .repository import RunRepository


STATE_FROM_LEGACY = {
    'pending': 'queued',
    'running': 'queued',
    'success': 'succeeded',
    'failed': 'failed',
    'paused': 'cancelled',
}


def import_legacy_runs(*, run_root: Path, repository: RunRepository) -> int:
    imported = 0
    for status_path in sorted(run_root.glob('*/status.json')):
        payload = json.loads(status_path.read_text(encoding='utf-8'))
        run_id = str(payload.get('run_id') or status_path.parent.name)
        if repository.exists(run_id):
            continue
        repository.import_legacy(
            run_id=run_id,
            state=STATE_FROM_LEGACY.get(str(payload.get('status')), 'failed'),
            config=dict(payload.get('config') or {}),
            dataset_id=payload.get('dataset_id'),
            legacy_data_path=payload.get('data_path'),
        )
        imported += 1
    return imported
