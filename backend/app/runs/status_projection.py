from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import RunRecord


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp')
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def project_status(run_dir: Path, record: RunRecord, **fields: Any) -> dict[str, Any]:
    payload = {
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'version': record.version,
        'dataset_id': record.dataset_id,
        **fields,
    }
    _atomic_json_write(run_dir / 'status.json', payload)
    return payload
