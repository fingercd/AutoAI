"""将只有 status.json 的历史 Run 幂等导入 SQLite 仓库。

历史 running 状态不能证明仍有活跃 worker，因此导入为 queued，由现行 lease/claim
机制重新领取；终态则按兼容映射保留。
"""

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
    """扫描历史目录并导入尚不存在的 Run，返回新增数量。"""
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
