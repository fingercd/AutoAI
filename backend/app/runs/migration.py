"""将只有 status.json 的历史 Run 幂等导入 SQLite 仓库。

历史 running 状态不能证明仍有活跃 worker，因此导入为 queued，由现行 lease/claim
机制重新领取；终态则按兼容映射保留。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .contracts import Principal
from .repository import RunRepository


STATE_FROM_LEGACY = {
    'pending': 'queued',
    'running': 'queued',
    'success': 'succeeded',
    'failed': 'failed',
    'paused': 'cancelled',
}


def import_legacy_runs(
    *,
    run_root: Path,
    repository: RunRepository,
    principal: Principal = Principal(),
    dry_run: bool = False,
    rebind_unowned: bool = False,
    warning_sink: Callable[[str], None] | None = None,
) -> int:
    """幂等扫描历史目录；dry-run 只返回可导入数量，不修改数据库。"""
    imported = 0
    for status_path in sorted(run_root.glob('*/status.json')):
        try:
            payload = json.loads(status_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            if warning_sink is not None:
                warning_sink(f'跳过损坏状态文件 {status_path}: {exc}')
            continue
        if not isinstance(payload, dict):
            if warning_sink is not None:
                warning_sink(f'跳过非对象状态文件 {status_path}')
            continue
        run_id = str(payload.get('run_id') or status_path.parent.name)
        if repository.exists(run_id):
            if rebind_unowned:
                current = repository.get(run_id)
                is_unowned = current.owner_id is None and current.tenant_id is None
                target_is_scoped = principal.owner_id is not None or principal.tenant_id is not None
                if is_unowned and target_is_scoped:
                    imported += 1
                    if not dry_run:
                        repository.assign_unowned(run_id, principal=principal)
            continue
        imported += 1
        if dry_run:
            continue
        snapshot = {
            key: payload[key]
            for key in ('dataset_name', 'dataset_sha256', 'curve_count', 'sample_id_count', 'class_count', 'feature_count')
            if payload.get(key) is not None
        }
        if 'dataset_name' in snapshot:
            snapshot['name'] = snapshot.pop('dataset_name')
        if 'dataset_sha256' in snapshot:
            snapshot['sha256'] = snapshot.pop('dataset_sha256')
        canonical_state = str(payload.get('state') or '')
        if canonical_state not in {'queued', 'running', 'succeeded', 'failed', 'cancelled'}:
            canonical_state = STATE_FROM_LEGACY.get(str(payload.get('status')), 'failed')
        elif canonical_state == 'running':
            canonical_state = 'queued'
        raw_config = payload.get('config')
        if not isinstance(raw_config, dict):
            if warning_sink is not None and raw_config is not None:
                warning_sink(f'{status_path} 的 config 无效，已按空配置导入')
            raw_config = {}
        repository.import_legacy(
            run_id=run_id,
            state=canonical_state,
            config=dict(raw_config),
            dataset_id=payload.get('dataset_id'),
            legacy_data_path=payload.get('data_path'),
            principal=principal,
            dataset_snapshot=snapshot,
            created_at=payload.get('created_at'),
            started_at=payload.get('started_at'),
            finished_at=payload.get('completed_at'),
            progress={
                key: payload[key]
                for key in ('current_fold', 'completed_folds', 'fold_progress_text', 'target_epochs')
                if payload.get(key) is not None
            },
            manifest_name=(
                str(payload.get('manifest_name'))
                if payload.get('manifest_name')
                else ('manifest.json' if (status_path.parent / 'manifest.json').is_file() else None)
            ),
            error=str(payload.get('error')) if payload.get('error') is not None else None,
            error_details=(
                dict(payload.get('error_details'))
                if isinstance(payload.get('error_details'), dict)
                else None
            ),
        )
    return imported


def main() -> None:
    """提供可审计的 dry-run/显式 owner 绑定入口，不在服务启动时自动迁移。"""
    from ..paths import RUNS_DATABASE, RUNS_DIR

    parser = argparse.ArgumentParser(description='幂等导入或绑定历史 AutoAI Run')
    parser.add_argument('--run-root', type=Path, default=RUNS_DIR)
    parser.add_argument('--database', type=Path, default=RUNS_DATABASE)
    parser.add_argument('--owner-id')
    parser.add_argument('--tenant-id')
    parser.add_argument('--rebind-unowned', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (args.owner_id is None) != (args.tenant_id is None):
        parser.error('--owner-id 与 --tenant-id 必须同时提供')
    if args.owner_id is not None and (not args.owner_id.strip() or not args.tenant_id.strip()):
        parser.error('--owner-id 与 --tenant-id 不能为空')
    if args.rebind_unowned and (not args.owner_id or not args.tenant_id):
        parser.error('--rebind-unowned 必须同时提供非空 --owner-id 和 --tenant-id')
    owner_id = args.owner_id.strip() if args.owner_id is not None else None
    tenant_id = args.tenant_id.strip() if args.tenant_id is not None else None
    repository = RunRepository(args.database)
    repository.initialize()
    count = import_legacy_runs(
        run_root=args.run_root,
        repository=repository,
        principal=Principal(owner_id=owner_id, tenant_id=tenant_id),
        dry_run=args.dry_run,
        rebind_unowned=args.rebind_unowned,
        warning_sink=lambda message: print(message, file=sys.stderr),
    )
    action = 'would import/bind' if args.dry_run else 'imported/bound'
    print(f'{action}: {count}')


if __name__ == '__main__':
    main()
