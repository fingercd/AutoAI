"""将只有 status.json 的历史 Run 幂等导入 SQLite 仓库。

历史 running 状态不能证明仍有活跃 worker，因此导入为 queued，由现行 lease/claim
机制重新领取；终态则按兼容映射保留。

系统位置：runs 子系统的一次性运维工具，把"纯文件时代"的 Run（目录里只有
status.json / manifest.json、没有 SQLite 记录）迁移进现行 RunRepository，
使旧结果能在新结果页按 run-result-v1 契约展示。

关键设计约束：
- 幂等：已存在的 run_id 跳过，可反复执行；
- 安全默认：不在服务启动时自动运行，必须人工显式触发，支持 --dry-run 审计；
- Principal 权限隔离：owner 为空的旧 Run 在 server 模式下默认不可见，只能通过
  --rebind-unowned 显式绑定到某个 owner_id/tenant_id。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .contracts import Principal
from .repository import RunRepository


# 历史 status 字段到现行 state 的映射。
# 关键决策：pending/running 一律导入为 queued —— 历史 running 无法证明当时仍有
# 活跃 worker，贸然恢复为 running 会成为永远无人认领的"僵尸"，交给现行
# lease/claim 机制重新排队领取才是安全做法；paused 没有对应现行状态，按 cancelled 处理。
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
    """幂等扫描历史目录；dry-run 只返回可导入数量，不修改数据库。

    参数:
        run_root: 历史 Run 根目录（默认 storage/runs），扫描其下 */status.json。
        repository: 目标 Run 仓库。
        principal: 导入后 Run 的归属（owner_id/tenant_id）；默认空 Principal 表示
            保持无归属（server 模式下不可见）。
        dry_run: 为 True 时只统计不写库，用于迁移前审计。
        rebind_unowned: 为 True 时，对已存在但无归属的 Run 重新绑定到 principal ——
            这是历史 Run 在 server 模式下可见的唯一途径。
        warning_sink: 可选告警回调（CLI 里接到 stderr），报告损坏文件。
    返回:
        本次导入或绑定（dry-run 时为"将会"）的 Run 数量。
    """
    imported = 0
    for status_path in sorted(run_root.glob('*/status.json')):
        # 按目录名排序遍历，保证多次运行顺序稳定，便于审计与日志对比
        try:
            payload = json.loads(status_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            # 损坏的状态文件不阻塞整体迁移：告警后跳过
            if warning_sink is not None:
                warning_sink(f'跳过损坏状态文件 {status_path}: {exc}')
            continue
        if not isinstance(payload, dict):
            if warning_sink is not None:
                warning_sink(f'跳过非对象状态文件 {status_path}')
            continue
        run_id = str(payload.get('run_id') or status_path.parent.name)
        # status.json 缺 run_id 时回退用目录名 —— 历史目录命名本身就是 run_id
        if repository.exists(run_id):
            # 已存在的 Run 默认跳过（幂等）；仅当显式 rebind 且该 Run 无归属、
            # 目标 principal 有归属时才改绑 —— 防止把已有主的 Run 误绑给别人
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
        # 只挑选快照需要的字段；dataset_name/dataset_sha256 在下面改名，
        # 对齐现行 dataset_snapshot 的 name/sha256 键
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
        # 优先信任现行 state 字段；缺失时回退按历史 status 映射；
        # 历史 running 一律降为 queued（设计说明见 STATE_FROM_LEGACY 上方注释）
        if canonical_state not in {'queued', 'running', 'succeeded', 'failed', 'cancelled'}:
            canonical_state = STATE_FROM_LEGACY.get(str(payload.get('status')), 'failed')
        elif canonical_state == 'running':
            canonical_state = 'queued'
        raw_config = payload.get('config')
        # config 必须是对象；历史脏数据按空配置导入并告警，不阻塞整体迁移
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
            # 训练进度字段原样保留，CV 场景（current_fold 等）在结果页仍可展示
            progress={
                key: payload[key]
                for key in ('current_fold', 'completed_folds', 'fold_progress_text', 'target_epochs')
                if payload.get(key) is not None
            },
            # 显式 manifest_name 优先；缺字段但磁盘上存在 manifest.json 时也登记，
            # 使新结果页能识别"Manifest 完整"的 Run
            manifest_name=(
                str(payload.get('manifest_name'))
                if payload.get('manifest_name')
                else ('manifest.json' if (status_path.parent / 'manifest.json').is_file() else None)
            ),
            error=str(payload.get('error')) if payload.get('error') is not None else None,
            # 只接受 dict 形态的 error_details，其他形态视为无
            error_details=(
                dict(payload.get('error_details'))
                if isinstance(payload.get('error_details'), dict)
                else None
            ),
        )
    return imported


def main() -> None:
    """提供可审计的 dry-run/显式 owner 绑定入口，不在服务启动时自动迁移。

    参数校验刻意严格：--owner-id 与 --tenant-id 必须成对出现且非空，
    --rebind-unowned 必须带归属信息 —— 迁移会改变数据可见性，
    宁可拒绝执行也不产生"半归属"的 Run。
    """
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
    # owner_id 与 tenant_id 在现行 Principal 模型中是不可分的整体
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
    # dry-run 与实际执行共用同一计数语义，输出措辞区分两者
    action = 'would import/bind' if args.dry_run else 'imported/bound'
    print(f'{action}: {count}')


if __name__ == '__main__':
    main()
