"""训练 Run 的创建、查询、取消、删除与 artifact 下载路由。

SQLite RunRecord 是规范状态；status.json 只补充旧前端字段。创建接口只写入 queued
Run，取消和删除均通过仓库状态机校验。成功 Run 的下载必须通过 manifest 条目，
不会把整个 storage/runs 目录暴露给通用文件接口。
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..contracts import TrainingConfigValidationError, TrainingRunRequest, TrainingSpec
from ..datasets.repository import DatasetRepository
from ..http.principal import get_principal
from ..paths import DATASETS_DATABASE, STORAGE_DIR
from ..runs.artifacts import ArtifactIntegrityError, ManifestCorruptError, RunArtifactWriter
from ..runs.contracts import Principal, RunRecord, public_error_message
from ..runs.repository import InvalidRunTransition, RunNotFound
from ..runs.result_projection import project_run_result
from ..runs.status_projection import project_status, recover_status_from_artifacts
from .deps import get_run_dir, get_run_repository, resolve_training_data_reference

router = APIRouter()


def _without_server_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_server_paths(item)
            for key, item in value.items()
            if key not in {'data_path', 'test_data_path', 'run_dir', 'path'}
            and not key.endswith('_path')
        }
    if isinstance(value, list):
        return [_without_server_paths(item) for item in value]
    return value


def _dataset_name_for_record(record: RunRecord) -> str | None:
    """优先读取 Run 快照，并为升级前的 Run 回查原始上传文件名。"""
    configured = record.config.get('dataset_name')
    if configured:
        return str(configured)
    if record.dataset_id:
        try:
            repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
            repository.initialize()
            return repository.resolve_system(record.dataset_id, legacy_path=None).original_name
        except (FileNotFoundError, PermissionError, OSError):
            pass
    if record.legacy_data_path:
        return Path(record.legacy_data_path).name
    return None


def _projection(record: RunRecord) -> dict[str, Any]:
    """合并规范 RunRecord、兼容 status.json 和可恢复的 artifact 摘要。"""
    status_file = get_run_dir(record.run_id) / 'status.json'
    payload: dict[str, Any] = {}
    if status_file.is_file():
        try:
            loaded = json.loads(status_file.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except json.JSONDecodeError:
            payload = {}
    payload = recover_status_from_artifacts(status_file.parent, payload)
    dataset_name = _dataset_name_for_record(record)
    payload.update(
        {
            'run_id': record.run_id,
            'status': record.legacy_status,
            'state': record.state,
            'version': record.version,
            'dataset_id': record.dataset_id,
            'dataset_name': dataset_name,
            'config': record.config,
            'created_at': record.created_at,
            'started_at': record.started_at,
            'completed_at': record.finished_at,
            **record.progress,
        }
    )
    if record.error:
        payload['error'] = public_error_message(record.error)
        if record.error_details:
            payload['error_details'] = {
                **record.error_details,
                'message': public_error_message(record.error_details.get('message') or record.error),
            }
    else:
        payload.pop('error', None)
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    return _without_server_paths(recover_status_from_artifacts(status_file.parent, payload))


def _dataset_snapshot_for_reference(
    *,
    dataset_id: str | None,
    legacy_path: str | None,
    dataset_name: str | None,
) -> dict[str, Any]:
    repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    repository.initialize()
    record = repository.resolve_system(dataset_id, legacy_path=legacy_path)
    actual_sha256 = repository.verify_integrity(record)
    return {
        'dataset_id': dataset_id,
        'name': dataset_name or record.original_name,
        'sha256': actual_sha256,
    }


def _summary_projection(record: RunRecord) -> dict[str, Any]:
    snapshot = record.dataset_snapshot
    result_state = {
        'queued': 'pending',
        'running': 'running',
        'failed': 'failed',
        'cancelled': 'cancelled',
    }.get(record.state)
    if result_state is None:
        run_dir = get_run_dir(record.run_id)
        try:
            _manifest, descriptors = RunArtifactWriter(run_dir).descriptors(run_id=record.run_id)
        except FileNotFoundError:
            result_state = 'missing_manifest'
        except ManifestCorruptError:
            result_state = 'corrupt_manifest'
        else:
            damaged = any(
                item['integrity'] in {'missing', 'corrupt'}
                and (item['required'] or item['applicable'])
                for item in descriptors
            )
            result_state = 'partial' if damaged else 'ready'
    return {
        'run_id': record.run_id,
        'state': record.state,
        'status': record.legacy_status,
        'result_state': result_state,
        'model_type': record.config.get('model_type'),
        'dataset_id': record.dataset_id,
        'dataset_name': snapshot.get('name') or record.config.get('dataset_name'),
        'created_at': record.created_at,
        'started_at': record.started_at,
        'finished_at': record.finished_at,
        'progress': _without_server_paths(record.progress),
        'error': public_error_message(record.error) if record.error else None,
    }


@router.post('/api/training/runs', status_code=202)
@router.post('/api/train', status_code=202)
def create_run(payload: TrainingRunRequest, principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """持久化 queued Run；本请求绝不直接调用训练器。"""
    data_ref = resolve_training_data_reference(payload, principal=principal)
    try:
        spec = TrainingSpec.from_legacy(payload.config).validated(
            has_external_test=bool(data_ref.test_dataset_id or data_ref.test_legacy_path)
        )
    except TrainingConfigValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={'code': 'invalid_training_config', 'message': str(exc)},
        ) from exc
    config = spec.to_legacy_dict()
    config['dataset_name'] = data_ref.dataset_name
    if data_ref.test_dataset_id:
        config['test_dataset_id'] = data_ref.test_dataset_id
    if data_ref.test_legacy_path:
        config['test_data_path'] = data_ref.test_legacy_path
    if data_ref.test_dataset_name:
        config['test_dataset_name'] = data_ref.test_dataset_name
    dataset_snapshot = _dataset_snapshot_for_reference(
        dataset_id=data_ref.dataset_id,
        legacy_path=data_ref.legacy_path,
        dataset_name=data_ref.dataset_name,
    )
    if data_ref.test_dataset_id or data_ref.test_legacy_path:
        test_snapshot = _dataset_snapshot_for_reference(
            dataset_id=data_ref.test_dataset_id,
            legacy_path=data_ref.test_legacy_path,
            dataset_name=data_ref.test_dataset_name,
        )
        config['test_dataset_sha256'] = test_snapshot['sha256']
    repository = get_run_repository()
    record = repository.create_queued(
        dataset_id=data_ref.dataset_id,
        legacy_data_path=data_ref.legacy_path,
        config=config,
        dataset_snapshot=dataset_snapshot,
        principal=principal,
    )
    project_status(get_run_dir(record.run_id), record, config=config)
    return {
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'warnings': list(spec.warnings),
    }


@router.get('/api/training/runs')
def get_runs(
    projection: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=100),
    cursor: str | None = Query(default=None),
    principal: Principal = Depends(get_principal),
) -> Any:
    """默认保持旧列表；projection=summary 返回轻量、可游标分页的结果。"""
    if projection not in {None, 'full', 'summary'}:
        raise HTTPException(status_code=422, detail='projection 仅支持 full 或 summary')
    repository = get_run_repository()
    try:
        records = repository.list_scoped(
            principal=principal,
            limit=(limit or 20) + 1 if projection == 'summary' else limit,
            cursor=cursor,
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='cursor 对应的 run 不存在') from exc
    if projection == 'summary':
        page_limit = limit or 20
        has_more = len(records) > page_limit
        page = records[:page_limit]
        return {
            'items': [_summary_projection(record) for record in page],
            'next_cursor': page[-1].run_id if has_more and page else None,
        }
    return [_projection(record) for record in records]


@router.get('/api/training/runs/{run_id}')
def get_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """返回单个 Run 的规范状态与兼容训练结果投影。"""
    try:
        record = get_run_repository().get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    return _projection(record)


@router.get('/api/training/runs/{run_id}/result')
def get_run_result(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """返回版本化、状态感知且可刷新恢复的建模结果页契约。"""
    try:
        record = get_run_repository().get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    return project_run_result(
        record,
        run_dir=get_run_dir(run_id),
        dataset_name=_dataset_name_for_record(record),
    )


@router.post('/api/training/runs/{run_id}/cancel')
def cancel_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """事务性取消 queued/running Run，并生成兼容 paused 投影。"""
    repository = get_run_repository()
    try:
        record = repository.cancel_scoped(run_id, now=datetime.now(timezone.utc), principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail='run 当前状态不能取消') from exc
    return project_status(get_run_dir(run_id), record)


@router.delete('/api/training/runs/{run_id}')
def delete_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """只删除终态 Run；目录先原子移入隔离区，DB 失败时可恢复。"""
    repository = get_run_repository()
    try:
        record = repository.get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    if record.state in {'queued', 'running'}:
        raise HTTPException(status_code=409, detail='run 当前状态不能删除')

    run_dir = get_run_dir(run_id)
    quarantine_dir: Path | None = None
    try:
        if run_dir.exists():
            # 先验证清理能力，避免目录已移走后才发现清理函数不可用。
            probe = run_dir.parent / f'.delete-probe-{uuid.uuid4().hex}'
            probe.mkdir(parents=True, exist_ok=False)
            try:
                shutil.rmtree(probe)
            finally:
                if probe.exists():
                    probe.rmdir()
            quarantine_dir = run_dir.parent / f'.{run_id}.deleting-{uuid.uuid4().hex}'
            run_dir.replace(quarantine_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail='run 产物删除失败，训练记录已保留') from exc
    try:
        repository.delete_terminal_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=409, detail='run 当前状态不能删除') from exc
    except Exception as exc:
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=500, detail='run 记录删除失败，产物目录已恢复') from exc
    if quarantine_dir is not None and quarantine_dir.exists():
        try:
            shutil.rmtree(quarantine_dir)
        except OSError as exc:
            repository.import_legacy(
                run_id=record.run_id,
                state=record.state,
                config=record.config,
                dataset_id=record.dataset_id,
                legacy_data_path=record.legacy_data_path,
                principal=Principal(owner_id=record.owner_id, tenant_id=record.tenant_id),
                dataset_snapshot=record.dataset_snapshot,
                created_at=record.created_at,
                started_at=record.started_at,
                finished_at=record.finished_at,
                progress=record.progress,
                manifest_name=record.manifest_name,
                error=record.error,
                error_details=record.error_details,
            )
            if quarantine_dir.exists() and not run_dir.exists():
                quarantine_dir.replace(run_dir)
            raise HTTPException(status_code=500, detail='run 清理失败，训练记录和剩余产物已恢复') from exc
    return {'run_id': run_id, 'deleted': True}


@router.get('/api/training/runs/{run_id}/artifact/{name}')
def get_run_artifact(
    run_id: str,
    name: str,
    principal: Principal = Depends(get_principal),
) -> FileResponse:
    """仅返回 Manifest 中存在且标为 downloadable 的单个产物。"""
    run_dir = get_run_dir(run_id)
    try:
        record = get_run_repository().get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        # 早期直接训练产物可能没有 SQLite 记录，仍保持只读兼容。
        # 正式删除会同时移除 Run 目录，因此不会被此兼容路径恢复访问。
        if principal.owner_id is not None or principal.tenant_id is not None or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail='run 不存在') from exc
    else:
        if record.state != 'succeeded':
            raise HTTPException(status_code=409, detail='run 尚未成功完成，产物不可下载')
    try:
        path = RunArtifactWriter(run_dir).resolve_download(name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail='不允许下载该文件') from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='文件不存在') from exc
    except (ManifestCorruptError, ArtifactIntegrityError) as exc:
        raise HTTPException(status_code=409, detail='文件完整性校验失败') from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='文件名无效') from exc
    return FileResponse(path, filename=Path(name).name)
