from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..contracts import TrainingRunRequest, TrainingSpec
from ..http.principal import get_principal
from ..runs.artifacts import RunArtifactWriter
from ..runs.contracts import Principal, RunRecord
from ..runs.repository import InvalidRunTransition, RunNotFound
from ..runs.status_projection import project_status, recover_status_from_artifacts
from .deps import get_run_dir, get_run_repository, resolve_training_data_reference

router = APIRouter()


def _projection(record: RunRecord) -> dict[str, Any]:
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
    payload.update(
        {
            'run_id': record.run_id,
            'status': record.legacy_status,
            'state': record.state,
            'version': record.version,
            'dataset_id': record.dataset_id,
            'config': record.config,
            **record.progress,
        }
    )
    if record.error:
        payload['error'] = record.error
    else:
        payload.pop('error', None)
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    return recover_status_from_artifacts(status_file.parent, payload)


@router.post('/api/training/runs', status_code=202)
@router.post('/api/train', status_code=202)
def create_run(payload: TrainingRunRequest, principal: Principal = Depends(get_principal)) -> dict[str, object]:
    data_ref = resolve_training_data_reference(payload, principal=principal)
    config = TrainingSpec.from_legacy(payload.config).to_legacy_dict()
    if data_ref.test_dataset_id:
        config['test_dataset_id'] = data_ref.test_dataset_id
    if data_ref.test_legacy_path:
        config['test_data_path'] = data_ref.test_legacy_path
    repository = get_run_repository()
    record = repository.create_queued(
        dataset_id=data_ref.dataset_id,
        legacy_data_path=data_ref.legacy_path,
        config=config,
    )
    project_status(get_run_dir(record.run_id), record, config=config)
    return {'run_id': record.run_id, 'status': record.legacy_status, 'state': record.state}


@router.get('/api/training/runs')
def get_runs() -> list[dict[str, Any]]:
    return [_projection(record) for record in get_run_repository().list()]


@router.get('/api/training/runs/{run_id}')
def get_run(run_id: str) -> dict[str, Any]:
    try:
        record = get_run_repository().get(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    return _projection(record)


@router.post('/api/training/runs/{run_id}/cancel')
def cancel_run(run_id: str) -> dict[str, Any]:
    repository = get_run_repository()
    try:
        record = repository.cancel(run_id, now=datetime.now(timezone.utc))
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail='run 当前状态不能取消') from exc
    return project_status(get_run_dir(run_id), record)


@router.delete('/api/training/runs/{run_id}')
def delete_run(run_id: str) -> dict[str, object]:
    repository = get_run_repository()
    try:
        record = repository.get(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    if record.state in {'queued', 'running'}:
        raise HTTPException(status_code=409, detail='run 当前状态不能删除')

    run_dir = get_run_dir(run_id)
    try:
        if run_dir.exists():
            shutil.rmtree(run_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail='run 产物删除失败，训练记录已保留') from exc
    try:
        repository.delete_terminal(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail='run 当前状态不能删除') from exc
    return {'run_id': run_id, 'deleted': True}


@router.get('/api/training/runs/{run_id}/artifact/{name}')
def get_run_artifact(run_id: str, name: str) -> FileResponse:
    run_dir = get_run_dir(run_id)
    try:
        get_run_repository().get(run_id)
    except RunNotFound as exc:
        # 早期直接训练产物可能没有 SQLite 记录，仍保持只读兼容。
        # 正式删除会同时移除 Run 目录，因此不会被此兼容路径恢复访问。
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail='run 不存在') from exc
    try:
        path = RunArtifactWriter(run_dir).resolve_download(name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail='不允许下载该文件') from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='文件不存在') from exc
    return FileResponse(path, filename=Path(name).name)
