from __future__ import annotations

import json
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
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    return payload


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


@router.get('/api/training/runs/{run_id}/artifact/{name}')
def get_run_artifact(run_id: str, name: str) -> FileResponse:
    try:
        path = RunArtifactWriter(get_run_dir(run_id)).resolve_download(name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail='不允许下载该文件') from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='文件不存在') from exc
    return FileResponse(path, filename=Path(name).name)
