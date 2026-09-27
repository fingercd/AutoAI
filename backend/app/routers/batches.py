"""Queued multi-model training batches and safe comparison endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..contracts import TrainingBatchRequest, TrainingConfigValidationError, TrainingSpec
from ..http.principal import get_principal
from ..models import canonical_model_type
from ..runs.batch_projection import batch_status_payload, project_model_comparison
from ..runs.contracts import Principal
from ..runs.repository import InvalidRunTransition, RunNotFound
from ..runs.status_projection import project_status
from .deps import get_run_dir, get_run_repository, resolve_training_data_reference
from .runs import _assert_worker_contract_compatible, _dataset_snapshot_for_reference

router = APIRouter()


def _batch_config(payload: TrainingBatchRequest, *, has_external_test: bool) -> tuple[list[str], dict[str, Any], list[str]]:
    raw = dict(payload.config or {})
    forbidden = {"model_type", "seed", "split_seed", "model_seed", "feature_selection_enabled", "explainability_enabled"}
    supplied = sorted(forbidden.intersection(raw))
    if supplied:
        raise TrainingConfigValidationError(f"Batch config 不允许包含：{', '.join(supplied)}")
    if isinstance(payload.base_seed, bool):
        raise TrainingConfigValidationError("base_seed 必须是整数")
    try:
        base_seed = int(payload.base_seed)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigValidationError("base_seed 必须是整数") from exc
    canonical: list[str] = []
    warnings: list[str] = []
    normalized_config: dict[str, Any] | None = None
    for raw_model in payload.model_types:
        model_type = canonical_model_type(str(raw_model))
        if model_type in canonical:
            continue
        spec = TrainingSpec.from_legacy({**raw, "model_type": model_type}).validated(has_external_test=has_external_test)
        canonical.append(str(spec.to_legacy_dict()["model_type"]))
        warnings.extend(spec.warnings)
        candidate = spec.to_legacy_dict()
        candidate.pop("model_type", None)
        if normalized_config is None:
            normalized_config = candidate
    if not canonical:
        raise TrainingConfigValidationError("至少选择一个不同的模型")
    if len(canonical) * int(payload.repeat_count) > 50:
        raise TrainingConfigValidationError("模型数 × 重复次数不能超过 50")
    config = dict(normalized_config or {})
    config["split_seed"] = base_seed
    config["feature_selection_enabled"] = False
    return canonical, config, warnings


@router.post("/api/training/batches", status_code=202)
def create_batch(payload: TrainingBatchRequest, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """Validate every child first, then atomically enqueue all queued Runs."""
    data_ref = resolve_training_data_reference(payload, principal=principal)
    try:
        model_types, config, warnings = _batch_config(payload, has_external_test=bool(data_ref.test_dataset_id or data_ref.test_legacy_path))
    except (TrainingConfigValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_training_batch", "message": str(exc)}) from exc
    repository = get_run_repository()
    _assert_worker_contract_compatible(repository)
    config["dataset_name"] = data_ref.dataset_name
    if data_ref.test_dataset_id:
        config["test_dataset_id"] = data_ref.test_dataset_id
    if data_ref.test_legacy_path:
        config["test_data_path"] = data_ref.test_legacy_path
    if data_ref.test_dataset_name:
        config["test_dataset_name"] = data_ref.test_dataset_name
    snapshot = _dataset_snapshot_for_reference(dataset_id=data_ref.dataset_id, legacy_path=data_ref.legacy_path, dataset_name=data_ref.dataset_name)
    if data_ref.test_dataset_id or data_ref.test_legacy_path:
        test_snapshot = _dataset_snapshot_for_reference(dataset_id=data_ref.test_dataset_id, legacy_path=data_ref.test_legacy_path, dataset_name=data_ref.test_dataset_name)
        config["test_dataset_sha256"] = test_snapshot["sha256"]
    try:
        batch, records = repository.create_batch_queued(
            dataset_id=data_ref.dataset_id, test_dataset_id=data_ref.test_dataset_id,
            legacy_data_path=data_ref.legacy_path, config=config, model_types=model_types,
            repeat_count=int(payload.repeat_count), base_seed=int(payload.base_seed),
            dataset_snapshot=snapshot, principal=principal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_training_batch", "message": str(exc)}) from exc
    for record in records:
        project_status(get_run_dir(record.run_id), record, config=record.config)
    return {
        "batch_id": batch.batch_id, "state": "queued", "model_count": len(model_types),
        "repeat_count": batch.repeat_count, "run_count": len(records),
        "runs": [{"model_type": record.config["model_type"], "repeat_index": record.batch_repeat_index, "run_id": record.run_id, "state": record.state} for record in records],
        "warnings": warnings,
    }


@router.get("/api/training/batches/{batch_id}")
def get_batch(batch_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    repository = get_run_repository()
    try:
        batch = repository.get_batch_scoped(batch_id, principal=principal)
        records = repository.list_batch_runs_scoped(batch_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail="batch 不存在") from exc
    return {"batch_id": batch.batch_id, "model_types": batch.model_types, "repeat_count": batch.repeat_count, "base_seed": batch.base_seed, "created_at": batch.created_at, **batch_status_payload(batch.batch_id, records)}


@router.post("/api/training/batches/{batch_id}/stop")
def stop_batch(batch_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    repository = get_run_repository()
    try:
        records = repository.cancel_batch_scoped(batch_id, now=datetime.now(timezone.utc), principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail="batch 不存在") from exc
    from ..runs.artifacts import discard_run_artifacts
    from ..runs.batch_archive import discard_batch_archive
    discard_batch_archive(repository, batch_id)
    for record in records:
        try:
            discard_run_artifacts(get_run_dir(record.run_id))
        except OSError:
            # A stopping worker may briefly hold a Windows file handle; its
            # shutdown/startup cleanup retries after releasing that handle.
            pass
    return batch_status_payload(batch_id, records)


@router.delete("/api/training/batches/{batch_id}")
def delete_batch(batch_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    repository = get_run_repository()
    try:
        run_ids = repository.delete_batch_terminal_scoped(batch_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail="batch 不存在") from exc
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail="批次仍有正在执行的任务，不能删除") from exc
    # DB deletion is deliberately transactional before best-effort file cleanup;
    # stale directories expose no download route once their records are gone.
    from ..runs.artifacts import discard_run_artifacts
    from ..runs.batch_archive import discard_batch_archive
    discard_batch_archive(repository, batch_id)
    for run_id in run_ids:
        try:
            discard_run_artifacts(get_run_dir(run_id))
        except OSError:
            pass
    return {"batch_id": batch_id, "deleted_run_count": len(run_ids)}


@router.get("/api/training/batches/{batch_id}/comparison")
def get_batch_comparison(batch_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    repository = get_run_repository()
    try:
        repository.get_batch_scoped(batch_id, principal=principal)
        records = repository.list_batch_runs_scoped(batch_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail="batch 不存在") from exc
    return project_model_comparison(batch_id=batch_id, records=records, run_dir_for=get_run_dir)


@router.get('/api/training/batches/{batch_id}/predictions.xlsx')
def download_batch_predictions(batch_id: str, principal: Principal = Depends(get_principal)):
    """导出批次内每个模型对每条记录的预测类别与逐类概率（两个工作表）。"""
    from ..runs.prediction_export import EXCEL_MEDIA_TYPE, PredictionExportError, build_prediction_workbook
    repository = get_run_repository()
    try:
        repository.get_batch_scoped(batch_id, principal=principal)
        records = repository.list_batch_runs_scoped(batch_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail="batch 不存在") from exc
    comparison = project_model_comparison(batch_id=batch_id, records=records, run_dir_for=get_run_dir)
    try:
        content = build_prediction_workbook(batch_id=batch_id, records=records, run_dir_for=get_run_dir, comparison=comparison)
    except PredictionExportError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    filename = f"predictions-{batch_id[:8]}.xlsx"
    return Response(content, media_type=EXCEL_MEDIA_TYPE, headers={
        'Content-Disposition': f'attachment; filename="{filename}"',
        'X-Content-Type-Options': 'nosniff',
        'Cache-Control': 'private, no-store',
    })


@router.get('/api/training/batches')
def list_batches(principal: Principal = Depends(get_principal), limit: int = Query(20, ge=1, le=100), cursor: str | None = None):
    from ..runs.batch_archive import archive_status
    repository = get_run_repository()
    try:
        batches, next_cursor = repository.list_batches_scoped(principal=principal, limit=limit, cursor=cursor)
        items = []
        for batch in batches:
            status = batch_status_payload(batch.batch_id, repository.list_batch_runs_scoped(batch.batch_id, principal=principal))
            items.append({'batch_id': batch.batch_id, 'created_at': batch.created_at, 'dataset_name': batch.dataset_snapshot.get('name') or batch.config.get('dataset_name') or batch.dataset_id,
                          'model_types': batch.model_types, 'state': status['state'], 'counts': status['counts'], 'archive': archive_status(repository, batch.batch_id, principal)})
        return {'items': items, 'next_cursor': next_cursor}
    except RunNotFound as exc:
        raise HTTPException(404, '批次不存在') from exc


def _archive_call(callback):
    from ..runs.batch_archive import ArchiveBusy, ArchiveUnavailable
    try:
        return callback()
    except RunNotFound as exc:
        raise HTTPException(404, '批次不存在') from exc
    except ArchiveBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ArchiveUnavailable, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, '对比图像或归档生成失败，请重试') from exc


@router.get('/api/training/batches/{batch_id}/archive')
def get_archive(batch_id: str, principal: Principal = Depends(get_principal)):
    from ..runs.batch_archive import archive_status, load_archive
    repository = get_run_repository()
    def read():
        status = archive_status(repository, batch_id, principal)
        return {**status, 'comparison': load_archive(repository, batch_id, principal) if status['state'] == 'ready' else None}
    return _archive_call(read)


@router.post('/api/training/batches/{batch_id}/archive')
def save_archive(batch_id: str, principal: Principal = Depends(get_principal), force: bool = False):
    from ..runs.batch_archive import ensure_archive
    return _archive_call(lambda: ensure_archive(get_run_repository(), batch_id, principal, get_run_dir, force=force))


@router.get('/api/training/batches/{batch_id}/archive/files/{name}')
def get_archive_file(batch_id: str, name: str, principal: Principal = Depends(get_principal)):
    from ..runs.batch_archive import archive_download
    content = _archive_call(lambda: archive_download(get_run_repository(), batch_id, principal, name))
    media = {'svg': 'image/svg+xml', 'png': 'image/png', 'csv': 'text/csv', 'json': 'application/json', 'zip': 'application/zip'}[name.rsplit('.', 1)[-1]]
    return Response(content, media_type=media, headers={'Content-Disposition': f'attachment; filename="{name}"', 'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'private, no-store'})


class ComparisonFigureRequest(BaseModel):
    kind: str = Field(pattern='^(overall|matrix|recall|precision|samples|features)$')
    metric: str = Field(default='balanced_accuracy', pattern='^(accuracy|balanced_accuracy|macro_f1|weighted_f1)$')
    model: str = Field(default='', max_length=100, pattern='^[a-zA-Z0-9_-]*$')
    sort: str = Field(default='balanced_accuracy', pattern='^(accuracy|balanced_accuracy|macro_f1|weighted_f1)$')
    page: int = Field(default=0, ge=0, le=100000)
    search: str = Field(default='', max_length=128)
    errors: bool = False
    matrix_mode: str = Field(default='percent', pattern='^(percent|count)$')
    width: int = Field(default=360, ge=260, le=1800)


@router.post('/api/training/batches/{batch_id}/figure')
def get_comparison_figure(batch_id: str, payload: ComparisonFigureRequest, principal: Principal = Depends(get_principal), format: str = Query('svg', pattern='^(svg|png)$')):
    from ..runs.batch_archive import archive_status, load_archive
    from ..runs.comparison_figures import render_figure
    repository = get_run_repository()
    def draw():
        status = archive_status(repository, batch_id, principal)
        if status['state'] == 'discarded':
            raise ValueError('批次已停止，不提供结果')
        data = load_archive(repository, batch_id, principal) if status['state'] == 'ready' else get_batch_comparison(batch_id, principal)
        content, metadata = render_figure(data, format=format, **payload.model_dump())
        if format == 'svg':
            return {'svg': content.decode('utf-8'), **metadata}
        return Response(content, media_type='image/png', headers={'Content-Disposition': 'attachment; filename="comparison.png"', 'Cache-Control': 'private, no-store'})
    return _archive_call(draw)
