from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from ..contracts import TrainingRunRequest
from ..datasets.repository import DatasetRepository
from ..paths import DATASETS_DATABASE, DEFAULT_DATA, PREPROCESSED_DIR, RUNS_DATABASE, RUNS_DIR, STORAGE_DIR, UPLOADS_DIR
from ..runs.contracts import Principal
from ..runs.repository import RunRepository


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_legacy_dataset_path(path: str | Path, *, allowed_roots: list[Path]) -> str:
    target = Path(path).resolve()
    if not target.is_file():
        raise HTTPException(status_code=404, detail='训练数据文件不存在')
    def allowed(root: Path) -> bool:
        resolved_root = root.resolve()
        return target == resolved_root if resolved_root.is_file() else _is_within(target, resolved_root)

    if not any(allowed(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail='旧 data_path 只能指向受控数据目录')
    return str(target)


@dataclass(frozen=True)
class TrainingDataReference:
    dataset_id: str | None
    legacy_path: str | None
    test_dataset_id: str | None
    test_legacy_path: str | None


def _dataset_repository() -> DatasetRepository:
    repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    repository.initialize()
    return repository


def resolve_training_data_reference(payload: TrainingRunRequest, *, principal: Principal) -> TrainingDataReference:
    if payload.dataset_id and payload.data_path:
        raise HTTPException(status_code=422, detail='dataset_id 与 data_path 不能同时提供')
    if payload.test_dataset_id and payload.test_data_path:
        raise HTTPException(status_code=422, detail='test_dataset_id 与 test_data_path 不能同时提供')

    datasets = _dataset_repository()
    if payload.dataset_id:
        datasets.resolve(payload.dataset_id, principal=principal)
        dataset_id = payload.dataset_id
        legacy_path = None
    else:
        path = payload.data_path or str(DEFAULT_DATA)
        dataset_id = None
        legacy_path = resolve_legacy_dataset_path(
            path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )

    if payload.test_dataset_id:
        datasets.resolve(payload.test_dataset_id, principal=principal)
        test_dataset_id = payload.test_dataset_id
        test_legacy_path = None
    elif payload.test_data_path:
        test_dataset_id = None
        test_legacy_path = resolve_legacy_dataset_path(
            payload.test_data_path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )
    else:
        test_dataset_id = None
        test_legacy_path = None

    return TrainingDataReference(dataset_id, legacy_path, test_dataset_id, test_legacy_path)


def get_run_repository() -> RunRepository:
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    return repository


def get_run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id
