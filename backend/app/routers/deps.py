"""路由共享依赖与数据引用解析。

新客户端优先传 dataset_id；data_path 只保留给受控本地兼容。两种引用最终都在
允许目录内解析，并由 Principal 限定作用域，不能借训练接口读取任意路径。
"""

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
    """验证兼容路径位于允许根目录并返回规范绝对路径。"""
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
    """已解析的主数据/独立测试数据引用及其上传文件名。"""
    dataset_id: str | None
    legacy_path: str | None
    dataset_name: str
    test_dataset_id: str | None
    test_legacy_path: str | None
    test_dataset_name: str | None


def _dataset_repository() -> DatasetRepository:
    repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    repository.initialize()
    return repository


def resolve_training_data_reference(payload: TrainingRunRequest, *, principal: Principal) -> TrainingDataReference:
    """校验互斥引用并解析一次训练所需的主数据和可选测试数据。"""
    if payload.dataset_id and payload.data_path:
        raise HTTPException(status_code=422, detail='dataset_id 与 data_path 不能同时提供')
    if payload.test_dataset_id and payload.test_data_path:
        raise HTTPException(status_code=422, detail='test_dataset_id 与 test_data_path 不能同时提供')

    datasets = _dataset_repository()
    if payload.dataset_id:
        dataset = datasets.resolve(payload.dataset_id, principal=principal)
        dataset_id = payload.dataset_id
        legacy_path = None
        dataset_name = dataset.original_name
    else:
        path = payload.data_path or str(DEFAULT_DATA)
        dataset_id = None
        legacy_path = resolve_legacy_dataset_path(
            path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )
        dataset_name = Path(legacy_path).name

    if payload.test_dataset_id:
        test_dataset = datasets.resolve(payload.test_dataset_id, principal=principal)
        test_dataset_id = payload.test_dataset_id
        test_legacy_path = None
        test_dataset_name = test_dataset.original_name
    elif payload.test_data_path:
        test_dataset_id = None
        test_legacy_path = resolve_legacy_dataset_path(
            payload.test_data_path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )
        test_dataset_name = Path(test_legacy_path).name
    else:
        test_dataset_id = None
        test_legacy_path = None
        test_dataset_name = None

    return TrainingDataReference(
        dataset_id=dataset_id,
        legacy_path=legacy_path,
        dataset_name=dataset_name,
        test_dataset_id=test_dataset_id,
        test_legacy_path=test_legacy_path,
        test_dataset_name=test_dataset_name,
    )


def get_run_repository() -> RunRepository:
    """为当前请求创建指向共享 SQLite 文件的轻量仓库对象。"""
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    return repository


def get_run_dir(run_id: str) -> Path:
    """校验 run_id 是安全单段名称并解析其受控产物目录。"""
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise HTTPException(status_code=404, detail='run 不存在')
    return RUNS_DIR / run_id
