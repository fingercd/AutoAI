from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..runs.contracts import Principal


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    path: Path
    sha256: str
    original_name: str
    owner_id: str | None
    tenant_id: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class DatasetRepository:
    def __init__(self, database_path: Path, *, storage_root: Path) -> None:
        self.database_path = Path(database_path)
        self.storage_root = Path(storage_root).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=str(row['dataset_id']),
            path=Path(row['path']).resolve(),
            sha256=str(row['sha256']),
            original_name=str(row['original_name']),
            owner_id=row['owner_id'],
            tenant_id=row['tenant_id'],
        )

    def register(self, path: Path, *, original_name: str, principal: Principal) -> DatasetRecord:
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        if not self._is_controlled_path(target):
            raise PermissionError(f'dataset path is outside controlled storage: {target}')
        dataset_id = f'ds_{uuid.uuid4().hex}'
        record = DatasetRecord(
            dataset_id=dataset_id,
            path=target,
            sha256=_sha256(target),
            original_name=original_name,
            owner_id=principal.owner_id,
            tenant_id=principal.tenant_id,
        )
        with self._connection() as connection:
            connection.execute(
                '''
                INSERT INTO datasets (dataset_id, path, sha256, original_name, owner_id, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    record.dataset_id,
                    str(record.path),
                    record.sha256,
                    record.original_name,
                    record.owner_id,
                    record.tenant_id,
                ),
            )
        return record

    def resolve(self, dataset_id: str, *, principal: Principal) -> DatasetRecord:
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM datasets WHERE dataset_id = ?', (dataset_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f'dataset {dataset_id} does not exist')
        record = self._record(row)
        if record.owner_id != principal.owner_id or record.tenant_id != principal.tenant_id:
            raise PermissionError(f'dataset {dataset_id} is outside the principal scope')
        if not record.path.is_file():
            raise FileNotFoundError(record.path)
        return record

    def resolve_system(self, dataset_id: str | None, *, legacy_path: str | None) -> DatasetRecord:
        if dataset_id:
            with self._connection() as connection:
                row = connection.execute('SELECT * FROM datasets WHERE dataset_id = ?', (dataset_id,)).fetchone()
            if row is None:
                raise FileNotFoundError(f'dataset {dataset_id} does not exist')
            record = self._record(row)
            if not record.path.is_file():
                raise FileNotFoundError(record.path)
            return record
        if not legacy_path:
            raise FileNotFoundError('no dataset reference supplied')
        target = Path(legacy_path).resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        if not self._is_controlled_path(target):
            raise PermissionError(f'dataset path is outside controlled storage: {target}')
        return DatasetRecord(
            dataset_id='',
            path=target,
            sha256=_sha256(target),
            original_name=target.name,
            owner_id=None,
            tenant_id=None,
        )

    def _is_controlled_path(self, path: Path) -> bool:
        allowed_roots = (
            self.storage_root,
            self.storage_root / 'uploads',
            self.storage_root / 'preprocessed',
            self.storage_root.parent / 'data.csv',
        )
        return any(_is_within(path, root.resolve()) for root in allowed_roots)
