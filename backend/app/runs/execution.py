from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import RunArtifactWriter
from .contracts import RunRecord


class TrainingExecution:
    def __init__(self, *, repository: Any, run_dir: Path) -> None:
        self.repository = repository
        self.run_dir = Path(run_dir)

    def cancel_check(self, record: RunRecord) -> None:
        self.repository.assert_active(
            record.run_id,
            claim_token=record.claim_token or '',
            now=datetime.now(timezone.utc),
        )

    def _run_legacy_training(
        self,
        record: RunRecord,
        *,
        data_path: Path,
        test_data_path: Path | None,
    ) -> dict[str, Any]:
        from ..training import _run_legacy_training

        config = dict(record.config)
        if test_data_path is not None:
            config['test_data_path'] = str(test_data_path)
        execution_record = replace(record, config=config)
        return _run_legacy_training(
            data_path=data_path,
            config_data=config,
            run_id=execution_record.run_id,
            repository=self.repository,
            record=execution_record,
        )

    def execute(
        self,
        record: RunRecord,
        *,
        data_path: Path,
        test_data_path: Path | None = None,
    ) -> dict[str, str]:
        self.cancel_check(record)
        writer = RunArtifactWriter(self.run_dir)
        result = self._run_legacy_training(
            record,
            data_path=data_path,
            test_data_path=test_data_path,
        )
        self.cancel_check(record)
        writer.write_run_result(result)
        self.cancel_check(record)
        metadata = {
            key: result[key]
            for key in ('model_type', 'model_family', 'evaluation_strategy', 'fold_count')
            if key in result
        }
        writer.finalize(run_id=record.run_id, metadata=metadata)
        return {'manifest_name': 'manifest.json'}


def execute_claimed_run(record: RunRecord, *, repository: Any) -> dict[str, str]:
    from ..datasets.repository import DatasetRepository
    from ..paths import DATASETS_DATABASE, RUNS_DIR, STORAGE_DIR

    dataset_repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    dataset_repository.initialize()
    dataset = dataset_repository.resolve_system(record.dataset_id, legacy_path=record.legacy_data_path)
    test_dataset_id = record.config.get('test_dataset_id')
    test_legacy_path = record.config.get('test_data_path')
    test_dataset = None
    if test_dataset_id or test_legacy_path:
        test_dataset = dataset_repository.resolve_system(test_dataset_id, legacy_path=test_legacy_path)
    execution = TrainingExecution(repository=repository, run_dir=RUNS_DIR / record.run_id)
    return execution.execute(
        record,
        data_path=dataset.path,
        test_data_path=test_dataset.path if test_dataset is not None else None,
    )
