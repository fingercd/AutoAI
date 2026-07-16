"""把训练函数包装进可取消、可校验 claim 的 Run 执行边界。

执行期间在训练前、写产物前和提交 Manifest 前多次确认 claim 仍有效，因此已取消
或 lease 已被回收的旧 worker 不能用迟到的成功结果覆盖最新状态。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import RunArtifactWriter
from .contracts import RunRecord
from .status_projection import project_status


class TrainingExecution:
    """协调一次已 claim Run 的训练、artifact 写入和状态投影。"""
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
        result_fields = {
            key: value
            for key, value in result.items()
            if key not in {'run_id', 'status', 'state', 'version', 'dataset_id', 'error', 'run_dir'}
        }
        project_status(self.run_dir, record, **result_fields)
        metadata = {
            key: result[key]
            for key in ('model_type', 'model_family', 'evaluation_strategy', 'fold_count')
            if key in result
        }
        writer.finalize(run_id=record.run_id, metadata=metadata)
        return {'manifest_name': 'manifest.json'}


def execute_claimed_run(record: RunRecord, *, repository: Any) -> dict[str, str]:
    """执行一个已 claim 的 Run，并在 Manifest 提交后返回完成信息。"""
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
