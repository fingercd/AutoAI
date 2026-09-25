"""把训练函数包装进可取消、可校验 claim 的 Run 执行边界。

执行期间在训练前、写产物前和提交 Manifest 前多次确认 claim 仍有效，因此已取消
或 lease 已被回收的旧 worker 不能用迟到的成功结果覆盖最新状态。

系统位置：runs 子系统的执行边界，被 worker.py 的 RunWorker 经 execute_claimed_run
调用，向下委托给 training 模块的 _run_legacy_training 完成真正的训练。

关键设计约束：
- claim 校验贯穿全程：训练前、写结果前、写 Manifest 前各一次 cancel_check，
  任何一次发现 claim 失效（用户取消 / lease 被回收）都会抛 InvalidRunTransition 中止。
- 数据集完整性：执行前按 dataset_snapshot 里的 sha256 重新校验数据文件，
  防止"排队后数据被替换"导致训练结果与登记快照不一致。
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
    """协调一次已 claim Run 的训练、artifact 写入和状态投影。

    参数:
        repository: Run 仓库（duck-typed，只需提供 assert_active 等方法），
            便于测试用假对象替换。
        run_dir: 该 Run 的产物目录（storage/runs/<run_id>），所有 artifact 写入这里。
    """
    def __init__(self, *, repository: Any, run_dir: Path) -> None:
        self.repository = repository
        self.run_dir = Path(run_dir)

    def cancel_check(self, record: RunRecord) -> None:
        """确认本 worker 仍持有该 Run 的 claim，否则抛 InvalidRunTransition。

        这是"可取消、防迟到提交"的核心检查点：用户取消 Run 或 lease 被回收后，
        下一次 cancel_check 就会中断执行，已写出的部分产物不会被 finalize 成有效结果。
        """
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
        """调用 training 模块的实际训练入口。

        把独立测试集路径注入 config（test_data_path），并用 dataclasses.replace
        生成带新 config 的 record 副本，不修改原 record。
        """
        from ..training import _run_legacy_training
        # 延迟导入：本模块被 worker 长驻加载，training 的重依赖（torch/sklearn）只在真正训练时引入

        config = dict(record.config)
        if test_data_path is not None:
            config['test_data_path'] = str(test_data_path)
        execution_record = replace(record, config=config)
        # replace 生成副本而不改原 record：test_data_path 注入只影响本次执行的视图
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
        """执行训练并把结果固化为可下载的 artifact 集合。

        流程：cancel_check → 训练 → cancel_check → 回填数据集快照 →
        写 run_result → cancel_check → 状态投影 → finalize Manifest。
        返回 {"manifest_name": "manifest.json"} 供 worker 提交 finish_success。
        """
        # 训练前第一次检查：排队期间被取消的 Run 直接中止，不浪费训练资源
        self.cancel_check(record)
        writer = RunArtifactWriter(self.run_dir)
        result = self._run_legacy_training(
            record,
            data_path=data_path,
            test_data_path=test_data_path,
        )
        # 训练完成后、写结果前再查一次：训练期间被取消则结果直接丢弃
        self.cancel_check(record)
        # 把训练实际统计出的数据规模回填到 dataset_snapshot
        # （旧训练结果字段名到新快照键的兼容映射，如历史 sample_count 即曲线数）
        snapshot = dict(record.dataset_snapshot)
        for source, target in (
            ('curve_count', 'curve_count'),
            ('sample_count', 'curve_count'),
            ('sample_id_count', 'sample_id_count'),
            ('class_count', 'class_count'),
            ('feature_count', 'feature_count'),
            ('test_sample_count', 'test_curve_count'),
        ):
            # 只填空缺字段：登记时已有的快照值优先，不被训练侧覆盖
            if result.get(source) is not None and snapshot.get(target) is None:
                snapshot[target] = result[source]
        # hasattr 防御：repository 可能是测试替身，不具备快照更新能力时静默跳过
        if snapshot != record.dataset_snapshot and hasattr(self.repository, 'update_dataset_snapshot'):
            self.repository.update_dataset_snapshot(
                record.run_id,
                claim_token=record.claim_token or '',
                now=datetime.now(timezone.utc),
                dataset_snapshot=snapshot,
            )
        writer.write_run_result(result)
        # run_result 是结果页 run-result-v1 契约的数据来源
        # 第三次检查：此后 finalize 落盘即成为"有效结果"，必须确保 claim 仍有效
        self.cancel_check(record)
        # 剔除 run_id/status 等与 record 重复或属于状态机的字段，剩余训练元数据用于刷新投影
        result_fields = {
            key: value
            for key, value in result.items()
            if key not in {'run_id', 'status', 'state', 'version', 'dataset_id', 'error', 'run_dir'}
        }
        project_status(self.run_dir, record, **result_fields)
        # 刷新磁盘上的 status.json 投影，兼容旧的结果读取路径
        # Manifest 只登记少量展示用元数据；完整结果已在 run_result 与 status.json 中
        metadata = {
            key: result[key]
            for key in ('model_type', 'model_family', 'evaluation_strategy', 'fold_count')
            if key in result
        }
        writer.finalize(run_id=record.run_id, metadata=metadata)
        return {'manifest_name': 'manifest.json'}


def execute_claimed_run(record: RunRecord, *, repository: Any,
                        storage_root: Path | None = None) -> dict[str, str]:
    """执行一个已 claim 的 Run，并在 Manifest 提交后返回完成信息。

    职责：解析主数据集与可选独立测试集（server 模式只认 dataset_id，
    legacy_path 仅为兼容历史 Run），按快照 sha256 校验完整性，
    然后交给 TrainingExecution 完成训练与产物固化。
    """
    from ..datasets.repository import DatasetRepository
    from ..paths import DATASETS_DATABASE, RUNS_DIR, STORAGE_DIR

    # A supervised child receives the same repository storage root explicitly;
    # the ordinary worker and existing direct callers retain the shared paths.
    root = Path(storage_root) if storage_root is not None else STORAGE_DIR
    datasets_database = root / 'datasets.sqlite3' if storage_root is not None else DATASETS_DATABASE
    runs_dir = root / 'runs' if storage_root is not None else RUNS_DIR

    dataset_repository = DatasetRepository(datasets_database, storage_root=root)
    dataset_repository.initialize()
    # 每次执行独立建 DatasetRepository：worker 是长驻进程，避免跨 Run 共享连接状态
    dataset = dataset_repository.resolve_system(record.dataset_id, legacy_path=record.legacy_data_path)
    expected_sha256 = record.dataset_snapshot.get('sha256') if record.dataset_snapshot else None
    # 排队时登记的数据集指纹；数据被改动会抛 DatasetIntegrityError，由 worker 映射为 dataset_changed
    dataset_repository.verify_integrity(dataset, expected_sha256=expected_sha256)
    test_dataset_id = record.config.get('test_dataset_id')
    test_legacy_path = record.config.get('test_data_path')
    test_dataset = None
    if test_dataset_id or test_legacy_path:
        # 独立测试集（external_test_holdout 口径）同样解析并校验；sha256 来自训练请求时的快照
        test_dataset = dataset_repository.resolve_system(test_dataset_id, legacy_path=test_legacy_path)
        dataset_repository.verify_integrity(
            test_dataset,
            expected_sha256=record.config.get('test_dataset_sha256'),
        )
    execution = TrainingExecution(repository=repository, run_dir=runs_dir / record.run_id)
    return execution.execute(
        record,
        data_path=dataset.path,
        test_data_path=test_dataset.path if test_dataset is not None else None,
    )
