"""HTTP 无关的 queued Run 提交服务。

人工训练路由与 Agent 实验入口都在这里完成配置校验、Worker 契约门禁、
数据集快照、queued Run 创建及 status.json 投影。该模块不依赖 FastAPI。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluation_plan import PreparedEvaluation

from ..contracts import TrainingConfigValidationError, TrainingSpec
from ..datasets.repository import DatasetIntegrityError, DatasetRepository
from ..version import WORKER_CONTRACT_VERSION
from .contracts import Principal, RunRecord
from .repository import (
    RunRepository,
    RunSubmissionKeyConflict,
    RunSubmissionMappingCorrupt,
)


SubmissionSource = Literal['human', 'agent']


class RunSubmissionError(RuntimeError):
    """可由 HTTP 层稳定映射的提交领域错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        retryable: bool = False,
        allowed_actions: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.allowed_actions = allowed_actions


class WorkerContractMismatch(RunSubmissionError):
    def __init__(self) -> None:
        super().__init__(
            'worker_contract_mismatch',
            '训练 Worker 版本与当前 Web 不兼容，请同时重启 Web 和 Worker',
            status_code=503,
            retryable=True,
            allowed_actions=('inspect_ml_capabilities',),
        )


class DatasetUnavailable(RunSubmissionError):
    def __init__(self, message: str = '数据集不存在或当前调用者无权访问') -> None:
        super().__init__('dataset_unavailable', message, status_code=404)


@dataclass(frozen=True)
class SubmissionDataReference:
    dataset_id: str | None
    legacy_path: str | None
    dataset_name: str
    test_dataset_id: str | None = None
    test_legacy_path: str | None = None
    test_dataset_name: str | None = None


@dataclass(frozen=True)
class RunSubmissionRequest:
    dataset_id: str | None
    legacy_data_path: str | None
    test_dataset_id: str | None
    test_legacy_data_path: str | None
    raw_config: dict[str, Any]
    principal: Principal
    submission_source: SubmissionSource
    submission_key: str | None = None
    expected_dataset_sha256: str | None = None
    prepared_evaluation: PreparedEvaluation | None = None


@dataclass(frozen=True)
class RunSubmissionResult:
    record: RunRecord
    spec: TrainingSpec
    effective_config: dict[str, Any]
    warnings: tuple[str, ...]
    dataset_snapshot: dict[str, Any]
    submission_payload_hash: str


class RunSubmissionService:
    """创建真实 queued Run；不启动训练。"""

    def __init__(
        self,
        *,
        run_repository: RunRepository,
        dataset_repository: DatasetRepository,
        run_dir: Callable[[str], Path],
        status_projector: Callable[..., Any],
    ) -> None:
        self.runs = run_repository
        self.datasets = dataset_repository
        self.run_dir = run_dir
        self.status_projector = status_projector

    def resolve_reference(self, request: RunSubmissionRequest) -> SubmissionDataReference:
        """解析并按 Principal 校验数据引用；越权与不存在使用相同错误。"""
        if request.dataset_id and request.legacy_data_path:
            raise RunSubmissionError('invalid_dataset_reference', 'dataset_id 与 data_path 不能同时提供')
        if request.test_dataset_id and request.test_legacy_data_path:
            raise RunSubmissionError(
                'invalid_dataset_reference', 'test_dataset_id 与 test_data_path 不能同时提供'
            )
        server_scoped = (
            request.principal.owner_id is not None or request.principal.tenant_id is not None
        )
        if server_scoped and (request.legacy_data_path or not request.dataset_id):
            raise RunSubmissionError(
                'invalid_dataset_reference', '服务器模式必须使用上传接口返回的 dataset_id'
            )
        if server_scoped and request.test_legacy_data_path:
            raise RunSubmissionError(
                'invalid_dataset_reference', '服务器模式的独立测试集必须使用 test_dataset_id'
            )
        try:
            if request.dataset_id:
                dataset = self.datasets.resolve(request.dataset_id, principal=request.principal)
                dataset_id, legacy_path = request.dataset_id, None
            else:
                dataset = self.datasets.resolve_system(None, legacy_path=request.legacy_data_path)
                dataset_id, legacy_path = None, str(dataset.path)
            test = None
            if request.test_dataset_id:
                test = self.datasets.resolve(request.test_dataset_id, principal=request.principal)
            elif request.test_legacy_data_path:
                test = self.datasets.resolve_system(None, legacy_path=request.test_legacy_data_path)
        except (FileNotFoundError, PermissionError):
            raise DatasetUnavailable() from None
        return SubmissionDataReference(
            dataset_id=dataset_id,
            legacy_path=legacy_path,
            dataset_name=dataset.original_name,
            test_dataset_id=request.test_dataset_id,
            test_legacy_path=(str(test.path) if test is not None and not request.test_dataset_id else None),
            test_dataset_name=test.original_name if test is not None else None,
        )

    def prepare_evaluation(self, *, dataset_id, raw_config, principal, expected_sha256=None, legacy_path=None, limits=None):
        from ..evaluation_plan import (build_evaluation_plan, load_dataset_view,
            validate_plan, evaluation_config, PreparationResourceExhausted)
        try:
            dataset = (self.datasets.resolve(dataset_id, principal=principal) if dataset_id
                else self.datasets.resolve_system(None, legacy_path=legacy_path))
            sha = self.datasets.verify_integrity(dataset, expected_sha256=expected_sha256)
            view = load_dataset_view(dataset.path,sha,limits)
            evaluation = evaluation_config(raw_config)
            seed = raw_config.get('seed',42)
            if raw_config.get('evaluation_plan_digest'):
                plan = self.runs.get_evaluation_plan(raw_config['evaluation_plan_digest'],principal=principal)
                validate_plan(plan,view,evaluation,seed)
            else:
                plan = build_evaluation_plan(view,evaluation,seed)
                self.runs.save_evaluation_plan(plan,principal=principal)
            return view, plan
        except PreparationResourceExhausted:
            raise RunSubmissionError('preparation_resource_exhausted','Preparation resource limit exceeded',status_code=422) from None
        except (FileNotFoundError, PermissionError, DatasetIntegrityError):
            raise DatasetUnavailable() from None
        except ValueError:
            raise RunSubmissionError('evaluation_plan_invalid','Dataset or evaluation plan is invalid',status_code=422) from None

    def submit(self, request: RunSubmissionRequest) -> RunSubmissionResult:
        return self.submit_prepared(
            reference=self.resolve_reference(request),
            raw_config=request.raw_config,
            principal=request.principal,
            submission_source=request.submission_source,
            submission_key=request.submission_key,
            expected_dataset_sha256=request.expected_dataset_sha256,
            prepared_evaluation=request.prepared_evaluation,
        )

    def submit_prepared(
        self,
        *,
        reference: SubmissionDataReference | Any,
        raw_config: dict[str, Any],
        principal: Principal,
        submission_source: SubmissionSource,
        submission_key: str | None = None,
        snapshot_provider: Callable[..., dict[str, Any]] | None = None,
        expected_dataset_sha256: str | None = None,
        prepared_evaluation: PreparedEvaluation | None = None,
    ) -> RunSubmissionResult:
        """提交已由 HTTP 兼容层解析的数据引用，供旧路由保持注入点兼容。"""
        audit_fields={'execution_recipe_digest','execution_catalog_digest','execution_evidence_digest','execution_search_digest',
            'execution_processing_policy_version','execution_processing_digest','execution_search_plan',
            'search_mode','max_trials'}
        audit_fields.update({'execution_budget_task_id', 'execution_budget_policy_digest'})
        audit_fields.update({'execution_guard_policy', 'execution_guard_policy_digest'})
        if submission_source!='agent' and audit_fields.intersection(raw_config):
            raise RunSubmissionError('invalid_training_config','Execution recipe audit is server-owned')
        if prepared_evaluation is not None:
            if (raw_config.get('evaluation_plan_digest')!=prepared_evaluation.plan_digest
                    or expected_dataset_sha256!=prepared_evaluation.dataset_sha256):
                raise RunSubmissionError('evaluation_plan_invalid','Prepared evaluation binding differs')
        if submission_key is not None and submission_source != 'agent':
            raise RunSubmissionError(
                'invalid_submission_key', 'submission key 仅供 Agent 内部提交使用'
            )
        has_external_test = bool(reference.test_dataset_id or reference.test_legacy_path)
        try:
            spec = TrainingSpec.from_legacy(raw_config).validated(
                has_external_test=has_external_test
            )
        except TrainingConfigValidationError as exc:
            raise RunSubmissionError('invalid_training_config', str(exc)) from exc

        health = self.runs.worker_health(
            now=datetime.now(timezone.utc),
            expected_contract_version=WORKER_CONTRACT_VERSION,
        )
        if health.get('available') and not health.get('compatible'):
            raise WorkerContractMismatch()

        def snapshot(dataset_id: str | None, legacy_path: str | None, name: str | None) -> dict[str, Any]:
            if snapshot_provider is not None:
                return snapshot_provider(
                    dataset_id=dataset_id, legacy_path=legacy_path, dataset_name=name
                )
            try:
                if dataset_id:
                    record = self.datasets.resolve(dataset_id, principal=principal)
                else:
                    record = self.datasets.resolve_system(None, legacy_path=legacy_path)
                digest = self.datasets.verify_integrity(record)
            except (FileNotFoundError, PermissionError, DatasetIntegrityError):
                raise DatasetUnavailable() from None
            return {'dataset_id': dataset_id, 'name': name or record.original_name, 'sha256': digest}

        dataset_snapshot = snapshot(
            reference.dataset_id, reference.legacy_path, reference.dataset_name
        )
        if (expected_dataset_sha256 is not None
                and dataset_snapshot.get('sha256') != expected_dataset_sha256):
            raise RunSubmissionError(
                'agent_dataset_fingerprint_mismatch',
                '数据指纹与冻结 Session 不一致，未创建训练任务', status_code=409,
            )
        if spec.values.get('model_type') == 'xgboost' and spec.values.get('class_balance') == 'class_weight':
            from ..parsers import load_modeling_csv
            from ..processing_policy import validate_processing
            try:
                source = (self.datasets.resolve(reference.dataset_id, principal=principal)
                          if reference.dataset_id else
                          self.datasets.resolve_system(None, legacy_path=reference.legacy_path))
                labels = load_modeling_csv(source.path).labels
                validate_processing('xgboost', spec.values['normalization'], 'class_weight',
                    class_count=len(set(labels)))
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                raise RunSubmissionError('invalid_training_config', str(exc), status_code=422) from exc
        effective_training_config = spec.to_legacy_dict()
        if not has_external_test and effective_training_config['split_mode'] == 'stratified_holdout':
            _, plan = self.prepare_evaluation(dataset_id=reference.dataset_id,
                raw_config=effective_training_config, principal=principal,
                expected_sha256=dataset_snapshot['sha256'],legacy_path=reference.legacy_path)
            effective_training_config['evaluation_plan_digest'] = plan.plan_digest
            spec = TrainingSpec(effective_training_config,warnings=spec.warnings)
        elif effective_training_config.get('evaluation_plan_digest'):
            raise RunSubmissionError('evaluation_plan_invalid','Plans apply only to grouped holdout')
        test_snapshot: dict[str, Any] | None = None
        if has_external_test:
            test_snapshot = snapshot(
                reference.test_dataset_id,
                reference.test_legacy_path,
                reference.test_dataset_name,
            )
        submission_payload = {
            'dataset': {
                'dataset_id': reference.dataset_id,
                'sha256': dataset_snapshot['sha256'],
            },
            'test_dataset': (
                {
                    'dataset_id': reference.test_dataset_id,
                    'sha256': test_snapshot['sha256'],
                }
                if test_snapshot is not None else None
            ),
            'effective_config': effective_training_config,
            'submission_source': submission_source,
        }
        submission_payload_hash = hashlib.sha256(
            json.dumps(
                submission_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()

        config = dict(effective_training_config)
        config['dataset_name'] = reference.dataset_name
        config['submission_source'] = submission_source
        if reference.test_dataset_id:
            config['test_dataset_id'] = reference.test_dataset_id
        if reference.test_legacy_path:
            config['test_data_path'] = reference.test_legacy_path
        if reference.test_dataset_name:
            config['test_dataset_name'] = reference.test_dataset_name
        if test_snapshot is not None:
            config['test_dataset_sha256'] = test_snapshot['sha256']

        try:
            from .guard import GuardPolicy
            record = self.runs.create_queued(
                dataset_id=reference.dataset_id,
                legacy_data_path=reference.legacy_path,
                config=config,
                dataset_snapshot=dataset_snapshot,
                principal=principal,
                guard_policy=GuardPolicy.model_validate(raw_config.get('execution_guard_policy',
                    {'fail_fast_guard': 'off'})).model_dump(),
                submission_key=submission_key,
                submission_source=(submission_source if submission_key is not None else None),
                submission_payload_hash=(
                    submission_payload_hash if submission_key is not None else None
                ),
            )
        except RunSubmissionKeyConflict as exc:
            raise RunSubmissionError(
                'agent_submission_key_conflict',
                '实验提交关联发生冲突，需要人工核对',
                status_code=409,
            ) from exc
        except RunSubmissionMappingCorrupt as exc:
            raise RunSubmissionError(
                'agent_submission_mapping_invalid',
                '实验提交关联不完整，需要人工核对',
                status_code=409,
            ) from exc
        self.status_projector(self.run_dir(record.run_id), record, config=config)
        return RunSubmissionResult(
            record=record,
            spec=spec,
            effective_config=config,
            warnings=tuple(spec.warnings),
            dataset_snapshot=dataset_snapshot,
            submission_payload_hash=submission_payload_hash,
        )
