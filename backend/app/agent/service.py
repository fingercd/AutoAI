"""Agent Session 服务层：组装 config、读 validation、按 selection_metric 选最佳。

职责边界：
- 严格不接触 metrics.json 之外的训练产物；不读取 test 子树、artifacts、explainability。
- 不重写训练流程；只通过现有 ``RunRepository`` + ``TrainingSpec`` 复用现有
  入队路径创建底层 Training Run，让 worker 与现有 Run 共享同一状态机。
- Validation 必须来自 ``run_dir / metrics.json`` 的 ``valid`` 子树；读不到
  valid（例如 Run 还在 queued/running 或失败）时显式返回 ``status="unknown"``，
  绝不从 test 子树、cv_summary 或其他产物借数。
- 所有 ``run_result`` / ``status`` 字段在 Agent 响应中一律只保留 state、progress、
  duration_seconds 三个安全子集。

注意：测试集合上的指标**已经在底层 metrics.json 里被计算出来**——这是
training.py 的既定产物结构。Agent 协议层的"隐藏 Test"通过白名单投影实现，
不修改训练流水线。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import TrainingConfigValidationError, TrainingSpec
from ..datasets.repository import DatasetRepository
from ..paths import DATASETS_DATABASE, STORAGE_DIR
from ..runs.contracts import Principal, RunRecord
from ..runs.repository import RunNotFound, RunRepository
from .contracts import CreateAgentExperimentRequest, CreateAgentSessionRequest
from .evidence import build_dataset_evidence_card
from .feedback import build_diagnosis
from .memory import (
    MAX_MEMORY_CONTEXT_ITEMS,
    build_memory_record,
    build_memory_signatures,
    build_terminal_outcome,
    freeze_memory_context,
)
from .decision_support import (
    VALIDATION_UNCERTAINTY_METHOD,
    build_decision_support,
    estimate_validation_metric_std,
)
from .policy import compile_proposal_catalog, resolve_proposal
from .repository import (
    AgentConfigCollision,
    AgentExperimentRecord,
    AgentSessionClosed,
    AgentSessionRecord,
    AgentSessionRepository,
    collect_run_states,
)


# validation metrics 白名单：只从 metrics.json 的 valid 子树挑这些标量。
# 其他任何字段（含 confusion_matrix、classification_report、pooled_*）一律
# 不出现在 AgentFeedback，避免泄漏 Test 集合的间接信号。
_VALIDATION_METRIC_KEYS = (
    'accuracy',
    'balanced_accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'weighted_f1',
)
_DEFAULT_MAX_RETRIES_PER_FAILURE = 1

# AgentFeedback / AgentSessionDetail / AgentExperimentSummary 通用敏感键黑名单：
# 任何包含以下子串（不分大小写）的键都被视为泄露，禁止出现在 Agent 响应里。
_FORBIDDEN_KEY_SUBSTRINGS = (
    'test',
    'artifact',
    'prediction',
    'confusion',
    'classification_report',
    'explainability',
    'samples',
    'sample_id',
    'roc',
    'precision_recall',
    'curve_length',
    'feature_count',  # 不让 Agent 推断数据规模
)
_ABSOLUTE_PATH = re.compile(
    r'(?:[A-Za-z]:[\\/]|/(?:users|home|var|tmp|opt|srv|etc|root|mnt|data)/)[^\s,;]*',
    flags=re.IGNORECASE,
)
_URL = re.compile(r'https?://[^\s,;]+', flags=re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r'\b(?:token|secret|password|credential|authorization)\s*[:=]\s*[^\s,;]+',
    flags=re.IGNORECASE,
)
_SAMPLE_ID_ASSIGNMENT = re.compile(
    r'\bsample[_\s-]*id\s*[:=]\s*'
    r'(?:(?:"[^"\r\n]*")|(?:\'[^\'\r\n]*\')|[^\r\n,;，；。)\]}]+)',
    flags=re.IGNORECASE,
)


def _safe_string(value: str) -> str:
    value = _SAMPLE_ID_ASSIGNMENT.sub('[redacted-sample-id]', value)
    value = re.sub(
        r'Bearer\s+\S+',
        '[redacted-auth]',
        value,
        flags=re.IGNORECASE,
    )
    value = _SECRET_ASSIGNMENT.sub('[redacted-secret]', value)
    value = _URL.sub('[redacted-url]', value)
    return _ABSOLUTE_PATH.sub('[redacted-path]', value)


def _scrub(value: Any) -> Any:
    """递归剥离任何包含敏感子串的键。

    这是 ``AgentFeedback`` 构造的最后一道防线——即使上游投影逻辑写漏了某个键，
    只要键名命中 ``_FORBIDDEN_KEY_SUBSTRINGS``，它就不会出现在响应里。
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if any(forbidden in key.lower() for forbidden in _FORBIDDEN_KEY_SUBSTRINGS):
                continue
            cleaned[key] = _scrub(item)
        return cleaned
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        return _safe_string(value)
    return value


def _compute_config_hash(*, session: AgentSessionRecord, action: dict[str, Any]) -> str:
    """稳定 hash：session 内由 selection_metric、seed、evaluation_config、action
    决定。任何对 Agent 可控字段的修改都会产生新 hash；Session 锁定的字段变化
    不会发生（因为它们由 Session 决定，不由 action 决定）。"""
    payload = {
        'dataset_id': session.dataset_id,
        'selection_metric': session.selection_metric,
        'seed': session.seed,
        'evaluation_config': session.evaluation_config,
        'action': {
            'model_type': action.get('model_type'),
            'normalization': action.get('normalization', 'zscore'),
            'class_balance': action.get('class_balance', 'none'),
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_progress(record: RunRecord) -> dict[str, Any]:
    """把 RunRecord.progress 中可能含 ``stop_*`` 等内部字段全部剔除。"""
    raw = _scrub(record.progress)
    if isinstance(raw, dict):
        return {key: value for key, value in raw.items() if not key.startswith('_')}
    return {}


def _safe_record_summary(record: RunRecord) -> dict[str, Any]:
    """RunRecord 安全子集：只保留 state/progress/started_at/finished_at/duration，
    绝不带 config（config 里可能含 selection_metric 之外的字段）。"""
    started = record.started_at
    finished = record.finished_at
    duration_seconds: float | None = None
    if started and finished:
        try:
            t_started = datetime.fromisoformat(started.replace('Z', '+00:00'))
            t_finished = datetime.fromisoformat(finished.replace('Z', '+00:00'))
            duration_seconds = max(0.0, (t_finished - t_started).total_seconds())
        except ValueError:
            duration_seconds = None
    return {
        'state': record.state,
        'progress': _safe_progress(record),
        'started_at': started,
        'finished_at': finished,
        'duration_seconds': duration_seconds,
        'error': None if not record.error else _scrub({'message': record.error}),
    }


def _guard_feedback(record: RunRecord) -> dict[str, Any] | None:
    progress_guard = (
        record.progress.get('agent_guard')
        if isinstance(record.progress.get('agent_guard'), dict)
        else None
    )
    failure_guard = (
        record.error_details.get('guard_result')
        if isinstance(record.error_details.get('guard_result'), dict)
        else None
    )
    if progress_guard is None and failure_guard is None:
        return None
    result = dict(progress_guard or {})
    if failure_guard is not None:
        result['failure'] = failure_guard
    return result


def _build_training_config(
    *,
    session: AgentSessionRecord,
    action: dict[str, Any],
) -> dict[str, Any]:
    """把 Session 固定字段 + action 合成现有 TrainingRunRequest.config。

    强制约束：
    - ``feature_selection_enabled=False``：禁止遮挡解释
    - ``model_type`` 来自 action，``normalization``、``class_balance`` 来自 action
    - 其他训练预算取 ``TrainingSpec`` 默认值（epochs=200 / batch_size=8 / lr=1e-3 等）
    - ``seed`` 来自 Session，保证每次实验可复现
    """
    config: dict[str, Any] = {
        'model_type': action['model_type'],
        'normalization': action.get('normalization', 'zscore'),
        'class_balance': action.get('class_balance', 'none'),
        'seed': session.seed,
        'feature_selection_enabled': False,
        'hpo_selection_metric': session.selection_metric,
    }
    hpo_policy = session.context.get('hpo_policy')
    if isinstance(hpo_policy, dict):
        config['hpo_profile'] = hpo_policy['profile']
        config['hpo_selection_metric'] = hpo_policy['selection_metric']
    guard_policy = session.context.get('guard_policy')
    if isinstance(guard_policy, dict):
        config['agent_execution'] = {
            'guard_version': guard_policy['schema_version'],
            'fail_fast_guard': True,
            'expected_model_type': action['model_type'],
            'hpo_profile': config.get('hpo_profile', 'standard'),
            'max_hpo_candidates': guard_policy['max_hpo_candidates'],
        }
    # 把 evaluation_config 的字段（如 split_mode/split_train/...）合并进来
    for key, value in session.evaluation_config.items():
        config.setdefault(key, value)
    return config


class AgentService:
    """Agent 接口的服务层。所有状态变更走 SQLite 事务；不在内存里维护任何
    跨请求共享 dict，避免后端重启丢状态。"""

    def __init__(
        self,
        *,
        session_repository: AgentSessionRepository,
        run_repository: RunRepository,
        dataset_repository: DatasetRepository,
    ) -> None:
        self.sessions = session_repository
        self.runs = run_repository
        self.datasets = dataset_repository

    def create_session(
        self, payload: CreateAgentSessionRequest, *, principal: Principal
    ) -> dict[str, Any]:
        """校验 dataset_id 存在 + 写 Session + 返回锁定配置。

        通过 Principal-scoped ``DatasetRepository.resolve`` 校验 dataset_id
        真实存在且属于当前调用者——Agent Session 不能绕过 Dataset scope。
        """
        payload.validate()
        try:
            dataset = self.datasets.resolve(payload.dataset_id, principal=principal)
        except FileNotFoundError as exc:
            raise TrainingConfigValidationError('dataset_id 不存在') from exc
        except PermissionError as exc:
            # 对外使用同一不可见语义，不泄露其他 Principal 的 dataset 是否存在。
            raise TrainingConfigValidationError('dataset_id 不存在或不可访问') from exc
        evaluation_config = payload.evaluation.model_dump()
        module_flags = payload.modules.model_dump()
        context_policy = payload.context_policy.model_dump()
        context = {
            'schema_version': 'agent-context-v1',
            'status': 'pending' if any(module_flags.values()) else 'disabled',
            'source_role': context_policy['source_role'],
        }
        if any(module_flags.values()):
            context['module_status'] = {
                name: 'pending'
                for name, enabled in module_flags.items()
                if enabled
            }
        if module_flags['evidence_card']:
            try:
                dataset_digest = self.datasets.verify_integrity(dataset)
                context['evidence_card'] = build_dataset_evidence_card(
                    dataset.path,
                    dataset_digest=dataset_digest,
                    seed=payload.seed,
                    evaluation_config=evaluation_config,
                )
            except Exception:
                raise TrainingConfigValidationError(
                    'evidence_card 无法基于当前数据构建'
                ) from None
            context['module_status']['evidence_card'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['restricted_strategy_pool']:
            try:
                context['proposal_catalog'] = compile_proposal_catalog(
                    allowed_models=list(payload.allowed_models),
                    evidence_card=context.get('evidence_card'),
                    dynamic_preprocessing=module_flags['dynamic_preprocessing'],
                )
            except Exception:
                raise TrainingConfigValidationError(
                    'proposal catalog 无法基于当前会话构建'
                ) from None
            context['module_status']['restricted_strategy_pool'] = 'ready'
            if module_flags['dynamic_preprocessing']:
                context['module_status']['dynamic_preprocessing'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['bounded_hpo']:
            context['hpo_policy'] = {
                'schema_version': 'bounded-hpo-v1',
                'profile': 'tiny',
                'selection_metric': payload.selection_metric,
                'max_candidates': 3,
            }
            context['module_status']['bounded_hpo'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['fail_fast_guard']:
            hpo_policy = context.get('hpo_policy')
            context['guard_policy'] = {
                'schema_version': 'agent-guard-v1',
                'max_hpo_candidates': (
                    int(hpo_policy['max_candidates'])
                    if isinstance(hpo_policy, dict)
                    else 18
                ),
                'postflight': True,
            }
            context['module_status']['fail_fast_guard'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['constrained_code_evolution']:
            context['code_evolution_policy'] = {
                'schema_version': 'code-evolution-v1',
                'status': 'experimental',
                'mode': 'candidate_generation_and_smoke_only',
                'allowed_base_models': [
                    'cnn1d', 'cnn1d_se', 'resnet1d',
                ],
                'output_scope': 'work/code-evolution',
                'max_source_lines': 220,
                'max_parameters': 5_000_000,
                'requires_guard': True,
                'registration': 'requires_reviewed_git_commit',
            }
            context['module_status']['constrained_code_evolution'] = 'experimental'
        if module_flags['feedback_diagnosis']:
            context['diagnosis_policy'] = {
                'schema_version': 'agent-diagnosis-v1',
                'scope': 'train_validation_only',
                'categories': [
                    'data', 'split', 'feature', 'model', 'hpo', 'training',
                    'environment', 'task_semantics',
                ],
            }
            context['module_status']['feedback_diagnosis'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['limited_replanning']:
            context['replanning_policy'] = {
                'schema_version': 'limited-replanning-v1',
                'max_retries_per_failure': _DEFAULT_MAX_RETRIES_PER_FAILURE,
                'max_replan_depth': 1,
                'allowed_action_ids': [
                    'choose_unused_proposal', 'stop', 'request_human',
                ],
                'requires_parent_failure': True,
            }
            context['module_status']['limited_replanning'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['uncertainty_selection']:
            context['selection_policy'] = {
                'schema_version': 'agent-decision-support-v1',
                'target_score': 0.90,
                'max_generalization_gap': 0.05,
                'minimum_improvement': 0.01,
                'uncertainty_multiplier': 1.96,
                'minimum_independent_validation_groups': 10,
                'minimum_independent_groups_per_class': 3,
                'gap_penalty_weight': 0.25,
                'model_fit_cost_weight': 0.01,
                'auto_finalize_requires_candidate_uncertainty': True,
            }
            context['module_status']['uncertainty_selection'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        if module_flags['case_memory']:
            task_signature, evidence_signature = build_memory_signatures(
                selection_metric=payload.selection_metric,
                allowed_models=payload.allowed_models,
                evaluation_config=evaluation_config,
                evidence_card=context.get('evidence_card'),
            )
            remembered = self.sessions.list_memory_payloads_scoped(
                task_signature=task_signature,
                evidence_signature=evidence_signature,
                principal=principal,
                limit=MAX_MEMORY_CONTEXT_ITEMS,
            )
            context['memory_policy'] = {
                'schema_version': 'agent-memory-policy-v1',
                'scope': 'principal',
                'top_k': MAX_MEMORY_CONTEXT_ITEMS,
                'frozen_at_session_creation': True,
                'write_enabled': bool(context_policy['case_write']),
            }
            context['memory_context'] = freeze_memory_context(remembered)
            context['module_status']['case_memory'] = 'ready'
            if all(
                status == 'ready'
                for status in context['module_status'].values()
            ):
                context['status'] = 'ready'
        session = self.sessions.create_session(
            dataset_id=payload.dataset_id,
            selection_metric=payload.selection_metric,
            allowed_models=list(payload.allowed_models),
            max_runs=payload.max_runs,
            seed=payload.seed,
            evaluation_config=evaluation_config,
            principal=principal,
            module_flags=module_flags,
            context_policy=context_policy,
            context=context,
        )
        return {
            'session_id': session.session_id,
            'state': session.state,
            'remaining_runs': session.max_runs,
            'locked_config': {
                'dataset_id': session.dataset_id,
                'selection_metric': session.selection_metric,
                'allowed_models': list(session.allowed_models),
                'max_runs': session.max_runs,
                'seed': session.seed,
                'evaluation_config': evaluation_config,
                'modules': session.module_flags,
                'context_policy': session.context_policy,
            },
            'context': session.context,
            'created_at': session.created_at,
        }

    def create_experiment(
        self,
        *,
        session_id: str,
        payload: CreateAgentExperimentRequest,
        principal: Principal,
    ) -> dict[str, Any]:
        """在 Session 内提交一次新实验。校验流程：

        1. Session 必须属于当前 Principal 且处于 open
        2. Session 内不能有非终态 Run（一次只跑一个）
        3. 实验数不能超过 max_runs
        4. config 不能与同 Session 内历史实验重复
        5. 复用 ``RunRepository.create_queued`` 入队底层 Training Run
        """
        payload.validate()
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        if session.state == 'finalized':
            raise AgentSessionClosed(f'session {session_id} 已经 finalized，不能继续提交')

        if payload.model_type not in session.allowed_models:
            raise TrainingConfigValidationError(
                f'model_type {payload.model_type} 不在 session 允许的模型集内'
            )

        action = payload.model_dump()
        if isinstance(action.get('rationale'), str):
            action['rationale'] = _safe_string(action['rationale'])
        if session.module_flags['restricted_strategy_pool']:
            try:
                recipe = resolve_proposal(session.context, payload.proposal_id)
            except ValueError:
                raise TrainingConfigValidationError(
                    'proposal_id 不在会话锁定策略目录中'
                ) from None
            for field in ('model_type', 'normalization', 'class_balance'):
                if action.get(field) != recipe[field]:
                    raise TrainingConfigValidationError(
                        '实验字段与 proposal_id 的 canonical recipe 不一致'
                    )
            action.update(recipe)
        elif payload.proposal_id is not None:
            raise TrainingConfigValidationError(
                '未启用 restricted_strategy_pool 时禁止 proposal_id'
            )
        else:
            action.pop('proposal_id', None)
        config_hash = _compute_config_hash(session=session, action=action)

        # 这些是快速失败的只读检查；最终 max_runs / config / reservation
        # 判定仍由下面的 BEGIN IMMEDIATE 原子 reservation 再校验一次。
        if self.sessions.find_duplicate_config(
            session_id=session_id, config_hash=config_hash, principal=principal
        ):
            raise AgentConfigCollision(
                f'config_hash {config_hash} 在该 session 内已存在'
            )
        experiment_count = self.sessions.count_experiments_scoped(
            session_id=session_id, principal=principal
        )
        if experiment_count >= session.max_runs:
            raise AgentConfigCollision(
                f'session {session_id} 已达 max_runs={session.max_runs} 上限'
            )

        # 同 Session 不能并发运行：queued/running 的实验存在时拒绝提交新实验
        existing = self.sessions.list_experiments_scoped(session_id, principal=principal)
        existing_run_ids = [item.run_id for item in existing]
        run_states = collect_run_states(self.runs, existing_run_ids)
        if self.sessions.has_non_terminal_experiment_scoped(
            session_id=session_id, run_states=run_states, principal=principal
        ):
            raise AgentConfigCollision(
                f'session {session_id} 仍有非终态 Run，请等待完成后再提交'
            )

        if session.module_flags.get('limited_replanning'):
            replanned_parents = {
                item.parent_run_id
                for item in existing
                if item.parent_run_id is not None
            }
            unresolved_failures = [
                item.run_id
                for item in existing
                if run_states.get(item.run_id) == 'failed'
                and item.run_id not in replanned_parents
            ]
            if unresolved_failures and payload.action_id is None:
                raise TrainingConfigValidationError(
                    '存在未处理失败实验时必须使用受控 REPLAN action'
                )

        # parent_run_id 若存在，必须属于同一 Session
        parent_record: RunRecord | None = None
        parent_experiment: AgentExperimentRecord | None = None
        if payload.parent_run_id is not None:
            try:
                parent_experiment = self.sessions.get_experiment_scoped(
                    session_id, payload.parent_run_id, principal=principal
                )
                parent_record = self._safe_get_run(
                    payload.parent_run_id, principal=principal
                )
            except Exception as exc:
                raise TrainingConfigValidationError(
                    f'parent_run_id {payload.parent_run_id} 不属于当前 session'
                ) from exc

        if payload.action_id is not None:
            if not session.module_flags.get('limited_replanning'):
                raise TrainingConfigValidationError(
                    '当前 session 未启用 limited_replanning'
                )
            if parent_record is None or parent_record.state != 'failed':
                raise TrainingConfigValidationError(
                    '有限重规划的 parent_run_id 必须是失败实验'
                )
            if parent_experiment is not None and parent_experiment.parent_run_id is not None:
                raise TrainingConfigValidationError(
                    '有限重规划不允许超过 max_replan_depth=1'
                )
            train_metrics, valid_metrics, _ = _extract_train_validation_metrics(
                parent_record.run_id
            )
            diagnosis = build_diagnosis(
                run_state=parent_record.state,
                selection_metric=session.selection_metric,
                training_metrics=train_metrics,
                validation_metrics=valid_metrics,
                error_details=parent_record.error_details,
                evidence_card=session.context.get('evidence_card'),
                restricted_actions_available=True,
            )
            if payload.action_id not in diagnosis.get('allowed_action_ids', []):
                raise TrainingConfigValidationError(
                    'action_id 不在父实验 diagnosis 允许集合中'
                )
            retry_count = sum(
                1
                for item in existing
                if item.parent_run_id == parent_record.run_id
                and item.action_json.get('action_id') == payload.action_id
            )
            replanning_policy = session.context.get('replanning_policy')
            retry_limit = (
                int(replanning_policy.get(
                    'max_retries_per_failure',
                    _DEFAULT_MAX_RETRIES_PER_FAILURE,
                ))
                if isinstance(replanning_policy, dict)
                else _DEFAULT_MAX_RETRIES_PER_FAILURE
            )
            if retry_count >= retry_limit:
                raise AgentConfigCollision('同一失败与 action_id 已达有限重试上限')
        elif (
            payload.parent_run_id is not None
            and session.module_flags.get('limited_replanning')
        ):
            raise TrainingConfigValidationError(
                'parent_run_id 只能用于有限重规划 action'
            )

        # 合成训练配置并通过 TrainingSpec.validated 校验（与现有 /api/training/runs 同口径）
        training_config = _build_training_config(session=session, action=action)
        try:
            spec = TrainingSpec.from_legacy(training_config).validated(has_external_test=False)
        except TrainingConfigValidationError as exc:
            raise TrainingConfigValidationError(
                f'合成后的训练配置不被现有 TrainingSpec 接受: {exc}'
            ) from exc

        try:
            dataset = self.datasets.resolve(session.dataset_id, principal=principal)
            dataset_snapshot = self.datasets.snapshot(
                dataset, dataset_id=session.dataset_id
            )
        except FileNotFoundError as exc:
            raise TrainingConfigValidationError('dataset_id 不存在') from exc
        except PermissionError as exc:
            raise TrainingConfigValidationError('dataset_id 不存在或不可访问') from exc

        # 先 reservation 再创建 queued Run。若后续 Run/Experiment 任一步失败，
        # except 分支会释放 reservation 并补偿取消刚创建的 queued Run。
        reservation = self.sessions.reserve_experiment(
            session_id=session_id,
            config_hash=config_hash,
            action_json=action,
            principal=principal,
            parent_run_id=payload.parent_run_id,
            action_id=payload.action_id,
            max_retries_per_failure=(
                retry_limit if payload.action_id is not None else None
            ),
            require_replan_for_failed=bool(
                session.module_flags.get('limited_replanning')
            ),
        )
        record: RunRecord | None = None
        try:
            # 复用 RunRepository.create_queued，让 Worker 与现有 Run 共享状态机。
            record = self.runs.create_queued(
                dataset_id=session.dataset_id,
                legacy_data_path=None,
                config=spec.to_legacy_dict(),
                dataset_snapshot=dataset_snapshot,
                principal=principal,
            )
            experiment = self.sessions.bind_reservation(
                reservation_id=reservation.reservation_id,
                run_id=record.run_id,
                parent_run_id=payload.parent_run_id,
                rationale=action.get('rationale'),
                principal=principal,
            )
        except Exception:
            self.sessions.release_reservation(
                reservation_id=reservation.reservation_id,
                principal=principal,
            )
            if record is not None:
                try:
                    self.runs.cancel_scoped(
                        record.run_id,
                        now=datetime.now(timezone.utc),
                        principal=principal,
                        reason='agent_bind_failed',
                        message='Agent experiment 绑定失败，已补偿取消',
                    )
                except Exception:
                    # 原始异常更能定位入队/绑定失败；Run 取消失败会由状态巡检
                    # 暴露，不把未知的补偿异常伪装成成功提交。
                    pass
            raise
        return {
            'session_id': session_id,
            'run_id': record.run_id,
            'attempt': experiment.attempt,
            'config_hash': experiment.config_hash,
            'state': record.state,
            'effective_action': _scrub(action),
        }

    def get_feedback(
        self,
        *,
        session_id: str,
        run_id: str,
        principal: Principal,
    ) -> dict[str, Any]:
        """返回安全版训练反馈：queued/running 只给 state+progress；succeeded
        只给 Validation 标量（无 Test、artifact、explainability）。

        Validation 来源：``storage/runs/<run_id>/metrics.json`` 的 ``valid`` 子树。
        读取失败（文件不存在、JSON 损坏）一律视为 ``status="unknown"``，绝不从
        其他子树（test、cv_summary、fold_mean）借数冒充 Validation。
        """
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiment = self.sessions.get_experiment_scoped(
            session_id, run_id, principal=principal
        )
        record = self._safe_get_run(run_id, principal=principal)

        feedback: dict[str, Any] = {
            'session_id': session_id,
            'run_id': run_id,
            'attempt': experiment.attempt,
            'state': record.state,
            'effective_action': _scrub(experiment.action_json),
            'selection_metric': session.selection_metric,
            'run_summary': _safe_record_summary(record),
            'validation': {'status': 'unknown', 'metrics': {}},
            'progress': _safe_progress(record),
            'remaining_runs': max(0, session.max_runs - self._experiment_count(session_id, principal)),
        }
        if session.module_flags.get('fail_fast_guard'):
            guard_result = _guard_feedback(record)
            if guard_result is not None:
                feedback['guard_result'] = _scrub(guard_result)

        if record.state == 'succeeded':
            validation_metrics, source = _extract_validation_metrics(run_id)
            feedback['validation'] = {
                'status': 'ready' if validation_metrics else 'unavailable',
                'metrics': validation_metrics,
                'aggregation': source.get('aggregation'),
                'fold_count': source.get('fold_count'),
            }
            feedback['validation_score'] = _selection_score(
                session.selection_metric, validation_metrics
            )
        elif record.state == 'failed':
            feedback['validation'] = {
                'status': 'failed',
                'metrics': {},
                'error': _scrub({'message': record.error or ''}),
            }
        else:
            feedback['validation'] = {'status': 'pending', 'metrics': {}}

        diagnosis: dict[str, Any] | None = None
        if session.module_flags.get('feedback_diagnosis'):
            train_metrics, validation_metrics, _ = _extract_train_validation_metrics(
                run_id
            )
            diagnosis = build_diagnosis(
                run_state=record.state,
                selection_metric=session.selection_metric,
                training_metrics=train_metrics,
                validation_metrics=validation_metrics,
                error_details=record.error_details,
                evidence_card=session.context.get('evidence_card'),
                restricted_actions_available=isinstance(
                    session.context.get('proposal_catalog'), dict
                ),
            )
            feedback['diagnosis'] = diagnosis

        if record.state in {'succeeded', 'failed', 'cancelled'}:
            self._persist_terminal_outcome(
                session=session,
                experiment=experiment,
                record=record,
                diagnosis=diagnosis,
                principal=principal,
            )

        return _scrub(feedback)

    def get_session(
        self, *, session_id: str, principal: Principal
    ) -> dict[str, Any]:
        """Session 详情 + 全部实验摘要。"""
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiments = self.sessions.list_experiments_scoped(session_id, principal=principal)
        run_ids = [item.run_id for item in experiments]
        run_states = collect_run_states(self.runs, run_ids)

        experiment_summaries: list[dict[str, Any]] = []
        validation_scores: list[tuple[float, str]] = []  # (score, run_id)
        for experiment in experiments:
            record = self._safe_get_run(experiment.run_id, principal=principal)
            summary = _safe_record_summary(record)
            validation_score: float | None = None
            validation_source: dict[str, Any] = {}
            if record.state == 'succeeded':
                metrics, validation_source = _extract_validation_metrics(
                    experiment.run_id
                )
                validation_score = _selection_score(session.selection_metric, metrics)
                if validation_score is not None:
                    validation_scores.append((validation_score, experiment.run_id))
            summary_entry = {
                'run_id': experiment.run_id,
                'attempt': experiment.attempt,
                'parent_run_id': experiment.parent_run_id,
                'rationale': experiment.rationale,
                'state': record.state,
                'effective_action': _scrub(experiment.action_json),
                'validation_score': validation_score,
                'config_hash': experiment.config_hash,
                'created_at': experiment.created_at,
            }
            metric_stds = validation_source.get('metric_std')
            validation_std = (
                metric_stds.get(session.selection_metric)
                if isinstance(metric_stds, dict)
                else None
            )
            if isinstance(validation_std, (int, float)) and not isinstance(
                validation_std, bool
            ):
                summary_entry['validation_std'] = float(validation_std)
                summary_entry['validation_uncertainty'] = {
                    'method': validation_source.get('uncertainty_method'),
                    'independent_group_count': validation_source.get(
                        'independent_group_count'
                    ),
                    'minimum_group_count_per_class': validation_source.get(
                        'minimum_group_count_per_class'
                    ),
                }
            guard = _guard_feedback(record)
            if isinstance(guard, dict):
                postflight = guard.get('postflight')
                if isinstance(postflight, dict):
                    fit_count = postflight.get('model_fit_count')
                    if isinstance(fit_count, int) and not isinstance(fit_count, bool):
                        summary_entry['model_fit_count'] = fit_count
            if session.module_flags.get('feedback_diagnosis') and record.state in {
                'succeeded', 'failed'
            }:
                train_metrics, valid_metrics, _ = _extract_train_validation_metrics(
                    experiment.run_id
                )
                summary_entry['diagnosis'] = build_diagnosis(
                    run_state=record.state,
                    selection_metric=session.selection_metric,
                    training_metrics=train_metrics,
                    validation_metrics=valid_metrics,
                    error_details=record.error_details,
                    evidence_card=session.context.get('evidence_card'),
                    restricted_actions_available=isinstance(
                        session.context.get('proposal_catalog'), dict
                    ),
                )
            experiment_summaries.append(_scrub(summary_entry))

        # 选 best_run_id：按 selection_metric 排序；同分时 attempt 最小优先
        best_run_id: str | None = None
        if validation_scores:
            # 因为 attempt 越大越后做，所以"先 attempt 再 score 升序"即可
            run_to_attempt = {item.run_id: item.attempt for item in experiments}
            validation_scores.sort(
                key=lambda pair: (pair[0], -run_to_attempt.get(pair[1], 0))
            )
            best_run_id = validation_scores[-1][1]

        remaining_runs = max(0, session.max_runs - len(experiments))
        response = {
            'session_id': session.session_id,
            'state': session.state,
            'locked_config': {
                'dataset_id': session.dataset_id,
                'selection_metric': session.selection_metric,
                'allowed_models': list(session.allowed_models),
                'max_runs': session.max_runs,
                'seed': session.seed,
                'evaluation_config': session.evaluation_config,
                'modules': session.module_flags,
                'context_policy': session.context_policy,
            },
            'context': session.context,
            'remaining_runs': remaining_runs,
            'best_run_id': best_run_id,
            'selected_run_id': session.selected_run_id,
            'created_at': session.created_at,
            'finalized_at': session.finalized_at,
            'experiments': experiment_summaries,
        }
        if session.module_flags.get('uncertainty_selection'):
            policy = session.context.get('selection_policy')
            policy = policy if isinstance(policy, dict) else {}
            support = build_decision_support(
                experiment_summaries,
                remaining_runs=remaining_runs,
                proposal_catalog=session.context.get('proposal_catalog'),
                target_score=float(policy.get('target_score', 0.90)),
                max_generalization_gap=float(
                    policy.get('max_generalization_gap', 0.05)
                ),
                minimum_improvement=float(
                    policy.get('minimum_improvement', 0.01)
                ),
                uncertainty_multiplier=float(
                    policy.get('uncertainty_multiplier', 1.96)
                ),
            )
            response['decision_support'] = support
            response['recommended_run_id'] = support['recommended_run_id']
        return _scrub(response)

    def finalize_session(
        self,
        *,
        session_id: str,
        selected_run_id: str,
        principal: Principal,
    ) -> dict[str, Any]:
        """Finalize：把 Session 切到 finalized + 记录 selected_run_id + 锁住。

        ``selected_run_id`` 必须属于该 Session + 训练成功。重复 finalize 是
        幂等的：相同 selected_run_id 返回旧记录；不同 selected_run_id 则报错。
        """
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiment = self.sessions.get_experiment_scoped(
            session_id, selected_run_id, principal=principal
        )
        record = self._safe_get_run(experiment.run_id, principal=principal)
        if record.state != 'succeeded':
            raise TrainingConfigValidationError(
                f'只能 finalize 训练成功的 Run，当前 run {selected_run_id} state={record.state}'
            )

        self._persist_terminal_outcome(
            session=session,
            experiment=experiment,
            record=record,
            diagnosis=None,
            principal=principal,
            require_validation_score=True,
        )

        updated = self.sessions.finalize_session(
            session_id=session_id,
            selected_run_id=selected_run_id,
            principal=principal,
        )

        # ``final_result_url`` 只是提示人类用户去前端结果页查看，不暴露给 Agent
        # 任何 Test / artifact 信息；前端结果页是现有的 /#/results?run_id=...
        return _scrub({
            'session_id': updated.session_id,
            'state': updated.state,
            'selected_run_id': updated.selected_run_id,
            'experiments_locked': True,
            'final_result_url': f'/#/results?run_id={selected_run_id}',
            'finalized_at': updated.finalized_at,
        })

    # ---- 内部 helpers ----

    @staticmethod
    def _memory_write_enabled(session: AgentSessionRecord) -> bool:
        flags = session.module_flags
        policy = session.context_policy
        return bool(
            flags.get('case_memory')
            and flags.get('evidence_card')
            and flags.get('fail_fast_guard')
            and flags.get('feedback_diagnosis')
            and policy.get('case_write') is True
            and policy.get('source_role') != 'benchmark'
        )

    def _persist_terminal_outcome(
        self,
        *,
        session: AgentSessionRecord,
        experiment: AgentExperimentRecord,
        record: RunRecord,
        diagnosis: dict[str, Any] | None,
        principal: Principal,
        require_validation_score: bool = False,
    ):
        """Persist an allow-listed terminal snapshot when Session policy permits.

        The returned repository record is internal.  It is used only to derive
        the selected case during finalize and is never serialized by the API.
        """
        if not self._memory_write_enabled(session):
            return None
        if record.state not in {'succeeded', 'failed', 'cancelled'}:
            return None
        train_metrics, validation_metrics, source = _extract_train_validation_metrics(
            record.run_id
        )
        if diagnosis is None:
            diagnosis = build_diagnosis(
                run_state=record.state,
                selection_metric=session.selection_metric,
                training_metrics=train_metrics,
                validation_metrics=validation_metrics,
                error_details=record.error_details,
                evidence_card=session.context.get('evidence_card'),
                restricted_actions_available=isinstance(
                    session.context.get('proposal_catalog'), dict
                ),
            )
        validation_score = _selection_score(
            session.selection_metric, validation_metrics
        )
        if record.state == 'succeeded' and validation_score is None:
            if require_validation_score:
                raise TrainingConfigValidationError(
                    '启用 case_memory 写入时，成功 Run 必须包含有效 Validation 分数'
                )
            return None
        metric_stds = source.get('metric_std')
        validation_std = (
            metric_stds.get(session.selection_metric)
            if isinstance(metric_stds, dict)
            else None
        )
        fit_count: int | None = None
        guard = _guard_feedback(record)
        if isinstance(guard, dict):
            postflight = guard.get('postflight')
            if isinstance(postflight, dict):
                observed_fit_count = postflight.get('model_fit_count')
                if isinstance(observed_fit_count, int) and not isinstance(
                    observed_fit_count, bool
                ):
                    fit_count = observed_fit_count
        duration_seconds = _safe_record_summary(record).get('duration_seconds')
        task_signature, evidence_signature = build_memory_signatures(
            selection_metric=session.selection_metric,
            allowed_models=session.allowed_models,
            evaluation_config=session.evaluation_config,
            evidence_card=session.context.get('evidence_card'),
        )
        outcome_payload = build_terminal_outcome(
            state=record.state,
            selection_metric=session.selection_metric,
            action=experiment.action_json,
            validation_score=validation_score,
            validation_std=validation_std,
            diagnosis=diagnosis,
            duration_seconds=duration_seconds,
            model_fit_count=fit_count,
        )
        failure_payload = (
            build_memory_record('failure', outcome_payload)
            if record.state == 'failed'
            else None
        )
        return self.sessions.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id=record.run_id,
            state=record.state,
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=outcome_payload,
            failure_memory_payload=failure_payload,
            principal=principal,
        )

    def _experiment_count(self, session_id: str, principal: Principal) -> int:
        return self.sessions.count_experiments_scoped(
            session_id=session_id, principal=principal
        )

    def _safe_get_run(self, run_id: str, *, principal: Principal) -> RunRecord:
        try:
            return self.runs.get_scoped(run_id, principal=principal)
        except RunNotFound as exc:
            raise TrainingConfigValidationError(
                f'run {run_id} 不属于当前 principal 作用域'
            ) from exc


# ---- 独立函数：metrics.json 解析与 validation 抽取 ----


def _extract_train_validation_metrics(
    run_id: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """只从 metrics.json 的 train/valid 子树读取有限白名单标量。

    返回 ``(train_metrics, valid_metrics, source_meta)``：
    - ``source_meta`` 是 ``{aggregation, fold_count}``，用于 AgentFeedback 字段
    - 读不到 / 解析失败时返回三个空 dict
    """
    run_dir = Path(STORAGE_DIR) / 'runs' / run_id
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.is_file():
        return {}, {}, {}
    try:
        payload = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {}, {}
    if not isinstance(payload, dict):
        return {}, {}, {}

    def selected(split_name: str) -> dict[str, float]:
        split = payload.get(split_name)
        if not isinstance(split, dict):
            return {}
        return {
            key: float(value)
            for key in _VALIDATION_METRIC_KEYS
            if isinstance((value := split.get(key)), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        }

    train_metrics = selected('train')
    valid_metrics = selected('valid')
    valid = payload.get('valid') if isinstance(payload.get('valid'), dict) else {}
    source_meta: dict[str, Any] = {}
    aggregation = valid.get('aggregation')
    if isinstance(aggregation, str):
        source_meta['aggregation'] = aggregation
    fold_count = valid.get('fold_count')
    if isinstance(fold_count, int):
        source_meta['fold_count'] = fold_count
    metric_stds: dict[str, float] = {}
    projected_group_count: int | None = None
    confusion_counts = valid.get('confusion_matrix')
    independent_group_count = valid.get('independent_group_count')
    minimum_group_count_per_class = valid.get(
        'minimum_group_count_per_class'
    )
    for metric in _VALIDATION_METRIC_KEYS:
        estimate = estimate_validation_metric_std(
            confusion_counts,
            metric,
            independent_group_count=independent_group_count,
            minimum_group_count_per_class=minimum_group_count_per_class,
        )
        if estimate is None:
            continue
        metric_std, group_count = estimate
        metric_stds[metric] = float(metric_std)
        projected_group_count = group_count
    if metric_stds and projected_group_count is not None:
        source_meta.update({
            'uncertainty_method': VALIDATION_UNCERTAINTY_METHOD,
            'independent_group_count': projected_group_count,
            'minimum_group_count_per_class': minimum_group_count_per_class,
            'metric_std': metric_stds,
        })
    return train_metrics, valid_metrics, source_meta


def _extract_validation_metrics(run_id: str) -> tuple[dict[str, float], dict[str, Any]]:
    _, valid_metrics, source_meta = _extract_train_validation_metrics(run_id)
    return valid_metrics, source_meta


def _selection_score(metric: str, validation_metrics: dict[str, float]) -> float | None:
    if not validation_metrics:
        return None
    value = validation_metrics.get(metric)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
