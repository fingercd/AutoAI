"""Agent API v1 服务：冻结 Session、预约实验、提交 Run 与安全 Observation。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..datasets.repository import DatasetRepository
from ..runs.contracts import Principal, RunRecord
from ..runs.repository import InvalidRunTransition, RunNotFound, RunRepository
from ..runs.submission import RunSubmissionError, RunSubmissionRequest, RunSubmissionService
from .capabilities import module_catalog
from .contracts import (
    AGENT_API_CONTRACT_VERSION,
    AGENT_RESERVATION_PROTOCOL_VERSION,
    AgentDomainError,
    CreateAgentExperimentRequest,
    CreateAgentSessionRequest,
)
from .observation import build_observation, validation_from_manifest
from .repository import (
    AgentExperimentRecord,
    AgentSessionClosed,
    AgentSessionRecord,
    AgentSessionRepository,
    collect_run_states,
)


def _hash_payload(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != 'client_request_id'}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compute_config_hash(*, session: AgentSessionRecord, action: dict[str, Any]) -> str:
    return _hash_payload({
        'dataset_id': session.dataset_id,
        'selection_metric': session.selection_metric,
        'seed': session.seed,
        'evaluation': session.evaluation_config,
        'action': {key: value for key, value in action.items() if key not in {'rationale', 'client_request_id'}},
    })[:16]


def _training_config(session: AgentSessionRecord, action: dict[str, Any]) -> dict[str, Any]:
    return {
        'model_type': action['model_type'],
        'normalization': action.get('normalization', 'zscore'),
        'class_balance': action.get('class_balance', 'none'),
        'seed': session.seed,
        'feature_selection_enabled': False,
        **session.evaluation_config,
    }


def _locked_config(session: AgentSessionRecord) -> dict[str, Any]:
    evaluation = session.evaluation_config
    return {
        'dataset_id': session.dataset_id,
        'selection_metric': session.selection_metric,
        'allowed_models': list(session.allowed_models),
        'max_runs': session.max_runs,
        'seed': session.seed,
        'evaluation_config': {
            'mode': evaluation.get('split_mode'),
            'train_weight': evaluation.get('split_train'),
            'validation_weight': evaluation.get('split_valid'),
            'heldout_weight': evaluation.get('split_test'),
        },
        'modules': list(session.modules),
        'context_policy': session.context_policy,
    }


class AgentService:
    def __init__(self, *, session_repository: AgentSessionRepository,
                 run_repository: RunRepository, dataset_repository: DatasetRepository,
                 submission_service: RunSubmissionService, run_root: Path) -> None:
        self.sessions = session_repository
        self.runs = run_repository
        self.datasets = dataset_repository
        self.submissions = submission_service
        self.run_root = Path(run_root)

    def create_session(self, payload: CreateAgentSessionRequest, *, principal: Principal) -> dict[str, Any]:
        payload.validate_business()
        catalog = module_catalog()
        unavailable = sorted({name for name in payload.modules if not catalog.get(name, {}).get('available')})
        if unavailable:
            raise AgentDomainError(
                'agent_module_unavailable', f'请求的模块当前不可用：{", ".join(unavailable)}', status_code=422
            )
        if payload.context_policy.case_write:
            raise AgentDomainError(
                'agent_module_unavailable', 'case_memory 不可用，不能启用 case_write', status_code=422
            )
        try:
            self.datasets.resolve(payload.dataset_id, principal=principal)
        except (FileNotFoundError, PermissionError):
            raise AgentDomainError('dataset_unavailable', '数据集不存在或当前调用者无权访问', status_code=404) from None
        body = payload.model_dump(mode='json')
        session, created = self.sessions.create_session(
            dataset_id=payload.dataset_id,
            selection_metric=payload.selection_metric,
            allowed_models=list(payload.allowed_models),
            max_runs=payload.max_runs,
            seed=payload.seed,
            evaluation_config=payload.evaluation.model_dump(mode='json'),
            modules=list(payload.modules),
            context_policy=payload.context_policy.model_dump(mode='json'),
            client_request_id=payload.client_request_id,
            payload_hash=_hash_payload(body),
            principal=principal,
        )
        return {
            'contract_version': AGENT_API_CONTRACT_VERSION,
            'session_id': session.session_id,
            'state': session.state,
            'remaining_runs': max(0, session.max_runs - self.sessions.count_budget_scoped(
                session_id=session.session_id, principal=principal
            )),
            'locked_config': _locked_config(session),
            'created_at': session.created_at,
            'idempotent_replay': not created,
        }

    def create_experiment(self, *, session_id: str, payload: CreateAgentExperimentRequest,
                          principal: Principal) -> dict[str, Any]:
        payload.validate_business()
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        if session.state != 'open':
            raise AgentSessionClosed()
        if payload.model_type not in session.allowed_models:
            raise AgentDomainError(
                'agent_invalid_action', 'model_type 不在 session 允许的模型集合内', status_code=422
            )
        body = payload.model_dump(mode='json')
        action = {
            'model_type': payload.model_type,
            'normalization': payload.normalization,
            'class_balance': payload.class_balance,
            'parent_run_id': payload.parent_run_id,
        }
        config_hash = _compute_config_hash(session=session, action=action)
        existing = self.sessions.list_experiments_scoped(session_id, principal=principal)
        bound_ids = [item.run_id for item in existing if item.run_id]
        states = collect_run_states(self.runs, bound_ids)
        active = {run_id for run_id, state in states.items() if state in {'queued', 'running'}}
        reservation, created = self.sessions.reserve_experiment(
            session_id=session_id,
            action_json=action,
            rationale=payload.rationale,
            parent_run_id=payload.parent_run_id,
            config_hash=config_hash,
            client_request_id=payload.client_request_id,
            payload_hash=_hash_payload(body),
            active_run_ids=active,
            principal=principal,
            protocol_version=AGENT_RESERVATION_PROTOCOL_VERSION,
        )
        if not created and reservation.state == 'bound' and reservation.run_id:
            record = self._run(reservation.run_id, principal)
            return self._experiment_response(session_id, reservation, record, replay=True)
        if not created and reservation.state == 'compensation_required':
            raise AgentDomainError(
                'agent_compensation_required', '先前提交需要人工核对，当前预算继续保留',
                status_code=409, allowed_actions=('inspect_ml_session',),
            )
        if not created and reservation.state == 'released':
            raise AgentDomainError(
                'agent_request_released',
                '该请求已经结束；如需重新训练，请使用新的 client_request_id',
                status_code=409,
                allowed_actions=('inspect_ml_session',),
            )

        try:
            submitted = self.submissions.submit(RunSubmissionRequest(
                dataset_id=session.dataset_id,
                legacy_data_path=None,
                test_dataset_id=None,
                test_legacy_data_path=None,
                raw_config=_training_config(session, action),
                principal=principal,
                submission_source='agent',
                submission_key=reservation.experiment_id,
            ))
        except RunSubmissionError as exc:
            if exc.code in {'agent_submission_key_conflict', 'agent_submission_mapping_invalid'}:
                self.sessions.require_compensation(
                    reservation.experiment_id, run_id=None,
                    failure_code=exc.code, principal=principal,
                )
            else:
                # 原子 mapping 能证明是否已有 Run：仅在确定没有 mapping 时返还预算。
                try:
                    mapping = self.runs.lookup_submission_mapping(
                        reservation.experiment_id, principal=principal
                    )
                except Exception:
                    mapping = None
                if mapping is not None and mapping.status == 'missing':
                    self.sessions.release_reservation(
                        reservation.experiment_id, failure_code=exc.code, principal=principal
                    )
                elif mapping is not None and mapping.status == 'scope_mismatch':
                    self.sessions.require_compensation(
                        reservation.experiment_id, run_id=None,
                        failure_code='agent_submission_scope_mismatch', principal=principal,
                    )
            raise AgentDomainError(
                exc.code, exc.message, status_code=exc.status_code,
                retryable=exc.retryable, allowed_actions=exc.allowed_actions,
            ) from exc
        except Exception as exc:
            # 未知异常后只有“mapping 明确不存在”才能证明没有 Run；否则保持 reserved。
            try:
                mapping = self.runs.lookup_submission_mapping(
                    reservation.experiment_id, principal=principal
                )
            except Exception:
                mapping = None
            if mapping is not None and mapping.status == 'missing':
                self.sessions.release_reservation(
                    reservation.experiment_id,
                    failure_code='agent_submission_failed', principal=principal,
                )
            elif mapping is not None and mapping.status == 'scope_mismatch':
                self.sessions.require_compensation(
                    reservation.experiment_id, run_id=None,
                    failure_code='agent_submission_scope_mismatch', principal=principal,
                )
            raise AgentDomainError(
                'agent_submission_failed', '训练任务提交失败', status_code=503, retryable=True,
                allowed_actions=('inspect_ml_capabilities',),
            ) from exc

        record = submitted.record
        try:
            bound = self.sessions.bind_experiment(
                reservation.experiment_id, run_id=record.run_id, principal=principal
            )
        except Exception as exc:
            try:
                current = self.sessions.get_reservation_scoped(
                    reservation.experiment_id, principal=principal
                )
                if current.state == 'bound' and current.run_id == record.run_id:
                    return self._experiment_response(
                        session_id, current, record, replay=(not created)
                    )
            except Exception:
                pass
            never_started = record.state == 'queued' and record.started_at is None
            compensated = False
            try:
                cancelled = self.runs.cancel_queued_unstarted_scoped(
                    record.run_id, now=datetime.now(timezone.utc), principal=principal,
                    reason='agent_binding_failed', message='Agent 实验绑定失败，Run 已取消',
                )
                compensated = never_started and cancelled.started_at is None
            except (RunNotFound, InvalidRunTransition):
                compensated = False
            if compensated:
                self.sessions.release_reservation(
                    reservation.experiment_id, failure_code='agent_binding_failed', principal=principal
                )
            else:
                self.sessions.require_compensation(
                    reservation.experiment_id, run_id=record.run_id,
                    failure_code='agent_binding_failed', principal=principal,
                )
            raise AgentDomainError(
                'agent_experiment_binding_failed', '实验绑定失败，已执行安全补偿',
                status_code=503, retryable=False,
            ) from exc
        return self._experiment_response(session_id, bound, record, replay=(not created))

    @staticmethod
    def _experiment_response(session_id: str, experiment: AgentExperimentRecord,
                             record: RunRecord, *, replay: bool) -> dict[str, Any]:
        return {
            'contract_version': AGENT_API_CONTRACT_VERSION,
            'session_id': session_id,
            'run_id': record.run_id,
            'attempt': experiment.attempt,
            'config_hash': experiment.config_hash,
            'state': record.state,
            'binding_state': experiment.state,
            'effective_action': experiment.action_json,
            'idempotent_replay': replay,
        }

    def get_feedback(self, *, session_id: str, run_id: str, principal: Principal) -> dict[str, Any]:
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiment = self.sessions.get_experiment_scoped(session_id, run_id, principal=principal)
        record = self._run(run_id, principal)
        remaining = max(0, session.max_runs - self.sessions.count_budget_scoped(
            session_id=session_id, principal=principal
        ))
        return build_observation(
            session_id=session_id,
            session_state=session.state,
            selection_metric=session.selection_metric,
            run_dir=self.run_root / run_id,
            record=record,
            attempt=experiment.attempt,
            effective_action=experiment.action_json,
            remaining_runs=remaining,
        )

    def get_session(self, *, session_id: str, principal: Principal) -> dict[str, Any]:
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiments = self.sessions.list_experiments_scoped(session_id, principal=principal)
        summaries: list[dict[str, Any]] = []
        scores: list[tuple[float, int, str]] = []
        for item in experiments:
            entry: dict[str, Any] = {
                'attempt': item.attempt, 'run_id': item.run_id, 'binding_state': item.state,
                'parent_run_id': item.parent_run_id, 'effective_action': item.action_json,
                'config_hash': item.config_hash, 'failure_code': item.failure_code,
                'created_at': item.created_at, 'state': 'binding_failed' if not item.run_id else 'missing',
                'validation_score': None,
            }
            if item.run_id:
                try:
                    record = self._run(item.run_id, principal)
                    entry['state'] = record.state
                    if record.state == 'succeeded':
                        status, metrics, _ = validation_from_manifest(
                            self.run_root / item.run_id, run_id=item.run_id
                        )
                        value = metrics.get(session.selection_metric) if status == 'ready' else None
                        if value is not None:
                            entry['validation_score'] = value
                            scores.append((value, -item.attempt, item.run_id))
                except AgentDomainError:
                    pass
            summaries.append(entry)
        best_run_id = max(scores)[2] if scores else None
        used = sum(1 for item in experiments if item.state != 'released')
        return {
            'contract_version': AGENT_API_CONTRACT_VERSION,
            'session_id': session.session_id,
            'state': session.state,
            'locked_config': _locked_config(session),
            'remaining_runs': max(0, session.max_runs - used),
            'best_run_id': best_run_id,
            'selected_run_id': session.selected_run_id,
            'created_at': session.created_at,
            'finalized_at': session.finalized_at,
            'experiments': summaries,
        }

    def finalize_session(self, *, session_id: str, selected_run_id: str,
                         principal: Principal) -> dict[str, Any]:
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        experiment = self.sessions.get_experiment_scoped(
            session_id, selected_run_id, principal=principal
        )
        record = self._run(selected_run_id, principal)
        if record.state != 'succeeded':
            raise AgentDomainError(
                'agent_invalid_action', f'只能 finalize 已成功且结果完整的 Run，state={record.state}', status_code=422
            )
        status, metrics, _ = validation_from_manifest(self.run_root / selected_run_id, run_id=selected_run_id)
        if status != 'ready' or session.selection_metric not in metrics:
            raise AgentDomainError(
                'agent_invalid_action', 'Run 的 Manifest 或 Validation 不完整', status_code=422
            )
        updated = self.sessions.finalize_session(
            session_id=session_id, selected_run_id=experiment.run_id or '', principal=principal
        )
        return {
            'contract_version': AGENT_API_CONTRACT_VERSION,
            'session_id': updated.session_id,
            'state': updated.state,
            'selected_run_id': updated.selected_run_id,
            'experiments_locked': True,
            'final_result_url': f'/#/results?run_id={selected_run_id}',
            'finalized_at': updated.finalized_at,
        }

    def _run(self, run_id: str, principal: Principal) -> RunRecord:
        try:
            return self.runs.get_scoped(run_id, principal=principal)
        except RunNotFound:
            raise AgentDomainError(
                'agent_experiment_not_found', 'agent experiment 不存在', status_code=404
            ) from None
