"""Agent API v1 服务：冻结 Session、预约实验、提交 Run 与安全 Observation。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..datasets.repository import DatasetIntegrityError, DatasetRepository
from ..runs.contracts import Principal, RunRecord
from ..runs.repository import InvalidRunTransition, RunNotFound, RunRepository
from ..runs.submission import RunSubmissionError, RunSubmissionRequest, RunSubmissionService
from .capabilities import module_catalog
from .contracts_v2 import V2
from . import policy as policy_v2
from .policy import base_training_config as _training_config
from .contracts import (
    AGENT_API_CONTRACT_VERSION,
    AGENT_METADATA_VERSION,
    AGENT_RESERVATION_PROTOCOL_VERSION,
    AgentDomainError,
    CreateAgentExperimentRequest,
    CreateAgentSessionRequest,
)
from .observation import build_observation, validation_from_manifest
from .metadata import dataset_metadata, run_metadata
from .repository import (
    AgentExperimentRecord,
    AgentExperimentNotFound,
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




def _locked_config(session: AgentSessionRecord) -> dict[str, Any]:
    evaluation = session.evaluation_config
    if session.metadata_version not in (None, AGENT_METADATA_VERSION, 'agent-metadata-v2'):
        raise AgentDomainError('agent_metadata_invalid', 'Session 元数据版本不兼容', status_code=409)
    return {
        **_preparation_wire(session),
        **({'capability_snapshot': session.capability_snapshot} if session.contract_version == V2 else {}),
        'dataset_id': session.dataset_id,
        **dataset_metadata(session.dataset_sha256, version=session.contract_version or AGENT_API_CONTRACT_VERSION),
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
                 submission_service: RunSubmissionService, run_root: Path,
                 contract_version: str = AGENT_API_CONTRACT_VERSION) -> None:
        self.contract_version = contract_version
        self.sessions = session_repository
        self.runs = run_repository
        self.datasets = dataset_repository
        self.submissions = submission_service
        self.run_root = Path(run_root)

    def create_session(self, payload: CreateAgentSessionRequest, *, principal: Principal) -> dict[str, Any]:
        payload.validate_business()
        catalog = module_catalog(getattr(payload, 'protocol_revision', None))
        requested_profile = getattr(payload, 'execution_profile', None)
        prepared_modules = {'train_evidence', 'legal_recipes'} if requested_profile else set()
        unavailable = sorted({name for name in payload.modules if name not in prepared_modules and not catalog.get(name, {}).get('available')})
        body = payload.model_dump(mode='json')
        if not requested_profile:
            body.pop('execution_profile', None); body.pop('protocol_revision', None)
        snapshot = None
        if self.contract_version == V2:
            previous = self.sessions.find_session_request_scoped(payload.client_request_id, payload_hash=None, principal=principal) if payload.client_request_id else None
            if previous is not None:
                policy_v2.require_version(previous, self.contract_version)
            snapshot = policy_v2.freeze_session(payload, previous.capability_snapshot if previous else None)
            body.update(contract_version=V2, model_configs=snapshot['model_configs'])
        body_hash = _hash_payload(body)
        if payload.client_request_id:
            existing = self.sessions.find_session_request_scoped(
                payload.client_request_id, payload_hash=body_hash, principal=principal,
            )
            if existing is not None:
                return self._session_creation_response(existing, created=False, principal=principal)
        if unavailable:
            raise AgentDomainError('agent_module_unavailable', f'请求的模块当前不可用：{", ".join(unavailable)}', status_code=422)
        if payload.context_policy.case_write:
            raise AgentDomainError('agent_module_unavailable', 'case_memory 不可用，不能启用 case_write', status_code=422)
        try:
            dataset = self.datasets.resolve(payload.dataset_id, principal=principal)
            digest = self.datasets.verify_integrity(dataset)
        except (FileNotFoundError, PermissionError, DatasetIntegrityError):
            raise AgentDomainError('dataset_unavailable', '数据集不存在或当前调用者无权访问', status_code=404) from None
        frozen_preparation = None
        if requested_profile:
            from ..evaluation_plan import PreparationLimits, PreparationResourceExhausted, select_train
            from ..train_evidence import compute_train_evidence
            from ..recipes import compile_recipe_catalog
            limits = PreparationLimits.configured()
            try:
                view, plan = self.submissions.prepare_evaluation(dataset_id=payload.dataset_id,
                    raw_config={**payload.evaluation.model_dump(), 'seed':payload.seed},
                    principal=principal, expected_sha256=digest, limits=limits)
                evidence = compute_train_evidence(select_train(view, plan), limits)
                recipes = compile_recipe_catalog(body, snapshot, evidence, plan)
                limits.check()
            except PreparationResourceExhausted as exc:
                raise AgentDomainError('preparation_resource_exhausted','Preparation resource limit exceeded',status_code=422) from exc
            except RunSubmissionError as exc:
                raise AgentDomainError(exc.code,exc.message,status_code=exc.status_code) from exc
            except ValueError as exc:
                raise AgentDomainError('agent_preparation_failed',
                    'Preparation failed: invalid data, constraints or resource limit',status_code=422) from exc
            frozen_preparation = dict(execution_profile=requested_profile,
                protocol_revision=payload.protocol_revision,
                preparation=dict(evaluation_plan=plan.safe_reference(),
                    evidence=evidence.model_dump(mode='json'),catalog=recipes.model_dump(mode='json')))
        if requested_profile and payload.protocol_revision == 'agent-recipes-revision-v2':
            from ..knowledge import freeze_knowledge, KnowledgeError
            try:
                knowledge = freeze_knowledge(enabled='knowledge' in payload.modules, evidence=evidence, catalog=recipes,
                    evidence_context=payload.context_policy.evidence, risk_context=payload.context_policy.risks)
            except KnowledgeError as exc:
                raise AgentDomainError('agent_knowledge_unavailable', 'Knowledge publication unavailable', status_code=503) from exc
            frozen_preparation['preparation']['knowledge'] = knowledge.model_dump(mode='json', exclude_unset=True)
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
            payload_hash=body_hash,
            principal=principal,
            dataset_sha256=digest,
            metadata_version='agent-metadata-v2' if self.contract_version == V2 else AGENT_METADATA_VERSION,
            contract_version=V2 if self.contract_version == V2 else None, capability_snapshot=snapshot,
            frozen_preparation=frozen_preparation,
        )
        return self._session_creation_response(session, created=created, principal=principal)

    def _session_creation_response(self, session: AgentSessionRecord, *, created: bool,
                                   principal: Principal) -> dict[str, Any]:
        return {
            'contract_version': session.contract_version or AGENT_API_CONTRACT_VERSION,
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
        session = self._session(session_id, principal)
        if self.contract_version != V2 and session.state != 'open':
            raise AgentSessionClosed()
        command = policy_v2.normalize_experiment(session, payload)
        action, body = command.action, command.request_body
        prepared = None
        full_digest = None
        replay_reservation = None
        if self.contract_version == V2:
            if payload.client_request_id:
                replay_reservation = self.sessions.find_experiment_request_scoped(
                    session_id, payload.client_request_id, payload_hash=_hash_payload(body), principal=principal)
            if replay_reservation is not None and replay_reservation.state == 'bound' and replay_reservation.run_id:
                return self._experiment_response(session, replay_reservation, self._run(replay_reservation.run_id, principal), replay=True)
            # A durable mapping wins over capability drift after a lost bind.
            mapping_exists = False
            if replay_reservation is not None and replay_reservation.state in ('reserved', 'compensation_required'):
                mapping = self.runs.lookup_submission_mapping(replay_reservation.experiment_id, principal=principal)
                mapping_exists = mapping.status != 'missing'
                if mapping.status == 'found' and mapping.submission_source == 'agent' and mapping.run_id and replay_reservation.state == 'reserved':
                    record = self._run(mapping.run_id, principal)
                    expected = replay_reservation.compiled_config
                    if expected is None or any(record.config.get(k) != v for k,v in expected.items()):
                        raise AgentDomainError('agent_submission_mapping_invalid', 'Mapped submission configuration differs', status_code=409)
                    try:
                        bound = self.sessions.bind_experiment(replay_reservation.experiment_id, run_id=record.run_id, principal=principal)
                    except AgentExperimentNotFound:
                        current = self.sessions.get_reservation_scoped(replay_reservation.experiment_id, principal=principal)
                        if current.state != 'bound' or current.run_id != record.run_id:
                            raise
                        bound = current
                    return self._experiment_response(session, bound, record, replay=True)
            if not mapping_exists:
                policy_v2.admit_command(session, command)
            prepared = (replay_reservation.compiled_config if replay_reservation else None)
            if prepared is None:
                prepared = policy_v2.compiled_config(session, action, _training_config(session, action))
                _, plan = self.submissions.prepare_evaluation(dataset_id=session.dataset_id,
                    raw_config=prepared,principal=principal,expected_sha256=session.dataset_sha256)
                prepared['evaluation_plan_digest'] = plan.plan_digest
            if command.recipe is not None:
                prepared.update(evaluation_plan_digest=session.frozen_preparation['preparation']['evaluation_plan']['plan_digest'],
                    execution_recipe_digest=command.recipe['recipe_digest'],
                    execution_catalog_digest=session.frozen_preparation['preparation']['catalog']['catalog_digest'],
                    execution_evidence_digest=session.frozen_preparation['preparation']['evidence']['evidence_digest'],
                    execution_search_digest=command.recipe['search_strategy_digest'])
            full_digest = policy_v2.command_digest(session, command)
        if session.state != 'open':
            raise AgentSessionClosed()
        config_hash = full_digest[:16] if full_digest else _compute_config_hash(session=session, action=action)
        existing = self.sessions.list_experiments_scoped(session_id, principal=principal)
        bound_ids = [item.run_id for item in existing if item.run_id]
        states = collect_run_states(self.runs, bound_ids)
        active = {run_id for run_id, state in states.items() if state in {'queued', 'running'}}
        reservation, created = self.sessions.reserve_experiment(
            session_id=session_id,
            action_json=action,
            rationale=command.rationale,
            parent_run_id=action['parent_run_id'],
            config_hash=config_hash,
            client_request_id=payload.client_request_id,
            payload_hash=_hash_payload(body),
            active_run_ids=active,
            principal=principal,
            protocol_version=AGENT_RESERVATION_PROTOCOL_VERSION,
            compiled_config=prepared, scientific_digest=full_digest, decision_metadata=command.decision_metadata,
        )
        if not created and reservation.state == 'bound' and reservation.run_id:
            record = self._run(reservation.run_id, principal)
            return self._experiment_response(session, reservation, record, replay=True)
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

        from ..evaluation_plan import PreparedEvaluation
        compiled = reservation.compiled_config
        prepared_evaluation = PreparedEvaluation(compiled['evaluation_plan_digest'],session.dataset_sha256) if compiled and compiled.get('evaluation_plan_digest') else None
        try:
            submitted = self.submissions.submit(RunSubmissionRequest(
                dataset_id=session.dataset_id,
                legacy_data_path=None,
                test_dataset_id=None,
                test_legacy_data_path=None,
                raw_config=reservation.compiled_config if self.contract_version == V2 else _training_config(session, action),
                principal=principal,
                submission_source='agent',
                submission_key=reservation.experiment_id,
                expected_dataset_sha256=session.dataset_sha256,
                prepared_evaluation=prepared_evaluation,
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
                        session, current, record, replay=(not created)
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
        return self._experiment_response(session, bound, record, replay=(not created))

    @staticmethod
    def _verify_run_metadata(session: AgentSessionRecord, record: RunRecord) -> None:
        if session.dataset_sha256 is not None and (
            record.dataset_id != session.dataset_id
            or record.dataset_snapshot.get('dataset_id') != session.dataset_id
            or record.dataset_snapshot.get('sha256') != session.dataset_sha256
        ):
            raise AgentDomainError(
                'agent_dataset_fingerprint_mismatch',
                '训练快照与冻结 Session 不一致，需要人工核对', status_code=409,
            )
        if session.frozen_preparation is not None:
            prepared=session.frozen_preparation['preparation']
            catalog=prepared['catalog']
            recipe=next((r for r in catalog['recipes'] if r['recipe_digest']==record.config.get('execution_recipe_digest')),None)
            if (recipe is None or any(record.config.get(k)!=v for k,v in recipe['fixed_execution_config'].items())
                    or record.config.get('evaluation_plan_digest')!=prepared['evaluation_plan']['plan_digest']
                    or record.config.get('execution_catalog_digest')!=catalog['catalog_digest']
                    or record.config.get('execution_evidence_digest')!=prepared['evidence']['evidence_digest']
                    or record.config.get('execution_search_digest')!=recipe['search_strategy_digest']):
                raise AgentDomainError('agent_metadata_invalid','Run differs from frozen execution recipe',status_code=409)

    @classmethod
    def _experiment_response(cls, session: AgentSessionRecord, experiment: AgentExperimentRecord,
                             record: RunRecord, *, replay: bool) -> dict[str, Any]:
        cls._verify_run_metadata(session, record)
        metadata = _decision_wire(session, experiment)
        return {
            **metadata,
            'contract_version': session.contract_version or AGENT_API_CONTRACT_VERSION,
            'session_id': session.session_id,
            **run_metadata(record, version=session.contract_version or AGENT_API_CONTRACT_VERSION, snapshot=session.capability_snapshot),
            'run_id': record.run_id,
            'attempt': experiment.attempt,
            'config_hash': experiment.config_hash,
            'state': record.state,
            'binding_state': experiment.state,
            'effective_action': experiment.action_json,
            'idempotent_replay': replay,
        }

    def get_feedback(self, *, session_id: str, run_id: str, principal: Principal) -> dict[str, Any]:
        session = self._session(session_id, principal)
        experiment = self.sessions.get_experiment_scoped(session_id, run_id, principal=principal)
        record = self._run(run_id, principal)
        self._verify_run_metadata(session, record)
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
            contract_version=session.contract_version or AGENT_API_CONTRACT_VERSION,
            capability_snapshot=session.capability_snapshot,
        )

    def get_session(self, *, session_id: str, principal: Principal) -> dict[str, Any]:
        session = self._session(session_id, principal)
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
                **run_metadata(None, pending=item.state == 'reserved', version=session.contract_version or AGENT_API_CONTRACT_VERSION, snapshot=session.capability_snapshot),
            }
            if item.run_id:
                try:
                    record = self._run(item.run_id, principal)
                    self._verify_run_metadata(session, record)
                    entry.update(run_metadata(record, version=session.contract_version or AGENT_API_CONTRACT_VERSION, snapshot=session.capability_snapshot))
                    entry['state'] = record.state
                    if record.state == 'succeeded':
                        status, metrics, _ = validation_from_manifest(
                            self.run_root / item.run_id, run_id=item.run_id
                        )
                        value = metrics.get(session.selection_metric) if status == 'ready' else None
                        if value is not None:
                            entry['validation_score'] = value
                            scores.append((value, -item.attempt, item.run_id))
                except AgentDomainError as exc:
                    if exc.code != 'agent_experiment_not_found':
                        raise
            summaries.append(entry)
        best_run_id = max(scores)[2] if scores else None
        used = sum(1 for item in experiments if item.state != 'released')
        return {
            'contract_version': session.contract_version or AGENT_API_CONTRACT_VERSION,
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
        session = self._session(session_id, principal)
        experiment = self.sessions.get_experiment_scoped(
            session_id, selected_run_id, principal=principal
        )
        record = self._run(selected_run_id, principal)
        self._verify_run_metadata(session, record)
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
            'contract_version': session.contract_version or AGENT_API_CONTRACT_VERSION,
            'session_id': updated.session_id,
            'state': updated.state,
            'selected_run_id': updated.selected_run_id,
            'experiments_locked': True,
            'final_result_url': f'/#/results?run_id={selected_run_id}',
            'finalized_at': updated.finalized_at,
        }

    def _session(self, session_id, principal):
        session = self.sessions.get_session_scoped(session_id, principal=principal)
        policy_v2.require_version(session, self.contract_version)
        return session

    def _run(self, run_id: str, principal: Principal) -> RunRecord:
        try:
            return self.runs.get_scoped(run_id, principal=principal)
        except RunNotFound:
            raise AgentDomainError(
                'agent_experiment_not_found', 'agent experiment 不存在', status_code=404
            ) from None


def _preparation_wire(session):
    frozen = session.frozen_preparation
    if frozen is None:
        return {}
    prepared = frozen['preparation']
    wire = {key: prepared[key] for key in ('evaluation_plan','evidence','catalog')}
    if frozen['protocol_revision'] == 'agent-recipes-revision-v2':
        wire['knowledge'] = prepared['knowledge'].wire()
    return dict(execution_profile=frozen['execution_profile'], protocol_revision=frozen['protocol_revision'], preparation=wire)


def _decision_wire(session, experiment):
    frozen = session.frozen_preparation
    if frozen is None or frozen['protocol_revision'] != 'agent-recipes-revision-v2':
        return {}
    return dict(decision_metadata=experiment.decision_metadata.model_dump(mode='json'))
