"""Finite HTTP client with strict validation and secret-free public errors."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from .contracts import ErrorResponse, RESPONSE_MODELS
from .http_transport import ResponseTooLarge, bounded_request

CONTRACT_VERSION = 'agent-session-v1'
OBSERVATION_VERSION = 'agent-observation-v1'
_IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
_KNOWN_ERRORS = frozenset({
    'model_retired','agent_recipe_invalid','agent_preparation_failed','evaluation_plan_invalid','preparation_resource_exhausted',
    'agent_session_not_found', 'agent_experiment_not_found', 'dataset_unavailable',
    'agent_session_finalized', 'agent_duplicate_config', 'agent_active_run_exists',
    'agent_run_budget_exhausted', 'agent_idempotency_conflict', 'agent_request_released',
    'agent_submission_key_conflict', 'agent_submission_mapping_invalid',
    'agent_module_unavailable', 'agent_invalid_action', 'worker_contract_mismatch',
    'agent_reconciliation_unavailable', 'agent_compensation_required',
    'agent_submission_scope_mismatch', 'agent_submission_failed', 'agent_binding_failed',
    'agent_experiment_binding_failed', 'agent_manifest_incomplete', 'agent_manifest_missing',
    'agent_manifest_invalid', 'agent_validation_unavailable', 'agent_run_failed',
    'agent_run_cancelled', 'agent_dataset_changed', 'agent_dataset_fingerprint_mismatch',
    'agent_metadata_invalid', 'agent_model_unavailable', 'agent_capability_changed', 'agent_version_incompatible',
})


class AgentClientError(RuntimeError):
    pass


class AgentConnectionError(AgentClientError):
    pass


class AgentTimeoutError(AgentClientError):
    pass


class AgentContractError(AgentClientError):
    pass


class AgentHTTPError(AgentClientError):
    def __init__(self, message: str, *, status_code: int, code: str | None,
                 retryable: bool, allowed_actions: list[str]) -> None:
        super().__init__(message)
        self.status_code, self.code = status_code, code
        self.retryable, self.allowed_actions = retryable, list(allowed_actions)


def validate_base_url(value: str) -> str:
    """Reject credentials and ambiguous routing without echoing rejected input."""
    try:
        if not isinstance(value, str) or value != value.strip() or any(ord(c) < 33 for c in value):
            raise ValueError
        parts = urlsplit(value)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or
                parts.username is not None or parts.password is not None or
                parts.query or parts.fragment or '?' in value or '#' in value or
                '\\' in value or '%' in parts.netloc or '%' in parts.path or
                any(piece in ('.', '..') for piece in parts.path.split('/'))):
            raise ValueError
        _ = parts.port
        if parts.path and not re.fullmatch(r'/[A-Za-z0-9/_-]*', parts.path):
            raise ValueError
    except Exception:
        raise ValueError('base_url 必须是无凭据、查询参数或片段的 HTTP(S) 地址') from None
    return value.rstrip('/')


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise AgentContractError('标识必须是合法的单一路径段') from None
    return value


def safe_error_code(value: str | None) -> str | None:
    return value if value in _KNOWN_ERRORS else ('agent_remote_error' if value else None)


def _sanitize_response(payload: dict[str, Any]) -> dict[str, Any]:
    # Free backend prose is never authoritative evidence and is not persisted.
    for module in payload.get('modules', {}).values():
        module['reason'] = '模块尚未实现'
    error = payload.get('error')
    if error is not None:
        error['code'], error['message'] = safe_error_code(error['code']), '后端报告执行或结果完整性问题'
    if 'progress' in payload:
        payload['progress'] = {key: value for key, value in payload['progress'].items()
                               if key in ('epoch', 'epochs', 'percent') and type(value) in (int, float)}
    for entry in payload.get('experiments', []):
        entry['failure_code'] = safe_error_code(entry['failure_code'])
    return payload


class Transport(Protocol):
    def request(self, method: str, url: str, *, headers: dict[str, str],
                json: dict[str, Any] | None, timeout: float) -> Any: ...


class _HttpxTransport:
    def request(self, method: str, url: str, *, headers: dict[str, str],
                json: dict[str, Any] | None, timeout: float) -> Any:
        return bounded_request(method, url, headers=headers, json=json, timeout=timeout,
                               max_response_bytes=1048576)


@dataclass(repr=False)
class AutoAIClient:
    base_url: str
    token: str | None = None
    timeout: float = 10.0
    max_retries: int = 2
    transport: Transport | None = None
    api_version: str = "v1"
    execution_profile: str | None = None
    protocol_revision: str = 'agent-recipes-revision-v1'
    decision_mode: str | None = None

    def __post_init__(self) -> None:
        if self.decision_mode not in (None, 'recipe_id', 'structured_config'):
            raise ValueError('Unknown decision mode')
        if self.decision_mode is not None and (not self.execution_profile or self.protocol_revision != 'agent-recipes-revision-v2'):
            raise ValueError('Decision mode requires current recipe protocol')
        if self.api_version not in ('v1','v2'):
            raise ValueError('Unknown API version')
        from copy import deepcopy
        from agent_poc.tools import TOOL_SCHEMAS
        self.tool_schemas = deepcopy(TOOL_SCHEMAS)
        from .execution_contracts import RESPONSE_MODELS as V2_RESPONSES
        self.response_models = V2_RESPONSES if self.api_version == 'v2' else RESPONSE_MODELS
        self.route_prefix = '/api/agent/v2/' if self.api_version == 'v2' else '/api/agent/'
        if self.execution_profile is not None:
            if self.execution_profile != 'train-evidence-recipes-v1' or self.api_version != 'v2':
                raise ValueError('Recipe profile requires v2')
            from .preparation import RECIPE_RESPONSE_MODELS
            if self.protocol_revision not in ('agent-recipes-revision-v1','agent-recipes-revision-v2'):
                raise ValueError('Unknown recipe revision')
            from .knowledge import KNOWLEDGE_RESPONSE_MODELS
            self.response_models = KNOWLEDGE_RESPONSE_MODELS if self.protocol_revision=='agent-recipes-revision-v2' else RECIPE_RESPONSE_MODELS
        self._sessions = {}
        self.base_url = validate_base_url(self.base_url)
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError('max_retries 必须在 0..5')
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout 必须为有限正数')
        if self.transport is None:
            self.transport = _HttpxTransport()

    def __repr__(self) -> str:
        return 'AutoAIClient(base_url=<configured>, token=<redacted>)'

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 *, operation: str, idempotent: bool = False,
                 bound_ids: dict[str, str] | None = None) -> dict[str, Any]:
        path = path.replace('/api/agent/', self.route_prefix, 1)
        headers = {'Accept': 'application/json'}
        if self.execution_profile:
            headers['X-AutoAI-Agent-Revision'] = self.protocol_revision
        if body is not None:
            headers['Content-Type'] = 'application/json'
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        attempts = 1 + (self.max_retries if method == 'GET' or idempotent else 0)
        for index in range(attempts):
            try:
                assert self.transport is not None
                response = self.transport.request(
                    method, self.base_url + path, headers=headers, json=body, timeout=self.timeout)
                break
            except ResponseTooLarge:
                raise AgentContractError('服务端响应超过大小上限') from None
            except Exception as exc:
                if index + 1 == attempts:
                    if isinstance(exc, TimeoutError) or 'timeout' in type(exc).__name__.lower():
                        raise AgentTimeoutError('SpecAutoAI 请求超时') from None
                    raise AgentConnectionError('无法连接 SpecAutoAI 服务') from None
        try:
            payload, status = response.json(), response.status_code
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError
        except Exception:
            raise AgentContractError('服务端返回了无效 HTTP/JSON 响应') from None
        if not 200 <= status < 300:
            try:
                detail = ErrorResponse.model_validate(payload).detail
            except Exception:
                raise AgentContractError('服务端错误响应结构不符合契约') from None
            raise AgentHTTPError(f'SpecAutoAI 请求失败（HTTP {status}）', status_code=status,
                                 code=safe_error_code(detail.code), retryable=detail.retryable,
                                 allowed_actions=list(detail.allowed_actions)) from None
        try:
            result = self.response_models[operation].model_validate(payload).model_dump(
                mode='json', exclude_unset=True)
            for key, value in (bound_ids or {}).items():
                if result.get(key) != value:
                    raise ValueError
        except Exception:
            raise AgentContractError('Agent API/Observation 版本或响应结构不符合契约') from None
        if self.api_version == 'v2':
            if operation == 'inspect_ml_capabilities':
                from agent_poc.tools import build_tool_schemas
                model_capabilities = {k:v for k,v in result.items() if k not in ('protocol_revision','execution_profiles')}
                model_capabilities['modules'] = {}
                self.tool_schemas = build_tool_schemas(model_capabilities)
                if self.execution_profile:
                    self._recipe_tools()
            if 'locked_config' in result:
                self._sessions[result['session_id']] = result['locked_config']
            if 'effective_action' in result:
                self._check_frozen_response(result, request_body=body)
        return _sanitize_response(result)

    def restore_frozen_session(self, session_id, snapshot, preparation=None):
        """Restore validated local bindings without performing any HTTP request."""
        from .capabilities import FrozenSnapshot
        if self.api_version != 'v2':
            raise AgentContractError('Frozen capabilities require v2')
        validate_identifier(session_id)
        frozen = FrozenSnapshot.model_validate(snapshot).model_dump(mode='json')
        self._sessions[session_id] = {'capability_snapshot': frozen}
        if self.execution_profile:
            from .preparation import Preparation
            from .knowledge import KnowledgePreparation
            schema = KnowledgePreparation if self.protocol_revision=='agent-recipes-revision-v2' else Preparation
            bound = schema.model_validate(preparation).model_dump(mode='json',exclude_unset=True)
            self._sessions[session_id]['preparation'] = bound
            self._recipe_tools()

    def restore_recipe_revision(self, revision):
        """Rebuild the negotiated client contract from a validated checkpoint."""
        from .preparation import RECIPE_RESPONSE_MODELS
        from .knowledge import KNOWLEDGE_RESPONSE_MODELS
        if self.api_version != 'v2' or revision not in ('agent-recipes-revision-v1','agent-recipes-revision-v2'):
            raise AgentContractError('Invalid checkpoint recipe revision')
        self.execution_profile='train-evidence-recipes-v1'
        self.protocol_revision=revision
        self.response_models=KNOWLEDGE_RESPONSE_MODELS if revision=='agent-recipes-revision-v2' else RECIPE_RESPONSE_MODELS

    def _recipe_tools(self):
        self.tool_schemas['submit_ml_experiment'] = dict(type='object',additionalProperties=False,
            properties={'session_id':{'type':'string'},'recipe_id':{'type':'string','pattern':'^recipe_[a-f0-9]{64}$'},
                        'rationale':{'type':'string','maxLength':2000},'client_request_id':{'type':'string'}},
            required=['session_id','recipe_id','rationale','client_request_id'])
        if self.protocol_revision=='agent-recipes-revision-v2':
            schema=self.tool_schemas['submit_ml_experiment']
            schema['properties']['knowledge_refs']=dict(type='array',items={'type':'string'},maxItems=6,uniqueItems=True)
            schema['required'].append('knowledge_refs')
            self.tool_schemas['start_ml_session']['properties']['modules']['items']['enum']=['train_evidence','legal_recipes','knowledge']
        for name in ('start_ml_session', 'submit_ml_experiment'):
            self.tool_schemas[name]['properties']['decision_mode'] = {'type':'string', 'enum':['recipe_id','structured_config']}
        context=self.tool_schemas['start_ml_session']['properties']['context_policy']['properties']
        context.update(evidence={'type':'boolean'},risks={'type':'boolean'})

    def _check_frozen_response(self, result, *, request_body=None):
        locked = self._sessions.get(result['session_id'])
        if locked is None:
            return
        action = result['effective_action']
        frozen = locked['capability_snapshot']['model_configs']
        if action['model_type'] not in frozen or action['model_params'] != frozen[action['model_type']]:
            raise AgentContractError('Response differs from frozen Session')
        config = result.get('effective_config')
        if config is not None and (config['model_type'] != action['model_type'] or config['model_params'] != action['model_params']):
            raise AgentContractError('Execution differs from frozen decision')
        if self.execution_profile and self.protocol_revision=='agent-recipes-revision-v2' and request_body is not None and 'knowledge_refs' in request_body:
            projection=locked['preparation']['knowledge']
            entries={e['entry_id']:e['entry_version'] for e in projection['projection']['entries']}
            metadata=result['decision_metadata']
            expected=[dict(entry_id=key,entry_version=entries[key]) for key in sorted(request_body['knowledge_refs'])]
            if metadata['projection_digest']!=projection['projection_digest'] or metadata['references']!=expected:
                raise AgentContractError('Response knowledge differs from submitted decision')

    def inspect_ml_capabilities(self) -> dict[str, Any]:
        return self._request('GET', '/api/agent/health', operation='inspect_ml_capabilities')

    def start_ml_session(self, *, dataset_id: str, selection_metric: str,
                         allowed_models: list[str], max_runs: int, seed: int = 42,
                         evaluation: dict[str, Any] | None = None,
                         modules: list[str] | None = None,
                         context_policy: dict[str, Any] | None = None,
                         client_request_id: str | None = None,
                         model_configs: dict[str, dict[str, Any]] | None = None, decision_mode: str | None = None) -> dict[str, Any]:
        validate_identifier(dataset_id)
        if client_request_id is not None:
            validate_identifier(client_request_id)
        body = {
            'dataset_id': dataset_id, 'selection_metric': selection_metric,
            'allowed_models': allowed_models, 'max_runs': max_runs, 'seed': seed,
            'evaluation': evaluation if evaluation is not None else {
                'split_mode': 'stratified_holdout', 'split_train': 8, 'split_valid': 1, 'split_test': 1},
            'modules': modules if modules is not None else [],
            'context_policy': context_policy if context_policy is not None else {
                'source_role': 'development', 'case_write': False},
        }
        if decision_mode != self.decision_mode:
            raise AgentContractError('Session expression differs from client binding')
        if decision_mode is not None:
            body['decision_mode'] = decision_mode
        if self.execution_profile:
            body.update(execution_profile=self.execution_profile,protocol_revision=self.protocol_revision,
                        modules=(modules if self.protocol_revision=='agent-recipes-revision-v2' and modules is not None else ['train_evidence','legal_recipes']))
            body['context_policy'].setdefault('evidence', True)
            body['context_policy'].setdefault('risks', True)
        if client_request_id is not None:
            body['client_request_id'] = client_request_id
        if model_configs is not None:
            if self.api_version != 'v2':
                raise AgentContractError('Model configurations require v2')
            body['model_configs'] = model_configs
        return self._request('POST', '/api/agent/sessions', body,
                             operation='start_ml_session', idempotent=bool(client_request_id))

    def inspect_ml_session(self, session_id: str) -> dict[str, Any]:
        validate_identifier(session_id)
        return self._request('GET', f'/api/agent/sessions/{session_id}',
                             operation='inspect_ml_session', bound_ids={'session_id': session_id})

    def submit_ml_experiment(self, session_id: str, *, model_type: str | None = None,
                             normalization: str = 'zscore', class_balance: str = 'none',
                             parent_run_id: str | None = None, rationale: str | None = None,
                             client_request_id: str | None = None,
                             model_params: dict[str, Any] | None = None,
                             recipe_id: str | None = None, knowledge_refs: list[str] | None = None, decision_mode: str | None = None) -> dict[str, Any]:
        validate_identifier(session_id)
        for value in (parent_run_id, client_request_id):
            if value is not None:
                validate_identifier(value)
        if decision_mode != self.decision_mode:
            raise AgentContractError('Prepared expression differs from client binding')
        if self.execution_profile:
            if model_type is not None or model_params is not None or parent_run_id is not None or normalization!='zscore' or class_balance!='none':
                raise AgentContractError('Recipe selection cannot contain free execution configuration')
            locked=self._sessions.get(session_id)
            catalog=locked.get('preparation',{}).get('catalog') if locked else None
            recipe=next((r for r in catalog['recipes'] if r['recipe_id']==recipe_id),None) if catalog else None
            if recipe is None:
                raise AgentContractError('Restore frozen recipe catalog before selection')
            body=dict(recipe_id=recipe_id,recipe_digest=recipe['recipe_digest'],
                catalog_digest=catalog['catalog_digest'],rationale=rationale,client_request_id=client_request_id)
            if self.decision_mode == 'structured_config':
                body = dict(model_id=recipe['model_id'], normalization='zscore', class_balance='none',
                    model_params=dict(locked['capability_snapshot']['model_configs'][recipe['model_id']]),
                    rationale=rationale, client_request_id=client_request_id)
            if self.protocol_revision=='agent-recipes-revision-v2':
                from .knowledge import KnowledgeWire
                if knowledge_refs is None:
                    raise AgentContractError('knowledge_refs is required')
                knowledge=KnowledgeWire.model_validate(self._sessions[session_id]['preparation']['knowledge'])
                knowledge.validate_refs(knowledge_refs)
                body['knowledge_refs']=knowledge_refs
            return self._request('POST',f'/api/agent/sessions/{session_id}/experiments',body,
                operation='submit_ml_experiment',idempotent=bool(client_request_id),bound_ids={'session_id':session_id})
        if recipe_id is not None:
            raise AgentContractError('Recipe requires negotiated profile')
        body: dict[str, Any] = {'model_type': model_type, 'normalization': normalization,
                                'class_balance': class_balance}
        for key, value in (('parent_run_id', parent_run_id), ('rationale', rationale),
                           ('client_request_id', client_request_id)):
            if value is not None:
                body[key] = value
        if self.api_version == 'v2':
            if session_id not in self._sessions:
                raise AgentContractError('Inspect or restore the frozen Session before submission')
            from .capabilities import FrozenSnapshot, validate_params
            frozen = FrozenSnapshot.model_validate(self._sessions[session_id]['capability_snapshot'])
            model = next((m for m in frozen.models if m.id == model_type), None)
            try:
                if model_type not in frozen.model_configs or model is None:
                    raise ValueError
                expected = frozen.model_configs[model_type]
                if model_params is not None and validate_params(model, model_params, complete=True) != expected:
                    raise ValueError
                if normalization != 'zscore' or class_balance != 'none' or parent_run_id is not None:
                    raise ValueError
            except (ValueError, TypeError):
                raise AgentContractError('Experiment differs from frozen Session') from None
            body['model_params'] = dict(expected)
        elif model_params is not None:
            raise AgentContractError('Model parameters require v2')
        return self._request('POST', f'/api/agent/sessions/{session_id}/experiments', body,
                             operation='submit_ml_experiment', idempotent=bool(client_request_id),
                             bound_ids={'session_id': session_id})

    def observe_ml_experiment(self, session_id: str, run_id: str) -> dict[str, Any]:
        validate_identifier(session_id)
        validate_identifier(run_id)
        return self._request('GET', f'/api/agent/sessions/{session_id}/experiments/{run_id}/feedback',
                             operation='observe_ml_experiment',
                             bound_ids={'session_id': session_id, 'run_id': run_id})

    def finalize_ml_session(self, session_id: str, selected_run_id: str) -> dict[str, Any]:
        validate_identifier(session_id)
        validate_identifier(selected_run_id)
        return self._request('POST', f'/api/agent/sessions/{session_id}/finalize',
                             {'selected_run_id': selected_run_id}, operation='finalize_ml_session',
                             idempotent=True, bound_ids={'session_id': session_id,
                                                        'selected_run_id': selected_run_id})

    def reconcile_ml_session(self, session_id: str) -> dict[str, Any]:
        """Recovery-only, fixed empty-body operation; absent from six tool schemas."""
        validate_identifier(session_id)
        return self._request('POST', f'/api/agent/sessions/{session_id}/reconcile', {},
                             operation='reconcile_ml_session', bound_ids={'session_id': session_id})
