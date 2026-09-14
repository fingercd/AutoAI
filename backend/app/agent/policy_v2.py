"""Frozen second-step semantics shared by Session creation and submission."""
from typing import Any
from ..contracts import TrainingSpec
from ..model_config import model_capability_snapshot, model_policy, resolve_model_params, semantic_digest, compatible_frozen_policy
from ..model_catalog import model_availability, RETIRED_MODEL_ALIASES
from .contracts import AgentDomainError
from .contracts_v2 import V2


def require_version(session, version):
    if (session.contract_version or 'agent-session-v1') != version:
        raise AgentDomainError('agent_version_incompatible', 'Session protocol is incompatible', status_code=409)


def freeze_session(payload, previous=None):
    snapshot = previous if previous is not None else model_capability_snapshot()
    policies = {m['id']: m for m in snapshot['models']}
    configs = {}
    for name in payload.allowed_models:
        if previous is None and name in RETIRED_MODEL_ALIASES:
            raise AgentDomainError('model_retired', 'DSCARNet has been retired', status_code=409)
        if name not in policies:
            raise AgentDomainError('agent_capability_changed', 'Frozen model policy unavailable', status_code=409)
        if previous is None and not policies[name]['available']:
            raise AgentDomainError('agent_model_unavailable', 'Requested model is not executable', status_code=422)
        configs[name] = resolve_model_params(name, payload.model_configs.get(name), policy=policies[name])
    return {**snapshot, 'model_configs': configs}


def frozen_action(session, payload):
    snapshot = session.capability_snapshot
    if snapshot is None or payload.model_type not in snapshot['model_configs']:
        raise AgentDomainError('agent_invalid_action', 'Model is outside frozen Session', status_code=422)
    frozen = snapshot['model_configs'][payload.model_type]
    if 'model_params' in payload.model_fields_set:
        # Explicit objects must be complete; normalization permits int vs float only for number fields.
        policy = next(m for m in snapshot['models'] if m['id'] == payload.model_type)
        if set(payload.model_params) != set(frozen) or resolve_model_params(payload.model_type, payload.model_params, policy=policy) != frozen:
            raise AgentDomainError('agent_invalid_action', 'Model parameters differ from frozen configuration', status_code=422)
    return dict(model_type=payload.model_type, normalization='zscore', class_balance='none',
                parent_run_id=None, model_params=dict(frozen))


def admit_action(session, action):
    model = action['model_type']
    if model in RETIRED_MODEL_ALIASES:
        raise AgentDomainError('model_retired', 'DSCARNet has been retired', status_code=409)
    policy = next(m for m in session.capability_snapshot['models'] if m['id'] == model)
    if not compatible_frozen_policy(model, policy, action['model_params'], current=model_policy(model)):
        raise AgentDomainError('agent_capability_changed', 'Model execution policy changed', status_code=409)
    if not model_availability(model)[0]:
        raise AgentDomainError('agent_model_unavailable', 'Model is no longer executable', status_code=409)


def compiled_config(session, action, base):
    policy = next(m for m in session.capability_snapshot['models'] if m['id'] == action['model_type'])
    return TrainingSpec.from_legacy({**base, **action['model_params'],
        'agent_config_policy_version': policy['config_policy_version'],
        'agent_config_policy_digest': policy['config_policy_digest'],
    }).validated(has_external_test=False).values


def scientific_digest(session, action):
    policy = next(m for m in session.capability_snapshot['models'] if m['id'] == action['model_type'])
    return semantic_digest(dict(protocol=V2, dataset_sha256=session.dataset_sha256,
        dataset_id=session.dataset_id, evaluation=session.evaluation_config, seed=session.seed,
        selection_metric=session.selection_metric, action=action, config_policy_digest=policy['config_policy_digest']))
