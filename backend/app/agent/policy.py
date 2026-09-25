"""Frozen second-step semantics shared by Session creation and submission."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..knowledge import KnowledgeDecision
from ..contracts import TrainingSpec
from ..model_config import model_capability_snapshot, model_policy, resolve_model_params, semantic_digest, compatible_frozen_policy
from ..model_catalog import model_availability, RETIRED_MODEL_ALIASES
from .contracts import AgentDomainError
from .contracts import V2


def require_version(session, version):
    if (session.contract_version or 'agent-session-v1') != version:
        raise AgentDomainError('agent_version_incompatible', 'Session protocol is incompatible', status_code=409)


def freeze_session(payload, previous=None):
    search_revision = getattr(payload, 'protocol_revision', None) in ('agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6')
    snapshot = previous if previous is not None else (model_capability_snapshot(search_revision=True)
        if search_revision else model_capability_snapshot())
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
    search_revision = bool(session.frozen_preparation and session.frozen_preparation['protocol_revision'] in ('agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6'))
    current_policy = (model_policy(model, search_revision=True)
                      if search_revision else model_policy(model))
    if not compatible_frozen_policy(model, policy, action['model_params'], current=current_policy):
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


def base_training_config(session, action: dict[str, Any]) -> dict[str, Any]:
    return {
        'model_type': action['model_type'],
        'normalization': action.get('normalization', 'zscore'),
        'class_balance': action.get('class_balance', 'none'),
        'seed': session.seed,
        'feature_selection_enabled': False,
        **session.evaluation_config,
    }



from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCommand:
    """Normalized execution action and its historical wire hash input."""
    action: dict[str, Any]
    request_body: dict[str, Any]
    rationale: str
    client_request_id: str | None
    recipe: dict[str, Any] | None = None
    decision_metadata: KnowledgeDecision | None = None


def normalize_experiment(session, payload) -> ExperimentCommand:
    from .contracts import CreateRecipeExperimentRequest, CreateAgentExperimentRequestV2, CreateStructuredExperimentRequest
    if session.frozen_preparation is not None:
        structured = session.frozen_preparation.get('decision_mode', 'recipe_id') == 'structured_config'
        if structured != isinstance(payload, CreateStructuredExperimentRequest):
            raise AgentDomainError('agent_invalid_action', 'Expression differs from frozen decision mode', status_code=422)
        if not isinstance(payload, (CreateRecipeExperimentRequest, CreateStructuredExperimentRequest)):
            raise AgentDomainError('agent_invalid_action', 'Recipe profile accepts only recipe selection', status_code=422)
        from ..recipes import RecipeCatalog, validate_catalog_binding
        catalog = validate_catalog_binding(RecipeCatalog.model_validate(session.frozen_preparation['preparation']['catalog']),session.capability_snapshot).model_dump(mode='json')
        processing_revision = session.frozen_preparation['protocol_revision'] in ('agent-recipes-revision-v3','agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6')
        if structured:
            if processing_revision:
                snapshot = session.capability_snapshot
                frozen = snapshot['model_configs'].get(payload.model_id)
                policy = next((m for m in snapshot['models'] if m['id'] == payload.model_id), None)
                if frozen is None or policy is None or set(payload.model_params) != set(frozen):
                    raise AgentDomainError('agent_recipe_invalid', 'Configuration is outside frozen catalog', status_code=422)
                try:
                    if resolve_model_params(payload.model_id, payload.model_params, policy=policy) != frozen:
                        raise ValueError('model parameters differ')
                except ValueError as exc:
                    raise AgentDomainError('agent_recipe_invalid', 'Configuration is outside frozen catalog', status_code=422) from exc
                recipe = next((r for r in catalog['recipes'] if
                    r['model_id'] == payload.model_id and
                    r['preprocessing']['normalization'] == payload.normalization and
                    r['class_balance'] == payload.class_balance and
                    {key: r['fixed_execution_config'][key] for key in frozen} == frozen), None)
            else:
                if payload.normalization != 'zscore' or payload.class_balance != 'none':
                    raise AgentDomainError('agent_recipe_invalid', 'Configuration is outside frozen catalog', status_code=422)
                canonical = CreateAgentExperimentRequestV2(model_type=payload.model_id, model_params=payload.model_params)
                frozen_action(session, canonical)  # Complete keys and strict finite numeric validation.
                recipe = next((r for r in catalog['recipes'] if r['model_id'] == payload.model_id), None)
            if recipe is None:
                raise AgentDomainError('agent_recipe_invalid', 'Configuration is outside frozen catalog', status_code=422)
        else:
            recipe = next((r for r in catalog['recipes'] if r['recipe_id']==payload.recipe_id), None)
            if recipe is None or catalog['catalog_digest']!=payload.catalog_digest or recipe['recipe_digest']!=payload.recipe_digest:
                raise AgentDomainError('agent_recipe_invalid', 'Recipe does not match frozen catalog', status_code=422)
        if processing_revision:
            action = dict(model_type=recipe['model_id'],
                normalization=recipe['preprocessing']['normalization'],
                class_balance=recipe['class_balance'],parent_run_id=None,
                model_params=dict(session.capability_snapshot['model_configs'][recipe['model_id']]))
        else:
            canonical = CreateAgentExperimentRequestV2(model_type=recipe['model_id'])
            action = frozen_action(session, canonical)
        body = payload.model_dump(mode='json')
        body.update(contract_version=V2,execution_profile=session.frozen_preparation['execution_profile'])
        from .contracts import CreateKnowledgeExperimentRequest
        metadata = None
        modern = session.frozen_preparation['protocol_revision'] in ('agent-recipes-revision-v2','agent-recipes-revision-v3','agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6')
        if modern != isinstance(payload, (CreateKnowledgeExperimentRequest, CreateStructuredExperimentRequest)):
            raise AgentDomainError('agent_version_incompatible', 'Knowledge request revision mismatch', status_code=409)
        if modern:
            from ..knowledge import resolve_decision
            try:
                metadata = resolve_decision(session.frozen_preparation['preparation']['knowledge'], payload.knowledge_refs)
            except ValueError as exc:
                raise AgentDomainError('agent_knowledge_reference_invalid', 'Reference not provided in frozen context', status_code=422) from exc
        return ExperimentCommand(action, body, payload.rationale, payload.client_request_id, recipe, metadata)
    if isinstance(payload, (CreateRecipeExperimentRequest, CreateStructuredExperimentRequest)):
        raise AgentDomainError('agent_version_incompatible', 'Recipe requires recipe profile', status_code=409)
    payload.validate_business()
    if payload.model_type not in session.allowed_models:
        raise AgentDomainError('agent_invalid_action', 'model_type 不在 session 允许的模型集合内', status_code=422)
    body = payload.model_dump(mode='json')
    if session.contract_version == V2:
        action = frozen_action(session, payload)
        body.update(contract_version=V2, model_params=action['model_params'])
    else:
        action = {key: getattr(payload, key) for key in
                  ('model_type', 'normalization', 'class_balance', 'parent_run_id')}
    return ExperimentCommand(action, body, payload.rationale, payload.client_request_id)


def admit_command(session, command):
    admit_action(session, command.action)
    if command.recipe is not None:
        if session.frozen_preparation['protocol_revision'] in ('agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6'):
            from ..search_policy import _digest_source
            plans = session.frozen_preparation['preparation']['search_plans']
            bound = command.recipe.get('search_plan_digest')
            if bound not in plans or plans[bound]['source_digest'] != _digest_source():
                raise AgentDomainError('agent_capability_changed', 'Execution search plan changed', status_code=409)
            return
        from ..model_config import compatible_search_strategy_binding
        if not compatible_search_strategy_binding(
            command.recipe['model_id'], command.recipe['search_strategy_digest'],
            command.action['normalization'], command.action['class_balance'],
        ):
            raise AgentDomainError('agent_capability_changed', 'Execution search policy changed', status_code=409)


def command_digest(session, command):
    if command.recipe is None:
        return scientific_digest(session, command.action)
    prepared=session.frozen_preparation['preparation']
    if session.frozen_preparation['protocol_revision'] in ('agent-recipes-revision-v4','agent-recipes-revision-v5','agent-recipes-revision-v6'):
        return semantic_digest(dict(profile=session.frozen_preparation['execution_profile'],
            dataset_sha256=session.dataset_sha256, plan_digest=prepared['evaluation_plan']['plan_digest'],
            model_id=command.recipe['model_id'], normalization=command.action['normalization'],
            class_balance=command.action['class_balance'], model_params=command.action['model_params'],
            search_plan_digest=command.recipe['search_plan_digest'], seed=session.seed))
    if session.frozen_preparation['protocol_revision'] == 'agent-recipes-revision-v3':
        return semantic_digest(dict(profile=session.frozen_preparation['execution_profile'],
            processing_policy_version=command.recipe['processing_policy_version'],
            dataset_sha256=session.dataset_sha256,plan_digest=prepared['evaluation_plan']['plan_digest'],
            model_id=command.recipe['model_id'],normalization=command.action['normalization'],
            class_balance=command.action['class_balance'],model_params=command.action['model_params'],
            search_strategy_digest=command.recipe['search_strategy_digest'],seed=session.seed))
    return semantic_digest(dict(profile=session.frozen_preparation['execution_profile'],
        dataset_sha256=session.dataset_sha256,plan_digest=prepared['evaluation_plan']['plan_digest'],
        catalog_digest=prepared['catalog']['catalog_digest'],recipe_digest=command.recipe['recipe_digest']))
