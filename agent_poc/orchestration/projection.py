"""Field allowlists, shared by both explicitly selected LLM protocols."""
from __future__ import annotations

from typing import Annotated, Any, Literal
from pydantic import Field, model_validator
from agent_poc.clients.contracts import (
    ClosedModel, Evaluation, Identifier, MetricName, ModelName, Score, ValidationMetrics,
)

CONTEXT_VERSION = 'agent-context-step1-v1'


class ProjectedTask(ClosedModel):
    task_type: Literal['classification']
    selection_metric: MetricName
    seed: Annotated[int, Field(ge=0)]
    evaluation_config: Evaluation


class ProjectedCapabilities(ClosedModel):
    models: Annotated[list[ModelName], Field(min_length=1, max_length=3)]

    @model_validator(mode='after')
    def unique_models(self):
        if len(self.models) != len(set(self.models)):
            raise ValueError('duplicate candidate model')
        return self


class SelectionBindings(ClosedModel):
    session_id: Identifier
    client_request_id: Identifier
    normalization: Literal['zscore']
    class_balance: Literal['none']


class SelectionContext(ClosedModel):
    context_version: Literal['agent-context-step1-v1']
    phase: Literal['submit']
    task: ProjectedTask
    capabilities: ProjectedCapabilities
    allowed_actions: list[Literal['submit_ml_experiment']]
    bindings: SelectionBindings


class FinalizationBindings(ClosedModel):
    session_id: Identifier
    selected_run_id: Identifier


class CandidateSummary(ClosedModel):
    run_id: Identifier
    validation_score: Score
    eligibility: Literal['valid']


class FinalizationContext(ClosedModel):
    context_version: Literal['agent-context-step1-v1']
    phase: Literal['finalize']
    task: ProjectedTask
    allowed_actions: list[Literal['finalize_ml_session']]
    bindings: FinalizationBindings
    validation: ValidationMetrics
    candidate: CandidateSummary

    @model_validator(mode='after')
    def coherent_candidate(self):
        if self.candidate.run_id != self.bindings.selected_run_id:
            raise ValueError('candidate does not match bound Run')
        if getattr(self.validation, self.task.selection_metric) != self.candidate.validation_score:
            raise ValueError('candidate does not match validation')
        return self


def _project_task(task: dict[str, Any]) -> dict[str, Any]:
    # Copy individual fields: no entire State or backend dictionary is serialized.
    evaluation = task.get('evaluation_config') or task.get('evaluation') or {}
    return ProjectedTask.model_validate({
        'task_type': task.get('task_type', 'classification'),
        'selection_metric': task['selection_metric'], 'seed': task['seed'],
        'evaluation_config': {key: evaluation[key] for key in (
            'mode', 'train_weight', 'validation_weight', 'heldout_weight')},
    }).model_dump(mode='json')


def selection_context(*, task: dict[str, Any], models: list[str], session_id: str,
                      client_request_id: str, model_configs: dict | None = None) -> dict[str, Any]:
    allowed = set(task['allowed_models'])
    candidates = [model for model in models if model in allowed]
    context_type = SelectionContext if model_configs is None else SelectionContextV2
    return context_type.model_validate({
        'context_version': CONTEXT_VERSION if model_configs is None else 'agent-context-step2-v1', 'phase': 'submit', 'task': _project_task(task),
        'capabilities': {'models': candidates, **({'fixed_model_params':{m:model_configs[m] for m in candidates}} if model_configs is not None else {})}, 'allowed_actions': ['submit_ml_experiment'],
        'bindings': {'session_id': session_id, 'client_request_id': client_request_id,
                     'normalization': 'zscore', 'class_balance': 'none'},
    }).model_dump(mode='json')


def finalization_context(*, task: dict[str, Any], session_id: str, run_id: str,
                         validation: dict[str, Any], validation_score: float,
                         allowed_actions: list[str], context_version: str = CONTEXT_VERSION) -> dict[str, Any]:
    if 'finalize_ml_session' not in allowed_actions:
        raise ValueError('finalize is not allowed')
    if 'status' in validation and validation['status'] != 'ready':
        raise ValueError('validation is not ready')
    metrics = validation.get('metrics', validation)
    projected_metrics = {key: metrics[key] for key in ValidationMetrics.model_fields if key in metrics}
    checked = ValidationMetrics.model_validate(projected_metrics).model_dump(exclude_none=True)
    if checked.get(task['selection_metric']) != validation_score:
        raise ValueError('candidate selection metric mismatch')
    context_type = (RecipeFinalizationContext if context_version in ('agent-context-recipes-v1','agent-context-knowledge-v1') else
                    FinalizationContext if context_version == CONTEXT_VERSION else FinalizationContextV2)
    return context_type.model_validate({
        'context_version': context_version, 'phase': 'finalize', 'task': _project_task(task),
        'allowed_actions': ['finalize_ml_session'],
        'bindings': {'session_id': session_id, 'selected_run_id': run_id},
        'validation': checked,
        'candidate': {'run_id': run_id, 'validation_score': validation_score, 'eligibility': 'valid'},
    }).model_dump(mode='json', exclude_none=True)


def validate_context(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    if context.get('context_version') == 'agent-context-knowledge-v1':
        model = {'submit':KnowledgeSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-recipes-v1':
        model = {'submit':RecipeSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-step2-v1':
        model = {'submit': SelectionContextV2, 'finalize': FinalizationContextV2}.get(phase)
    else:
        model = {'submit': SelectionContext, 'finalize': FinalizationContext}.get(phase)
    if model is None:
        raise ValueError('unknown decision phase')
    checked = model.model_validate(context).model_dump(mode='json', exclude_none=context.get('context_version')!='agent-context-knowledge-v1')
    if checked['phase'] != phase or len(checked['allowed_actions']) != 1:
        raise ValueError('invalid allowed action')
    return checked


class ProjectedCapabilitiesV2(ProjectedCapabilities):
    models: Annotated[list[Identifier],Field(min_length=1)]
    fixed_model_params: dict[Identifier,dict[Identifier,int | float | str]]

    @model_validator(mode='after')
    def bindings_complete(self):
        if set(self.models)!=set(self.fixed_model_params):
            raise ValueError('missing fixed model configuration')
        return self


class SelectionContextV2(SelectionContext):
    context_version: Literal['agent-context-step2-v1']
    capabilities: ProjectedCapabilitiesV2


class FinalizationContextV2(FinalizationContext):
    context_version: Literal['agent-context-step2-v1']


from agent_poc.clients.contracts_v2 import Recipe, TrainStatistics, Risk, Preparation


class RecipeSelectionBindings(ClosedModel):
    session_id: Identifier


class RecipeSelectionContext(ClosedModel):
    context_version: Literal['agent-context-recipes-v1']
    phase: Literal['submit']
    task: ProjectedTask
    recipes: Annotated[list[Recipe],Field(min_length=1)]
    train_statistics: TrainStatistics | None=None
    train_risks: list[Risk] | None=None
    allowed_actions: list[Literal['submit_ml_experiment']]
    bindings: RecipeSelectionBindings


class RecipeFinalizationContext(FinalizationContext):
    context_version: Literal['agent-context-recipes-v1','agent-context-knowledge-v1']


def recipe_selection_context(*,task,session_id,preparation,context_policy):
    from agent_poc.clients.contracts_v2 import KnowledgePreparation
    modern=context_policy.get('projection')=='agent-context-knowledge-v1'
    prepared=(KnowledgePreparation if modern else Preparation).model_validate(preparation)
    payload=dict(context_version='agent-context-recipes-v1',phase='submit',task=_project_task(task),
        recipes=[r.model_dump(mode='json') for r in prepared.catalog.recipes],
        allowed_actions=['submit_ml_experiment'],bindings={'session_id':session_id})
    if context_policy['evidence']:
        payload['train_statistics']=prepared.evidence.statistics.model_dump(mode='json')
    if context_policy['risks']:
        payload['train_risks']=[r.model_dump(mode='json') for r in prepared.evidence.risks]
    if modern:
        payload.update(context_version='agent-context-knowledge-v1',knowledge=prepared.knowledge.projection.model_dump(mode='json'))
    return (KnowledgeSelectionContext if modern else RecipeSelectionContext).model_validate(payload).model_dump(mode='json',exclude_none=not modern)


from agent_poc.clients.contracts_v2 import KnowledgeProjection


class KnowledgeSelectionContext(RecipeSelectionContext):
    context_version: Literal['agent-context-knowledge-v1']
    knowledge: KnowledgeProjection
