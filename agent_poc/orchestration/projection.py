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
                      client_request_id: str) -> dict[str, Any]:
    allowed = set(task['allowed_models'])
    candidates = [model for model in models if model in allowed]
    return SelectionContext.model_validate({
        'context_version': CONTEXT_VERSION, 'phase': 'submit', 'task': _project_task(task),
        'capabilities': {'models': candidates}, 'allowed_actions': ['submit_ml_experiment'],
        'bindings': {'session_id': session_id, 'client_request_id': client_request_id,
                     'normalization': 'zscore', 'class_balance': 'none'},
    }).model_dump(mode='json')


def finalization_context(*, task: dict[str, Any], session_id: str, run_id: str,
                         validation: dict[str, Any], validation_score: float,
                         allowed_actions: list[str]) -> dict[str, Any]:
    if 'finalize_ml_session' not in allowed_actions:
        raise ValueError('finalize is not allowed')
    if 'status' in validation and validation['status'] != 'ready':
        raise ValueError('validation is not ready')
    metrics = validation.get('metrics', validation)
    projected_metrics = {key: metrics[key] for key in ValidationMetrics.model_fields if key in metrics}
    checked = ValidationMetrics.model_validate(projected_metrics).model_dump(exclude_none=True)
    if checked.get(task['selection_metric']) != validation_score:
        raise ValueError('candidate selection metric mismatch')
    return FinalizationContext.model_validate({
        'context_version': CONTEXT_VERSION, 'phase': 'finalize', 'task': _project_task(task),
        'allowed_actions': ['finalize_ml_session'],
        'bindings': {'session_id': session_id, 'selected_run_id': run_id},
        'validation': checked,
        'candidate': {'run_id': run_id, 'validation_score': validation_score, 'eligibility': 'valid'},
    }).model_dump(mode='json', exclude_none=True)


def validate_context(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    model = {'submit': SelectionContext, 'finalize': FinalizationContext}.get(phase)
    if model is None:
        raise ValueError('unknown decision phase')
    checked = model.model_validate(context).model_dump(mode='json', exclude_none=True)
    if checked['phase'] != phase or len(checked['allowed_actions']) != 1:
        raise ValueError('invalid allowed action')
    return checked
