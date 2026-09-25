"""Field allowlists, shared by both explicitly selected LLM protocols."""
from __future__ import annotations

from typing import Annotated, Any, Literal
from pydantic import Field, model_validator, model_serializer
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
                         allowed_actions: list[str], context_version: str = CONTEXT_VERSION,
                         budget_awareness: str | None = None,
                         budget_balance: dict | None = None) -> dict[str, Any]:
    if 'finalize_ml_session' not in allowed_actions:
        raise ValueError('finalize is not allowed')
    if 'status' in validation and validation['status'] != 'ready':
        raise ValueError('validation is not ready')
    metrics = validation.get('metrics', validation)
    projected_metrics = {key: metrics[key] for key in ValidationMetrics.model_fields if key in metrics}
    checked = ValidationMetrics.model_validate(projected_metrics).model_dump(exclude_none=True)
    if checked.get(task['selection_metric']) != validation_score:
        raise ValueError('candidate selection metric mismatch')
    context_type = (BudgetFinalizationContext if context_version == 'agent-context-budget-v1' else
                    RecipeFinalizationContext if context_version in ('agent-context-recipes-v1','agent-context-knowledge-v1','agent-context-processing-v1','agent-context-search-v1') else
                    FinalizationContext if context_version == CONTEXT_VERSION else FinalizationContextV2)
    return context_type.model_validate({
        'context_version': context_version, 'phase': 'finalize', 'task': _project_task(task),
        'allowed_actions': ['finalize_ml_session','stop_ml_session']
            if budget_awareness == 'on' and context_version == 'agent-context-budget-v1'
            else ['finalize_ml_session'],
        'bindings': {'session_id': session_id, 'selected_run_id': run_id},
        'validation': checked,
        'candidate': {'run_id': run_id, 'validation_score': validation_score, 'eligibility': 'valid'},
        **({'budget_balance':budget_balance} if budget_balance is not None else {}),
    }).model_dump(mode='json', exclude_none=True)


def validate_context(phase: str, context: dict[str, Any]) -> dict[str, Any]:
    if context.get('context_version') == 'agent-context-budget-v1':
        model = {'submit':BudgetSelectionContext,'finalize':BudgetFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-search-v1':
        model = {'submit':SearchSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-processing-v1':
        model = {'submit':ProcessingSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-knowledge-v1':
        model = {'submit':KnowledgeSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-recipes-v1':
        model = {'submit':RecipeSelectionContext,'finalize':RecipeFinalizationContext}.get(phase)
    elif context.get('context_version') == 'agent-context-step2-v1':
        model = {'submit': SelectionContextV2, 'finalize': FinalizationContextV2}.get(phase)
    else:
        model = {'submit': SelectionContext, 'finalize': FinalizationContext}.get(phase)
    if model is None:
        raise ValueError('unknown decision phase')
    checked = model.model_validate(context).model_dump(mode='json', exclude_none=context.get('context_version') not in ('agent-context-knowledge-v1','agent-context-processing-v1','agent-context-search-v1','agent-context-budget-v1'))
    if context.get('context_version') == 'agent-context-budget-v1' and checked.get('budget_card') is None:
        checked.pop('budget_card', None)
    if context.get('context_version') == 'agent-context-budget-v1' and checked.get('budget_balance') is None:
        checked.pop('budget_balance', None)
    if (checked['phase'] != phase or
            checked['allowed_actions'] not in (['submit_ml_experiment'],
                                               ['finalize_ml_session'],
                                               ['submit_ml_experiment','stop_ml_session'],
                                               ['finalize_ml_session','stop_ml_session'])):
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


from agent_poc.clients.preparation import Recipe, TrainStatistics, Risk, Preparation


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
    context_version: Literal['agent-context-recipes-v1','agent-context-knowledge-v1','agent-context-processing-v1','agent-context-search-v1','agent-context-budget-v1']


class BudgetFinalizationBalance(ClosedModel):
    llm_calls_remaining: int = Field(ge=0)
    api_calls_remaining: int = Field(ge=0)
    llm_calls_unknown_held: int = Field(ge=0)
    api_calls_unknown_held: int = Field(ge=0)
    task_seconds_remaining: float = Field(ge=0, allow_inf_nan=False)


class BudgetFinalizationContext(RecipeFinalizationContext):
    context_version: Literal['agent-context-budget-v1']
    allowed_actions: list[Literal['finalize_ml_session','stop_ml_session']]
    budget_balance: BudgetFinalizationBalance | None = None

    @model_validator(mode='after')
    def stop_visibility(self):
        expected = (['finalize_ml_session','stop_ml_session']
                    if self.budget_balance is not None else ['finalize_ml_session'])
        if self.allowed_actions != expected:
            raise ValueError('finalization budget visibility mismatch')
        return self


def recipe_selection_context(*,task,session_id,preparation,context_policy,model_configs=None,
                             budget_card=None):
    from agent_poc.clients.knowledge import KnowledgePreparation
    budget_revision=context_policy.get('projection')=='agent-context-budget-v1'
    searching=context_policy.get('projection') in ('agent-context-search-v1','agent-context-budget-v1')
    processing=context_policy.get('projection') in ('agent-context-processing-v1','agent-context-search-v1','agent-context-budget-v1')
    modern=context_policy.get('projection') in ('agent-context-knowledge-v1','agent-context-processing-v1','agent-context-search-v1','agent-context-budget-v1')
    prepared=(KnowledgePreparation if modern else Preparation).model_validate(preparation)
    full_recipes=[r.model_dump(mode='json') for r in prepared.catalog.recipes]
    payload=dict(context_version='agent-context-recipes-v1',phase='submit',task=_project_task(task),
        recipes=full_recipes,
        allowed_actions=['submit_ml_experiment'],bindings={'session_id':session_id})
    if task.get('decision_mode') is not None:
        payload.update(decision_mode=task['decision_mode'], fixed_model_params=model_configs or task.get('capability_snapshot',{}).get('model_configs',{}))
    if context_policy['evidence']:
        payload['train_statistics']=prepared.evidence.statistics.model_dump(mode='json')
    if context_policy['risks']:
        payload['train_risks']=[r.model_dump(mode='json') for r in prepared.evidence.risks]
    if modern:
        payload.update(context_version='agent-context-budget-v1' if budget_revision else 'agent-context-search-v1' if searching else 'agent-context-processing-v1' if processing else 'agent-context-knowledge-v1',
            knowledge=prepared.knowledge.projection.model_dump(mode='json'))
    if processing:
        profiles={}
        for recipe in full_recipes:
            profiles.setdefault(recipe['model_id'], dict(
                architecture_version=recipe['architecture_version'],
                model_policy_version=recipe['model_policy_version'],
                search_strategy_version=recipe['search_strategy_version'],
                cost_description=recipe['cost_description']))
        payload.update(processing_mode=task['processing_mode'],fixed_processing=task['fixed_processing'],
            model_profiles=profiles,recipes=[dict(recipe_id=recipe['recipe_id'],
                model_id=recipe['model_id'],normalization=recipe['preprocessing']['normalization'],
                class_balance=recipe['class_balance']) for recipe in full_recipes])
    if searching:
        payload.update(search_mode=task['search_mode'],max_trials=task['max_trials'])
    if budget_revision and budget_card is not None:
        payload['budget_card']=budget_card
        payload['allowed_actions']=['submit_ml_experiment','stop_ml_session']
    projected = (BudgetSelectionContext if budget_revision else SearchSelectionContext if searching else ProcessingSelectionContext if processing else KnowledgeSelectionContext if modern else RecipeSelectionContext).model_validate(payload).model_dump(mode='json',exclude_none=not modern)
    if budget_revision and projected.get('budget_card') is None:
        projected.pop('budget_card', None)
    return projected


from agent_poc.clients.knowledge import KnowledgeProjection


class KnowledgeSelectionContext(RecipeSelectionContext):
    decision_mode: Literal['recipe_id','structured_config'] | None = None
    fixed_model_params: dict[str,dict[str,int|float|str]] | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_expression(self, handler):
        result = handler(self)
        if self.decision_mode is None:
            result.pop('decision_mode',None)
            result.pop('fixed_model_params',None)
        return result

    context_version: Literal['agent-context-knowledge-v1']
    knowledge: KnowledgeProjection


class CompactRecipe(ClosedModel):
    recipe_id: str
    model_id: str
    normalization: Literal['zscore','minmax','area','none']
    class_balance: Literal['none','class_weight']


class ModelProfile(ClosedModel):
    architecture_version: str
    model_policy_version: str
    search_strategy_version: str
    cost_description: str


class ProcessingSelectionContext(KnowledgeSelectionContext):
    context_version: Literal['agent-context-processing-v1']
    recipes: Annotated[list[CompactRecipe],Field(min_length=1)]
    model_profiles: dict[str,ModelProfile]
    processing_mode: Literal['fixed','dynamic']
    fixed_processing: dict[str,dict[str,str]]

    @model_validator(mode='after')
    def complete_processing_domain(self):
        members={(r.model_id,r.normalization,r.class_balance) for r in self.recipes}
        if len(members)!=len(self.recipes):
            raise ValueError('duplicate processing member')
        eligible={r.model_id for r in self.recipes}
        if not eligible.issubset(self.fixed_processing):
            raise ValueError('eligible model has no frozen processing')
        for model_id in eligible:
            choice=self.fixed_processing[model_id]
            if (model_id,choice['normalization'],choice['class_balance']) not in members:
                raise ValueError('fixed processing outside catalog')
        if self.processing_mode=='fixed' and len(members)!=len(eligible):
            raise ValueError('fixed catalog has extra processing members')
        if set(self.model_profiles)!=eligible:
            raise ValueError('processing model profiles incomplete')
        return self


class SearchSelectionContext(ProcessingSelectionContext):
    context_version: Literal['agent-context-search-v1']
    search_mode: Literal['fixed','bounded']
    max_trials: int


class BudgetRecipeCost(ClosedModel):
    recipe_id: Identifier
    model_fits_upper: int = Field(ge=0)
    training_epochs_upper: int = Field(ge=0)
    feasible: bool
    estimated_seconds: None = None


class BudgetCostCard(ClosedModel):
    model_fits_remaining: int = Field(ge=0)
    training_epochs_remaining: int = Field(ge=0)
    llm_calls_remaining: int = Field(ge=0)
    api_calls_remaining: int = Field(ge=0)
    llm_calls_unknown_held: int = Field(ge=0)
    api_calls_unknown_held: int = Field(ge=0)
    work_seconds_remaining: float = Field(ge=0, allow_inf_nan=False)
    finalization_llm_calls_protected: int = Field(ge=0)
    finalization_api_calls_protected: int = Field(ge=0)
    recipes: list[BudgetRecipeCost]


class BudgetSelectionContext(SearchSelectionContext):
    context_version: Literal['agent-context-budget-v1']
    allowed_actions: list[Literal['submit_ml_experiment','stop_ml_session']]
    budget_card: BudgetCostCard | None = None

    @model_validator(mode='after')
    def stop_visibility(self):
        expected = (['submit_ml_experiment','stop_ml_session'] if self.budget_card is not None
                    else ['submit_ml_experiment'])
        if self.allowed_actions != expected:
            raise ValueError('budget action visibility mismatch')
        return self
