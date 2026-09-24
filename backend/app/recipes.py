"""Finite execution recipes compiled from authoritative frozen model policy."""
from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel,ConfigDict,model_validator,model_serializer

from .evaluation_plan import EvaluationPlan, digest
from .model_config import resolve_model_params, search_strategy_binding
from .train_evidence import TrainEvidence
from .processing_policy import (
    PROCESSING_POLICY_VERSION, freeze_fixed_processing, legal_processing,
    validate_processing,
)

PROFILE='train-evidence-recipes-v1'
REVISION='agent-recipes-revision-v1'


class Recipe(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    recipe_schema_version: Literal['execution-recipe-v1','execution-recipe-v2','execution-recipe-v3']='execution-recipe-v1'
    recipe_id: str
    recipe_digest: str
    model_id: str
    architecture_version: str
    preprocessing: dict[str,str]
    class_balance: Literal['none','class_weight']='none'
    processing_policy_version: str | None = None
    fixed_execution_config: dict[str,int|float|str|bool]
    model_policy_version: str
    model_policy_digest: str
    search_strategy_ref: str
    search_strategy_version: str
    search_strategy_digest: str
    search_plan_digest: str | None = None
    search_policy_version: str | None = None
    hard_constraints: list[str]
    evidence_applicability: Literal['eligible']='eligible'
    cost_description: str

    @model_serializer(mode='wrap')
    def legacy_shape(self, handler):
        result=handler(self)
        if self.recipe_schema_version == 'execution-recipe-v1':
            result.pop('processing_policy_version', None)
        if self.recipe_schema_version != 'execution-recipe-v3':
            result.pop('search_plan_digest', None)
            result.pop('search_policy_version', None)
        return result

    @model_validator(mode='after')
    def executable_digest(self):
        body={k:v for k,v in self.model_dump(mode='json').items() if k not in
              ('recipe_id','recipe_digest','hard_constraints','evidence_applicability','cost_description')}
        if digest(body)!=self.recipe_digest or self.recipe_id!='recipe_'+self.recipe_digest:
            raise ValueError('recipe executable content mismatch')
        return self


class RecipeCatalog(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    catalog_version: Literal['recipe-catalog-v1','recipe-catalog-v2','recipe-catalog-v3']='recipe-catalog-v1'
    catalog_digest: str
    dataset_sha256: str
    plan_digest: str
    evidence_digest: str
    allowed_models: list[str]
    recipes: list[Recipe]
    excluded_models: dict[str,str]
    processing_mode: Literal['fixed','dynamic'] | None = None
    fixed_processing: dict[str,dict[str,str]] | None = None
    search_mode: Literal['fixed','bounded'] | None = None
    max_trials: int | None = None

    @model_serializer(mode='wrap')
    def legacy_shape(self, handler):
        result=handler(self)
        if self.catalog_version == 'recipe-catalog-v1':
            result.pop('processing_mode', None)
            result.pop('fixed_processing', None)
        if self.catalog_version != 'recipe-catalog-v3':
            result.pop('search_mode', None)
            result.pop('max_trials', None)
        return result


def canonical_execution(value):
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('nonfinite execution value')
        return 0.0 if value==0 else value
    if type(value) is dict:return {k:canonical_execution(v) for k,v in sorted(value.items())}
    if type(value) is list:return [canonical_execution(v) for v in value]
    if type(value) in (str,int,bool):return value
    raise ValueError('invalid execution value')


def compile_recipe_catalog(task: dict, frozen_capabilities: dict, evidence: TrainEvidence,
                           evaluation_plan: EvaluationPlan, *, search_plans: dict | None = None) -> RecipeCatalog:
    if evidence.plan_digest!=evaluation_plan.plan_digest or evidence.dataset_sha256!=evaluation_plan.dataset_sha256:
        raise ValueError('recipe_evidence_binding_mismatch')
    policies={m['id']:m for m in frozen_capabilities['models']}
    recipes=[];excluded={};allowed=sorted(set(task['allowed_models']))
    if len(allowed)!=len(task['allowed_models']):raise ValueError('duplicate_allowed_model')
    search_revision = task.get('protocol_revision') == 'agent-recipes-revision-v4'
    modern = task.get('protocol_revision') in ('agent-recipes-revision-v3','agent-recipes-revision-v4')
    processing_mode = task.get('processing_mode', 'fixed') if modern else None
    fixed_processing = freeze_fixed_processing(allowed, task.get('fixed_processing')) if modern else None
    class_count = len(evidence.statistics.classes)
    for model_id in allowed:
        policy=policies[model_id]
        params=resolve_model_params(model_id,frozen_capabilities['model_configs'][model_id],policy=policy)
        if not policy['available']:
            excluded[model_id]=policy['reason_code'] or 'model_unavailable';continue
        # The existing PCA-LDA search starts at two components; sklearn LDA
        # requires more observations than classes. These are hard fit limits,
        # not small-sample/high-dimension advisory risk thresholds.
        stats=evidence.statistics
        if model_id=='pca_lda' and (stats.feature_count<2 or stats.observation_count<=len(stats.classes)):
            excluded[model_id]='pca_lda_insufficient_training_dimensions';continue
        # Data risk flags are advisory, never model exclusion rules.
        search=search_strategy_binding(model_id)
        if modern:
            frozen = fixed_processing[model_id]
            validate_processing(model_id, frozen['normalization'], frozen['class_balance'], class_count=class_count)
            choices = (legal_processing(model_id, class_count=class_count) if processing_mode == 'dynamic'
                       else ((frozen['normalization'], frozen['class_balance']),))
        else:
            choices = (('zscore','none'),)
        for normalization, balance in choices:
            plan = None
            if search_revision:
                from .search_policy import build_search_plan
                plan = build_search_plan(model_id=model_id, baseline=params,
                    mode=task['search_mode'], max_trials=task['max_trials'],
                    train_count=stats.observation_count, feature_count=stats.feature_count,
                    class_count=class_count, dataset_sha256=evidence.dataset_sha256,
                    evaluation_plan_digest=evaluation_plan.plan_digest,
                    normalization=normalization, class_balance=balance, seed=task['seed'],
                    architecture_version=policy['architecture_version'],
                    processing_policy_version=PROCESSING_POLICY_VERSION)
                if search_plans is not None:
                    search_plans[plan['plan_digest']] = plan
            config=canonical_execution(dict(model_type=model_id,normalization=normalization,class_balance=balance,
                seed=task['seed'],feature_selection_enabled=False,**task['evaluation'],**params))
            body=dict(recipe_schema_version='execution-recipe-v3' if search_revision else 'execution-recipe-v2' if modern else 'execution-recipe-v1',
                model_id=model_id,architecture_version=policy['architecture_version'],
                preprocessing={'normalization':normalization,'implementation_ref':
                    'finite-processing-v1' if modern else 'backend-train-feature-zscore-v1'},
                class_balance=balance,fixed_execution_config=config,
                model_policy_version=policy['config_policy_version'],model_policy_digest=policy['config_policy_digest'],
                search_strategy_ref=search['ref'],search_strategy_version=search['version'],search_strategy_digest=search['digest'])
            if modern:
                body['processing_policy_version']=PROCESSING_POLICY_VERSION
            if search_revision:
                body.update(search_plan_digest=plan['plan_digest'], search_policy_version=plan['policy_version'])
            key=digest(body)
            recipes.append(Recipe(**body,recipe_id='recipe_'+key,recipe_digest=key,
                hard_constraints=['classification','class-complete-group-holdout'],
                cost_description=('One fixed candidate' if task['search_mode']=='fixed' else 'At most '+str(plan['effective_trials'])+' candidates') if search_revision else 'Existing bounded validation/OOB search' if policy['execution_family']=='traditional_ml'
                                 else 'Bounded epochs; runtime and memory depend on data and architecture'))
    recipes=sorted(recipes,key=lambda r:r.recipe_id)
    if not recipes:raise ValueError('no_legal_recipes')
    binding=dict(dataset_sha256=evidence.dataset_sha256,plan_digest=evaluation_plan.plan_digest,
        evidence_digest=evidence.evidence_digest,allowed_models=allowed,
        frozen_policies=[{k:v for k,v in policies[m].items() if k not in ('display_name','available','reason_code','availability_basis')} for m in allowed],
        recipe_digests=[r.recipe_digest for r in recipes])
    if modern:
        binding.update(processing_mode=processing_mode,fixed_processing=fixed_processing,
                       processing_policy_version=PROCESSING_POLICY_VERSION)
    if search_revision:
        binding.update(search_mode=task['search_mode'],max_trials=task['max_trials'])
    return RecipeCatalog(catalog_version='recipe-catalog-v3' if search_revision else 'recipe-catalog-v2' if modern else 'recipe-catalog-v1',
        processing_mode=processing_mode,fixed_processing=fixed_processing,
        search_mode=task.get('search_mode') if search_revision else None,
        max_trials=task.get('max_trials') if search_revision else None,
        catalog_digest=digest(binding),dataset_sha256=evidence.dataset_sha256,
        plan_digest=evaluation_plan.plan_digest,evidence_digest=evidence.evidence_digest,
        allowed_models=allowed,recipes=recipes,excluded_models=excluded)


def validate_catalog_binding(catalog, snapshot):
    policies={m['id']:m for m in snapshot['models']}
    allowed=catalog.allowed_models
    binding=dict(dataset_sha256=catalog.dataset_sha256,plan_digest=catalog.plan_digest,
        evidence_digest=catalog.evidence_digest,allowed_models=allowed,
        frozen_policies=[{k:v for k,v in policies[m].items() if k not in
            ('display_name','available','reason_code','availability_basis')} for m in allowed],
        recipe_digests=[r.recipe_digest for r in catalog.recipes])
    if catalog.catalog_version in ('recipe-catalog-v2','recipe-catalog-v3'):
        binding.update(processing_mode=catalog.processing_mode,fixed_processing=catalog.fixed_processing,
                       processing_policy_version=PROCESSING_POLICY_VERSION)
    if catalog.catalog_version == 'recipe-catalog-v3':
        binding.update(search_mode=catalog.search_mode,max_trials=catalog.max_trials)
    if (allowed!=sorted(set(allowed)) or catalog.catalog_digest!=digest(binding)
            or [r.recipe_id for r in catalog.recipes]!=sorted(set(r.recipe_id for r in catalog.recipes))):
        raise ValueError('catalog binding mismatch')
    return catalog
