"""Frozen Train evidence, finite recipes and their response bindings."""
from __future__ import annotations
from typing import Annotated, Literal
import math
from pydantic import ConfigDict, Field, field_validator, model_validator
from . import contracts as v1
from .contracts import ClosedModel, Identifier
from .capabilities import digest
from .execution_contracts import LockedConfig, CreatedSessionResponse, SessionResponse, HealthResponse, RESPONSE_MODELS

class ClassCount(ClosedModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    class_id: int
    observation_count: int
    sample_group_count: int
    observation_fraction: float
    sample_group_fraction: float

class TrainStatistics(ClosedModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    observation_count: int
    sample_group_count: int
    feature_count: int
    classes: list[ClassCount]
    repeated_measurement_group_count: int
    group_size_min: int
    group_size_max: int
    group_size_median: float
    unequal_repeats: bool
    zero_variance_feature_count: int
    duplicate_feature_group_count: int
    redundant_feature_count: int
    duplicate_vector_group_count: int
    duplicate_vector_observation_count: int
    cross_sample_duplicate_vector_group_count: int
    conflicting_vector_group_count: int
    conflicting_vector_observation_count: int

class Risk(ClosedModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    code: Literal['small_sample','high_dimension','imbalance','zero_variance','duplicate_features','conflicting_vectors','repeated_measurement']
    severity: Literal['warning','informational']

class TrainEvidence(ClosedModel):
    model_config = ConfigDict(extra='forbid',strict=True,frozen=True)
    schema_version: Literal['train-evidence-v1'] = 'train-evidence-v1'
    statistics_version: Literal['train-statistics-v1'] = 'train-statistics-v1'
    risk_version: Literal['train-risk-rules-v1'] = 'train-risk-rules-v1'
    status: Literal['ready'] = 'ready'
    scope: Literal['train'] = 'train'
    input_representation: Literal['raw-pre-normalization-float32'] = 'raw-pre-normalization-float32'
    dataset_sha256: str
    plan_digest: str
    train_content_digest: str
    statistics_digest: str
    evidence_digest: str
    statistics: TrainStatistics
    risks: list[Risk]


    @model_validator(mode='after')
    def evidence_content_digest(self):
        objective=dict(statistics=self.statistics.model_dump(mode='json'),risks=[r.model_dump() for r in self.risks],
            statistics_version=self.statistics_version,risk_version=self.risk_version)
        body=dict(dataset_sha256=self.dataset_sha256,plan_digest=self.plan_digest,
            train_content_digest=self.train_content_digest,statistics_digest=self.statistics_digest,**objective)
        if self.statistics_digest!=digest(objective) or self.evidence_digest!=digest(body):
            raise ValueError('evidence content mismatch')
        return self

class Recipe(ClosedModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    recipe_schema_version: Literal['execution-recipe-v1']='execution-recipe-v1'
    recipe_id: str
    recipe_digest: str
    model_id: str
    architecture_version: str
    preprocessing: dict[str,str]
    class_balance: Literal['none']='none'
    fixed_execution_config: dict[str,int|float|str|bool]
    model_policy_version: str
    model_policy_digest: str
    search_strategy_ref: str
    search_strategy_version: str
    search_strategy_digest: str
    hard_constraints: list[str]
    evidence_applicability: Literal['eligible']='eligible'
    cost_description: str

    @model_validator(mode='after')
    def executable_digest(self):
        body={k:v for k,v in self.model_dump(mode='json').items() if k not in
              ('recipe_id','recipe_digest','hard_constraints','evidence_applicability','cost_description')}
        if digest(body)!=self.recipe_digest or self.recipe_id!='recipe_'+self.recipe_digest:
            raise ValueError('recipe executable content mismatch')
        return self


class RecipeCatalog(ClosedModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)
    catalog_version: Literal['recipe-catalog-v1']='recipe-catalog-v1'
    catalog_digest: str
    dataset_sha256: str
    plan_digest: str
    evidence_digest: str
    allowed_models: list[str]
    recipes: list[Recipe]
    excluded_models: dict[str,str]


class EvaluationPlanReference(ClosedModel):
    schema_version: Literal['evaluation-plan-v1']
    split_algorithm_version: Literal['group-stratified-test-first-v1']
    dataset_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    axis_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    evaluation_config_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    seed: int
    partition_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    plan_digest: str = Field(pattern=r'^[a-f0-9]{64}$')


class Preparation(ClosedModel):
    evaluation_plan: EvaluationPlanReference
    evidence: TrainEvidence
    catalog: RecipeCatalog

    @model_validator(mode='after')
    def bound(self):
        if not (self.evaluation_plan.dataset_sha256==self.evidence.dataset_sha256==self.catalog.dataset_sha256
                and self.evaluation_plan.plan_digest==self.evidence.plan_digest==self.catalog.plan_digest
                and self.evidence.evidence_digest==self.catalog.evidence_digest):
            raise ValueError('preparation binding mismatch')
        if not self.catalog.recipes or len({r.recipe_id for r in self.catalog.recipes})!=len(self.catalog.recipes):
            raise ValueError('invalid finite recipe IDs')
        for recipe in self.catalog.recipes:
            if recipe.recipe_id != 'recipe_'+recipe.recipe_digest or recipe.model_id not in self.catalog.allowed_models:
                raise ValueError('invalid recipe binding')
        return self


class RecipeContextPolicy(ClosedModel):
    source_role: Literal['development','benchmark','domain']
    case_write: Literal[False]
    evidence: bool
    risks: bool


class RecipeLockedConfig(LockedConfig):
    execution_profile: Literal['train-evidence-recipes-v1']
    protocol_revision: Literal['agent-recipes-revision-v1']
    preparation: Preparation
    context_policy: RecipeContextPolicy

    @field_validator('modules')
    @classmethod
    def known_modules(cls,value):
        if value!=['train_evidence','legal_recipes']:
            raise ValueError('recipe capabilities required')
        return value

    @model_validator(mode='after')
    def prepared_binding(self):
        validate_catalog_binding(self.preparation.catalog,self.capability_snapshot.model_dump(mode='json'))
        if self.dataset_sha256 != self.preparation.catalog.dataset_sha256:
            raise ValueError('dataset preparation mismatch')
        if set(self.allowed_models)!=set(self.preparation.catalog.allowed_models):
            raise ValueError('catalog allowed model mismatch')
        return self


class RecipeCreatedSessionResponse(CreatedSessionResponse):
    locked_config: RecipeLockedConfig


class RecipeSessionResponse(SessionResponse):
    locked_config: RecipeLockedConfig


class RecipeHealthResponse(HealthResponse):
    protocol_revision: Literal['agent-recipes-revision-v1']
    execution_profiles: list[Literal['train-evidence-recipes-v1']]


RECIPE_RESPONSE_MODELS = {**RESPONSE_MODELS,
    'inspect_ml_capabilities':RecipeHealthResponse,
    'start_ml_session':RecipeCreatedSessionResponse,
    'inspect_ml_session':RecipeSessionResponse}


def validate_catalog_binding(catalog, snapshot):
    policies={m['id']:m for m in snapshot['models']}
    allowed=catalog.allowed_models
    binding=dict(dataset_sha256=catalog.dataset_sha256,plan_digest=catalog.plan_digest,
        evidence_digest=catalog.evidence_digest,allowed_models=allowed,
        frozen_policies=[{k:v for k,v in policies[m].items() if k not in
            ('display_name','available','reason_code','availability_basis')} for m in allowed],
        recipe_digests=[r.recipe_digest for r in catalog.recipes])
    if (allowed!=sorted(set(allowed)) or catalog.catalog_digest!=digest(binding)
            or [r.recipe_id for r in catalog.recipes]!=sorted(set(r.recipe_id for r in catalog.recipes))):
        raise ValueError('catalog binding mismatch')
    return catalog


