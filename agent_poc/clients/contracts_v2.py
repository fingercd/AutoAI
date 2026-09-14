"""Closed v2 wire types. No backend or training imports are permitted."""
from __future__ import annotations
from typing import Annotated, Literal
import hashlib
import json
import math
from pydantic import ConfigDict, Field, field_validator, model_validator, model_serializer
from . import contracts as v1
from .contracts import ClosedModel, Identifier

Scalar = int | float | str
CanonicalID = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class Parameter(v1.ClosedModel):
    name: CanonicalID
    value_type: Literal['integer','number','string','boolean']
    nullable: bool
    default_status: Literal['fixed','derived','unavailable']
    default: Scalar | bool | None
    minimum: int | float | None
    maximum: int | float | None
    exclusive_maximum: int | float | None = None
    exclusive_minimum: int | float | None
    choices: list[Scalar | bool] | None
    role: Literal['operator_fixed','backend_search','train_profile','fixed']
    applies_to: list[CanonicalID]
    condition_refs: list[Annotated[str,Field(pattern=r'^[a-z0-9_-]+$')]]

    @model_serializer(mode='wrap')
    def preserve_legacy_shape(self, handler):
        result = handler(self)
        if self.exclusive_maximum is None:
            result.pop('exclusive_maximum', None)
        return result

    @model_validator(mode='after')
    def consistent(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('invalid bounds')
        if self.default_status == 'fixed':
            validate_value(self, self.default)
        elif self.default is not None:
            raise ValueError('derived value cannot claim default')
        return self


def validate_value(parameter, value):
    if value is None:
        if parameter.nullable:
            return value
        raise ValueError('parameter is not nullable')
    types = {'integer':(int,), 'number':(int,float), 'string':(str,), 'boolean':(bool,)}
    if type(value) not in types[parameter.value_type]:
        raise ValueError('parameter type mismatch')
    if parameter.value_type in ('integer','number'):
        if not math.isfinite(value):
            raise ValueError('nonfinite parameter')
        if parameter.minimum is not None and value < parameter.minimum:
            raise ValueError('below minimum')
        if parameter.maximum is not None and value > parameter.maximum:
            raise ValueError('above maximum')
        if parameter.exclusive_maximum is not None and value >= parameter.exclusive_maximum:
            raise ValueError('above exclusive maximum')
        if parameter.exclusive_minimum is not None and value <= parameter.exclusive_minimum:
            raise ValueError('below exclusive minimum')
    if parameter.choices is not None and value not in parameter.choices:
        raise ValueError('unknown choice')
    return float(value) if parameter.value_type == 'number' else value


class ModelCapability(v1.ClosedModel):
    id: CanonicalID
    display_name: Annotated[str,Field(min_length=1,max_length=80)]
    family: Literal['traditional_ml','basic_deep','convolutional','long_range','two_dimensional_mapping']
    execution_family: Literal['traditional_ml','deep_learning']
    architecture_version: Literal['docx-classification-v2']
    implemented: bool
    available: bool
    availability_basis: Literal['declaration-module-dependency-probe']
    reason_code: Annotated[str,Field(pattern=r'^(not_implemented_for_version|implementation_missing|dependency_missing_[a-z0-9_]+)$')] | None
    config_policy_version: Literal['agent-model-config-v1','agent-model-config-v2']
    config_policy_digest: v1.Digest
    fixed_execution_defaults: dict[CanonicalID, Scalar]
    parameters: list[Parameter]
    compatibility_rules: list[Annotated[str,Field(pattern=r'^[a-z0-9_-]+$')]]

    @model_validator(mode='after')
    def consistent(self):
        if self.available and (not self.implemented or self.reason_code is not None):
            raise ValueError('invalid availability')
        if not self.available and self.reason_code is None:
            raise ValueError('missing unavailable reason')
        specs={p.name:p for p in self.parameters}
        if len(specs)!=len(self.parameters):
            raise ValueError('duplicate parameter')
        expected={p.name:p.default for p in self.parameters if p.role=='operator_fixed' and p.default_status=='fixed'}
        if expected != self.fixed_execution_defaults:
            raise ValueError('inconsistent defaults')
        if any(self.id not in p.applies_to for p in self.parameters):
            raise ValueError('inapplicable parameter')
        policy={key:self.model_dump(mode='json')[key] for key in ('config_policy_version','fixed_execution_defaults','parameters','compatibility_rules')}
        if digest(policy)!=self.config_policy_digest:
            raise ValueError('invalid policy digest')
        return self


def validate_params(model, params, *, complete=False):
    specs={p.name:p for p in model.parameters if p.role=='operator_fixed'}
    if type(params) is not dict or set(params)-set(specs) or (complete and set(params)!=set(specs)):
        raise ValueError('invalid parameter keys')
    return {key:validate_value(specs[key],value) for key,value in params.items()}


class CapabilitySnapshot(v1.ClosedModel):
    catalog_version: Literal['agent-model-catalog-v1']
    catalog_digest: v1.Digest
    availability_digest: v1.Digest
    models: list[ModelCapability]

    @model_validator(mode='after')
    def consistent(self):
        if len({m.id for m in self.models})!=len(self.models):
            raise ValueError('duplicate model ID')
        semantic=[]; available=[]
        for m in self.models:
            raw=m.model_dump(mode='json')
            semantic.append({k:v for k,v in raw.items() if k not in ('available','reason_code','availability_basis')})
            available.append(dict(id=m.id,available=m.available,reason_code=m.reason_code))
        if digest(semantic)!=self.catalog_digest or digest(available)!=self.availability_digest:
            raise ValueError('invalid catalog digest')
        return self


class FrozenSnapshot(CapabilitySnapshot):
    model_configs: dict[CanonicalID,dict[CanonicalID,Scalar]]

    @model_validator(mode='after')
    def frozen(self):
        models={m.id:m for m in self.models}
        if set(self.model_configs)-set(models):
            raise ValueError('unknown model config')
        for name, params in self.model_configs.items():
            if not models[name].available:
                raise ValueError('unavailable frozen model')
            validate_params(models[name],params,complete=True)
        return self


class EffectiveAction(v1.EffectiveAction):
    model_type: CanonicalID
    normalization: Literal['zscore']
    class_balance: Literal['none']
    parent_run_id: None
    model_params: dict[CanonicalID,Scalar]


class EffectiveConfig(v1.EffectiveConfig):
    model_type: CanonicalID
    normalization: Literal['zscore']
    class_balance: Literal['none']
    feature_selection_enabled: Literal[False]
    config_stage: Literal['submission']
    config_policy_version: Literal['agent-model-config-v1','agent-model-config-v2']
    config_policy_digest: v1.Digest
    model_params: dict[CanonicalID,Scalar]


class LockedConfig(v1.LockedConfig):
    metadata_version: Literal['agent-metadata-v2']
    allowed_models: Annotated[list[CanonicalID],Field(min_length=1)]
    max_runs: Literal[1]
    capability_snapshot: FrozenSnapshot

    @model_validator(mode='after')
    def consistent_models(self):
        if len(set(self.allowed_models))!=len(self.allowed_models) or set(self.allowed_models)!=set(self.capability_snapshot.model_configs):
            raise ValueError('frozen model set mismatch')
        return self


class HealthResponse(v1.HealthResponse, CapabilitySnapshot):
    contract_version: Literal['agent-session-v2']
    models: list[ModelCapability]


class CreatedSessionResponse(v1.CreatedSessionResponse):
    contract_version: Literal['agent-session-v2']
    locked_config: LockedConfig


class ExperimentResponse(v1.ExperimentResponse):
    contract_version: Literal['agent-session-v2']
    metadata_version: Literal['agent-metadata-v2']
    effective_config: EffectiveConfig | None = None
    effective_action: EffectiveAction


class ExperimentSummary(v1.ExperimentSummary):
    metadata_version: Literal['agent-metadata-v2']
    effective_config: EffectiveConfig | None = None
    effective_action: EffectiveAction


class SessionResponse(v1.SessionResponse):
    contract_version: Literal['agent-session-v2']
    locked_config: LockedConfig
    experiments: list[ExperimentSummary]


class ResolvedExecution(v1.ClosedModel):
    status: Literal['pending','unavailable','ready']
    parameters: dict[CanonicalID,Scalar | bool | list[int]] | None

    @model_validator(mode='after')
    def consistency(self):
        if (self.status=='ready')!=(self.parameters is not None):
            raise ValueError('invalid execution projection')
        numeric = {'pls_components','pca_components','logistic_c','svm_c','random_forest_n_estimators',
            'random_forest_max_depth','random_forest_min_samples_leaf','xgboost_n_estimators','xgboost_max_depth',
            'xgboost_learning_rate','xgboost_subsample','xgboost_colsample_bytree','xgboost_reg_lambda',
            'xgboost_min_child_weight','xgboost_gamma','dropout','n_components','heads','layers',
            'embedding_dim','hidden_size','kernel_size','pool_size','num_layers','num_heads'}
        arrays = {'channels','kernel_sizes','pool_sizes','hidden_sizes','dilations'}
        choices = {'execution_device':('cpu','cuda:0'), 'svm_kernel':('linear',), 'svm_gamma':('scale',),
            'random_forest_max_features':('sqrt','log2'), 'dscarnet_input_mode':('sar','car','dual'),
            'classification_head':('binary_single_logit','multiclass_logits'),
            'loss_function':('BCEWithLogitsLoss','CrossEntropyLoss')}
        for key,value in (self.parameters or {}).items():
            if key in numeric and type(value) in (int,float) and math.isfinite(value):
                continue
            if key in arrays and type(value) is list and len(value)<=16 and all(type(n) is int and n>0 for n in value):
                continue
            if key in choices and value in choices[key]:
                continue
            if key=='random_forest_oob_score' and type(value) is bool:
                continue
            raise ValueError('unknown or invalid resolved execution field')
        return self


class ObservationResponse(v1.ObservationResponse):
    contract_version: Literal['agent-session-v2']
    observation_version: Literal['agent-observation-v2']
    metadata_version: Literal['agent-metadata-v2']
    effective_config: EffectiveConfig | None = None
    effective_action: EffectiveAction
    resolved_execution: ResolvedExecution


class FinalizeResponse(v1.FinalizeResponse):
    contract_version: Literal['agent-session-v2']


class ReconcileResponse(v1.ReconcileResponse):
    contract_version: Literal['agent-session-v2']


RESPONSE_MODELS = dict(inspect_ml_capabilities=HealthResponse,start_ml_session=CreatedSessionResponse,
    inspect_ml_session=SessionResponse,submit_ml_experiment=ExperimentResponse,observe_ml_experiment=ObservationResponse,
    finalize_ml_session=FinalizeResponse,reconcile_ml_session=ReconcileResponse)


# Negotiated recipe profile wire codec. Legacy response models stay closed.

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


class KnowledgeSource(ClosedModel):
    source_id: Identifier
    description: str = Field(min_length=1,max_length=240)


class KnowledgeReason(ClosedModel):
    condition_id: Identifier
    field: Literal['task_type','feature_count','sample_group_count','repeated_measurement_group_count','class_count','risks','legal_models']
    actual: int | str | list[str] | None = None
    compared_group_count: int | None = None


class KnowledgeEntryProjection(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    related_recipe_ids: list[str]
    stance: Literal['consider','caution']
    advice: str = Field(min_length=1,max_length=180)
    basis: str = Field(min_length=1,max_length=240)
    limitations: str = Field(min_length=1,max_length=300)
    sources: list[KnowledgeSource] = Field(min_length=1,max_length=4)
    evidence: list[KnowledgeReason] = Field(min_length=1,max_length=8)
    conflict_group: Identifier | None
    conflict_notice: Literal['存在适用范围不同的建议'] | None


class KnowledgeProjection(ClosedModel):
    status: Literal['disabled','ready']
    projection_version: Literal['knowledge-projection-v1'] | None
    matched_count: int = Field(ge=0,le=32)
    provided_count: int = Field(ge=0,le=6)
    omitted_count: int = Field(ge=0,le=32)
    provided_entry_ids: list[Identifier]
    entries: list[KnowledgeEntryProjection] = Field(max_length=6)

    @model_validator(mode='after')
    def coherent(self):
        ids=[e.entry_id for e in self.entries]
        if (ids!=self.provided_entry_ids or len(set(ids))!=len(ids) or len(ids)!=self.provided_count
                or self.matched_count!=self.provided_count+self.omitted_count):
            raise ValueError('knowledge projection count mismatch')
        if self.status=='disabled' and (self.matched_count or self.projection_version is not None):
            raise ValueError('disabled knowledge has outputs')
        if self.status=='ready' and self.projection_version is None:
            raise ValueError('ready knowledge lacks version')
        if len(self.model_dump_json(exclude_unset=True).encode())>16384:
            raise ValueError('knowledge projection too large')
        return self


class KnowledgeMatchSummary(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    related_recipe_ids: list[str]


class KnowledgeProvenance(ClosedModel):
    dataset_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    plan_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    evidence_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    catalog_digest: str = Field(pattern=r'^[a-f0-9]{64}$')


class KnowledgeWire(ClosedModel):
    schema_version: Literal['knowledge-snapshot-v1']
    status: Literal['disabled','ready']
    knowledge_set_version: Identifier | None
    knowledge_set_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    matcher_version: Literal['knowledge-match-v1'] | None
    projection_version: Literal['knowledge-projection-v1'] | None
    semantic_input_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    match_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    matches: list[KnowledgeMatchSummary] = Field(max_length=32)
    projection: KnowledgeProjection
    projection_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    provenance: KnowledgeProvenance | None

    @model_validator(mode='after')
    def integrity(self):
        if self.projection_digest!=digest(self.projection.model_dump(mode='json',exclude_unset=True)):
            raise ValueError('knowledge projection digest mismatch')
        values=(self.knowledge_set_version,self.knowledge_set_digest,self.matcher_version,self.projection_version,
                self.semantic_input_digest,self.match_digest,self.provenance)
        if self.status=='disabled' and (any(v is not None for v in values) or self.matches):
            raise ValueError('disabled knowledge has bindings')
        if self.status=='ready' and any(v is None for v in values):
            raise ValueError('ready knowledge lacks bindings')
        matches={m.entry_id:m for m in self.matches}
        if len(matches)!=len(self.matches) or len(matches)!=self.projection.matched_count or self.status!=self.projection.status:
            raise ValueError('knowledge matches inconsistent')
        for entry in self.projection.entries:
            if entry.entry_id not in matches:
                raise ValueError('unmatched projection entry')
            match=matches[entry.entry_id]
            if entry.entry_version!=match.entry_version or entry.related_recipe_ids!=match.related_recipe_ids:
                raise ValueError('projection match binding mismatch')
        return self

    def validate_refs(self,refs):
        if len(refs)>6 or len(refs)!=len(set(refs)) or set(refs)-set(self.projection.provided_entry_ids):
            raise ValueError('knowledge reference not provided')


class KnowledgePreparation(Preparation):
    knowledge: KnowledgeWire

    @model_validator(mode='after')
    def knowledge_binding(self):
        k=self.knowledge
        if k.status=='ready':
            p=k.provenance
            if (p.dataset_sha256!=self.catalog.dataset_sha256 or p.catalog_digest!=self.catalog.catalog_digest
                    or p.plan_digest!=self.evaluation_plan.plan_digest or p.evidence_digest!=self.evidence.evidence_digest):
                raise ValueError('knowledge provenance mismatch')
            legal={r.recipe_id for r in self.catalog.recipes}
            if any(set(m.related_recipe_ids)-legal for m in k.matches):
                raise ValueError('knowledge outside legal recipes')
        return self


class KnowledgeLockedConfig(RecipeLockedConfig):
    protocol_revision: Literal['agent-recipes-revision-v2']
    preparation: KnowledgePreparation

    @field_validator('modules')
    @classmethod
    def known_modules(cls,value):
        if value not in (['train_evidence','legal_recipes'],['train_evidence','legal_recipes','knowledge']):
            raise ValueError('invalid knowledge modules')
        return value

    @model_validator(mode='after')
    def module_binding(self):
        if ('knowledge' in self.modules)!=(self.preparation.knowledge.status=='ready'):
            raise ValueError('knowledge switch mismatch')
        return self


class KnowledgeCreatedSessionResponse(CreatedSessionResponse):
    locked_config: KnowledgeLockedConfig


class KnowledgeSessionResponse(SessionResponse):
    locked_config: KnowledgeLockedConfig


class KnowledgeCapability(ClosedModel):
    available: Literal[True]
    status: Literal['ready']
    schema_version: Literal['knowledge-snapshot-v1']
    reason: str | None


class KnowledgeHealthResponse(HealthResponse):
    protocol_revision: Literal['agent-recipes-revision-v2']
    execution_profiles: list[Literal['train-evidence-recipes-v1']]
    modules: dict[str, v1.ModuleCapability | KnowledgeCapability]

    @field_validator('modules')
    @classmethod
    def known_modules(cls,value):
        from agent_poc.tools import MODULES
        if set(value)-set(MODULES)-{'knowledge'} or 'knowledge' not in value:
            raise ValueError('unknown module')
        if not isinstance(value['knowledge'],KnowledgeCapability):
            raise ValueError('knowledge capability missing')
        return value


class KnowledgeReference(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')


class KnowledgeDecision(ClosedModel):
    schema_version: Literal['knowledge-decision-v1']
    references: list[KnowledgeReference] = Field(max_length=6)
    projection_digest: str = Field(pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def unique_references(self):
        if len({r.entry_id for r in self.references})!=len(self.references):
            raise ValueError('duplicate knowledge decision reference')
        return self


class KnowledgeExperimentResponse(ExperimentResponse):
    decision_metadata: KnowledgeDecision


KNOWLEDGE_RESPONSE_MODELS = {**RECIPE_RESPONSE_MODELS,
    'inspect_ml_capabilities':KnowledgeHealthResponse,'start_ml_session':KnowledgeCreatedSessionResponse,
    'inspect_ml_session':KnowledgeSessionResponse,'submit_ml_experiment':KnowledgeExperimentResponse}
