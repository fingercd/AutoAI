"""Closed v2 wire types. No backend or training imports are permitted."""
from __future__ import annotations
from typing import Annotated, Literal
import hashlib
import json
import math
from pydantic import Field, model_validator, model_serializer
from . import contracts as v1

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
