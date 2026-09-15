"""Current execution response codecs; legacy wire models remain in contracts."""
from __future__ import annotations
from typing import Annotated, Literal
import math
from pydantic import ConfigDict, Field, field_validator, model_validator
from . import contracts as v1
from .contracts import ClosedModel, Identifier
from .capabilities import CanonicalID, Scalar, FrozenSnapshot, CapabilitySnapshot, ModelCapability

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


