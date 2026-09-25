"""Closed, independent wire models for the Agent HTTP boundary.

These models deliberately do not import backend repositories or trainers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

ModelName = Literal['logistic_regression', 'svm', 'random_forest']
MetricName = Literal['macro_f1', 'balanced_accuracy']
ToolName = Literal['inspect_ml_capabilities', 'start_ml_session', 'inspect_ml_session',
                   'submit_ml_experiment', 'observe_ml_experiment', 'finalize_ml_session']
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
NonNegative = Annotated[int, Field(ge=0)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
RunState = Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']
BindingState = Literal['reserved', 'bound', 'released', 'compensation_required']


def _timestamp(value: str) -> str:
    datetime.fromisoformat(value.replace('Z', '+00:00'))
    return value


Timestamp = Annotated[str, AfterValidator(_timestamp)]


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True,
                              allow_inf_nan=False)


class Versioned(ClosedModel):
    contract_version: Literal['agent-session-v1']


class Evaluation(ClosedModel):
    mode: Literal['stratified_holdout']
    train_weight: Annotated[int, Field(ge=1, le=9)]
    validation_weight: Annotated[int, Field(ge=1, le=9)]
    heldout_weight: Annotated[int, Field(ge=1, le=9)]

    @model_validator(mode='after')
    def check_weights(self):
        if self.train_weight + self.validation_weight + self.heldout_weight != 10:
            raise ValueError('invalid evaluation weights')
        return self


class ContextPolicy(ClosedModel):
    source_role: Literal['development', 'benchmark', 'domain']
    case_write: bool


class DatasetMetadata(ClosedModel):
    # Defaults support old v1 responses without claiming their data was frozen.
    metadata_version: Literal['agent-metadata-v1'] | None = None
    dataset_sha256: Digest | None = None
    dataset_fingerprint_status: Literal['ready', 'unavailable'] = 'unavailable'

    @model_validator(mode='after')
    def check_dataset(self):
        if self.dataset_fingerprint_status == 'ready':
            if self.metadata_version is None or self.dataset_sha256 is None:
                raise ValueError('incomplete dataset metadata')
        elif self.dataset_sha256 is not None:
            raise ValueError('unavailable dataset has fingerprint')
        return self


class EffectiveAction(ClosedModel):
    model_type: ModelName
    normalization: Literal['zscore', 'minmax', 'area', 'none']
    class_balance: Literal['none', 'class_weight']
    parent_run_id: Identifier | None


class EffectiveConfig(ClosedModel):
    model_type: ModelName
    normalization: Literal['zscore', 'minmax', 'area', 'none']
    class_balance: Literal['none', 'class_weight']
    seed: NonNegative
    feature_selection_enabled: bool
    evaluation_config: Evaluation


class ExperimentMetadata(DatasetMetadata):
    effective_config_status: Literal['ready', 'pending', 'unavailable'] = 'unavailable'
    effective_config: EffectiveConfig | None = None

    @model_validator(mode='after')
    def check_effective(self):
        if (self.effective_config_status == 'ready') != (self.effective_config is not None):
            raise ValueError('inconsistent effective config metadata')
        return self


class LockedConfig(DatasetMetadata):
    dataset_id: Identifier
    selection_metric: MetricName
    allowed_models: Annotated[list[ModelName], Field(min_length=1, max_length=3)]
    max_runs: Annotated[int, Field(ge=1, le=10)]
    seed: NonNegative
    evaluation_config: Evaluation
    modules: list[str]
    context_policy: ContextPolicy

    @field_validator('modules')
    @classmethod
    def known_modules(cls, value):
        from agent_poc.tools import MODULES
        if set(value) - set(MODULES):
            raise ValueError('unknown module')
        return value


class ModuleCapability(ClosedModel):
    available: Literal[False]
    status: Literal['unavailable']
    schema_version: None
    reason: Annotated[str, Field(max_length=2000)]


class Capabilities(ClosedModel):
    create_session: bool
    create_experiment: bool
    read_session: bool
    read_feedback: bool
    finalize_session: bool


class HealthResponse(Versioned):
    status: Literal['ready', 'unavailable', 'degraded']
    capabilities: Capabilities
    models: Annotated[list[ModelName], Field(min_length=1, max_length=3)]
    modules: dict[str, ModuleCapability]

    @field_validator('modules')
    @classmethod
    def known_modules(cls, value):
        from agent_poc.tools import MODULES
        if set(value) - set(MODULES):
            raise ValueError('unknown module')
        return value


class CreatedSessionResponse(Versioned):
    session_id: Identifier
    state: Literal['open', 'finalized']
    remaining_runs: NonNegative
    locked_config: LockedConfig
    created_at: Timestamp
    idempotent_replay: bool

    @field_validator('created_at')
    @classmethod
    def timestamp(cls, value):
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value


class ExperimentResponse(Versioned, ExperimentMetadata):
    session_id: Identifier
    run_id: Identifier
    attempt: Annotated[int, Field(ge=1)]
    config_hash: Annotated[str, Field(pattern=r'^[a-f0-9]{16}$')]
    state: RunState
    binding_state: BindingState
    effective_action: EffectiveAction
    idempotent_replay: bool


class ExperimentSummary(ExperimentMetadata):
    attempt: Annotated[int, Field(ge=1)]
    run_id: Identifier | None
    binding_state: BindingState
    parent_run_id: Identifier | None
    effective_action: EffectiveAction
    config_hash: Annotated[str, Field(pattern=r'^[a-f0-9]{16}$')]
    failure_code: str | None
    created_at: Timestamp
    state: Literal['queued', 'running', 'succeeded', 'failed', 'cancelled', 'binding_failed', 'missing']
    validation_score: Score | None


class SessionResponse(Versioned):
    session_id: Identifier
    state: Literal['open', 'finalized', 'terminated']
    locked_config: LockedConfig
    remaining_runs: NonNegative
    best_run_id: Identifier | None
    selected_run_id: Identifier | None
    created_at: Timestamp
    finalized_at: Timestamp | None
    experiments: list[ExperimentSummary]


class ValidationMetrics(ClosedModel):
    accuracy: Score | None = None
    balanced_accuracy: Score | None = None
    macro_precision: Score | None = None
    macro_recall: Score | None = None
    macro_f1: Score | None = None
    weighted_f1: Score | None = None


class Validation(ClosedModel):
    status: Literal['ready', 'pending', 'unavailable', 'failed']
    metrics: ValidationMetrics


class Progress(ClosedModel):
    stage: str | int | float | None = None
    phase: str | int | float | None = None
    epoch: str | int | float | None = None
    epochs: str | int | float | None = None
    percent: str | int | float | None = None
    message: str | int | float | None = None


class SafeError(ClosedModel):
    code: Annotated[str, Field(min_length=1, max_length=128)]
    message: Annotated[str, Field(max_length=2000)]
    retryable: bool


class ObservationResponse(Versioned, ExperimentMetadata):
    observation_version: Literal['agent-observation-v1']
    session_id: Identifier
    run_id: Identifier
    attempt: Annotated[int, Field(ge=1)]
    state: RunState
    effective_action: EffectiveAction
    selection_metric: MetricName
    progress: Progress
    validation: Validation
    validation_score: Score | None
    remaining_runs: NonNegative
    allowed_actions: list[ToolName]
    error: SafeError | None
    extensions: ClosedModel
    retry_after_seconds: Annotated[int, Field(ge=1, le=300)] | None = None

    @model_validator(mode='after')
    def coherent_validation(self):
        metrics = self.validation.metrics.model_dump(exclude_none=True)
        if self.validation.status == 'ready':
            if self.state != 'succeeded' or not metrics:
                raise ValueError('invalid ready observation')
            if self.validation_score != metrics.get(self.selection_metric):
                raise ValueError('inconsistent validation score')
        elif metrics or self.validation_score is not None:
            raise ValueError('unready observation contains scores')
        if self.state in ('queued', 'running') and self.retry_after_seconds is None:
            raise ValueError('waiting observation requires retry delay')
        return self


class FinalizeResponse(Versioned):
    session_id: Identifier
    state: Literal['finalized']
    selected_run_id: Identifier
    experiments_locked: Literal[True]
    final_result_url: Annotated[str, Field(max_length=512)]
    finalized_at: Timestamp

    @model_validator(mode='after')
    def result_link(self):
        if self.final_result_url != f'/#/results?run_id={self.selected_run_id}':
            raise ValueError('invalid result link')
        datetime.fromisoformat(self.finalized_at.replace('Z', '+00:00'))
        return self


class ErrorDetail(SafeError):
    allowed_actions: list[ToolName]


class ErrorResponse(ClosedModel):
    detail: ErrorDetail


class ReconcileSummary(ClosedModel):
    examined: NonNegative
    recovered_bound: NonNegative
    released: NonNegative
    unchanged: NonNegative
    manual_review: NonNegative


class ReconcileItem(ClosedModel):
    attempt: Annotated[int, Field(ge=1)]
    state: BindingState
    resolution_code: Literal['stale_without_run', 'recovered_binding', 'legacy_protocol_unknown',
                             'reservation_timestamp_invalid', 'cancelled_before_start',
                             'run_still_active', 'recovered_terminal_binding',
                             'recovered_cancelled_binding', 'submission_scope_mismatch',
                             'submission_mapping_invalid', 'mapped_run_unavailable',
                             'run_reference_missing', 'run_state_uncertain',
                             'cancellation_uncertain', 'reservation_fresh',
                             'concurrent_state_changed', 'cancellation_pending']
    requires_manual_review: bool


class ReconcileResponse(Versioned):
    session_id: Identifier
    status: Literal['unchanged', 'manual_review_required', 'reconciled']
    summary: ReconcileSummary
    items: list[ReconcileItem]


RESPONSE_MODELS = {
    'inspect_ml_capabilities': HealthResponse,
    'start_ml_session': CreatedSessionResponse,
    'inspect_ml_session': SessionResponse,
    'submit_ml_experiment': ExperimentResponse,
    'observe_ml_experiment': ObservationResponse,
    'finalize_ml_session': FinalizeResponse,
    'reconcile_ml_session': ReconcileResponse,
}


class DecisionModeBinding(ClosedModel):
    decision_mode: Literal['recipe_id', 'structured_config'] | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_mode_shape(self, handler):
        result = handler(self)
        if self.decision_mode is None:
            result.pop('decision_mode', None)
        return result


class KnowledgeQueryConfig(ClosedModel):
    query_mode: Literal['train_template', 'user_text'] = 'train_template'
    user_text: str | None = Field(default=None, min_length=1, max_length=4096)
    domain: Identifier | None = None

    @model_validator(mode='after')
    def explicit_text(self):
        if self.query_mode == 'user_text':
            if self.user_text is None or not self.user_text.strip():
                raise ValueError('user_text requires a nonempty description')
        elif self.user_text is not None:
            raise ValueError('template query cannot contain user_text')
        return self


class KnowledgeQueryBinding(DecisionModeBinding):
    knowledge_query: KnowledgeQueryConfig | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_query_shape(self, handler):
        result = handler(self)
        for key in ('decision_mode', 'knowledge_query'):
            if getattr(self, key) is None:
                result.pop(key, None)
        return result
