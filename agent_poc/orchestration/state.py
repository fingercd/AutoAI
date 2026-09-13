"""Strict, JSON-only checkpoint contract for the single-experiment graph.

The models define the complete future-facing state.  GraphState describes the
JSON serialization boundary; no Pydantic instance is sent to a checkpointer.
Only validated, whitelist-projected public data belongs in these structures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from typing import Annotated, Literal, TypeAlias, TypedDict, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


Availability = Literal['unavailable', 'disabled', 'pending', 'ready', 'invalid']
RunStatus = Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']
ValidationStatus = Literal['pending', 'ready', 'invalid', 'unavailable']
MetricName = Literal['macro_f1', 'balanced_accuracy']
ToolName = Literal[
    'inspect_ml_capabilities', 'start_ml_session', 'inspect_ml_session',
    'submit_ml_experiment', 'observe_ml_experiment', 'finalize_ml_session',
]
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list['JsonValue'] | dict[str, 'JsonValue']
JsonObject: TypeAlias = dict[str, JsonValue]


def _identifier(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value) or '..' in value:
        raise ValueError('invalid public identifier')
    return value


def validate_safe_text(value: str) -> str:
    if len(value) > 2000 or '\x00' in value:
        raise ValueError('invalid safe text')
    # These are boundary checks, not an attempt to redact arbitrary raw output.
    # Callers must construct safe text and references by whitelist projection.
    if re.search(r'(?i)(?:[a-z]:[\\/]|file://|https?://|(?<!\w)/(?:[^/\s]+/)+|'
                 r'traceback|authorization\s*[:=]|bearer\s+|api[_-]?key\s*[:=]|'
                 r'password\s*[:=]|token\s*[:=]|\btest\b|prediction|artifact)', value):
        raise ValueError('unsafe text is not checkpointable')
    return value


Identifier = Annotated[str, AfterValidator(_identifier)]
SafeText = Annotated[str, AfterValidator(validate_safe_text)]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
NonnegativeInt = Annotated[int, Field(ge=0)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True,
                              allow_inf_nan=False, hide_input_in_errors=True)


class EvaluationConfig(StrictModel):
    mode: Literal['stratified_holdout'] = 'stratified_holdout'
    train_weight: Annotated[int, Field(gt=0)] = 8
    validation_weight: Annotated[int, Field(gt=0)] = 1
    heldout_weight: Annotated[int, Field(gt=0)] = 1


class ContextPolicy(StrictModel):
    source_role: Literal['development', 'benchmark', 'domain'] = 'development'
    case_read: bool = False
    case_write: bool = False
    projection: Literal['agent-context-step1-v1'] = 'agent-context-step1-v1'


class VersionRef(StrictModel):
    status: Availability = 'unavailable'
    version: Identifier | None = None

    @model_validator(mode='after')
    def ready_version(self) -> 'VersionRef':
        if self.status == 'ready' and self.version is None:
            raise ValueError('ready version requires a value')
        return self


class IdentityState(StrictModel):
    task_id: Identifier
    thread_id: Identifier
    session_id: Identifier | None = None
    current_experiment_id: Identifier | None = None
    session_operation_id: Identifier | None = None
    session_request_id: Identifier | None = None
    experiment_operation_id: Identifier | None = None
    experiment_request_id: Identifier | None = None
    finalize_operation_id: Identifier | None = None
    startup_config_fingerprint: Digest
    backend_fingerprint: Digest
    principal_fingerprint: Digest
    runtime_config_fingerprint: Digest


class LifecycleState(StrictModel):
    status: Literal['initializing', 'running', 'waiting', 'recovering', 'completed',
                    'failed', 'cancelled', 'timed_out', 'needs_attention'] = 'initializing'
    stage: Identifier = 'initialize'
    next_action: Identifier | None = 'capabilities'
    started_at: Nonnegative
    ended_at: Nonnegative | None = None
    reason_code: Identifier | None = None


class TaskState(StrictModel):
    dataset_id: Identifier
    dataset_fingerprint: Digest | None = None
    dataset_fingerprint_status: Availability = 'pending'
    task_type: Literal['classification'] = 'classification'
    selection_metric: MetricName = 'macro_f1'
    allowed_models: Annotated[list[Identifier], Field(min_length=1)]
    seed: NonnegativeInt = 42
    evaluation_config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    evaluation_config_fingerprint: Digest
    split_fingerprint: Digest | None = None
    split_fingerprint_status: Availability = 'unavailable'
    normalization: Literal['zscore'] = 'zscore'
    class_balance: Literal['none'] = 'none'

    @model_validator(mode='after')
    def valid_fingerprints(self) -> 'TaskState':
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise ValueError('allowed models must be unique')
        for name in ('dataset', 'split'):
            value = getattr(self, f'{name}_fingerprint')
            status = getattr(self, f'{name}_fingerprint_status')
            if (status == 'ready') != (value is not None):
                raise ValueError('fingerprint availability is inconsistent')
        if self.evaluation_config_fingerprint != fingerprint(self.evaluation_config.model_dump(mode='json')):
            raise ValueError('evaluation configuration fingerprint mismatch')
        return self


class VersionsState(StrictModel):
    state: Literal['agent-state-v1'] = 'agent-state-v1'
    graph: Literal['agent-single-experiment-v1'] = 'agent-single-experiment-v1'
    api: Literal['agent-session-v1'] = 'agent-session-v1'
    observation: Literal['agent-observation-v1'] = 'agent-observation-v1'
    metadata: Literal['agent-metadata-v1'] = 'agent-metadata-v1'
    capabilities: VersionRef = Field(default_factory=lambda: VersionRef(status='ready', version='agent-session-v1'))
    evidence: VersionRef = Field(default_factory=VersionRef)
    recipes: VersionRef = Field(default_factory=VersionRef)
    search: VersionRef = Field(default_factory=VersionRef)
    rules: VersionRef = Field(default_factory=VersionRef)
    knowledge: VersionRef = Field(default_factory=VersionRef)
    memory: VersionRef = Field(default_factory=VersionRef)
    context_projection: Literal['agent-context-step1-v1'] = 'agent-context-step1-v1'
    prompt: Identifier = 'agent-decision-step1-v1'
    llm_config: Identifier = 'agent-llm-http-v1'
    llm_config_fingerprint: Digest


class ModuleSwitch(StrictModel):
    enabled: bool = False
    implementation_status: Literal['unavailable'] = 'unavailable'

    @model_validator(mode='after')
    def unavailable_disabled(self) -> 'ModuleSwitch':
        if self.enabled:
            raise ValueError('unavailable module cannot be enabled')
        return self


class ModulePolicyState(StrictModel):
    evidence_card: ModuleSwitch = Field(default_factory=ModuleSwitch)
    dynamic_preprocessing: ModuleSwitch = Field(default_factory=ModuleSwitch)
    restricted_strategy_pool: ModuleSwitch = Field(default_factory=ModuleSwitch)
    bounded_hpo: ModuleSwitch = Field(default_factory=ModuleSwitch)
    fail_fast_guard: ModuleSwitch = Field(default_factory=ModuleSwitch)
    feedback_diagnosis: ModuleSwitch = Field(default_factory=ModuleSwitch)
    limited_replanning: ModuleSwitch = Field(default_factory=ModuleSwitch)
    uncertainty_selection: ModuleSwitch = Field(default_factory=ModuleSwitch)
    case_memory: ModuleSwitch = Field(default_factory=ModuleSwitch)
    budget_control: ModuleSwitch = Field(default_factory=ModuleSwitch)
    constrained_code_evolution: ModuleSwitch = Field(default_factory=ModuleSwitch)
    knowledge: ModuleSwitch = Field(default_factory=ModuleSwitch)
    ablation_id: Identifier | None = None
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)

    @model_validator(mode='after')
    def no_case_access(self) -> 'ModulePolicyState':
        if self.context_policy.case_read or self.context_policy.case_write:
            raise ValueError('case memory is unavailable')
        return self


class ParameterCapability(StrictModel):
    name: Identifier
    value_type: Literal['integer', 'number', 'string', 'boolean']
    required: bool = False
    choices: list[SafeText | int | float | bool] = Field(default_factory=list)
    minimum: Finite | None = None
    maximum: Finite | None = None
    condition_refs: list[Identifier] = Field(default_factory=list)


class CompatibilityRule(StrictModel):
    rule_id: Identifier
    version: Identifier
    model_types: list[Identifier] = Field(default_factory=list)
    normalizations: list[Identifier] = Field(default_factory=list)
    class_balances: list[Identifier] = Field(default_factory=list)
    allowed: bool
    reason_code: Identifier


class ModelCapability(StrictModel):
    model_type: Identifier
    available: bool
    status: Availability = 'ready'
    parameter_status: Availability = 'unavailable'
    parameters: list[ParameterCapability] = Field(default_factory=list)
    compatibility_status: Availability = 'unavailable'
    compatibility: list[CompatibilityRule] = Field(default_factory=list)

    @model_validator(mode='after')
    def available_details(self) -> 'ModelCapability':
        if self.parameters and self.parameter_status != 'ready':
            raise ValueError('parameter details require ready capability metadata')
        if self.compatibility and self.compatibility_status != 'ready':
            raise ValueError('compatibility rules require ready capability metadata')
        return self


class CapabilityItem(StrictModel):
    name: Identifier
    available: bool
    status: Availability
    version: Identifier | None = None


class CapabilitiesState(StrictModel):
    status: Availability = 'pending'
    source: Literal['agent_api'] = 'agent_api'
    snapshot_fingerprint: Digest | None = None
    observed_at: Nonnegative | None = None
    models: list[ModelCapability] = Field(default_factory=list)
    eligible_models: list[Identifier] = Field(default_factory=list)
    tools: list[CapabilityItem] = Field(default_factory=list)
    modules: list[CapabilityItem] = Field(default_factory=list)


class EvidenceReference(StrictModel):
    reference_id: Identifier
    kind: Literal['train_statistic', 'risk', 'rule', 'knowledge', 'case', 'observation',
                  'guard', 'decision', 'event', 'recipe', 'search', 'config']
    version: Identifier | None = None


class ClassStatistic(StrictModel):
    class_id: Identifier
    observation_count: NonnegativeInt | None = None
    sample_group_count: NonnegativeInt | None = None
    fraction: Score | None = None


class TrainStatistics(StrictModel):
    status: Availability = 'unavailable'
    observation_count: NonnegativeInt | None = None
    sample_group_count: NonnegativeInt | None = None
    feature_count: NonnegativeInt | None = None
    class_count: NonnegativeInt | None = None
    classes: list[ClassStatistic] = Field(default_factory=list)
    repeated_group_count: NonnegativeInt | None = None
    zero_variance_feature_count: NonnegativeInt | None = None
    duplicate_feature_count: NonnegativeInt | None = None
    conflicting_feature_count: NonnegativeInt | None = None


class RiskItem(StrictModel):
    risk_id: Identifier
    kind: Literal['small_sample', 'high_dimension', 'imbalance', 'repeated_measurement',
                  'zero_variance', 'duplicate_features', 'conflicting_features']
    severity: Literal['informational', 'warning', 'blocking']
    algorithm_version: Identifier
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    summary: SafeText


class EvidenceState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    algorithm_version: Identifier | None = None
    validity: Availability = 'unavailable'
    statistics: TrainStatistics = Field(default_factory=TrainStatistics)
    risks: list[RiskItem] = Field(default_factory=list)
    summary: SafeText | None = None
    references: list[EvidenceReference] = Field(default_factory=list)


class RecipeReference(StrictModel):
    recipe_id: Identifier
    version: Identifier
    digest: Digest
    model_type: Identifier
    preprocessing_ref: Identifier | None = None
    search_config_ref: Identifier | None = None
    compatibility_refs: list[Identifier] = Field(default_factory=list)


class RecipeUse(StrictModel):
    use_id: Identifier
    recipe_id: Identifier
    decision_id: Identifier
    experiment_id: Identifier


class RecipesState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    selection_mode: Literal['direct_action', 'recipe_selection'] = 'direct_action'
    catalog_reference: Identifier | None = None
    catalog_version: Identifier | None = None
    catalog_digest: Digest | None = None
    legal_recipes: list[RecipeReference] = Field(default_factory=list)
    uses: list[RecipeUse] = Field(default_factory=list)


class ApplicabilityCondition(StrictModel):
    condition_id: Identifier
    field: Literal['task_type', 'model_type', 'feature_count', 'sample_group_count',
                   'class_count', 'risk_kind', 'source_role']
    operator: Literal['eq', 'in', 'lt', 'le', 'gt', 'ge', 'present']
    values: list[SafeText | int | float | bool] = Field(default_factory=list)


class KnowledgeMatch(StrictModel):
    entry_id: Identifier
    version: Identifier
    source_reference: Identifier
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    conditions: list[ApplicabilityCondition] = Field(default_factory=list)
    rationale: SafeText


class KnowledgeState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    prior_version: Identifier | None = None
    matches: list[KnowledgeMatch] = Field(default_factory=list)


class CaseRecall(StrictModel):
    case_id: Identifier
    version: Identifier
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    rationale: SafeText


class MemoryState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    snapshot_version: Identifier | None = None
    snapshot_digest: Digest | None = None
    recalls: list[CaseRecall] = Field(default_factory=list)
    publication_operation_id: Identifier | None = None
    published_case_id: Identifier | None = None
    write_status: Availability = 'disabled'


class DirectAction(StrictModel):
    model_type: Identifier
    normalization: Literal['zscore', 'minmax', 'area', 'none'] = 'zscore'
    class_balance: Literal['none', 'class_weight'] = 'none'


class DecisionState(StrictModel):
    status: Availability = 'pending'
    decision_id: Identifier | None = None
    kind: Literal['direct_action', 'recipe_selection', 'search_selection', 'finalize'] | None = None
    action: DirectAction | None = None
    selected_run_id: Identifier | None = None
    recipe_id: Identifier | None = None
    search_id: Identifier | None = None
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    rationale: SafeText | None = None
    validation_status: Availability = 'pending'
    reason_code: Identifier | None = None
    tool_name: ToolName | None = None
    tool_call_id: Identifier | None = None
    response_id: Identifier | None = None


class EffectiveConfig(StrictModel):
    model_type: Identifier
    normalization: Literal['zscore', 'minmax', 'area', 'none']
    class_balance: Literal['none', 'class_weight']
    seed: NonnegativeInt
    feature_selection_enabled: bool
    evaluation_config: EvaluationConfig


class SafeProgress(StrictModel):
    stage: Identifier | None = None
    percent: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] | None = None
    current: NonnegativeInt | None = None
    total: NonnegativeInt | None = None


class SessionRequest(StrictModel):
    dataset_id: Identifier
    selection_metric: MetricName
    allowed_models: Annotated[list[Identifier], Field(min_length=1)]
    max_runs: Annotated[int, Field(ge=1)] = 1
    seed: NonnegativeInt
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    modules: list[Identifier] = Field(default_factory=list)
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    client_request_id: Identifier


class ExperimentRequest(StrictModel):
    session_id: Identifier
    model_type: Identifier
    normalization: Literal['zscore', 'minmax', 'area', 'none'] = 'zscore'
    class_balance: Literal['none', 'class_weight'] = 'none'
    rationale: SafeText
    client_request_id: Identifier


class FinalizeRequest(StrictModel):
    session_id: Identifier
    selected_run_id: Identifier


class ExecutionState(StrictModel):
    experiment_id: Identifier | None = None
    purpose: Literal['initial', 'recovery', 'improvement', 'supplemental_validation'] = 'initial'
    reservation_id: Identifier | None = None
    reservation_status: Availability = 'unavailable'
    submission_key: Identifier | None = None
    submission_key_status: Availability = 'unavailable'
    submission_request_id: Identifier | None = None
    submission_content: ExperimentRequest | None = None
    submission_fingerprint: Digest | None = None
    run_id: Identifier | None = None
    effective_action: DirectAction | None = None
    effective_config: EffectiveConfig | None = None
    effective_config_status: Availability = 'pending'
    metadata_version: Literal['agent-metadata-v1'] = 'agent-metadata-v1'
    dataset_fingerprint: Digest | None = None
    dataset_fingerprint_status: Availability = 'pending'
    progress: SafeProgress = Field(default_factory=SafeProgress)
    run_status: RunStatus | None = None
    terminal_reason_code: Identifier | None = None

    @model_validator(mode='after')
    def available_metadata(self) -> 'ExecutionState':
        if (self.effective_config_status == 'ready') != (self.effective_config is not None):
            raise ValueError('effective configuration availability is inconsistent')
        if (self.dataset_fingerprint_status == 'ready') != (self.dataset_fingerprint is not None):
            raise ValueError('Run dataset fingerprint availability is inconsistent')
        if self.reservation_id is not None or self.submission_key is not None:
            raise ValueError('internal reservation and submission keys are not exposed by the current API')
        if self.reservation_status != 'unavailable' or self.submission_key_status != 'unavailable':
            raise ValueError('internal references remain unavailable in the current API')
        return self


class BudgetDimension(StrictModel):
    limit: Nonnegative | None = None
    reserved: Nonnegative | None = None
    actual: Nonnegative | None = None
    unknown_pending: NonnegativeInt = 0
    remaining: Nonnegative | None = None
    unit: Literal['runs', 'calls', 'tokens', 'seconds', 'fits', 'epochs']
    source: Literal['backend_remaining_runs', 'local_call_journal', 'provider_usage',
                    'wall_clock', 'unavailable']
    measurement_status: Availability = 'unavailable'


class UsageRecord(StrictModel):
    usage_id: Identifier
    operation_id: Identifier
    kind: Literal['llm', 'api']
    status: Literal['pending', 'confirmed', 'unknown']
    calls: Annotated[int, Field(ge=1)] = 1
    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None
    cached_tokens: NonnegativeInt | None = None
    token_status: Literal['pending', 'known', 'partial', 'unknown', 'not_applicable'] = 'unknown'
    response_id: Identifier | None = None

    @model_validator(mode='after')
    def known_token_usage(self) -> 'UsageRecord':
        if self.token_status == 'known' and (self.input_tokens is None or self.output_tokens is None):
            raise ValueError('known token usage requires measured input and output counts')
        if self.token_status == 'not_applicable' and any(
                value is not None for value in (self.input_tokens, self.output_tokens, self.total_tokens, self.cached_tokens)):
            raise ValueError('non-token operation cannot carry token usage')
        return self


def _local_calls(limit: int | None = None) -> BudgetDimension:
    return BudgetDimension(limit=limit, actual=0, reserved=0, remaining=limit,
                           unit='calls', source='local_call_journal', measurement_status='ready')


class BudgetState(StrictModel):
    control_status: Literal['disabled'] = 'disabled'
    deadline_at: Nonnegative
    max_operation_attempts: Annotated[int, Field(ge=1)] = 3
    max_repair_attempts: NonnegativeInt = 2
    runs: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        limit=1, unit='runs', source='backend_remaining_runs', measurement_status='pending'))
    llm_calls: BudgetDimension = Field(default_factory=lambda: _local_calls(6))
    api_calls: BudgetDimension = Field(default_factory=lambda: _local_calls(60))
    input_tokens: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='tokens', source='provider_usage', measurement_status='pending'))
    output_tokens: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='tokens', source='provider_usage', measurement_status='pending'))
    cached_tokens: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='tokens', source='provider_usage', measurement_status='pending'))
    wall_time: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='seconds', source='wall_clock', measurement_status='pending'))
    model_fits: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='fits', source='unavailable'))
    training_epochs: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='epochs', source='unavailable'))
    resource_seconds: BudgetDimension = Field(default_factory=lambda: BudgetDimension(
        unit='seconds', source='unavailable'))
    usage: list[UsageRecord] = Field(default_factory=list)


class ValidationMetrics(StrictModel):
    accuracy: Score | None = None
    balanced_accuracy: Score | None = None
    macro_precision: Score | None = None
    macro_recall: Score | None = None
    macro_f1: Score | None = None
    weighted_f1: Score | None = None


class SafeError(StrictModel):
    code: Identifier
    retryable: bool = False


class DiagnosticSummary(StrictModel):
    status: Availability = 'unavailable'
    source_version: Identifier | None = None
    signal_codes: list[Identifier] = Field(default_factory=list)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


class CostSummary(StrictModel):
    status: Availability = 'unavailable'
    model_fits: NonnegativeInt | None = None
    epochs: NonnegativeInt | None = None
    resource_seconds: Nonnegative | None = None
    source_version: Identifier | None = None


class FeedbackState(StrictModel):
    status: Availability = 'pending'
    observation_version: Literal['agent-observation-v1'] = 'agent-observation-v1'
    run_status: RunStatus | None = None
    validation_status: ValidationStatus = 'pending'
    validation_metrics: ValidationMetrics = Field(default_factory=ValidationMetrics)
    selection_score: Score | None = None
    error: SafeError | None = None
    progress: SafeProgress = Field(default_factory=SafeProgress)
    allowed_actions: list[Identifier] = Field(default_factory=list)
    integrity: Availability = 'pending'
    retry_after_seconds: Nonnegative | None = None
    diagnostic_summary: DiagnosticSummary = Field(default_factory=DiagnosticSummary)
    cost: CostSummary = Field(default_factory=CostSummary)


class GuardCheck(StrictModel):
    check_id: Identifier
    phase: Literal['pre', 'post']
    kind: Literal['contract', 'action', 'manifest', 'selection_metric', 'scope', 'config']
    version: Identifier
    status: Literal['passed', 'failed', 'pending']
    reason_code: Identifier | None = None
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


class GuardState(StrictModel):
    research_guard_status: Literal['disabled'] = 'disabled'
    checks: list[GuardCheck] = Field(default_factory=list)
    candidate_eligible: bool | None = None
    reason_codes: list[Identifier] = Field(default_factory=list)


class DiagnosisFact(StrictModel):
    fact_id: Identifier
    summary: SafeText
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


class DiagnosisHypothesis(StrictModel):
    hypothesis_id: Identifier
    summary: SafeText
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    uncertainty: Literal['unknown', 'low', 'medium', 'high'] = 'unknown'
    limitations: list[SafeText] = Field(default_factory=list)


class AdjustmentSuggestion(StrictModel):
    suggestion_id: Identifier
    change_kind: Literal['model', 'preprocessing', 'class_balance', 'search', 'stop']
    target_reference: Identifier | None = None
    rationale: SafeText
    condition_refs: list[Identifier] = Field(default_factory=list)


class DiagnosisState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    facts: list[DiagnosisFact] = Field(default_factory=list)
    problem_type: Literal['dependency', 'data', 'configuration', 'training_failure',
                          'underfitting', 'overfitting', 'unstable', 'resource_limit', 'unknown'] | None = None
    hypotheses: list[DiagnosisHypothesis] = Field(default_factory=list)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    uncertainty: Literal['unknown', 'low', 'medium', 'high'] = 'unknown'
    suggestions: list[AdjustmentSuggestion] = Field(default_factory=list)


class ReplanningChange(StrictModel):
    change_id: Identifier
    field: Literal['model_type', 'normalization', 'class_balance', 'recipe_id', 'search_id']
    previous: Identifier | None = None
    proposed: Identifier


class ReplanningState(StrictModel):
    status: Availability = 'disabled'
    implementation_status: Literal['unavailable'] = 'unavailable'
    trigger: Literal['failure_recovery', 'performance_improvement', 'uncertainty', 'budget'] | None = None
    parent_experiment_id: Identifier | None = None
    child_experiment_id: Identifier | None = None
    changes: list[ReplanningChange] = Field(default_factory=list)
    max_count: NonnegativeInt = 0
    used_count: NonnegativeInt = 0
    max_depth: NonnegativeInt = 0
    used_depth: NonnegativeInt = 0
    stop_reason: Identifier | None = None


class UncertaintyEstimate(StrictModel):
    status: Availability = 'unavailable'
    method_version: Identifier | None = None
    estimate: Nonnegative | None = None
    lower_bound: Finite | None = None
    upper_bound: Finite | None = None
    confidence_level: Score | None = None


class Candidate(StrictModel):
    candidate_id: Identifier
    session_id: Identifier
    run_id: Identifier
    experiment_id: Identifier
    status: Literal['valid', 'invalid', 'pending']
    validation_metrics: ValidationMetrics
    selection_score: Score | None = None
    uncertainty: UncertaintyEstimate = Field(default_factory=UncertaintyEstimate)
    cost: CostSummary = Field(default_factory=CostSummary)
    reason_code: Identifier | None = None


class Recommendation(StrictModel):
    recommendation_id: Identifier
    decision_id: Identifier
    run_id: Identifier
    rationale: SafeText


class CandidatesState(StrictModel):
    status: Availability = 'pending'
    items: list[Candidate] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


class HistoryEvent(StrictModel):
    event_id: Identifier
    kind: Identifier
    sequence: NonnegativeInt
    occurred_at: Nonnegative
    decision_id: Identifier | None = None
    operation_id: Identifier | None = None
    experiment_id: Identifier | None = None
    run_id: Identifier | None = None
    reason_code: Identifier | None = None
    safe_record_ref: Identifier | None = None


class HistoryState(StrictModel):
    events: list[HistoryEvent] = Field(default_factory=list)


class PendingOperation(StrictModel):
    operation_id: Identifier
    kind: Literal['session', 'experiment', 'finalize', 'reconcile', 'inspect', 'observe', 'llm']
    request_id: Identifier | None = None
    tool_name: ToolName | None = None
    content: SessionRequest | ExperimentRequest | FinalizeRequest | None = None
    content_fingerprint: Digest | None = None
    status: Literal['prepared', 'in_flight', 'unknown', 'confirmed', 'failed'] = 'prepared'

    @model_validator(mode='after')
    def content_matches(self) -> 'PendingOperation':
        types = {'session': SessionRequest, 'experiment': ExperimentRequest, 'finalize': FinalizeRequest}
        tools = {'session': 'start_ml_session', 'experiment': 'submit_ml_experiment',
                 'finalize': 'finalize_ml_session'}
        if self.kind in types:
            if not isinstance(self.content, types[self.kind]):
                raise ValueError('operation content does not match its kind')
            if self.content_fingerprint != fingerprint(self.content.model_dump(mode='json')):
                raise ValueError('operation content fingerprint mismatch')
            if self.tool_name != tools[self.kind]:
                raise ValueError('operation tool does not match its kind')
            if self.kind in ('session', 'experiment') and self.request_id != self.content.client_request_id:
                raise ValueError('operation request identifier does not match content')
        elif self.content is not None:
            raise ValueError('read and reconciliation operations do not carry mutation content')
        return self


class OperationAttempts(StrictModel):
    operation_id: Identifier
    network_attempts: NonnegativeInt = 0
    repair_attempts: NonnegativeInt = 0
    reconciliation_attempts: NonnegativeInt = 0
    last_error_code: Identifier | None = None


class ConfirmedBackendState(StrictModel):
    session_state: Literal['open', 'finalized'] | None = None
    run_status: RunStatus | None = None
    selected_run_id: Identifier | None = None
    remaining_runs: NonnegativeInt | None = None
    confirmed_at: Nonnegative | None = None


class ReconciliationResult(StrictModel):
    status: Availability = 'pending'
    outcome: Literal['bound', 'released', 'compensation_required', 'pending', 'unavailable'] | None = None
    resolution_code: Identifier | None = None
    checked_at: Nonnegative | None = None


class RecoveryState(StrictModel):
    attempts: list[OperationAttempts] = Field(default_factory=list)
    pending_operation: PendingOperation | None = None
    last_confirmed_backend_state: ConfirmedBackendState = Field(default_factory=ConfirmedBackendState)
    next_wake_at: Nonnegative | None = None
    reconciliation: ReconciliationResult = Field(default_factory=ReconciliationResult)
    needs_human_review: bool = False
    reason_code: Identifier | None = None


class FinalizationState(StrictModel):
    selected_run_id: Identifier | None = None
    rationale: SafeText | None = None
    status: Literal['pending', 'confirmed', 'unselected'] = 'pending'
    backend_session_state: Literal['open', 'finalized'] | None = None
    locked_at: Nonnegative | None = None
    termination_reason: Identifier | None = None

    @model_validator(mode='after')
    def confirmed_lock(self) -> 'FinalizationState':
        if self.status == 'confirmed':
            if self.backend_session_state != 'finalized' or self.selected_run_id is None or self.locked_at is None:
                raise ValueError('confirmation requires a backend lock')
        elif self.locked_at is not None:
            raise ValueError('only a confirmed backend lock has a timestamp')
        if self.status == 'unselected' and self.selected_run_id is not None:
            raise ValueError('unselected finalization cannot name a Run')
        return self


class StateModel(StrictModel):
    identity: IdentityState
    lifecycle: LifecycleState
    task: TaskState
    versions: VersionsState
    module_policy: ModulePolicyState = Field(default_factory=ModulePolicyState)
    capabilities: CapabilitiesState = Field(default_factory=CapabilitiesState)
    evidence: EvidenceState = Field(default_factory=EvidenceState)
    recipes: RecipesState = Field(default_factory=RecipesState)
    knowledge: KnowledgeState = Field(default_factory=KnowledgeState)
    memory: MemoryState = Field(default_factory=MemoryState)
    decision: DecisionState = Field(default_factory=DecisionState)
    execution: ExecutionState = Field(default_factory=ExecutionState)
    budget: BudgetState
    feedback: FeedbackState = Field(default_factory=FeedbackState)
    guard: GuardState = Field(default_factory=GuardState)
    diagnosis: DiagnosisState = Field(default_factory=DiagnosisState)
    replanning: ReplanningState = Field(default_factory=ReplanningState)
    candidates: CandidatesState = Field(default_factory=CandidatesState)
    history: HistoryState = Field(default_factory=HistoryState)
    recovery: RecoveryState = Field(default_factory=RecoveryState)
    finalization: FinalizationState = Field(default_factory=FinalizationState)

    @model_validator(mode='after')
    def consistent_projections(self) -> 'StateModel':
        frozen_payload = _startup_payload(
            self.task.model_dump(mode='json'), self.versions.model_dump(mode='json'),
            self.module_policy.model_dump(mode='json'), self.budget.model_dump(mode='json'),
            self.identity.backend_fingerprint, self.identity.principal_fingerprint,
            self.identity.runtime_config_fingerprint)
        if self.identity.startup_config_fingerprint != fingerprint(frozen_payload):
            raise ValueError('frozen startup configuration fingerprint mismatch')
        if not set(self.capabilities.eligible_models).issubset(self.task.allowed_models):
            raise ValueError('eligible models exceed frozen user permission')
        if self.capabilities.status == 'ready':
            available = {item.model_type for item in self.capabilities.models if item.available and item.status == 'ready'}
            if not set(self.capabilities.eligible_models).issubset(available):
                raise ValueError('eligible models exceed Agent server capability')
        if self.execution.dataset_fingerprint is not None and self.execution.dataset_fingerprint != self.task.dataset_fingerprint:
            raise ValueError('Run dataset fingerprint differs from frozen dataset')
        config = self.execution.effective_config
        if config is not None:
            if (config.seed != self.task.seed or config.evaluation_config != self.task.evaluation_config
                    or config.normalization != self.task.normalization or config.class_balance != self.task.class_balance):
                raise ValueError('effective configuration differs from frozen task')
            if config.model_type not in self.capabilities.eligible_models:
                raise ValueError('effective model is outside Agent candidate capability')
            if self.execution.effective_action is not None:
                action = self.execution.effective_action
                if (config.model_type, config.normalization, config.class_balance) != (
                        action.model_type, action.normalization, action.class_balance):
                    raise ValueError('effective action differs from effective configuration')
        decision = self.decision
        if decision.validation_status == 'ready' and decision.kind == 'direct_action':
            if decision.action is None or decision.action.model_type not in self.capabilities.eligible_models:
                raise ValueError('validated decision is outside Agent candidate capability')
            if (decision.action.normalization != self.task.normalization
                    or decision.action.class_balance != self.task.class_balance):
                raise ValueError('validated decision changes frozen processing')
        if self.execution.submission_content is not None:
            content = self.execution.submission_content
            if content.session_id != self.identity.session_id or content.client_request_id != self.execution.submission_request_id:
                raise ValueError('submission identity mismatch')
            if self.execution.submission_fingerprint != fingerprint(content.model_dump(mode='json')):
                raise ValueError('submission content fingerprint mismatch')
        if self.budget.runs.limit != 1:
            raise ValueError('first-step protocol permits one scientific experiment')
        if len(self.candidates.items) > 1:
            raise ValueError('single-experiment protocol has at most one candidate')
        for candidate in self.candidates.items:
            if candidate.session_id != self.identity.session_id or candidate.run_id != self.execution.run_id:
                raise ValueError('candidate is outside the current scientific experiment')
            if candidate.status == 'valid' and candidate.selection_score is None:
                raise ValueError('valid candidate requires a finite selection score')
        if self.finalization.status == 'confirmed':
            backend = self.recovery.last_confirmed_backend_state
            if backend.session_state != 'finalized' or backend.selected_run_id != self.finalization.selected_run_id:
                raise ValueError('finalization needs a matching backend inspection')
        if self.feedback.validation_status == 'ready':
            if self.feedback.run_status != 'succeeded' or self.feedback.integrity != 'ready':
                raise ValueError('ready validation requires a complete successful Run')
        if self.feedback.selection_score is not None:
            if self.feedback.validation_status != 'ready':
                raise ValueError('selection score requires ready validation')
            if self.feedback.selection_score != getattr(self.feedback.validation_metrics, self.task.selection_metric):
                raise ValueError('selection score does not match the frozen metric')
        for name in ('evidence', 'recipes', 'knowledge', 'memory', 'diagnosis', 'replanning'):
            if getattr(self, name).status != 'disabled':
                raise ValueError('unavailable research module cannot publish an enabled status')
        if (self.evidence.algorithm_version is not None or self.evidence.risks or self.evidence.summary is not None
                or self.evidence.references or self.evidence.validity != 'unavailable'
                or self.evidence.statistics != TrainStatistics()):
            raise ValueError('disabled evidence cannot publish fabricated outputs')
        if (self.recipes.catalog_reference is not None or self.recipes.catalog_version is not None
                or self.recipes.catalog_digest is not None or self.recipes.legal_recipes or self.recipes.uses
                or self.recipes.selection_mode != 'direct_action'):
            raise ValueError('disabled recipes cannot publish fabricated outputs')
        if self.knowledge.prior_version is not None or self.knowledge.matches:
            raise ValueError('disabled knowledge cannot publish fabricated outputs')
        if (self.memory.snapshot_version is not None or self.memory.snapshot_digest is not None
                or self.memory.recalls or self.memory.publication_operation_id is not None
                or self.memory.published_case_id is not None or self.memory.write_status != 'disabled'):
            raise ValueError('disabled memory cannot publish fabricated outputs')
        if (self.diagnosis.facts or self.diagnosis.problem_type is not None or self.diagnosis.hypotheses
                or self.diagnosis.evidence_refs or self.diagnosis.suggestions or self.diagnosis.uncertainty != 'unknown'):
            raise ValueError('disabled diagnosis cannot publish fabricated outputs')
        if (self.replanning.trigger is not None or self.replanning.parent_experiment_id is not None
                or self.replanning.child_experiment_id is not None or self.replanning.changes
                or any((self.replanning.max_count, self.replanning.used_count,
                        self.replanning.max_depth, self.replanning.used_depth))):
            raise ValueError('disabled replanning cannot create scientific retries')
        _validate_unique_records(self.model_dump(mode='json'))
        return self


class GraphState(TypedDict):
    """JSON checkpoint channels; each channel is validated by its named model.

    JsonObject is deliberately a primitive-only recursive type, never Any.
    StateModel supplies the full field-level schema and forbids extra fields.
    """

    identity: JsonObject
    lifecycle: JsonObject
    task: JsonObject
    versions: JsonObject
    module_policy: JsonObject
    capabilities: JsonObject
    evidence: JsonObject
    recipes: JsonObject
    knowledge: JsonObject
    memory: JsonObject
    decision: JsonObject
    execution: JsonObject
    budget: JsonObject
    feedback: JsonObject
    guard: JsonObject
    diagnosis: JsonObject
    replanning: JsonObject
    candidates: JsonObject
    history: JsonObject
    recovery: JsonObject
    finalization: JsonObject


def fingerprint(value: object) -> str:
    """Hash a finite canonical JSON value; never serialize arbitrary objects."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def new_state(*, dataset_id: str, allowed_models: list[str], backend_fingerprint: str,
              principal_fingerprint: str, llm_config_fingerprint: str,
              runtime_config_fingerprint: str | None = None,
              task_id: str | None = None, thread_id: str | None = None,
              seed: int = 42, selection_metric: MetricName = 'macro_f1',
              max_llm_calls: int = 6, max_api_calls: int = 60,
              max_operation_attempts: int = 3, max_repair_attempts: int = 2,
              timeout_seconds: float = 3600.0, now: float | None = None,
              prompt_version: str = 'agent-decision-step1-v1',
              llm_config_version: str = 'agent-llm-http-v1',
              source_role: Literal['development', 'benchmark', 'domain'] = 'development') -> GraphState:
    """Create a complete empty state without fabricating unavailable outputs."""
    started_at = time.time() if now is None else now
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout must be positive and finite')
    for count in (max_llm_calls, max_api_calls, max_operation_attempts):
        if type(count) is not int or count < 1:
            raise ValueError('runtime call limits must be positive integers')
    evaluation = EvaluationConfig()
    task = TaskState(dataset_id=dataset_id, allowed_models=allowed_models, seed=seed,
                     selection_metric=selection_metric,
                     evaluation_config_fingerprint=fingerprint(evaluation.model_dump(mode='json')))
    versions = VersionsState(prompt=prompt_version, llm_config=llm_config_version,
                             llm_config_fingerprint=llm_config_fingerprint)
    policy = ModulePolicyState(context_policy=ContextPolicy(source_role=source_role))
    budget = BudgetState(deadline_at=started_at + timeout_seconds,
                         max_operation_attempts=max_operation_attempts,
                         max_repair_attempts=max_repair_attempts,
                         llm_calls=_local_calls(max_llm_calls), api_calls=_local_calls(max_api_calls))
    budget.wall_time.limit = timeout_seconds
    frozen_config = _startup_payload(task.model_dump(mode='json'), versions.model_dump(mode='json'),
                                     policy.model_dump(mode='json'), budget.model_dump(mode='json'),
                                     backend_fingerprint, principal_fingerprint,
                                     runtime_config_fingerprint or fingerprint({'api_timeout': 10.0}))
    state = StateModel(identity=IdentityState(task_id=task_id or str(uuid.uuid4()),
                                             thread_id=thread_id or str(uuid.uuid4()),
                                             startup_config_fingerprint=fingerprint(frozen_config),
                                             backend_fingerprint=backend_fingerprint,
                                             principal_fingerprint=principal_fingerprint,
                                             runtime_config_fingerprint=runtime_config_fingerprint
                                             or fingerprint({'api_timeout': 10.0})),
                       lifecycle=LifecycleState(started_at=started_at), task=task, versions=versions,
                       module_policy=policy, budget=budget)
    return cast(GraphState, state.model_dump(mode='json'))


_RECORD_IDS = {('history', 'events'): 'event_id', ('recipes', 'uses'): 'use_id',
               ('budget', 'usage'): 'usage_id', ('recovery', 'attempts'): 'operation_id',
               ('guard', 'checks'): 'check_id', ('candidates', 'items'): 'candidate_id',
               ('candidates', 'recommendations'): 'recommendation_id'}


def _validate_unique_records(state: dict[str, object]) -> None:
    for (block, field), key in _RECORD_IDS.items():
        records = state[block][field]
        ids = [record[key] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate stable record identifier')
    events = state['history']['events']
    sequences = [event['sequence'] for event in events]
    if len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
        raise ValueError('history logical sequence must be unique and increasing')


def _frozen_budget(budget: dict[str, object]) -> dict[str, object]:
    frozen = {key: budget[key] for key in ('deadline_at', 'max_operation_attempts', 'max_repair_attempts')}
    for name, value in budget.items():
        if isinstance(value, dict) and 'unit' in value:
            frozen[name] = {key: value[key] for key in ('limit', 'unit', 'source')}
    return frozen


def _startup_payload(task: dict, versions: dict, policy: dict, budget: dict,
                     backend_fingerprint: str, principal_fingerprint: str,
                     runtime_config_fingerprint: str) -> dict:
    # The trusted backend binds dataset metadata once after Session creation.
    # Exclude only these two fields from the pre-request configuration digest.
    frozen_task = {key: value for key, value in task.items()
                   if key not in ('dataset_fingerprint', 'dataset_fingerprint_status')}
    return {'task': frozen_task, 'versions': versions, 'module_policy': policy,
            'budget': _frozen_budget(budget), 'backend_fingerprint': backend_fingerprint,
            'principal_fingerprint': principal_fingerprint,
            'runtime_config_fingerprint': runtime_config_fingerprint}


def _merge_records(old: list[dict], new: list[dict], path: tuple[str, ...]) -> list[dict]:
    key = _RECORD_IDS[path]
    result = copy.deepcopy(old)
    indexes = {record[key]: index for index, record in enumerate(result)}
    for record in new:
        record_id = record.get(key)
        if record_id not in indexes:
            indexes[record_id] = len(result)
            result.append(copy.deepcopy(record))
            continue
        previous = result[indexes[record_id]]
        if previous == record:
            continue
        if path == ('recovery', 'attempts'):
            merged = {**previous, **record}
            for counter in ('network_attempts', 'repair_attempts', 'reconciliation_attempts'):
                if merged[counter] < previous[counter]:
                    raise ValueError('operation attempt counters cannot decrease')
            result[indexes[record_id]] = merged
        elif path == ('budget', 'usage'):
            merged = {**previous, **record}
            if merged == previous:
                continue
            immutable = ('usage_id', 'operation_id', 'kind', 'calls')
            if any(previous[field] != merged[field] for field in immutable):
                raise ValueError('usage identity conflicts with an existing record')
            if previous['status'] == 'confirmed' or (previous['status'] == 'unknown' and merged['status'] == 'pending'):
                raise ValueError('usage settlement cannot be rewritten or moved backward')
            result[indexes[record_id]] = merged
        elif path == ('guard', 'checks') and previous['status'] == 'pending':
            merged = {**previous, **record}
            if any(previous[field] != merged[field] for field in ('check_id', 'phase', 'kind', 'version')):
                raise ValueError('guard check identity conflict')
            result[indexes[record_id]] = merged
        else:
            raise ValueError('stable record identifier has conflicting content')
    return result


def _deep_merge(old: dict, patch: dict, path: tuple[str, ...] = ()) -> dict:
    result = copy.deepcopy(old)
    for key, value in patch.items():
        here = (*path, key)
        if here in _RECORD_IDS and isinstance(value, list):
            result[key] = _merge_records(result.get(key, []), value, here)
        elif (here == ('recovery', 'pending_operation') and isinstance(value, dict)
              and isinstance(result.get(key), dict)
              and value.get('operation_id') != result[key].get('operation_id')):
            result[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value, here)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_patch(state: GraphState | StateModel, patch: dict[str, object]) -> GraphState:
    """Validate a partial node patch, preserve fields, and reject frozen changes.

    Events and immutable usage records merge by stable ID. The only legal
    same-ID changes are monotonic operation counters, pending guard outcomes,
    and pending/unknown usage settlement. A new snapshot replaces its field.
    """
    old = StateModel.model_validate(state).model_dump(mode='json')
    merged = _deep_merge(old, patch)
    updated = StateModel.model_validate(merged).model_dump(mode='json')
    for block in ('versions', 'module_policy'):
        if old[block] != updated[block]:
            raise ValueError(f'{block} is frozen')
    for key in ('task_id', 'thread_id', 'startup_config_fingerprint', 'backend_fingerprint', 'principal_fingerprint'):
        if old['identity'][key] != updated['identity'][key]:
            raise ValueError('startup identity is frozen')
    for key, value in old['identity'].items():
        if value is not None and value != updated['identity'][key]:
            raise ValueError('bound identity cannot be changed')
    for key, value in old['task'].items():
        if key in ('dataset_fingerprint', 'dataset_fingerprint_status') and old['task']['dataset_fingerprint_status'] == 'pending':
            continue
        if value != updated['task'][key]:
            raise ValueError('task configuration is frozen')
    if _frozen_budget(old['budget']) != _frozen_budget(updated['budget']):
        raise ValueError('runtime limits and deadline are frozen')
    if old['lifecycle']['started_at'] != updated['lifecycle']['started_at']:
        raise ValueError('task start time is frozen')
    for key in ('submission_request_id', 'submission_content', 'submission_fingerprint', 'run_id', 'experiment_id'):
        if old['execution'][key] is not None and old['execution'][key] != updated['execution'][key]:
            raise ValueError('scientific experiment submission is immutable')
    previous_op = old['recovery']['pending_operation']
    next_op = updated['recovery']['pending_operation']
    if previous_op is not None and next_op is not None and previous_op['operation_id'] == next_op['operation_id']:
        if any(previous_op[key] != next_op[key] for key in ('kind', 'request_id', 'tool_name', 'content', 'content_fingerprint')):
            raise ValueError('prepared operation content is immutable')
    for dimension in ('llm_calls', 'api_calls'):
        previous = old['budget'][dimension]['actual']
        current = updated['budget'][dimension]['actual']
        if previous is not None and (current is None or current < previous):
            raise ValueError('durable call counters cannot decrease')
    if old['finalization']['status'] == 'confirmed' and old['finalization'] != updated['finalization']:
        raise ValueError('confirmed backend lock is immutable')
    return cast(GraphState, updated)
