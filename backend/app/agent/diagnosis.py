"""Closed feedback and explanation contracts; explanations never dispatch tools."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator, model_serializer
from ..diagnostic_evidence import Strict, Digest, Count, Score, Support, digest, paired_metrics
from ..runs.guard import GuardProjection, project_guard
from ..train_evidence import TrainStatistics, Risk
from ..knowledge import KnowledgeProjection

Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')]
Text = Annotated[str, Field(min_length=1, max_length=600)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Reason = Literal['diagnosis_budget_insufficient', 'diagnosis_work_deadline',
    'diagnosis_evidence_unavailable', 'diagnosis_context_too_long', 'diagnosis_output_invalid',
    'diagnosis_repair_exhausted', 'diagnosis_response_unknown', 'diagnosis_source_changed',
    'diagnosis_persistence_unavailable', 'diagnosis_protocol_unsupported']
RULES_VERSION = 'feedback-diagnosis-rules-v1'
PROMPT_VERSION = 'agent-diagnosis-prompt-v1'


def rules_digest():
    root = Path(__file__).parents[3]
    names = ('backend/app/agent/diagnosis.py', 'backend/app/diagnostic_evidence.py',
        'backend/app/agent/observation.py', 'agent_poc/orchestration/diagnosis.py',
        'agent_poc/orchestration/llm.py', 'agent_poc/orchestration/projection.py',
        'agent_poc/orchestration/state.py', 'agent_poc/orchestration/graph.py',
        'agent_poc/orchestration/persistence.py', 'backend/app/agent/revisions.py',
        'backend/app/agent/service.py', 'backend/app/agent/repository.py',
        'agent_poc/orchestration/measurements.py', 'agent_poc/orchestration/runtime.py')
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in names)).hexdigest()


class DiagnosisPolicy(Strict):
    schema_version: Literal['feedback-diagnosis-policy-v1'] = 'feedback-diagnosis-policy-v1'
    enabled: bool
    rules_version: Literal['feedback-diagnosis-rules-v1'] = RULES_VERSION
    rules_digest: Digest
    prompt_version: Literal['agent-diagnosis-prompt-v1'] = PROMPT_VERSION
    context_version: Literal['agent-context-diagnosis-v1'] = 'agent-context-diagnosis-v1'
    max_physical_calls: Literal[2] = 2
    max_output_repairs: Literal[1] = 1
    max_network_retries: Literal[1] = 1
    max_generations: Literal[2] = 2


class MetricPair(Strict):
    fold_index: Count
    metric: Literal['macro_f1', 'balanced_accuracy', 'accuracy']
    train: Score
    valid: Score
    delta: Finite
    aggregation: Literal['direct_fold']
    same_snapshot: Literal[True]
    fit_scope: Literal['train']

    @model_validator(mode='after')
    def coherent(self):
        if self.delta != self.train-self.valid:
            raise ValueError('metric delta mismatch')
        return self


class FoldSupport(Strict):
    fold_index: Count
    train: Support
    valid: Support


class Failure(Strict):
    status: Literal['ready', 'unavailable', 'not_applicable']
    stage: Literal['admission', 'pre_fit', 'training', 'publication', 'observation', 'unknown']
    reason_code: Literal['training_failure', 'cancelled', 'dependency_unavailable', 'model_retired',
        'data_invalid', 'integrity_unavailable', 'budget_rejected', 'unknown'] | None
    source_ref: Identifier
    retryable: Literal[False] = False


class EffectiveExecution(Strict):
    status: Literal['ready', 'unavailable']
    model_id: Identifier | None = None
    normalization: Literal['zscore', 'minmax', 'area', 'none'] | None = None
    class_balance: Literal['none', 'class_weight'] | None = None
    search_mode: Literal['fixed', 'bounded'] | None = None
    selection_criterion: Literal['balanced_accuracy', 'oob_balanced_accuracy', 'best_valid_loss'] | None = None
    selected_trial: Count | None = None
    best_epoch: Count | None = None
    selected_parameters: dict[Identifier, Count | Finite | Identifier | None] = Field(default_factory=dict)


class CostDimension(Strict):
    known_actual: Count
    held_reserved: Count
    held_unknown: Count
    unknown_count: Count
    source_ref: Identifier


class Measurement(Strict):
    status: Literal['known', 'unknown', 'withheld', 'unavailable']
    dimensions: dict[Literal['experiments','model_fits','training_epochs'], CostDimension] = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def hidden(self, handler):
        return {'status':self.status} if self.status=='withheld' else handler(self)

    @model_validator(mode='after')
    def coherent(self):
        if self.status in ('withheld','unavailable') and self.dimensions:
            raise ValueError('unavailable or withheld cost has quantities')
        return self


class FeedbackEvidence(Strict):
    schema_version: Literal['agent-feedback-evidence-v1'] = 'agent-feedback-evidence-v1'
    status: Literal['ready', 'pending', 'unavailable']
    reason_code: Literal['audit_missing', 'audit_invalid', 'audit_input_unavailable',
        'class_support_incomplete', 'guard_unavailable', 'unsupported_evaluation'] | None
    subject_event_id: Identifier
    evidence_digest: Digest
    source_versions: dict[Identifier, Digest]
    outcome: Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']
    guard_ref: GuardProjection | None
    failure: Failure
    metric_pairs: list[MetricPair] = Field(max_length=3)
    support: list[FoldSupport] = Field(max_length=1)
    effective_execution: EffectiveExecution
    training_cost: Measurement
    limitations: list[Identifier]

    @model_validator(mode='after')
    def coherent(self):
        if self.metric_pairs and (self.status != 'ready' or self.outcome != 'succeeded'
                or self.guard_ref is None or self.guard_ref.status != 'passed'):
            raise ValueError('performance requires current trusted Guard')
        body = self.model_dump(exclude={'evidence_digest'})
        if digest(body) != self.evidence_digest:
            raise ValueError('evidence digest mismatch')
        return self


def feedback_evidence(*, record, assessment, awareness, training_cost=None):
    """Scope and Run/Session binding must have been checked by AgentService first."""
    guard = project_guard(assessment.report) if assessment and assessment.report else None
    trusted = (record.state == 'succeeded' and assessment is not None and assessment.status == 'ready'
               and guard is not None and guard['status'] == 'passed')
    status, reason, pairs, support = 'unavailable', 'guard_unavailable', [], []
    execution = EffectiveExecution(status='unavailable')
    source_versions = {}
    if trusted:
        docs = assessment.documents
        compared = paired_metrics(docs)
        status, reason = compared['status'], compared['reason_code']
        if docs['config.json']['evaluation_strategy'] != 'stratified_holdout':
            status, reason = 'unavailable', 'unsupported_evaluation'
        elif status == 'ready':
            pairs, support = compared['pairs'], compared['support']
            audit = docs['training_validation_audit.json']
            fold = audit['folds'][0]
            cfg = docs['config.json']
            execution = EffectiveExecution(status='ready', model_id=cfg['model_type'],
                normalization=cfg['normalization'], class_balance=cfg['class_balance'],
                search_mode=docs.get('search_summary.json', {}).get('mode'),
                selection_criterion=fold['selection_criterion'],
                selected_trial=fold['selected_trial_index'], best_epoch=fold['best_epoch'],
                selected_parameters=docs['split.json'][0]['best_params'])
            source_versions['training_validation_audit'] = digest(audit)
    elif record.state in ('queued', 'running'):
        status, reason = 'pending', None
    failure_reason = None
    if record.state == 'failed':
        # No string matching of arbitrary exception messages or type names.
        failure_reason = 'training_failure'
    elif record.state == 'cancelled':
        failure_reason = 'cancelled'
    elif record.state == 'succeeded' and not trusted:
        failure_reason = 'integrity_unavailable'
    if guard and guard['status'] == 'failed':
        codes = {c['reason_code'] for c in guard['checks']}
        if codes & {'guard_dataset_invalid', 'guard_dataset_changed', 'guard_split_invalid'}:
            failure_reason = 'data_invalid'
    event = 'event-' + digest(dict(run=record.run_id, outcome=record.state,
        guard=guard['report_id'] if guard else None))
    payload = dict(schema_version='agent-feedback-evidence-v1', status=status, reason_code=reason,
        subject_event_id=event, source_versions=source_versions, outcome=record.state,
        guard_ref=guard, failure=Failure(status='ready' if failure_reason else 'not_applicable',
            stage='observation', reason_code=failure_reason, source_ref=event).model_dump(),
        metric_pairs=pairs, support=support, effective_execution=execution.model_dump(),
        training_cost=Measurement.model_validate(training_cost or {'status':'withheld' if awareness=='off' else 'unavailable'}).model_dump(),
        limitations=['no_causal_inference', 'no_statistical_interval'] +
            (['validation_used_for_selection'] if execution.selection_criterion != 'oob_balanced_accuracy' else []))
    return FeedbackEvidence(**payload, evidence_digest=digest(payload)).model_dump()


class Fact(Strict):
    fact_id: Identifier
    code: Identifier
    value: bool | Count | Finite | Identifier | None
    unit: Identifier | None = None
    source_ref: Identifier
    source_version: Identifier
    source_digest: Digest
    trust: Literal['verified', 'unavailable']
    scope: Literal['current_experiment', 'current_task'] = 'current_experiment'
    limitation_codes: list[Identifier]


class SuggestionAction(Strict):
    action_id: Identifier
    kind: Literal['review_recipe', 'inspect_environment', 'inspect_data', 'inspect_integrity', 'retain_current']
    target: Identifier | None = None
    recipe_digest: Digest | None = None
    configuration_const: dict[Identifier, bool | Count | Finite | str] | None = None
    recipe_summary: dict[Literal['model_id', 'normalization', 'class_balance'], Identifier] | None = None

    @model_serializer(mode='wrap')
    def preserve_historical_action(self, handler):
        result = handler(self)
        if 'recipe_summary' not in self.model_fields_set:
            result.pop('recipe_summary', None)
        return result
    condition_refs: list[Identifier]
    executable_now: Literal[False] = False


class Hypothesis(Strict):
    hypothesis_id: Identifier
    hypothesis_code: Literal['overfitting_risk', 'model_data_mismatch', 'optimization_limitation', 'environment_related', 'unknown']
    explanation: Text
    evidence_refs: list[Identifier] = Field(min_length=1, max_length=6)
    uncertainty: Literal['tentative', 'unknown']
    limitation_codes: list[Identifier] = Field(min_length=1, max_length=6)


class Suggestion(Strict):
    suggestion_id: Identifier
    action_id: Identifier
    evidence_refs: list[Identifier] = Field(min_length=1, max_length=6)
    rationale: Text
    condition_refs: list[Identifier] = Field(max_length=6)


class DiagnosisProposal(Strict):
    schema_version: Literal['diagnosis-proposal-v1'] = 'diagnosis-proposal-v1'
    input_digest: Digest
    assessment: Literal['hypotheses_present', 'no_issue_identified', 'insufficient_evidence']
    hypotheses: list[Hypothesis] = Field(max_length=3)
    suggestions: list[Suggestion] = Field(max_length=3)

    def bind(self, context):
        if self.input_digest != context['input_digest']:
            raise ValueError('diagnosis input mismatch')
        facts = {f['fact_id']: f for f in context['facts']}
        knowledge_ids = set((context.get('knowledge') or {}).get('provided_entry_ids', []))
        displayed = set(facts) | knowledge_ids
        actions = {a['action_id']: a for a in context['suggestion_catalog']}
        limits = set(context['limitations'])
        for group, key in ((self.hypotheses, 'hypothesis_id'), (self.suggestions, 'suggestion_id')):
            ids = [getattr(item, key) for item in group]
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate diagnosis ID')
            for item in group:
                if not set(item.evidence_refs) <= displayed or len(set(item.evidence_refs)) != len(item.evidence_refs):
                    raise ValueError('evidence not displayed')
        if (self.assessment == 'hypotheses_present') != bool(self.hypotheses):
            raise ValueError('inconsistent assessment')
        for item in self.hypotheses:
            if not set(item.limitation_codes) <= limits:
                raise ValueError('unknown limitation')
            refs = [facts[key] for key in item.evidence_refs if key in facts]
            if item.hypothesis_code == 'overfitting_risk' and not any(
                    f['code'].endswith('_gap') and type(f['value']) in (int, float) and f['value'] > 0 for f in refs):
                raise ValueError('no positive paired gap')
            if item.hypothesis_code in ('model_data_mismatch', 'optimization_limitation') and not any(
                    f['code'].startswith(('train_', 'valid_')) and f['trust']=='verified' for f in refs):
                raise ValueError('no training evidence')
            if item.hypothesis_code == 'environment_related' and not any(
                    f['code']=='failure' and f['value']=='dependency_unavailable' for f in refs):
                raise ValueError('no environment evidence')
        for item in self.suggestions:
            action = actions.get(item.action_id)
            if action is None or item.condition_refs != action['condition_refs']:
                raise ValueError('suggestion not in frozen catalog')
        # Apply the existing safe-text boundary to all model prose; never retain raw rejects.
        from agent_poc.orchestration.state import validate_safe_text
        for item in self.hypotheses:
            validate_safe_text(item.explanation)
        for item in self.suggestions:
            validate_safe_text(item.rationale)
        return self


class DiagnosisContextFields(Strict):
    context_version: Literal['agent-context-diagnosis-v1'] = 'agent-context-diagnosis-v1'
    phase: Literal['diagnose'] = 'diagnose'
    input_digest: Digest
    facts: list[Fact] = Field(max_length=32)
    problem_codes: list[Identifier]
    limitations: list[Identifier]
    suggestion_catalog: list[SuggestionAction]
    call_cost: dict[Literal['llm_calls','api_calls','input_tokens','output_tokens','cached_tokens','output_repairs','network_retries'], CostDimension] | None = None
    training_cost: Measurement | None = None
    train_statistics: TrainStatistics | None = None
    train_risks: list[Risk] = Field(default_factory=list)
    knowledge: KnowledgeProjection | None = None

    @field_validator('knowledge', mode='before')
    @classmethod
    def decode_knowledge(cls, value):
        if isinstance(value, dict):
            return KnowledgeProjection.model_validate_json(json.dumps(value, allow_nan=False))
        return value


class DiagnosisContext(DiagnosisContextFields):
    @model_validator(mode='after')
    def bounded(self):
        if len(self.model_dump_json().encode()) > 65536:
            raise ValueError('diagnosis context too large')
        return self


class UnavailableDiagnosisContext(Strict):
    """Bounded audit receipt, never a sendable context or a partial action catalog."""
    context_version: Literal['agent-context-diagnosis-unavailable-v1'] = 'agent-context-diagnosis-unavailable-v1'
    phase: Literal['diagnose'] = 'diagnose'
    input_digest: Digest
    reason_code: Literal['diagnosis_context_too_long'] = 'diagnosis_context_too_long'
    capacity_limit: Literal['bytes', 'tokens']
    source_context_digest: Digest
    source_context_bytes: Count
    fact_count: Count
    action_count: Count
    # Facts remain bound through the complete context hash and scoped evidence binding.
    # An unavailable receipt makes no claims and cannot be submitted to a model.
    facts: list[Fact] = Field(default_factory=list, max_length=0)
    problem_codes: list[Identifier] = Field(default_factory=list, max_length=0)
    limitations: list[Literal['diagnosis_context_unavailable']] = ['diagnosis_context_unavailable']
    suggestion_catalog: list[SuggestionAction] = Field(default_factory=list, max_length=0)


class DiagnosisInput(Strict):
    schema_version: Literal['agent-diagnosis-input-v1'] = 'agent-diagnosis-input-v1'
    subject_event_id: Identifier
    generation: Annotated[int, Field(strict=True, ge=0, le=1)]
    bindings: dict[Identifier, Digest | None]
    context: Annotated[DiagnosisContext | UnavailableDiagnosisContext, Field(discriminator='context_version')]
    displayed_context_digest: Digest
    after_diagnosis: Literal['finalize_decision', 'terminated']

    @model_validator(mode='after')
    def verify_identity(self):
        context=self.context.model_dump(mode='json')
        expected=digest(dict(bindings=self.bindings,subject=self.subject_event_id,generation=self.generation))
        if self.context.input_digest!=expected or digest(context)!=self.displayed_context_digest:
            raise ValueError('diagnostic input identity mismatch')
        required={'task','thread','session','principal','run','budget_policy','diagnosis_policy','llm',
            'evidence','effective_config','train_evidence','knowledge','catalog'}
        if set(self.bindings)!=required or any(v is None for k,v in self.bindings.items() if k!='run'):
            raise ValueError('diagnostic input bindings incomplete')
        return self


class ReportSuggestion(Suggestion):
    executable_now: Literal[False] = False


class DiagnosisReport(Strict):
    schema_version: Literal['agent-diagnosis-report-v1'] = 'agent-diagnosis-report-v1'
    diagnosis_id: Identifier
    input_digest: Digest
    rules_version: Literal['feedback-diagnosis-rules-v1'] = RULES_VERSION
    rules_digest: Digest
    status: Literal['ready', 'unavailable', 'invalid']
    mode: Literal['llm', 'deterministic_only']
    reason_code: Reason | None
    subject_event_id: Identifier
    bindings: dict[Identifier, Digest | None]
    facts: list[Fact] = Field(max_length=32)
    problem_codes: list[Identifier]
    hypotheses: list[Hypothesis] = Field(max_length=3)
    suggestions: list[ReportSuggestion] = Field(max_length=3)
    limitations: list[Identifier]
    proposal_ref: Identifier | None
    call_refs: list[Identifier] = Field(max_length=2)
    created_at: Finite
    freshness: Literal['snapshot_only', 'current', 'stale', 'unverified']
    assessment: Literal['hypotheses_present', 'no_issue_identified', 'insufficient_evidence', 'historically_unrecorded'] | None = 'historically_unrecorded'
    proposal_digest: Digest | None = None

    @model_serializer(mode='wrap')
    def preserve_historical_bytes(self, handler):
        result = handler(self)
        # Old immutable Journal hashes include neither field. Annotate only after verification.
        for name in ('assessment', 'proposal_digest'):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result

    @model_validator(mode='after')
    def coherent(self):
        if self.status != 'ready' and (self.hypotheses or self.suggestions or self.mode != 'deterministic_only'):
            raise ValueError('unavailable explanation has model claims')
        return self


    def bind(self, snapshot):
        context=snapshot['context']
        if (self.input_digest!=context['input_digest'] or self.bindings!=snapshot['bindings']
                or self.subject_event_id!=snapshot['subject_event_id']
                or [f.model_dump(mode='json') for f in self.facts]!=context['facts']
                or self.problem_codes!=context['problem_codes'] or self.limitations!=context['limitations']):
            raise ValueError('diagnosis report differs from immutable input')
        if context['context_version']=='agent-context-diagnosis-unavailable-v1' and (
                self.status!='unavailable' or self.reason_code!=context['reason_code'] or self.call_refs):
            raise ValueError('capacity receipt must remain unavailable with zero calls')
        if self.status=='ready':
            if context['context_version']=='agent-context-diagnosis-unavailable-v1':
                raise ValueError('capacity receipt cannot have a model proposal')
            historical = 'assessment' not in self.model_fields_set or self.assessment=='historically_unrecorded'
            proposal = DiagnosisProposal(schema_version='diagnosis-proposal-v1',input_digest=self.input_digest,
                # Historical claims still undergo reference validation, without inferring a recorded assessment.
                assessment=('hypotheses_present' if self.hypotheses else 'insufficient_evidence') if historical else self.assessment,
                hypotheses=self.hypotheses,suggestions=[Suggestion.model_validate(s.model_dump(exclude={'executable_now'})) for s in self.suggestions]).bind(context)
            if not historical and self.proposal_digest != digest(proposal.model_dump(mode='json')):
                raise ValueError('diagnosis report differs from validated proposal')
        elif self.assessment not in (None, 'historically_unrecorded') or self.proposal_digest is not None:
            raise ValueError('unavailable report has a model assessment')
        return self
