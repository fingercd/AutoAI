"""Local diagnostic snapshots and proposals. There is no tool dispatcher here."""
from __future__ import annotations

import json
from pydantic import Field

from backend.app.agent.diagnosis import (Strict, Digest, Identifier, DiagnosisInput,
    DiagnosisContext, DiagnosisProposal, DiagnosisReport, Fact, SuggestionAction, digest)


class StoredDiagnosisProposal(Strict):
    schema_version: str = 'stored-diagnosis-proposal-v1'
    input_digest: Digest
    context_digest: Digest
    proposal: DiagnosisProposal
    response_id: Identifier | None
    tool_call_id: Identifier | None

    def bind(self, context):
        from .llm import Proposal, TokenUsage
        if (self.schema_version != 'stored-diagnosis-proposal-v1' or
                self.input_digest != context['input_digest'] or self.context_digest != digest(context)):
            raise ValueError('stored diagnosis binding mismatch')
        self.proposal.bind(context)
        return Proposal('report_feedback_diagnosis', self.proposal.model_dump(),
            'Validated diagnostic explanation', self.tool_call_id, self.response_id, TokenUsage())


def store_diagnosis(proposal, context):
    parsed = DiagnosisProposal.model_validate(proposal.arguments).bind(context)
    return StoredDiagnosisProposal(input_digest=context['input_digest'], context_digest=digest(context),
        proposal=parsed, response_id=proposal.response_id, tool_call_id=proposal.tool_call_id).model_dump()


def build_input(state, *, after, generation, adapter, journal):
    """Build from closed scoped evidence and the Session's frozen Train-only cards."""
    identity, task = state['identity'], state['task']
    evidence = state['diagnosis']['feedback_evidence']
    source = evidence['evidence_digest'] if evidence else digest(state['decision']['reason_code'])
    subject = evidence['subject_event_id'] if evidence else 'rejection-' + digest(dict(
        task=identity['task_id'], decision=state['decision']['decision_id'],
        reason=state['decision']['reason_code'] or state['finalization']['termination_reason']))
    bindings = dict(task=digest(identity['task_id']), thread=digest(identity['thread_id']),
        session=digest(identity['session_id']), principal=identity['principal_fingerprint'],
        run=digest(state['execution']['run_id']) if state['execution']['run_id'] else None,
        budget_policy=task['budget_policy_digest'], diagnosis_policy=digest(task['diagnosis_policy']),
        llm=state['versions']['llm_config_fingerprint'], evidence=source,
        effective_config=digest(state['execution']['effective_config']),
        train_evidence=state['evidence']['content']['evidence_digest'],
        knowledge=digest(state['knowledge']['snapshot']), catalog=state['recipes']['catalog']['catalog_digest'])
    facts, problems = [], []
    limitations = ['no_causal_inference', 'no_statistical_interval', 'suggestions_not_executed']
    def fact(code, value, *, trust='verified', unit=None):
        facts.append(Fact(fact_id=f'e{len(facts)+1}', code=code, value=value, unit=unit,
            source_ref=subject, source_version='agent-feedback-evidence-v1', source_digest=source,
            trust=trust, limitation_codes=limitations).model_dump())
    if evidence:
        fact('outcome', evidence['outcome'])
        if evidence['failure']['reason_code']:
            fact('failure', evidence['failure']['reason_code'])
            problems.append(evidence['failure']['reason_code'])
        if evidence['guard_ref']:
            fact('candidate_eligible', evidence['guard_ref']['eligibility']=='eligible')
        for pair in evidence['metric_pairs']:
            fact('train_'+pair['metric'], pair['train'], unit='fraction')
            fact('valid_'+pair['metric'], pair['valid'], unit='fraction')
            fact(pair['metric']+'_gap', pair['delta'], unit='fraction')
        if not evidence['metric_pairs']:
            fact('paired_metric_unavailable', evidence['reason_code'], trust='unavailable')
        for key,value in evidence['effective_execution'].items():
            if key not in ('status','selected_parameters') and value is not None:
                fact('execution_'+key,value)
        for key,value in evidence['effective_execution']['selected_parameters'].items():
            fact('parameter_'+key,value)
        for support in evidence['support']:
            for role in ('train','valid'):
                value=support[role]
                fact(role+'_observations',value['observation_count'],unit='observations')
                fact(role+'_groups',value['sample_group_count'],unit='groups')
                fact(role+'_minimum_class_groups', min(c['sample_group_count'] for c in value['class_support']),unit='groups')
        limitations = list(dict.fromkeys([*limitations, *evidence['limitations']]))
    else:
        reason = state['decision']['reason_code']
        classification = 'budget_rejected' if reason=='recipe_training_budget_exceeded' else 'unknown'
        guard_codes={check['reason_code'] for report in state['guard']['reports'] for check in report['checks']}
        if guard_codes & {'guard_dataset_changed','guard_dataset_invalid','guard_split_invalid'}:
            classification='data_invalid'
        fact('failure', classification)
        problems.append(classification)
    actions=[]
    if 'data_invalid' in problems:
        actions.append(SuggestionAction(action_id='inspect-data',kind='inspect_data',condition_refs=['manual_review']))
    if 'integrity_unavailable' in problems:
        actions.append(SuggestionAction(action_id='inspect-integrity',kind='inspect_integrity',condition_refs=['manual_review']))
    if 'dependency_unavailable' in problems:
        actions.append(SuggestionAction(action_id='inspect-environment',kind='inspect_environment',condition_refs=['manual_review']))
    for i, recipe in enumerate(state['recipes']['catalog']['recipes']):
        # Catalog was verified against the frozen capability snapshot; no new values are generated.
        actions.append(SuggestionAction(action_id=f'review-recipe-{i+1}', kind='review_recipe',
            target=recipe['recipe_id'], recipe_digest=recipe['recipe_digest'],
            configuration_const=recipe['fixed_execution_config'],
            condition_refs=['requires_new_experiment_protocol']))
    actions.append(SuggestionAction(action_id='retain-current',kind='retain_current',condition_refs=[]))
    knowledge=(state['knowledge']['snapshot'] or {}).get('projection')
    context_policy=state['module_policy']['context_policy']
    call_cost=None
    if task['budget_awareness']=='on':
        call_cost=call_cost_evidence(journal.budget_summary(),journal.snapshot(),identity['task_id'])
    context=DiagnosisContext(call_cost=call_cost, training_cost=evidence['training_cost'] if evidence and task['budget_awareness']=='on' else None, input_digest=digest(dict(bindings=bindings,subject=subject,generation=generation)),
        facts=facts,problem_codes=problems,limitations=limitations,
        suggestion_catalog=actions,
        train_statistics=state['evidence']['content']['statistics'] if context_policy['evidence'] else None,
        train_risks=state['evidence']['content']['risks'] if context_policy['risks'] else [],
        knowledge=knowledge).model_dump(mode='json')
    # prepare_context removes only optional whole cards and checks repair/output tokens.
    if adapter is not None:
        context=adapter.prepare_context('diagnose',context)
    return DiagnosisInput(subject_event_id=subject,generation=generation,bindings=bindings,
        context=DiagnosisContext.model_validate(context), displayed_context_digest=digest(context),
        after_diagnosis=after).model_dump(mode='json')


def report_for(snapshot, *, policy, calls, now, proposal=None, reason=None):
    context=snapshot['context']
    parsed=DiagnosisProposal.model_validate(proposal.arguments).bind(context) if proposal else None
    return DiagnosisReport(diagnosis_id='diagnosis-'+digest([context['input_digest'],snapshot['generation']]),
        input_digest=context['input_digest'],rules_digest=policy['rules_digest'],
        status='ready' if parsed else 'unavailable', mode='llm' if parsed else 'deterministic_only',
        reason_code=reason, subject_event_id=snapshot['subject_event_id'], bindings=snapshot['bindings'],
        facts=context['facts'],problem_codes=context['problem_codes'],
        hypotheses=parsed.hypotheses if parsed else [],
        suggestions=[{**row.model_dump(),'executable_now':False} for row in parsed.suggestions] if parsed else [],
        limitations=context['limitations'],proposal_ref=f'call-{calls[-1]["id"]}' if parsed and calls else None,
        call_refs=[f'call-{row["id"]}' for row in calls],created_at=float(now),freshness='current').model_dump(mode='json')


def call_cost_evidence(balances,calls,task_id):
    from .measurements import measure
    result={}
    for name,row in balances.items():
        value={key:row[key] for key in ('known_actual','held_reserved','held_unknown','unknown_count')}
        if row['limit'] is None and name in ('input_tokens','output_tokens','cached_tokens'):
            measured=measure(calls,name,applicable=lambda call:call['kind']=='llm' and call['status'] not in ('prepared','prepare_failed'))
            value.update(known_actual=measured['known_subtotal'],unknown_count=measured['unknown_count'])
        result[name]={**value,'source_ref':'journal-'+digest(task_id)}
    return result
