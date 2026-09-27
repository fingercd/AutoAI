"""Local diagnostic snapshots and proposals. There is no tool dispatcher here."""
from __future__ import annotations

import json
from pydantic import Field

from backend.app.agent.diagnosis import (Strict, Digest, Identifier, DiagnosisInput,
    DiagnosisContext, DiagnosisContextFields, UnavailableDiagnosisContext, DiagnosisProposal, DiagnosisReport, Fact, SuggestionAction, digest)


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
            # The complete recipe (including fixed configuration) is bound by this digest.
            recipe_summary=dict(model_id=recipe['model_id'],normalization=recipe['preprocessing']['normalization'],
                class_balance=recipe['class_balance']),
            condition_refs=['requires_new_experiment_protocol']))
    actions.append(SuggestionAction(action_id='retain-current',kind='retain_current',condition_refs=[]))
    knowledge=(state['knowledge']['snapshot'] or {}).get('projection')
    context_policy=state['module_policy']['context_policy']
    call_cost=None
    if task['budget_awareness']=='on':
        call_cost=call_cost_evidence(journal.budget_summary(),journal.snapshot(),identity['task_id'])
    context=DiagnosisContextFields(call_cost=call_cost, training_cost=evidence['training_cost'] if evidence and task['budget_awareness']=='on' else None, input_digest=digest(dict(bindings=bindings,subject=subject,generation=generation)),
        facts=facts,problem_codes=problems,limitations=limitations,
        suggestion_catalog=actions,
        train_statistics=state['evidence']['content']['statistics'] if context_policy['evidence'] else None,
        train_risks=state['evidence']['content']['risks'] if context_policy['risks'] else [],
        knowledge=knowledge).model_dump(mode='json')
    # Structural validation precedes capacity decisions; corrupt input is never downgraded.
    source_context = context
    capacity = None
    while len(json.dumps(context,ensure_ascii=False,separators=(',',':')).encode('utf8')) > 65536:
        knowledge = context.get('knowledge')
        if not knowledge or not knowledge['entries']:
            capacity = 'bytes'
            break
        # Clone before clipping: retain the exact full projection digest for an audit receipt.
        context = json.loads(json.dumps(context))
        knowledge = context['knowledge']
        knowledge['entries'].pop()
        knowledge['provided_entry_ids'] = [entry['entry_id'] for entry in knowledge['entries']]
        knowledge['provided_count'] = len(knowledge['entries'])
        knowledge['omitted_count'] = knowledge['matched_count'] - knowledge['provided_count']
    if capacity is None:
        context = DiagnosisContext.model_validate(context).model_dump(mode='json')
        if adapter is not None:
            from .llm import LLMError
            try:
                context=adapter.prepare_context('diagnose',context)
            except LLMError as error:
                if error.code != 'llm_context_too_long':
                    raise
                capacity = 'tokens'
    if capacity is not None:
        context = UnavailableDiagnosisContext(input_digest=source_context['input_digest'],
            capacity_limit=capacity, source_context_digest=digest(source_context),
            source_context_bytes=len(json.dumps(source_context,ensure_ascii=False,separators=(',',':')).encode('utf8')),
            fact_count=len(source_context['facts']), action_count=len(source_context['suggestion_catalog'])).model_dump(mode='json')
    return DiagnosisInput(subject_event_id=subject,generation=generation,bindings=bindings,
        context=context, displayed_context_digest=digest(context),
        after_diagnosis=after).model_dump(mode='json')


def report_for(snapshot, *, policy, calls, now, proposal=None, reason=None):
    context=snapshot['context']
    if context['context_version']=='agent-context-diagnosis-unavailable-v1':
        calls=[]  # Earlier generations remain in the independent cost ledger, not this unsent receipt.
    parsed=DiagnosisProposal.model_validate(proposal.arguments).bind(context) if proposal else None
    return DiagnosisReport(diagnosis_id='diagnosis-'+digest([context['input_digest'],snapshot['generation']]),
        input_digest=context['input_digest'],rules_digest=policy['rules_digest'],
        status='ready' if parsed else 'unavailable', mode='llm' if parsed else 'deterministic_only',
        assessment=parsed.assessment if parsed else None,
        proposal_digest=digest(parsed.model_dump(mode='json')) if parsed else None,
        reason_code=reason, subject_event_id=snapshot['subject_event_id'], bindings=snapshot['bindings'],
        facts=context['facts'],problem_codes=context['problem_codes'],
        hypotheses=parsed.hypotheses if parsed else [],
        suggestions=[{**row.model_dump(),'executable_now':False} for row in parsed.suggestions] if parsed else [],
        limitations=context['limitations'],proposal_ref=f'call-{calls[-1]["id"]}' if parsed and calls else None,
        call_refs=[f'call-{row["id"]}' for row in calls],created_at=float(now),freshness='current').model_dump(mode='json')


def verify_report_proposal(connection, thread_id, report, snapshot):
    """Bind modern reports to the actual immutable confirmed call, not just its ID."""
    if report['status'] != 'ready' or report.get('assessment', 'historically_unrecorded') == 'historically_unrecorded':
        return
    reference = report['proposal_ref']
    if reference is None:
        # Pure report construction may have no Journal call; production reports always carry it.
        return
    if not reference.startswith('call-') or not reference[5:].isdigit():
        raise ValueError('diagnosis proposal reference invalid')
    row = connection.execute('SELECT proposal_json,status,name,kind FROM orchestration_calls_v1 WHERE thread_id=? AND id=?',
        (thread_id,int(reference[5:]))).fetchone()
    if row is None or tuple(row[1:]) != ('confirmed','diagnose','llm') or not row[0]:
        raise ValueError('diagnosis confirmed proposal missing')
    parsed = StoredDiagnosisProposal.model_validate_json(row[0])
    parsed.bind(snapshot['context'])
    if digest(parsed.proposal.model_dump(mode='json')) != report['proposal_digest']:
        raise ValueError('diagnosis confirmed proposal mismatch')


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
