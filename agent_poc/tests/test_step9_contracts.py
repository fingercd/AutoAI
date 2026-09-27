"""Closed diagnosis contracts and hostile-output boundary (no real provider)."""
from copy import deepcopy
import json
import pytest
from pydantic import ValidationError
from backend.app.agent.diagnosis import (DiagnosisContext, DiagnosisProposal, FeedbackEvidence, MetricPair, Fact, digest)


def context():
    fact=Fact(fact_id='e1',code='macro_f1_gap',value=0.2,unit='fraction',source_ref='event-1',
        source_version='agent-feedback-evidence-v1',source_digest='a'*64,trust='verified',
        limitation_codes=['no_causal_inference']).model_dump()
    return DiagnosisContext(input_digest='b'*64,facts=[fact],problem_codes=[],
        limitations=['no_causal_inference'],suggestion_catalog=[dict(action_id='retain-current',
        kind='retain_current',condition_refs=[])]).model_dump(mode='json')


def proposal(ctx):
    return dict(schema_version='diagnosis-proposal-v1',input_digest=ctx['input_digest'],assessment='hypotheses_present',
        hypotheses=[dict(hypothesis_id='h1',hypothesis_code='overfitting_risk',explanation='A paired positive gap may indicate overfitting.',
            evidence_refs=['e1'],uncertainty='tentative',limitation_codes=['no_causal_inference'])],
        suggestions=[dict(suggestion_id='s1',action_id='retain-current',evidence_refs=['e1'],rationale='Review this observation.',condition_refs=[])])


@pytest.mark.parametrize('attack',['facts','test','action','execute','extra_param','parallel','false_ref','duplicate','nan','inf','bool',
    'missing_uncertainty','causal_certainty','empty_evidence','unknown_limit','input','conditions','path','secret','environment'])
def test_hostile_proposals_rejected(attack):
    ctx=context();p=proposal(ctx)
    if attack=='facts':p['facts']=ctx['facts']
    elif attack=='test':p['hypotheses'][0]['evidence_refs']=['test_accuracy']
    elif attack=='action':p['suggestions'][0]['action_id']='submit_ml_experiment'
    elif attack=='execute':p['suggestions'][0]['executable_now']=True
    elif attack=='extra_param':p['suggestions'][0]['model_params']={'epochs':100}
    elif attack=='parallel':p['tool_calls']=[{},{}]
    elif attack=='false_ref':p['suggestions'][0]['evidence_refs']=['unshown-knowledge']
    elif attack=='duplicate':p['hypotheses']*=2
    elif attack in ('nan','inf','bool'):p['hypotheses'][0]['explanation']={'nan':float('nan'),'inf':float('inf'),'bool':True}[attack]
    elif attack=='missing_uncertainty':del p['hypotheses'][0]['uncertainty']
    elif attack=='causal_certainty':p['hypotheses'][0]['uncertainty']='certain'
    elif attack=='empty_evidence':p['hypotheses'][0]['evidence_refs']=[]
    elif attack=='unknown_limit':p['hypotheses'][0]['limitation_codes']=['invented']
    elif attack=='input':p['input_digest']='c'*64
    elif attack=='conditions':p['suggestions'][0]['condition_refs']=['turn_guard_off']
    elif attack=='path':p['hypotheses'][0]['explanation']='Read /users/private/data.csv now'
    elif attack=='secret':p['hypotheses'][0]['explanation']='Authorization: Bearer private-token'
    elif attack=='environment':p['hypotheses'][0]['hypothesis_code']='environment_related'
    with pytest.raises((ValueError,ValidationError)):
        DiagnosisProposal.model_validate(p).bind(ctx)


@pytest.mark.parametrize('gap',[0.0,-0.2])
def test_overfit_requires_positive_paired_evidence(gap):
    ctx=context();ctx['facts'][0]['value']=gap
    with pytest.raises(ValueError,match='positive paired gap'):DiagnosisProposal.model_validate(proposal(ctx)).bind(ctx)


@pytest.mark.parametrize('score',[0.0,0.01,0.5,1.0])
def test_normal_scores_have_no_threshold_or_required_problem(score):
    pair=MetricPair(fold_index=0,metric='macro_f1',train=score,valid=score,delta=0.0,
        aggregation='direct_fold',same_snapshot=True,fit_scope='train')
    assert pair.delta==0
    ctx=context()
    p=dict(schema_version='diagnosis-proposal-v1',input_digest=ctx['input_digest'],assessment='no_issue_identified',hypotheses=[],suggestions=[])
    assert DiagnosisProposal.model_validate(p).bind(ctx).hypotheses==[]


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-0.1,1.1])
def test_metric_pair_rejects_non_scores(value):
    with pytest.raises(ValueError):MetricPair(fold_index=0,metric='accuracy',train=value,valid=0.5,delta=0.0,
        aggregation='direct_fold',same_snapshot=True,fit_scope='train')


@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_real_tokenizer_schema_and_output_capacity(budget_config,protocol):
    from agent_poc.orchestration.llm import LLMAdapter,LLMConfig,LLMError
    from dataclasses import replace
    cfg=LLMConfig('http://fixture.invalid','fixture',protocol=protocol,diagnosis_phase=True,**budget_config)
    adapter=LLMAdapter(cfg)
    ctx=context();frozen=adapter.prepare_context('diagnose',ctx)
    request=adapter._build_request('diagnose',frozen)[0]
    assert request['max_tokens']==cfg.max_tokens
    assert json.loads(request['messages'][-1]['content'])['facts']==ctx['facts']
    tiny=LLMAdapter(replace(cfg,context_window=1025))
    with pytest.raises(LLMError,match='llm_context_too_long'):tiny.prepare_context('diagnose',ctx)
    if protocol=='native_tools':
        assert [t['function']['name'] for t in request['tools']]==['report_feedback_diagnosis']
        assert request['parallel_tool_calls'] is False



@pytest.mark.parametrize('assessment',['no_issue_identified','insufficient_evidence'])
def test_observed_inconsistent_assessment_is_not_repaired_silently(assessment):
    ctx=context();value=proposal(ctx)
    value['assessment']=assessment
    value['hypotheses'][0]['hypothesis_code']='unknown'
    with pytest.raises(ValueError,match='inconsistent assessment'):
        DiagnosisProposal.model_validate(value).bind(ctx)


def test_single_action_finalization_example_roundtrips_original_envelope():
    from agent_poc.orchestration.llm import LLMAdapter,LLMConfig
    from agent_poc.orchestration.projection import finalization_context
    from agent_poc.tests.test_llm import TASK,envelope,FakeTransport,Response
    from dataclasses import replace
    ctx=finalization_context(task=TASK,session_id='session-1',run_id='run-1',validation={'macro_f1':.3},
        validation_score=.3,allowed_actions=['finalize_ml_session'],context_version='agent-context-budget-v1',budget_awareness='off')
    cfg=LLMConfig('http://fixture.invalid','local-model',prompt_version='agent-decision-budget-v1',diagnosis_phase=True)
    adapter=LLMAdapter(cfg)
    system=adapter._build_request('finalize',ctx)[0]['messages'][0]['content']
    value=json.loads(system.split('Complete JSON object example: ',1)[1].split('\n',1)[0])
    assert value['arguments']==ctx['bindings']
    adapter=LLMAdapter(cfg,transport=FakeTransport([Response(200,envelope(json.dumps(value)))]))
    parsed=adapter.propose('finalize',ctx)
    assert parsed.tool_name=='finalize_ml_session' and parsed.arguments==ctx['bindings']
    legacy=LLMAdapter(replace(cfg,diagnosis_phase=False))._build_request('finalize',ctx)[0]['messages'][0]['content']
    native=LLMAdapter(replace(cfg,protocol='native_tools'))._build_request('finalize',ctx)[0]
    assert 'Complete JSON object example:' not in legacy
    assert 'Complete JSON object example:' not in native['messages'][0]['content']
    assert native['parallel_tool_calls'] is False


@pytest.mark.parametrize('content',['"finalize_ml_session"','"finalize_ml_session"{"session_id":"session-1"}Finish.'])
def test_real_invalid_finalization_forms_remain_rejected(content):
    from agent_poc.orchestration.llm import LLMAdapter,LLMConfig,LLMError
    from agent_poc.orchestration.projection import finalization_context
    from agent_poc.tests.test_llm import TASK,envelope,FakeTransport,Response
    ctx=finalization_context(task=TASK,session_id='session-1',run_id='run-1',validation={'macro_f1':.3},
        validation_score=.3,allowed_actions=['finalize_ml_session'],context_version='agent-context-budget-v1',budget_awareness='off')
    cfg=LLMConfig('http://fixture.invalid','local-model',prompt_version='agent-decision-budget-v1',diagnosis_phase=True)
    adapter=LLMAdapter(cfg,transport=FakeTransport([Response(200,envelope(content))]))
    with pytest.raises(LLMError,match='llm_output_invalid'):adapter.propose('finalize',ctx)
