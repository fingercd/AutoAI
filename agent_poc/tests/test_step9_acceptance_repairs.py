"""Acceptance regressions: full real catalog and preserved assessments."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from backend.app.model_config import model_capability_snapshot, resolve_model_params
from backend.app.processing_policy import EXECUTABLE_MODELS
from backend.app.evaluation_plan import DatasetView, build_evaluation_plan, select_train
from backend.app.train_evidence import compute_train_evidence
from backend.app.recipes import compile_recipe_catalog
from agent_poc.orchestration.diagnosis import build_input, report_for
from agent_poc.tests.test_step9_journal import setup


def full_state(awareness='off'):
    caps=model_capability_snapshot(search_revision=True)
    allowed=[m['id'] for m in caps['models'] if m['available'] and m['id'] in EXECUTABLE_MODELS]
    caps['model_configs']={m['id']:resolve_model_params(m['id'],policy=m) for m in caps['models'] if m['id'] in allowed}
    evaluation=dict(split_mode='stratified_holdout',split_train=8,split_valid=1,split_test=1)
    data=DatasetView(np.random.default_rng(3).normal(size=(60,128)).astype(np.float32),
        tuple(str(i//30) for i in range(60)),tuple('g'+str(i//2) for i in range(60)),tuple(float(i) for i in range(128)),'a'*64)
    plan=build_evaluation_plan(data,evaluation,42)
    evidence=compute_train_evidence(select_train(data,plan))
    task=dict(allowed_models=allowed,protocol_revision='agent-recipes-revision-v7',processing_mode='dynamic',
        search_mode='fixed',max_trials=1,seed=42,evaluation=evaluation,budget_policy_digest='a'*64,
        diagnosis_policy={},budget_awareness=awareness)
    catalog=compile_recipe_catalog(task,caps,evidence,plan).model_dump(mode='json')
    assert len(allowed)==13 and len(catalog['recipes'])==96
    return dict(identity=dict(task_id='task-1',thread_id='thread-1',session_id='session-1',principal_fingerprint='a'*64),
        task=task,diagnosis=dict(feedback_evidence=None),decision=dict(reason_code='recipe_training_budget_exceeded',decision_id='decision-1'),
        finalization=dict(termination_reason='budget_exhausted'),execution=dict(run_id=None,effective_config=None),
        versions=dict(llm_config_fingerprint='a'*64),evidence=dict(content=evidence.model_dump(mode='json')),
        knowledge=dict(snapshot=None),recipes=dict(catalog=catalog),guard=dict(reports=[]),
        module_policy=dict(context_policy=dict(evidence=True,risks=True)))


@pytest.mark.parametrize('awareness',['on','off'])
def test_complete_catalog_fits_bytes(awareness):
    state=full_state(awareness)
    balances={k:dict(known_actual=0,held_reserved=0,held_unknown=0,unknown_count=0,limit=100)
        for k in ('llm_calls','api_calls','input_tokens','output_tokens','cached_tokens','output_repairs','network_retries')}
    snapshot=build_input(state,after='terminated',generation=0,adapter=None,
        journal=SimpleNamespace(budget_summary=lambda:balances,snapshot=lambda:[]))
    ctx=snapshot['context']
    actions=[a for a in ctx['suggestion_catalog'] if a['kind']=='review_recipe']
    assert len(actions)==96
    assert {(a['target'],a['recipe_digest']) for a in actions}=={
        (r['recipe_id'],r['recipe_digest']) for r in state['recipes']['catalog']['recipes']}
    assert len(json.dumps(ctx,ensure_ascii=False,separators=(',',':')).encode())<=65536


@pytest.mark.parametrize('assessment',['no_issue_identified','insufficient_evidence'])
def test_assessment_journal_roundtrip(tmp_path,assessment):
    journal,_,snapshot=setup(tmp_path)
    journal.freeze_diagnosis_input(snapshot)
    proposal=dict(schema_version='diagnosis-proposal-v1',input_digest=snapshot['context']['input_digest'],
        assessment=assessment,hypotheses=[],suggestions=[])
    report=report_for(snapshot,policy={'rules_digest':'a'*64},calls=[],now=1.,proposal=SimpleNamespace(arguments=proposal))
    journal.save_diagnosis_report(report)
    restored=journal.diagnosis_report(snapshot['context']['input_digest'])
    assert restored['assessment']==assessment
    journal.close()



def success_evidence():
    # Explicit synthetic valid projected facts; no claim of real training evidence.
    role=dict(observation_count=20,sample_group_count=10,class_support=[dict(sample_group_count=5)])
    return dict(evidence_digest='a'*64,subject_event_id='event-1',outcome='succeeded',failure=dict(reason_code=None),
        guard_ref=dict(eligibility='eligible'),metric_pairs=[dict(metric=m,train=.8,valid=.6,delta=.2)
            for m in ('macro_f1','balanced_accuracy','accuracy')],reason_code=None,
        effective_execution=dict(status='ready',model_id='logistic_regression',normalization='zscore',class_balance='none',
            search_mode='fixed',selection_criterion='balanced_accuracy',selected_trial=0,best_epoch=None,
            selected_parameters={'C':1.0}),support=[dict(train=role,valid=role)],limitations=['no_causal_inference'],
        training_cost=dict(status='unavailable'))


def knowledge_fixture(n):
    entries=[dict(entry_id='synthetic-'+str(i),entry_version='1',related_recipe_ids=[],title='Synthetic capacity fixture',
        body=('Advisory context only. '*50)[:1100],sources=[dict(source_id='fixture',description='Synthetic fixture, not retrieval.')]) for i in range(n)]
    return dict(status='ready',projection_version='knowledge-rag-projection-v1',matched_count=n,provided_count=n,
        omitted_count=0,provided_entry_ids=[e['entry_id'] for e in entries],entries=entries)


@pytest.mark.parametrize('protocol',['json_action','native_tools'])
@pytest.mark.parametrize('awareness',['on','off'])
@pytest.mark.parametrize('outcome',['success','terminated'])
@pytest.mark.parametrize('cards',[0,6])
def test_full_pool_token_boundaries(tmp_path,budget_config,protocol,awareness,outcome,cards):
    from dataclasses import replace
    from agent_poc.orchestration.llm import LLMAdapter,LLMConfig
    state=full_state(awareness)
    if outcome=='success':state['diagnosis']['feedback_evidence']=success_evidence()
    state['knowledge']['snapshot']={'projection':knowledge_fixture(cards)}
    journal,_,_=setup(tmp_path)
    after='finalize_decision' if outcome=='success' else 'terminated'
    raw=build_input(state,after=after,generation=0,adapter=None,journal=journal)
    cfg=LLMConfig('http://fixture.invalid','fixture',protocol=protocol,diagnosis_phase=True,**budget_config)
    adapter=LLMAdapter(cfg)
    request=adapter._build_request('diagnose',raw['context'],'llm_output_invalid')[0]
    total=adapter._prompt_tokens(request)+cfg.max_tokens
    exact=LLMAdapter(replace(cfg,context_window=total));exact._budget_tokenizer=adapter._budget_tokenizer
    assert build_input(state,after=after,generation=0,adapter=exact,journal=journal)==raw
    tight=LLMAdapter(replace(cfg,context_window=total-1));tight._budget_tokenizer=adapter._budget_tokenizer
    result=build_input(state,after=after,generation=0,adapter=tight,journal=journal)
    if cards:
        assert result['context']['knowledge']['provided_count']<cards
        assert result['context']['suggestion_catalog']==raw['context']['suggestion_catalog']
        assert result['context']['facts']==raw['context']['facts']
        ids=result['context']['knowledge']['provided_entry_ids']
        assert ids==[c['entry_id'] for c in result['context']['knowledge']['entries']]
    else:
        assert result['context']['reason_code']=='diagnosis_context_too_long'
        assert result['context']['action_count']==97
        journal.freeze_diagnosis_input(result)
        assert journal.diagnosis_input(0)==result
    assert not journal.snapshot()
    journal.close()


def test_byte_capacity_is_distinct_from_corruption(tmp_path):
    from copy import deepcopy
    state=full_state()
    state['evidence']['content']['risks']*=2000
    snapshot=build_input(state,after='terminated',generation=0,adapter=None,journal=None)
    assert snapshot['context']['capacity_limit']=='bytes'
    assert snapshot['context']['source_context_bytes']>65536
    assert len(json.dumps(snapshot).encode())<4096
    journal,_,_=setup(tmp_path);journal.freeze_diagnosis_input(snapshot)
    assert journal.diagnosis_input(0)==snapshot
    report=report_for(snapshot,policy={'rules_digest':'a'*64},calls=[],now=1.,reason='diagnosis_context_too_long')
    journal.save_diagnosis_report(report)
    assert journal.diagnosis_report(snapshot['context']['input_digest'])['status']=='unavailable'
    broken=deepcopy(state);broken['evidence']['content']['risks'][0]['forged']=True
    with pytest.raises(ValueError):build_input(broken,after='terminated',generation=0,adapter=None,journal=None)
    assert not journal.snapshot();journal.close()


def test_historical_report_hash_and_unrecorded_assessment(tmp_path):
    from backend.app.agent.diagnosis import DiagnosisReport,digest
    journal,_,snapshot=setup(tmp_path);journal.freeze_diagnosis_input(snapshot)
    proposal=dict(schema_version='diagnosis-proposal-v1',input_digest=snapshot['context']['input_digest'],
        assessment='insufficient_evidence',hypotheses=[],suggestions=[])
    old=report_for(snapshot,policy={'rules_digest':'a'*64},calls=[],now=1.,proposal=SimpleNamespace(arguments=proposal))
    del old['assessment'];del old['proposal_digest']
    parsed=DiagnosisReport.model_validate(old).bind(snapshot)
    assert parsed.assessment=='historically_unrecorded'
    assert parsed.model_dump(mode='json')==old
    journal.connection.execute('INSERT INTO diagnosis_reports_v1 VALUES(?,?,?,?,?)',
        (journal.thread_id,old['diagnosis_id'],old['input_digest'],json.dumps(old),digest(old)))
    journal.connection.commit()
    assert journal.diagnosis_report(old['input_digest'])==old
    journal.close()


def test_assessment_cannot_change_without_original_proposal(tmp_path):
    from backend.app.agent.diagnosis import DiagnosisReport
    journal,_,snapshot=setup(tmp_path)
    proposal=dict(schema_version='diagnosis-proposal-v1',input_digest=snapshot['context']['input_digest'],
        assessment='insufficient_evidence',hypotheses=[],suggestions=[])
    report=report_for(snapshot,policy={'rules_digest':'a'*64},calls=[],now=1.,proposal=SimpleNamespace(arguments=proposal))
    report['assessment']='no_issue_identified'
    with pytest.raises(ValueError,match='validated proposal'):DiagnosisReport.model_validate(report).bind(snapshot)
    journal.close()
