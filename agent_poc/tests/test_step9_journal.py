"""Canonical immutable diagnosis storage and report reconstruction invariants."""
from copy import deepcopy
import json,shutil,time,sqlite3
import pytest
from backend.app.agent.diagnosis import DiagnosisInput,DiagnosisContext,DiagnosisPolicy,digest
from agent_poc.orchestration.runtime import _default_budget_policy
from agent_poc.orchestration.persistence import CallJournal,PersistenceError
from agent_poc.orchestration.diagnosis import report_for
from agent_poc.tests.test_step9_contracts import context


def setup(tmp_path):
    policy=_default_budget_policy(task_id='task-1',started_at=time.time(),timeout_seconds=600,
        allowed_models=['logistic_regression'],model_configs={},max_trials=1,max_llm_calls=6,max_api_calls=60,
        max_repair_attempts=2,max_operation_attempts=3,max_output_tokens=1024)
    journal=CallJournal(tmp_path/'calls.sqlite','thread-1');journal.bind_budget(policy)
    bindings={k:'a'*64 for k in ('task','thread','session','principal','run','budget_policy','diagnosis_policy','llm','evidence','effective_config','train_evidence','knowledge','catalog')}
    bindings.update(task=digest('task-1'),thread=digest('thread-1'),run=None)
    ctx=context();ctx['input_digest']=digest(dict(bindings=bindings,subject='event-1',generation=0))
    snapshot=DiagnosisInput(subject_event_id='event-1',generation=0,bindings=bindings,
        context=ctx,displayed_context_digest=digest(ctx),after_diagnosis='terminated').model_dump(mode='json')
    return journal,policy,snapshot


def test_immutable_inputs_reports_and_idempotent_migration(tmp_path):
    j,policy,snapshot=setup(tmp_path)
    j.freeze_diagnosis_input(snapshot);j.freeze_diagnosis_input(snapshot)
    report=report_for(snapshot,policy=DiagnosisPolicy(enabled=True,rules_digest='a'*64).model_dump(),calls=[],now=1.,reason='diagnosis_work_deadline')
    j.save_diagnosis_report(report);j.save_diagnosis_report(report)
    changed=deepcopy(report);changed['facts'][0]['value']=.5
    with pytest.raises(PersistenceError,match='immutable'):j.save_diagnosis_report(changed)
    j.close()
    for _ in range(2):
        j=CallJournal(tmp_path/'calls.sqlite','thread-1');j.bind_budget(policy)
        assert j.diagnosis_input(0)==snapshot
        assert j.diagnosis_report(snapshot['context']['input_digest'])==report
        assert j.connection.execute('SELECT count(*) FROM diagnosis_inputs_v1').fetchone()[0]==1
        assert not j.snapshot();j.close()


@pytest.mark.parametrize('table',['diagnosis_inputs_v1','diagnosis_reports_v1'])
def test_corrupt_persistence_not_reinterpreted_as_missing(tmp_path,table):
    j,policy,snapshot=setup(tmp_path);j.freeze_diagnosis_input(snapshot)
    report=report_for(snapshot,policy=DiagnosisPolicy(enabled=True,rules_digest='a'*64).model_dump(),calls=[],now=1.,reason='diagnosis_work_deadline')
    j.save_diagnosis_report(report)
    j.connection.execute(f"UPDATE {table} SET payload_json='{{}}'");j.connection.commit()
    with pytest.raises(PersistenceError,match='corrupt'):
        j.diagnosis_input(0) if table=='diagnosis_inputs_v1' else j.diagnosis_report(snapshot['context']['input_digest'])
    j.close()


def test_other_task_and_copied_journal_rejected(tmp_path):
    j,policy,snapshot=setup(tmp_path)
    changed=deepcopy(snapshot);changed['bindings']['task']=digest('other')
    changed['context']['input_digest']=digest(dict(bindings=changed['bindings'],subject='event-1',generation=0))
    changed['displayed_context_digest']=digest(changed['context'])
    with pytest.raises(PersistenceError,match='scope'):j.freeze_diagnosis_input(changed)
    j.close();copy=tmp_path/'copy';copy.mkdir();shutil.copyfile(tmp_path/'calls.sqlite',copy/'calls.sqlite')
    other=CallJournal(copy/'calls.sqlite','thread-1')
    with pytest.raises(PersistenceError,match='binding_conflict'):other.bind_budget(policy)
    other.close()


def test_mismatched_report_never_published(tmp_path):
    j,policy,snapshot=setup(tmp_path);j.freeze_diagnosis_input(snapshot)
    report=report_for(snapshot,policy=DiagnosisPolicy(enabled=True,rules_digest='a'*64).model_dump(),calls=[],now=1.,reason='diagnosis_work_deadline')
    report['facts'][0]['value']=.6
    with pytest.raises(ValueError,match='differs from immutable'):j.save_diagnosis_report(report)
    assert j.diagnosis_report(snapshot['context']['input_digest']) is None
    j.close()
