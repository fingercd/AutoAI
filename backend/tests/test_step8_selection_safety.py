"""Selection protocol faults. Injected scores are not model quality evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import sqlite3
import pytest

from backend.tests.test_agent_model_sessions import api
from backend.tests.test_step8_guard_integration import submit, HEADERS
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker, execute_with_budget_supervision


def test_zero_validation_and_private_test_payload_can_finalize_without_leak(api):
    from backend.app.runs.artifacts import RunArtifactWriter
    from backend.app.runs.guard import METRICS, check_outputs
    client, storage, _ = api
    base, run_id, _ = submit(api)
    repo = RunRepository(storage/'runs.sqlite3')
    def execute(record):
        result = execute_with_budget_supervision(record,repository=repo,agent_database=storage/'agent.sqlite3')
        root = storage/'runs'/run_id
        metrics = json.loads((root/'metrics.json').read_bytes())
        # Deterministic score fixture, after actual fitting. This does not claim
        # that the trained model measured zero on its validation dataset.
        metrics['valid'].update({key:0.0 for key in METRICS})
        metrics['test']['private_canary']='TEST_PRIVATE_CANARY'
        cv = json.loads((root/'cv_metrics.json').read_bytes()); cv['metrics']=metrics
        metadata = json.loads((root/'model_metadata.json').read_bytes())
        metadata['private_note']='ARTIFACT_PATH_CANARY'
        for name, body in [('metrics.json',metrics),('cv_metrics.json',cv),('model_metadata.json',metadata)]:
            (root/name).write_text(json.dumps(body),encoding='utf-8')
        RunArtifactWriter(root).finalize(run_id=run_id)
        # Simulate the child's valid final snapshot, using the actual guard.
        checked=check_outputs(record=record,repository=repo,run_dir=root,stage='pre_publish',usage=result['_guard_usage'])
        result['guard_report_id']=checked.report_id
        return result
    runner=RunWorker(repository=repo,worker_id='zero-protocol',now=lambda:datetime.now(timezone.utc),execute=execute)
    assert runner.run_once()
    assert repo.get(run_id).state=='succeeded'
    feedback=client.get(base+f'/experiments/{run_id}/feedback',headers=HEADERS).json()
    assert feedback['validation_score']==0.0
    assert feedback['extensions']['guard']['eligibility']=='eligible'
    assert 'CANARY' not in json.dumps(feedback)
    assert client.get(base,headers=HEADERS).json()['best_run_id']==run_id
    response=client.post(base+'/finalize',headers=HEADERS,json={'selected_run_id':run_id})
    assert response.status_code==200,response.text


def test_finalize_terminate_race_has_one_terminal_state(api):
    client, storage, _ = api
    base, run_id, _=submit(api)
    repo=RunRepository(storage/'runs.sqlite3')
    runner=RunWorker(repository=repo,worker_id='terminal-race',now=lambda:datetime.now(timezone.utc),
        execute=lambda r:execute_with_budget_supervision(r,repository=repo,agent_database=storage/'agent.sqlite3'))
    assert runner.run_once() and repo.get(run_id).state=='succeeded'
    with ThreadPoolExecutor(max_workers=2) as pool:
        final=pool.submit(client.post,base+'/finalize',headers=HEADERS,json={'selected_run_id':run_id})
        terminate=pool.submit(client.post,base+'/terminate',headers=HEADERS,
            json={'client_request_id':'terminal-race','reason':'operator_stop'})
        responses=[final.result(),terminate.result()]
    assert sorted(r.status_code for r in responses)==[200,409],[r.text for r in responses]
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        selected,finalized,terminated=db.execute('SELECT selected_run_id,finalized_at,terminated_at FROM agent_sessions_v1').fetchone()
    assert (finalized is not None)!=(terminated is not None)
    assert (selected==run_id)==(finalized is not None)


def test_guard_on_off_preserves_scientific_recipe_and_disabled_is_not_passed(api):
    from backend.tests.test_step7_v5_session import _request
    client,storage,dataset=api
    recipes=[]
    for switch in ('on','off'):
        body=_request(dataset)
        body.update(protocol_revision='agent-recipes-revision-v6',fail_fast_guard=switch,client_request_id=switch)
        body['budget_policy']['task_id']=switch
        if switch=='on':body['modules'].append('fail_fast_guard')
        response=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
        assert response.status_code==201,response.text
        session=response.json(); prep=session['locked_config']['preparation']; catalog=prep['catalog']
        recipes.append(catalog)
        recipe=catalog['recipes'][0]
        queued=client.post('/api/agent/v2/sessions/'+session['session_id']+'/experiments',headers=HEADERS,json={
            'recipe_id':recipe['recipe_id'],'recipe_digest':recipe['recipe_digest'],
            'catalog_digest':catalog['catalog_digest'],'knowledge_refs':[],'client_request_id':switch})
        assert queued.status_code==202,queued.text
    assert recipes[0]==recipes[1]
    with sqlite3.connect(storage/'runs.sqlite3') as db:
        statuses=[json.loads(v)['status'] for v, in db.execute("SELECT report_json FROM run_guard_reports_v1 WHERE stage='admission'")]
    assert sorted(statuses)==['disabled','passed']


def test_incomplete_usage_check_is_read_only_and_cannot_pass(tmp_path):
    from backend.tests.test_step7_training_budget import _setup, _summary
    from backend.app.runs.guard import GuardError
    _,_,_,ledger=_setup(tmp_path)
    ledger.bind()
    ledger.enter('fit-1',dimension='model_fits',kind='trial')
    before=_summary(ledger.database_path,'model_fits')
    for _ in range(2):
        with pytest.raises(GuardError):ledger.guard_snapshot(settled=False)
    assert _summary(ledger.database_path,'model_fits')==before


@pytest.mark.parametrize('field',['execution_guard_policy','execution_guard_policy_digest'])
def test_human_request_cannot_forge_guard_marker(api,field):
    client,storage,dataset=api
    response=client.post('/api/training/runs',json={'dataset_id':dataset,
        'config':{'model_type':'logistic_regression',field:'forged'}})
    assert response.status_code==422,response.text
    assert not RunRepository(storage/'runs.sqlite3').list()
