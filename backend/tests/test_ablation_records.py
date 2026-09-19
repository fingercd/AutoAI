import json
import pytest
from scripts.agent_ablation import baseline, save


class Response:
    def __init__(self,body):self.body=body
    def raise_for_status(self):pass
    def json(self):return self.body


class Client:
    def __init__(self,path,lose=False):self.path=path;self.posts=0;self.lose=lose
    def post(self,path,json):
        self.posts+=1
        assert self.path.exists()
        assert __import__('json').loads(self.path.read_text())['status']=='attempt_started'
        if self.lose:raise TimeoutError('response lost')
        return Response({'run_id':'run-one'})
    def get(self,path):return Response({'state':'succeeded'})


def test_baseline_lost_response_never_reposts(tmp_path):
    path=tmp_path/'record.json';record={'status':'prepared','training_request':{}}
    client=Client(path,lose=True)
    baseline(client,record,path,1)
    assert record['status']=='submission_uncertain' and client.posts==1
    restored=json.loads(path.read_text())
    baseline(client,restored,path,1)
    assert client.posts==1 and restored['status']=='submission_uncertain'


def test_baseline_crash_before_response_is_ambiguous(tmp_path):
    path=tmp_path/'record.json';record={'status':'attempt_started','training_request':{}}
    save(path,record);client=Client(path)
    baseline(client,record,path,1)
    assert client.posts==0 and record['status']=='submission_uncertain'


def test_baseline_bound_run_resumes_without_new_post(tmp_path):
    path=tmp_path/'record.json';record={'status':'prepared','training_request':{}}
    client=Client(path)
    baseline(client,record,path,1)
    assert client.posts==1 and record['status']=='completed'
    record['status']='submitted'
    baseline(client,record,path,1)
    assert record['run_id']=='run-one' and client.posts==1


def test_terminal_record_keeps_only_aggregate_metrics_and_bound_artifacts():
    from scripts.agent_ablation import terminal_results
    class Results:
        def get(self,path):
            if path.endswith('/result'):
                return Response({'metrics':{'primary':{'macro_f1':.7,'confusion_matrix':[]},
                    'direct':{'valid':{'macro_f1':.8,'classification_report':{}}}},
                    'artifacts':[{'name':'model_metadata.json','download_url':'/api/training/runs/run-one/artifact/model_metadata.json'}]})
            return Response({'execution_audit':{'evaluation_plan':{'dataset_sha256':'a'*64},'partition_digest':'b'*64}})
    record={'run_id':'run-one'}
    terminal_results(Results(),record)
    assert record['validation']=={'macro_f1':.8}
    assert record['offline_test']=={'macro_f1':.7}
    assert record['dataset_digest']=='a'*64 and record['actual_split_digest']=='b'*64


def test_terminal_record_rejects_unbound_artifact_url():
    from scripts.agent_ablation import terminal_results
    class Results:
        def get(self,path):
            return Response({'artifacts':[{'name':'split.json','download_url':'https://unbound.invalid/split'}]})
    with pytest.raises(ValueError):terminal_results(Results(),{'run_id':'run-one'})


@pytest.fixture
def recorded_cli(tmp_path, monkeypatch):
    """Real CLI persistence with an in-memory HTTP transport; no remote requests."""
    import httpx
    from scripts import agent_ablation as ablation
    from scripts import agent_step2_acceptance as acceptance
    binding=dict(head='a'*40,dirty_diff_sha256='b'*64,source_digest='c'*64)
    monkeypatch.setattr(acceptance,'code_binding',lambda:dict(binding))
    monkeypatch.setattr(ablation,'terminal_results',lambda client,record:None)
    requests=[];window=[None]
    def respond(request):
        requests.append(request.method)
        if request.method=='POST':
            if window[0]=='post':raise SystemExit('simulated process termination')
            return httpx.Response(200,json={'run_id':'run-one'})
        if window[0]=='poll':raise SystemExit('simulated process termination')
        return httpx.Response(200,json={'state':'succeeded'})
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(**kwargs,transport=httpx.MockTransport(respond)))
    argv=['--experiment-id','row','--kind','baseline','--backend-url','http://fixture.invalid',
        '--dataset-id','ds-one','--scope-key','test','--storage',str(tmp_path)]
    return ablation,argv,tmp_path/'row'/'record.json',binding,requests,window


@pytest.mark.parametrize('field',['source_digest','head','dirty_diff_sha256'])
def test_resume_rejects_changed_code_before_http(recorded_cli,field):
    ablation,argv,path,binding,requests,_=recorded_cli
    assert ablation.main(argv)==0
    original=json.loads(path.read_text())['source']
    count=len(requests);binding[field]='d'*len(binding[field])
    assert ablation.main(argv)==2
    record=json.loads(path.read_text())
    assert len(requests)==count and record['source']==original
    assert record['attempts'][-1]['status']=='rejected_source_change'
    assert record['attempts'][-1]['source'][field]==binding[field]
    assert record['attempts'][-1]['remote_execution_versions']=={'web':'unknown','worker':'unknown'}


def test_same_source_cli_resume_keeps_exact_counts_and_run(recorded_cli):
    ablation,argv,path,_,requests,_=recorded_cli
    assert ablation.main(argv)==ablation.main(argv)==0
    record=json.loads(path.read_text())
    assert requests==['POST','GET'] and record['run_id']=='run-one'
    assert record['api_calls']==2 and record['measurement']['status']=='complete'
    assert [a['measurement']['http_calls'] for a in record['attempts']]==[2,0]
    assert record['elapsed_seconds']>=0


def test_agent_start_failure_keeps_original_reason_without_frozen_config(recorded_cli,monkeypatch):
    from agent_poc.orchestration import runtime
    from agent_poc.orchestration.state import new_state
    ablation,argv,path,_,requests,_=recorded_cli
    state=new_state(dataset_id='ds-one',allowed_models=['logistic_regression'],backend_fingerprint='a'*64,
        principal_fingerprint='b'*64,llm_config_fingerprint='c'*64,wire_version='agent-state-v4')
    state['lifecycle'].update(status='needs_attention',reason_code='contract_validation_failed')
    monkeypatch.setattr(runtime,'start_task',lambda *args,**kwargs:state)
    argv[argv.index('baseline')]='agent'
    argv+=['--llm-url','http://fixture.invalid/v1','--llm-model','fixture']
    assert ablation.main(argv)==2
    record=json.loads(path.read_text())
    assert record['failure_reason']=='contract_validation_failed'
    assert record['frozen_model_configs'] is None and record['run_id'] is None


@pytest.mark.parametrize('window_name',['post','poll'])
def test_unsettled_post_and_poll_recovery_preserves_uncertainty(recorded_cli,monkeypatch,window_name):
    ablation,argv,path,_,requests,window=recorded_cli
    finish=ablation.finish_attempt
    window[0]=window_name
    # SIGKILL would skip finally. Keep the exact persisted snapshot by suppressing settlement.
    monkeypatch.setattr(ablation,'finish_attempt',lambda *args:None)
    with pytest.raises(SystemExit):ablation.main(argv)
    pending=json.loads(path.read_text())
    assert pending['attempts'][-1]['status']=='running'
    assert pending['status']==('attempt_started' if window_name=='post' else 'submitted')
    monkeypatch.setattr(ablation,'finish_attempt',finish);window[0]=None
    assert ablation.main(argv)==(2 if window_name=='post' else 0)
    record=json.loads(path.read_text())
    assert requests.count('POST')==1
    assert record['record_http_calls']==record['api_calls']==record['elapsed_seconds']=='unknown'
    assert record['measurement']['status']=='incomplete'
    assert record['measurement']['known_http_calls']==(0 if window_name=='post' else 1)
    assert record['attempts'][0]['measurement']['status']=='incomplete'
    assert record['attempts'][1]['measurement']['status']=='complete'


def test_prior_known_counts_survive_an_interrupted_attempt(tmp_path):
    from scripts.agent_ablation import begin_attempt,finish_attempt
    path=tmp_path/'record.json';source={'head':'a'}
    record=dict(source=source,attempts=[],configuration={'kind':'baseline'},status='submitted')
    first=begin_attempt(record,path,source);finish_attempt(record,path,first,7,2.5)
    begin_attempt(record,path,source)  # Never settled: emulate the last durable write before SIGKILL.
    record=json.loads(path.read_text())
    resumed=begin_attempt(record,path,source);finish_attempt(record,path,resumed,1,1.5)
    assert record['api_calls']==record['elapsed_seconds']=='unknown'
    assert record['measurement']['known_http_calls']==8
    assert record['measurement']['known_elapsed_seconds']==4


def test_legacy_record_does_not_gain_false_measurement_certainty(tmp_path):
    from scripts.agent_ablation import begin_attempt,finish_attempt
    path=tmp_path/'record.json';source={'head':'a'}
    record=dict(source=source,configuration={'kind':'baseline'},status='attempt_started',record_http_calls=5,elapsed_seconds=2)
    attempt=begin_attempt(record,path,source);finish_attempt(record,path,attempt,0,1)
    assert record['source']==source and record['api_calls']=='unknown'
    assert record['measurement']['known_http_calls']==5
    assert record['measurement']['known_elapsed_seconds']==3
    assert record['legacy_measurements']['status']=='unverified'
