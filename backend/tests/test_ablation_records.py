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
