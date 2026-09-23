import json

import pytest

from backend.app.processing_policy import EXECUTABLE_MODELS
from backend.app.models import model_family
from scripts.agent_ablation import (
    DECISION_TERMINAL_STATES,
    collect_processing_plan,
    digest,
    load_processing_plan,
    register_processing_plan,
    save,
    summarize_processing_plan,
)

BACKEND_URL='http://fixture.invalid'


def plan_config():
    pools = {
        family: [model for model in sorted(EXECUTABLE_MODELS)
                 if model_family(model) == kind]
        for family, kind in [('ML', 'traditional_ml'), ('DL', 'deep_learning')]
    }
    return dict(
        tasks=[dict(task_id=f'task-{index}', dataset_id=f'ds-{index}',
                    dataset_sha256=f'{index:064x}') for index in range(10)],
        model_pools=pools,
        fixed_processing={model:dict(normalization='zscore', class_balance='none')
                          for model in EXECUTABLE_MODELS},
        seed=42, repeat=0,
        conditions={name:letter*64 for name,letter in [
            ('backend_binding','a'),('scope_binding','b'),
            ('llm_binding','c'),('model_configs_digest','d')]},
    )


def registered_config():
    config=plan_config()
    config['conditions']['backend_binding']=digest(BACKEND_URL)
    config['conditions']['model_configs_digest']=digest({})
    return config


def write_record_rows(tmp_path,plan,*,status='completed'):
    for row in plan['rows']:
        config=dict(plan_digest=plan['plan_digest'],processing_mode=row['processing_mode'],
            dataset_id=row['dataset_id'],seed=42,allowed_models=sorted(row['allowed_models']),
            fixed_processing={model:plan['fixed_processing'][model] for model in row['allowed_models']},
            model_configs={},backend_binding=plan['conditions']['backend_binding'],
            scope_binding=plan['conditions']['scope_binding'],llm_binding=plan['conditions']['llm_binding'])
        save(tmp_path/row['experiment_id']/'record.json',dict(
            experiment_id=row['experiment_id'],configuration=config,status=status,
            run_id='run-'+row['experiment_id'],dataset_digest=row['dataset_sha256'],
            actual_split_digest='a'*64,frozen_model_configs={'model':'fixed'},
            actual_processing=dict(model_type='svm',normalization='zscore',class_balance='none'),
            measurement=dict(status='complete',known_http_calls=1)))


def fake_checkpoint(row,record,*,needs_human_review=False):
    return dict(identity={'thread_id':row['experiment_id']},
        task={'dataset_id':row['dataset_id'],'dataset_fingerprint':record['dataset_digest'],
              'processing_mode':row['processing_mode']},
        lifecycle={'status':record['status'],'next_action':None,'ended_at':1},
        recovery={'needs_human_review':needs_human_review,'pending_operation':None},
        execution={'run_id':record['run_id']},
        finalization={'status':'confirmed','selected_run_id':record['run_id'],
                      'backend_session_state':'finalized'})


def mock_checkpoints(monkeypatch,storage,plan,*,needs_human_review=False):
    from agent_poc.orchestration import runtime
    rows={row['experiment_id']:row for row in plan['rows']}
    def read(*,storage,thread_id):
        row=rows[thread_id]
        record=json.loads((storage.parent/'record.json').read_text())
        return fake_checkpoint(row,record,needs_human_review=needs_human_review)
    monkeypatch.setattr(runtime,'read_status',read)


def test_processing_registration_freezes_forty_interleaved_rows(tmp_path):
    plan=register_processing_plan(plan_config(),'comparison')
    path=tmp_path/'plan.json'
    save(path,plan)
    assert load_processing_plan(path)==plan
    assert len(plan['rows'])==40
    assert {(row['task_id'],row['family']) for row in plan['rows']}=={
        (f'task-{index}',family) for index in range(10) for family in ('ML','DL')}
    assert all(plan['rows'][index]['processing_mode']!=plan['rows'][index+1]['processing_mode']
               for index in range(0,40,2))
    assert plan['rows'][0]['processing_mode']!=plan['rows'][2]['processing_mode']
    changed=json.loads(path.read_text())
    changed['rows'][0]['seed']=43
    save(path,changed)
    with pytest.raises(ValueError,match='digest mismatch'):
        load_processing_plan(path)


def test_processing_collection_waits_until_every_decision_is_terminal(tmp_path):
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    with pytest.raises(ValueError,match='all planned decisions'):
        collect_processing_plan(path,tmp_path,BACKEND_URL)


def test_processing_summary_keeps_missing_rows_and_pair_denominator():
    plan=register_processing_plan(plan_config(),'comparison')
    first,second=plan['rows'][:2]
    records={
        first['experiment_id']:dict(status='completed',run_id='run-a',dataset_digest=first['dataset_sha256'],
            actual_split_digest='a'*64,frozen_model_configs={'model':'fixed'},
            offline_test=dict(macro_f1=.8,balanced_accuracy=.7,accuracy=.9),
            actual_processing=dict(model_type='svm',normalization='minmax',class_balance='none'),
            measurement=dict(status='complete',known_http_calls=3)),
        second['experiment_id']:dict(status='completed',run_id='run-b',dataset_digest=second['dataset_sha256'],
            actual_split_digest='a'*64,frozen_model_configs={'model':'fixed'},
            offline_test=dict(macro_f1=.6,balanced_accuracy=.5,accuracy=.7),
            actual_processing=dict(model_type='svm',normalization='zscore',class_balance='none'),
            measurement=dict(status='complete',known_http_calls=4)),
    }
    result=summarize_processing_plan(plan,records)
    assert result['paired_count']==1 and result['unpaired_count']==19
    assert result['paired_mean_difference']['macro_f1']==pytest.approx(.2)
    assert all(group['planned']==20 and group['succeeded']==1
               for group in result['groups'].values())
    assert all(group['metrics']['macro_f1']['all']['complete_mean'] is None
               for group in result['groups'].values())
    assert result['complete'] is False


def test_offline_collection_records_failure_without_erasing_other_rows(tmp_path,monkeypatch):
    import httpx
    from scripts import agent_ablation
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    write_record_rows(tmp_path,plan)
    mock_checkpoints(monkeypatch,tmp_path,plan)
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
    monkeypatch.setattr(httpx,'Client',Client)
    def fake_result(client,record):
        if record['experiment_id'].endswith('-01'):
            raise RuntimeError('offline read failed')
        record['offline_test']=dict(macro_f1=.5,balanced_accuracy=.5,accuracy=.5)
    monkeypatch.setattr(agent_ablation,'terminal_results',fake_result)
    result=collect_processing_plan(path,tmp_path,BACKEND_URL)
    failed=json.loads((tmp_path/plan['rows'][0]['experiment_id']/'record.json').read_text())
    assert failed['offline_test_failure']=='RuntimeError' and 'offline_test' not in failed
    assert result['paired_count']==19 and result['complete'] is False


@pytest.mark.parametrize('status',['waiting','initializing','recovering','unknown','running','submitted'])
def test_nonterminal_row_blocks_all_test_reads(tmp_path,monkeypatch,status):
    import httpx
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    write_record_rows(tmp_path,plan)
    first=tmp_path/plan['rows'][0]['experiment_id']/'record.json'
    record=json.loads(first.read_text())
    record['status']=status
    save(first,record)
    mock_checkpoints(monkeypatch,tmp_path,plan)
    calls=[]
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(ValueError,match='all planned decisions'):
        collect_processing_plan(path,tmp_path,BACKEND_URL)
    assert calls==[]


@pytest.mark.parametrize('status',sorted(DECISION_TERMINAL_STATES))
def test_closed_terminal_row_allows_collection(tmp_path,monkeypatch,status):
    import httpx
    from scripts import agent_ablation
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    write_record_rows(tmp_path,plan)
    first=tmp_path/plan['rows'][0]['experiment_id']/'record.json'
    record=json.loads(first.read_text())
    record['status']=status
    save(first,record)
    mock_checkpoints(monkeypatch,tmp_path,plan)
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
    monkeypatch.setattr(httpx,'Client',Client)
    monkeypatch.setattr(agent_ablation,'terminal_results',lambda client,record:record.update(
        offline_test=dict(macro_f1=.5,balanced_accuracy=.5,accuracy=.5)))
    result=collect_processing_plan(path,tmp_path,BACKEND_URL)
    assert result['groups']['dynamic']['succeeded']+result['groups']['fixed']['succeeded']==(
        40 if status=='completed' else 39)


def test_unresolved_terminal_checkpoint_blocks_collection(tmp_path,monkeypatch):
    import httpx
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    write_record_rows(tmp_path,plan)
    mock_checkpoints(monkeypatch,tmp_path,plan,needs_human_review=True)
    calls=[]
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(ValueError,match='checkpoint is not closed'):
        collect_processing_plan(path,tmp_path,BACKEND_URL)
    assert calls==[]


def test_different_collection_backend_is_rejected_before_http(tmp_path,monkeypatch):
    import httpx
    plan=register_processing_plan(registered_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    calls=[]
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(ValueError,match='backend differs'):
        collect_processing_plan(path,tmp_path,'http://different-backend.invalid')
    assert calls==[]


def test_invalid_pair_is_excluded_from_formal_aggregates():
    plan=register_processing_plan(registered_config(),'comparison')
    records={}
    for row in plan['rows']:
        records[row['experiment_id']]=dict(status='completed',run_id='run-'+row['experiment_id'],
            dataset_digest=row['dataset_sha256'],actual_split_digest='a'*64,
            frozen_model_configs={'model':'fixed'},
            offline_test=dict(macro_f1=.5,balanced_accuracy=.5,accuracy=.5))
    records[plan['rows'][0]['experiment_id']]['actual_split_digest']='b'*64
    result=summarize_processing_plan(plan,records)
    assert result['condition_counts']=={'matched':19,'mismatch':1,'unavailable':0}
    assert result['paired_count']==19 and result['unpaired_count']==1
    affected=next(pair for pair in result['paired'] if pair['task_id']=='task-0' and pair['family']=='ML')
    assert affected['difference'] is None
    assert affected['exclusion_reason']=='split_mismatch'
    assert result['paired_complete_mean_difference']['macro_f1'] is None
    assert result['groups']['dynamic']['metrics']['macro_f1']['all']['complete_mean'] is None
    assert result['groups']['fixed']['metrics']['macro_f1']['all']['complete_mean'] is None
    assert result['groups']['dynamic']['metrics']['macro_f1']['DL']['complete_mean']==.5
