import json

import pytest

from backend.app.processing_policy import EXECUTABLE_MODELS
from backend.app.models import model_family
from scripts.agent_ablation import (
    collect_processing_plan,
    digest,
    load_processing_plan,
    register_processing_plan,
    save,
    summarize_processing_plan,
)


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
    plan=register_processing_plan(plan_config(),'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    with pytest.raises(ValueError,match='all planned decisions'):
        collect_processing_plan(path,tmp_path,'http://unused.invalid')


def test_processing_summary_keeps_missing_rows_and_pair_denominator():
    plan=register_processing_plan(plan_config(),'comparison')
    first,second=plan['rows'][:2]
    records={
        first['experiment_id']:dict(status='completed',run_id='run-a',
            offline_test=dict(macro_f1=.8,balanced_accuracy=.7,accuracy=.9),
            actual_processing=dict(model_type='svm',normalization='minmax',class_balance='none'),
            measurement=dict(status='complete',known_http_calls=3)),
        second['experiment_id']:dict(status='completed',run_id='run-b',
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
    config=plan_config()
    config['conditions']['model_configs_digest']=digest({})
    plan=register_processing_plan(config,'comparison')
    path=tmp_path/'comparison'/'plan.json'
    save(path,plan)
    for row in plan['rows']:
        config=dict(plan_digest=plan['plan_digest'],processing_mode=row['processing_mode'],
            dataset_id=row['dataset_id'],seed=42,allowed_models=sorted(row['allowed_models']),
            fixed_processing={model:plan['fixed_processing'][model] for model in row['allowed_models']},
            model_configs={},backend_binding=plan['conditions']['backend_binding'],
            scope_binding=plan['conditions']['scope_binding'],llm_binding=plan['conditions']['llm_binding'])
        save(tmp_path/row['experiment_id']/'record.json',dict(
            experiment_id=row['experiment_id'],configuration=config,status='completed',
            run_id='run-'+row['experiment_id'],dataset_digest=row['dataset_sha256'],
            actual_split_digest='a'*64,frozen_model_configs={'model':'fixed'},
            actual_processing=dict(model_type='svm',normalization='zscore',class_balance='none'),
            measurement=dict(status='complete',known_http_calls=1)))
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
    result=collect_processing_plan(path,tmp_path,'http://fixture.invalid')
    failed=json.loads((tmp_path/plan['rows'][0]['experiment_id']/'record.json').read_text())
    assert failed['offline_test_failure']=='RuntimeError' and 'offline_test' not in failed
    assert result['paired_count']==19 and result['complete'] is False
