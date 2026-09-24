"""The formal comparison can be registered and summarized without running it."""
import copy
from backend.app.model_catalog import MODELS_BY_ID
from backend.app.model_config import model_policy
from backend.app.processing_policy import EXECUTABLE_MODELS, freeze_fixed_processing
from scripts.agent_ablation import register_search_plan, summarize_search_plan


def test_search_registration_has_forty_distinct_pending_rows():
    models=sorted(EXECUTABLE_MODELS)
    config=dict(
        tasks=[dict(task_id=f'task{i}',dataset_id=f'dataset{i}',
                    dataset_sha256=f'{i:064x}') for i in range(10)],
        model_pools={family:[m for m in models if MODELS_BY_ID[m].execution_family==execution]
                     for family,execution in (('ML','traditional_ml'),('DL','deep_learning'))},
        fixed_processing=freeze_fixed_processing(models,None),
        model_configs={m:{item['name']:item['default'] for item in model_policy(m,search_revision=True)['parameters']}
                       for m in models},
        seed=42,repeat=0,bounded_max_trials=6,
        conditions={key:'a'*64 for key in ('backend_binding','scope_binding','llm_binding',
                                          'source_binding','tokenizer_binding','prompt_binding')})
    plan=register_search_plan(config,'step6-formal')
    assert len(plan['rows'])==40
    assert len({r['experiment_id'] for r in plan['rows']})==40
    assert {r['search_mode'] for r in plan['rows']}=={'fixed','bounded'}
    assert all(r['processing_mode']=='fixed' for r in plan['rows'])
    assert all(r['max_trials']==(1 if r['search_mode']=='fixed' else 6)
               for r in plan['rows'])
    summary=summarize_search_plan(plan,{})
    assert summary['paired_count']==0
    assert not summary['complete']
    assert summary['unpaired_count']==20


def _complete_search_records(plan):
    records={}
    for row in plan['rows']:
        model='logistic_regression' if row['family']=='ML' else 'cnn1d'
        records[row['experiment_id']]=dict(
            status='completed', run_id='run-'+row['experiment_id'],
            dataset_digest=row['dataset_sha256'], actual_split_digest='b'*64,
            frozen_model_configs={m:plan['baseline_configs'][m] for m in row['allowed_models']},
            candidate_pool=sorted(row['allowed_models']),
            actual_processing=dict(model_type=model, **plan['fixed_processing'][model]),
            source={'source_digest':'a'*64},
            configuration={k:plan['conditions'][k] for k in
                           ('backend_binding','scope_binding','llm_binding')},
            offline_test=dict(macro_f1=.5, balanced_accuracy=.5, accuracy=.5),
            budget=dict(llm_calls={'actual':2.0}, api_calls={'actual':10.0},
                        input_tokens={'actual':100.0}, output_tokens={'actual':5.0},
                        cached_tokens={'actual':0.0}),
            measurement=dict(status='complete',known_http_calls=0),
            search_summary=dict(integrity='complete', candidate_fit_count=1,
                                final_refit_count=1, actual_epochs=0,
                                training_batches=0))
    return records


def test_search_pairs_allow_different_legal_models_and_reject_processing_drift():
    models=sorted(EXECUTABLE_MODELS)
    config=dict(
        tasks=[dict(task_id=f'task{i}',dataset_id=f'dataset{i}',
                    dataset_sha256=f'{i:064x}') for i in range(10)],
        model_pools={family:[m for m in models if MODELS_BY_ID[m].execution_family==execution]
                     for family,execution in (('ML','traditional_ml'),('DL','deep_learning'))},
        fixed_processing=freeze_fixed_processing(models,None),
        model_configs={m:{item['name']:item['default'] for item in
                          model_policy(m,search_revision=True)['parameters']} for m in models},
        seed=42,repeat=0,bounded_max_trials=6,
        conditions={key:'a'*64 for key in ('backend_binding','scope_binding','llm_binding',
                                          'source_binding','tokenizer_binding','prompt_binding')})
    plan=register_search_plan(config,'step6-comparison')
    records=_complete_search_records(plan)
    bounded=next(r for r in plan['rows'] if r['task_id']=='task0' and
                 r['family']=='ML' and r['search_mode']=='bounded')
    records[bounded['experiment_id']]['actual_processing']['model_type']='svm'
    matched=summarize_search_plan(plan,records)
    assert matched['paired_count']==20 and matched['complete']
    pair=next(p for p in matched['paired'] if p['task_id']=='task0' and p['family']=='ML')
    assert (pair['fixed_model'],pair['bounded_model'])==('logistic_regression','svm')
    drift=copy.deepcopy(records)
    drift[bounded['experiment_id']]['actual_processing']['normalization']='minmax'
    rejected=summarize_search_plan(plan,drift)
    assert rejected['paired_count']==19 and not rejected['complete']
    assert next(p for p in rejected['paired'] if p['task_id']=='task0' and
                p['family']=='ML')['exclusion_reason']=='processing_mismatch'
    outside=copy.deepcopy(records)
    outside[bounded['experiment_id']]['actual_processing']['model_type']='cnn1d'
    rejected=summarize_search_plan(plan,outside)
    assert rejected['paired_count']==19
    assert next(p for p in rejected['paired'] if p['task_id']=='task0' and
                p['family']=='ML')['exclusion_reason']=='selected_model_outside_pool'


def test_search_costs_keep_separate_known_and_unknown_denominators():
    models=sorted(EXECUTABLE_MODELS)
    config=dict(tasks=[dict(task_id=f'task{i}',dataset_id=f'dataset{i}',
                            dataset_sha256=f'{i:064x}') for i in range(10)],
        model_pools={family:[m for m in models if MODELS_BY_ID[m].execution_family==execution]
                     for family,execution in (('ML','traditional_ml'),('DL','deep_learning'))},
        fixed_processing=freeze_fixed_processing(models,None),
        model_configs={m:{item['name']:item['default'] for item in
                          model_policy(m,search_revision=True)['parameters']} for m in models},
        seed=42,repeat=0,bounded_max_trials=6,
        conditions={key:'a'*64 for key in ('backend_binding','scope_binding','llm_binding',
                                          'source_binding','tokenizer_binding','prompt_binding')})
    plan=register_search_plan(config,'step6-costs')
    records=_complete_search_records(plan)
    complete=summarize_search_plan(plan,records)['costs']
    for mode in ('fixed','bounded'):
        assert complete[mode]['llm_call_count_known']==40
        assert complete[mode]['graph_api_calls_known']==200
        assert complete[mode]['http_calls_known']==200
        assert complete[mode]['input_tokens_known']==2000
        assert complete[mode]['llm_call_count_known_count']==20
        assert complete[mode]['llm_call_count_unknown_count']==0
        assert complete[mode]['cached_tokens_known_count']==20
    fixed=next(row for row in plan['rows'] if row['search_mode']=='fixed')
    partial=copy.deepcopy(records)
    record=partial[fixed['experiment_id']]
    record['search_summary'].update(integrity='incomplete',final_refit_count=None,
        actual_epochs=None,actual_epochs_known=2,actual_epochs_known_trials=1,
        training_batches=None,training_batches_known=3,training_batches_known_trials=1)
    record['budget']['llm_calls']={'actual':None}
    record['budget']['api_calls']={'actual':4.0,'unknown_pending':1}
    record['budget']['input_tokens']={'actual':None}
    record['measurement']={'status':'incomplete','known_http_calls':1}
    costs=summarize_search_plan(plan,partial)['costs']['fixed']
    assert costs['candidate_fits_known']==20 and costs['candidate_fits_unknown_count']==0
    assert costs['final_refits_known']==19 and costs['final_refits_unknown_count']==1
    assert costs['epochs_known']==2 and costs['epochs_unknown_count']==1
    assert costs['training_batches_known']==3 and costs['training_batches_unknown_count']==1
    assert costs['llm_call_count_known']==38 and costs['llm_call_count_unknown_count']==1
    assert costs['graph_api_calls_known']==194 and costs['graph_api_calls_unknown_count']==1
    assert costs['record_http_calls_known']==1 and costs['record_http_calls_unknown_count']==1
    assert costs['http_calls_known']==195 and costs['http_calls_unknown_count']==1
    assert costs['input_tokens_known']==1900 and costs['input_tokens_unknown_count']==1
    zero=copy.deepcopy(records)
    zero[fixed['experiment_id']]['budget']['llm_calls']['actual']=0.0
    zero[fixed['experiment_id']]['budget']['api_calls']['actual']=0.0
    zero_cost=summarize_search_plan(plan,zero)['costs']['fixed']
    assert zero_cost['llm_call_count_known']==38
    assert zero_cost['llm_call_count_known_count']==20
    assert zero_cost['llm_call_count_unknown_count']==0
