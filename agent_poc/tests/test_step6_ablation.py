"""The formal comparison can be registered and summarized without running it."""
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
