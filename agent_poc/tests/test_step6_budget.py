"""Full sixth-step candidate pool under initial and repair prompt budgets."""
import pytest

from backend.tests.test_agent_model_sessions import api
from backend.app.processing_policy import EXECUTABLE_MODELS
from backend.app.model_catalog import MODELS_BY_ID
from agent_poc.orchestration.projection import recipe_selection_context
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig


@pytest.mark.parametrize('pool', ['ML','DL','all'])
@pytest.mark.parametrize('processing_mode', ['fixed','dynamic'])
@pytest.mark.parametrize('search_mode', ['fixed','bounded'])
@pytest.mark.parametrize('protocol', ['json_action','native_tools'])
@pytest.mark.parametrize('decision_mode', ['recipe_id','structured_config'])
@pytest.mark.parametrize('knowledge', [False,True])
def test_step6_full_candidates_fit_both_request_budgets(
        api,budget_config,pool,processing_mode,search_mode,protocol,decision_mode,knowledge):
    client,_,dataset=api
    models=sorted(model for model in EXECUTABLE_MODELS if pool=='all' or
                  MODELS_BY_ID[model].execution_family==
                  ('traditional_ml' if pool=='ML' else 'deep_learning'))
    headers={'X-AutoAI-Agent-Revision':'agent-recipes-revision-v4'}
    modules=['train_evidence','legal_recipes']
    if processing_mode=='dynamic':modules.append('dynamic_preprocessing')
    if search_mode=='bounded':modules.append('bounded_hpo')
    if knowledge:modules.append('knowledge')
    response=client.post('/api/agent/v2/sessions',headers=headers,json=dict(
        dataset_id=dataset,selection_metric='macro_f1',allowed_models=models,
        max_runs=1,seed=42,execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4',processing_mode=processing_mode,
        search_mode=search_mode,max_trials=1 if search_mode=='fixed' else 6,
        decision_mode=decision_mode,modules=modules,
        context_policy=dict(source_role='development',case_write=False,evidence=True,risks=True)))
    assert response.status_code==201,response.text
    locked=response.json()['locked_config']
    context=recipe_selection_context(task=locked,session_id=response.json()['session_id'],
        preparation=locked['preparation'],
        context_policy={**locked['context_policy'],'projection':'agent-context-search-v1'},
        model_configs=locked['capability_snapshot']['model_configs'])
    catalog_ids={recipe['recipe_id'] for recipe in locked['preparation']['catalog']['recipes']}
    assert {recipe['recipe_id'] for recipe in context['recipes']}==catalog_ids
    assert len(context['model_profiles'])==len(models)
    if knowledge:assert context['knowledge']['entries']
    adapter=LLMAdapter(LLMConfig('http://fixture.invalid/v1','fixture',protocol=protocol,
        max_tokens=1024,prompt_version='agent-decision-search-v1',**budget_config))
    prepared=adapter.prepare_context('submit',context)
    assert {recipe['recipe_id'] for recipe in prepared['recipes']}==catalog_ids
    for repair in (None,'llm_output_invalid'):
        request,*_=adapter._build_request('submit',prepared,repair)
        assert adapter._prompt_tokens(request)+adapter.config.max_tokens<=adapter.config.context_window
