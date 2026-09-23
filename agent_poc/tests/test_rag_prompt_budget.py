from copy import deepcopy
from dataclasses import replace

import pytest

from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig, LLMError, StoredProposal, store_proposal
from agent_poc.orchestration.runtime import RuntimeConfig, RuntimeErrorCode, start_task


@pytest.mark.parametrize('decision_mode',['recipe_id','structured_config'])
@pytest.mark.parametrize('knowledge',[False,True])
@pytest.mark.parametrize('pool,mode,protocol,expected_count',[
    ('all','dynamic','json_action',96),
    ('ML','dynamic','native_tools',40),
    ('DL','dynamic','json_action',56),
    ('all','fixed','native_tools',13),
])
def test_processing_full_catalog_fits_real_tokenizer_without_dropping_members(
        api,budget_config,decision_mode,knowledge,pool,mode,protocol,expected_count):
    from backend.app.processing_policy import EXECUTABLE_MODELS
    from agent_poc.orchestration.projection import recipe_selection_context
    from backend.app.recipes import PROFILE

    client,_,dataset=api
    headers={'X-AutoAI-Agent-Revision':'agent-recipes-revision-v3'}
    from backend.app.models import model_family
    models=sorted(model for model in EXECUTABLE_MODELS if pool=='all' or
                  model_family(model)==('traditional_ml' if pool=='ML' else 'deep_learning'))
    response=client.post('/api/agent/v2/sessions',headers=headers,json=dict(
        dataset_id=dataset,selection_metric='macro_f1',allowed_models=models,
        max_runs=1,seed=42,execution_profile=PROFILE,protocol_revision='agent-recipes-revision-v3',
        processing_mode=mode,decision_mode=decision_mode,
        modules=['train_evidence','legal_recipes']+(['dynamic_preprocessing'] if mode=='dynamic' else [])+(['knowledge'] if knowledge else []),
        context_policy={'source_role':'development','case_write':False,'evidence':True,'risks':True},
        client_request_id='full-'+pool+mode+protocol+decision_mode+str(knowledge)))
    assert response.status_code==201,response.text
    locked=response.json()['locked_config']
    context=recipe_selection_context(task=locked,session_id=response.json()['session_id'],
        preparation=locked['preparation'],context_policy={**locked['context_policy'],
        'projection':'agent-context-processing-v1'},model_configs=locked['capability_snapshot']['model_configs'])
    expected={recipe['recipe_id'] for recipe in locked['preparation']['catalog']['recipes']}
    assert len(expected)==expected_count
    assert {recipe['recipe_id'] for recipe in context['recipes']}==expected
    assert len(context['model_profiles'])==len(models)
    if knowledge:
        assert context['knowledge']['entries']
    config=LLMConfig('http://fixture.invalid/v1','fixture',protocol=protocol,max_tokens=64,
        prompt_version='agent-decision-processing-v1',**budget_config)
    adapter=LLMAdapter(config)
    prepared=adapter.prepare_context('submit',context)
    assert {recipe['recipe_id'] for recipe in prepared['recipes']}==expected
    request,*_=adapter._build_request('submit',prepared,'llm_output_invalid')
    assert adapter._prompt_tokens(request)+config.max_tokens<=config.context_window


@pytest.mark.parametrize('protocol', ['json_action', 'native_tools'])
def test_cli_requires_budget_before_building_graph(tmp_path, monkeypatch, capsys, protocol):
    from agent_poc.orchestration import runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('Must fail before graph construction')
    monkeypatch.setattr(runtime, '_build', forbidden)
    storage = tmp_path / 'checkpoint'
    result = runtime.main(['start', '--dataset-id', 'fixture-data', '--knowledge', 'on',
        '--storage', str(storage), '--protocol', protocol], environ={
        'AUTOAI_LLM_BASE_URL': 'http://fixture.invalid/v1',
        'AUTOAI_LLM_MODEL': 'fixture', 'AUTOAI_PRINCIPAL_SCOPE': 'local'})
    assert result == 2
    assert 'rag_prompt_budget_configuration_required' in capsys.readouterr().err
    assert not storage.exists()


@pytest.mark.parametrize('protocol', ['json_action', 'native_tools'])
@pytest.mark.parametrize('failure', ['missing', 'unreadable', 'invalid_template', 'adapter_mismatch'])
def test_start_rejects_budget_configuration_before_any_effect(tmp_path, budget_config, protocol, failure):
    class NoRequests:
        calls = 0
        def request(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError('No network request is allowed')

    options = dict(budget_config)
    if failure == 'missing':
        options = {}
    elif failure == 'unreadable':
        options['tokenizer_path'] = str(tmp_path / 'missing')
    elif failure == 'invalid_template':
        import json
        import shutil
        root = tmp_path / 'invalid-tokenizer'
        root.mkdir()
        from pathlib import Path
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            shutil.copyfile(Path(options['tokenizer_path']) / name, root / name)
        config_path = root / 'tokenizer_config.json'
        config = json.loads(config_path.read_text())
        config['chat_template'] = '{{ raise_exception("invalid template") }}'
        config_path.write_text(json.dumps(config))
        options['tokenizer_path'] = str(root)
    cfg = LLMConfig('http://fixture.invalid/v1', 'fixture', protocol=protocol,
                    prompt_version='agent-decision-knowledge-v1', **options)
    wire = NoRequests()
    adapter_cfg = replace(cfg, context_window=64000) if failure == 'adapter_mismatch' else cfg
    adapter = LLMAdapter(adapter_cfg, transport=wire)
    storage = tmp_path / 'checkpoint'
    client = AutoAIClient('http://backend.invalid', transport=wire, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v2')
    with pytest.raises(RuntimeErrorCode, match='rag_prompt_budget_configuration_'):
        start_task(RuntimeConfig('http://backend.invalid', 'local', cfg),
                   dataset_id='fixture-data', allowed_models=['logistic_regression'],
                   knowledge=True, storage=storage, llm=adapter, client=client)
    assert wire.calls == 0
    assert not storage.exists()


@pytest.mark.parametrize('protocol', ['json_action', 'native_tools'])
def test_real_tokenizer_trimming_references_and_stored_context(api, tmp_path, budget_config, protocol):
    from backend.tests.test_knowledge_session import create
    from agent_poc.orchestration.projection import recipe_selection_context
    from agent_poc.clients.capabilities import digest
    created, _ = create(api)
    locked = created['locked_config']
    context = recipe_selection_context(task=locked, session_id=created['session_id'],
        preparation=locked['preparation'], context_policy={**locked['context_policy'],
        'projection': 'agent-context-knowledge-v1'})
    assert len(context['knowledge']['entries']) == 2
    cfg = LLMConfig('http://fixture.invalid/v1', 'fixture', protocol=protocol, max_tokens=64,
                    prompt_version='agent-decision-knowledge-v1', **budget_config)
    provider = KnowledgeProvider(protocol)
    adapter = LLMAdapter(cfg, transport=provider)
    adapter.validate_prompt_budget()
    one = deepcopy(context)
    one['knowledge']['entries'].pop()
    one['knowledge']['provided_entry_ids'] = [one['knowledge']['entries'][0]['entry_id']]
    one['knowledge']['provided_count'] = 1
    one['knowledge']['omitted_count'] = one['knowledge']['matched_count'] - 1
    request, *_ = adapter._build_request('submit', one, 'llm_output_invalid')
    limit = adapter._prompt_tokens(request) + cfg.max_tokens
    adapter.config = replace(cfg, context_window=limit)
    prepared = adapter.prepare_context('submit', context)
    assert prepared == one
    assert prepared['recipes'] == context['recipes']
    proposal = adapter.propose('submit', prepared)
    assert provider.contexts == [prepared]
    stored = store_proposal(proposal, prepared)
    assert stored['displayed_context'] == prepared
    assert stored['context_digest'] == digest(prepared)
    from agent_poc.orchestration.persistence import CallJournal
    journal_path = tmp_path / 'calls.sqlite'
    journal = CallJournal(journal_path, 'budget-test')
    call = journal.begin(operation_id='choose', kind='llm', name='submit',
                         maximum=6, max_attempts=3, deadline=100, now=1)
    journal.finish(call, proposal=stored)
    journal.close()
    journal = CallJournal(journal_path, 'budget-test')
    try:
        restored = StoredProposal.model_validate_json(journal.proposal('choose'))
    finally:
        journal.close()
    assert restored.bind(restored.displayed_context) == proposal
    with pytest.raises(ValueError):
        restored.bind(context)
    bad = deepcopy(stored)
    bad['arguments']['knowledge_refs'] = [context['knowledge']['entries'][1]['entry_id']]
    with pytest.raises(ValueError):
        StoredProposal.model_validate(bad).bind(prepared)
    adapter.config = replace(cfg, context_window=cfg.max_tokens + 1)
    with pytest.raises(LLMError, match='llm_context_too_long'):
        adapter.prepare_context('submit', context)
    with pytest.raises(LLMError, match='llm_context_too_long'):
        adapter.propose('submit', context)
    assert len(provider.contexts) == 1
