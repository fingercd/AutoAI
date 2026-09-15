"""Finite-domain expression equivalence, hard guards and frozen recovery."""
import json
from copy import deepcopy

import pytest
from backend.tests.test_agent_model_sessions import api
from backend.app.agent.repository import AgentSessionRepository
from backend.app.agent.contracts import CreateKnowledgeExperimentRequest, CreateStructuredExperimentRequest
from backend.app.agent.policy import normalize_experiment, command_digest
from backend.app.agent.service import _hash_payload
from backend.app.runs.contracts import Principal
from agent_poc.orchestration.projection import recipe_selection_context
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig, store_proposal, StoredProposal
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider

HEADERS = {'X-AutoAI-Agent-Revision':'agent-recipes-revision-v2'}

def create(api, mode, *, evidence=True, risks=True, knowledge=False, models=None):
    client, _, dataset = api
    body = dict(dataset_id=dataset, selection_metric='macro_f1', allowed_models=models or ['logistic_regression','svm'],
        max_runs=1, seed=42, execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v2',
        decision_mode=mode, modules=['train_evidence','legal_recipes']+(['knowledge'] if knowledge else []),
        context_policy=dict(source_role='benchmark',case_write=False,evidence=evidence,risks=risks))
    response = client.post('/api/agent/v2/sessions', json=body, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json(), body

def expression(created, recipe, mode):
    if mode == 'recipe_id':
        return dict(recipe_id=recipe['recipe_id'], recipe_digest=recipe['recipe_digest'],
            catalog_digest=created['locked_config']['preparation']['catalog']['catalog_digest'], knowledge_refs=[], client_request_id='chosen')
    return dict(model_id=recipe['model_id'], normalization='zscore', class_balance='none',
        model_params=created['locked_config']['capability_snapshot']['model_configs'][recipe['model_id']],
        knowledge_refs=[],client_request_id='chosen')

def test_same_domain_commands_and_distinct_request_hashes(api):
    pairs = [create(api, mode)[0] for mode in ('recipe_id','structured_config')]
    assert pairs[0]['locked_config']['preparation']['catalog'] == pairs[1]['locked_config']['preparation']['catalog']
    repo = AgentSessionRepository(api[1]/'agent.sqlite3')
    for recipe in pairs[0]['locked_config']['preparation']['catalog']['recipes']:
        commands=[]
        for created, mode, schema in zip(pairs, ('recipe_id','structured_config'), (CreateKnowledgeExperimentRequest,CreateStructuredExperimentRequest)):
            session = repo.get_session_scoped(created['session_id'], principal=Principal())
            command = normalize_experiment(session, schema(**expression(created,recipe,mode)))
            commands.append((command,command_digest(session,command)))
        assert commands[0][0].action == commands[1][0].action
        assert commands[0][1] == commands[1][1]
        assert _hash_payload(commands[0][0].request_body) != _hash_payload(commands[1][0].request_body)

@pytest.mark.parametrize('change',[
    {'normalization':'none'}, {'class_balance':'class_weight'}, {'model_id':'cnn1d'},
    {'extra':1}, {'model_params':{'epochs':True}}, {'model_params':{'learning_rate':0.1}},
    {'knowledge_refs':['unknown']}, {'knowledge_refs':['x','x']},
])
def test_illegal_structured_choice_has_no_reservation(api, change):
    created,_ = create(api,'structured_config')
    recipe=created['locked_config']['preparation']['catalog']['recipes'][0]
    body={**expression(created,recipe,'structured_config'),**change}
    path='/api/agent/v2/sessions/'+created['session_id']
    response=api[0].post(path+'/experiments',json=body,headers=HEADERS)
    assert response.status_code==422, response.text
    assert api[0].get(path,headers=HEADERS).json()['experiments']==[]

@pytest.mark.parametrize('mode',['recipe_id','structured_config'])
def test_mode_frozen_and_submission_replays(api, mode):
    created,body=create(api,mode)
    body['client_request_id']='frozen-session'
    created=api[0].post('/api/agent/v2/sessions',json=body,headers=HEADERS).json()
    changed={**body,'decision_mode':'structured_config' if mode=='recipe_id' else 'recipe_id'}
    assert api[0].post('/api/agent/v2/sessions',json=changed,headers=HEADERS).status_code==409
    recipe=created['locked_config']['preparation']['catalog']['recipes'][0]
    path='/api/agent/v2/sessions/'+created['session_id']+'/experiments'
    wrong=expression(created,recipe,'structured_config' if mode=='recipe_id' else 'recipe_id')
    assert api[0].post(path,json=wrong,headers=HEADERS).status_code==422
    request=expression(created,recipe,mode)
    first=api[0].post(path,json=request,headers=HEADERS)
    second=api[0].post(path,json=request,headers=HEADERS)
    assert first.status_code==second.status_code==202
    assert first.json()['run_id']==second.json()['run_id']

class ExpressionProvider(KnowledgeProvider):
    def request(self,*args,**kwargs):
        response=super().request(*args,**kwargs)
        context=self.contexts[-1]
        if context.get('decision_mode')!='structured_config' or context['phase']!='submit':return response
        body=response.json();message=body['choices'][0]['message']
        if self.protocol=='json_action':
            action=json.loads(message['content']);arguments=action['arguments']
        else:
            arguments=json.loads(message['tool_calls'][0]['function']['arguments'])
        model=context['recipes'][0]['model_id']
        arguments.pop('recipe_id')
        arguments.update(model_id=model,normalization='zscore',class_balance='none',model_params=context['fixed_model_params'][model])
        if self.protocol=='json_action':message['content']=json.dumps(action)
        else:message['tool_calls'][0]['function']['arguments']=json.dumps(arguments)
        import httpx
        return httpx.Response(200,json=body)

@pytest.mark.parametrize('evidence,risks,knowledge',[(e,r,k) for e in (False,True) for r in (False,True) for k in (False,True)])
@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_independent_contexts_and_proposal_restore(api,evidence,risks,knowledge,protocol):
    created,_=create(api,'structured_config',evidence=evidence,risks=risks,knowledge=knowledge)
    locked=created['locked_config']
    context=recipe_selection_context(task=locked,session_id=created['session_id'],preparation=locked['preparation'],
        context_policy={**locked['context_policy'],'projection':'agent-context-knowledge-v1'})
    assert (context['train_statistics'] is not None)==evidence
    assert (context['train_risks'] is not None)==risks
    assert context['knowledge']['status']==('ready' if knowledge else 'disabled')
    cfg=LLMConfig('http://fixture.invalid/v1','fixture',protocol=protocol,prompt_version='agent-decision-knowledge-v1')
    proposal=LLMAdapter(cfg,transport=ExpressionProvider(protocol)).propose('submit',context)
    saved=store_proposal(proposal,context)
    assert StoredProposal.model_validate(saved).bind(context).arguments==proposal.arguments
    bad=deepcopy(context);bad['decision_mode']='recipe_id'
    with pytest.raises(ValueError):StoredProposal.model_validate(saved).bind(bad)

@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),1.5])
def test_numeric_aliases_rejected(api,value):
    created,_=create(api,'structured_config',models=['cnn1d'])
    recipe=created['locked_config']['preparation']['catalog']['recipes'][0]
    body=expression(created,recipe,'structured_config');body['model_params']=dict(body['model_params'],epochs=value)
    session=AgentSessionRepository(api[1]/'agent.sqlite3').get_session_scoped(created['session_id'],principal=Principal())
    with pytest.raises((ValueError,RuntimeError)):
        normalize_experiment(session,CreateStructuredExperimentRequest(**body))

