"""Independent client schemas against real v2 HTTP, with v1 kept unchanged."""
from copy import deepcopy
import pytest
from agent_poc.clients.autoai_client import AutoAIClient, AgentContractError
from agent_poc.clients.execution_contracts import HealthResponse
from agent_poc.tools import ToolDispatcher, build_tool_schemas, validate_tool_arguments, TOOL_SCHEMAS
from agent_poc.tests.test_orchestration_integration import BackendTransport
from backend.tests.test_agent_model_sessions import api


def test_real_v2_client_freezes_all_candidates(api):
    transport, storage, dataset=api
    client=AutoAIClient('http://backend.invalid',transport=BackendTransport(transport),api_version='v2')
    dispatcher=ToolDispatcher(client)
    health=dispatcher.dispatch('inspect_ml_capabilities',{})
    available=[m['id'] for m in health['models'] if m['available']]
    configs={m['id']:{'epochs':2} for m in health['models'] if m['available'] and m['execution_family']=='deep_learning'}
    for model in available:
        created=dispatcher.dispatch('start_ml_session',dict(dataset_id=dataset,selection_metric='macro_f1',
            allowed_models=[model],max_runs=1,model_configs={model:configs[model]} if model in configs else {}))
        response=dispatcher.dispatch('submit_ml_experiment',dict(session_id=created['session_id'],model_type=model,client_request_id='model'))
        assert response['effective_config_status']=='ready'
        assert response['effective_action']['model_params']==created['locked_config']['capability_snapshot']['model_configs'][model]
        observed=dispatcher.dispatch('observe_ml_experiment',dict(session_id=created['session_id'],run_id=response['run_id']))
        assert observed['state']=='queued'
        assert observed['resolved_execution']['status']=='pending'
        inspected=dispatcher.dispatch('inspect_ml_session',dict(session_id=created['session_id']))
        assert inspected['experiments'][0]['run_id']==response['run_id']


def test_schema_instances_and_closed_types(api):
    transport,_,_=api
    original=deepcopy(TOOL_SCHEMAS)
    one=AutoAIClient('http://backend.invalid',transport=BackendTransport(transport),api_version='v2')
    two=AutoAIClient('http://backend.invalid',transport=BackendTransport(transport),api_version='v2')
    health=one.inspect_ml_capabilities()
    two.inspect_ml_capabilities()
    one.tool_schemas['submit_ml_experiment']['properties']['model_type']['enum'].clear()
    assert two.tool_schemas['submit_ml_experiment']['properties']['model_type']['enum']
    assert TOOL_SCHEMAS==original
    tampered=deepcopy(health);tampered['models'][0]['parameters'].append({'name':'device'})
    with pytest.raises(ValueError): HealthResponse.model_validate(tampered)
    tampered=deepcopy(health);tampered['models'].append(tampered['models'][0])
    with pytest.raises(ValueError): HealthResponse.model_validate(tampered)
    for value in (True, None, '0.1', float('nan'), float('inf'), -1):
        with pytest.raises(AgentContractError):
            validate_tool_arguments('submit_ml_experiment',{'session_id':'s','model_type':'cnn1d','model_params':{'learning_rate':value}},two.tool_schemas)


def test_parameter_binding_rejects_cross_model_and_partial(api):
    transport,_,dataset=api
    client=AutoAIClient('http://backend.invalid',transport=BackendTransport(transport),api_version='v2')
    client.inspect_ml_capabilities()
    created=client.start_ml_session(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['cnn1d'],
        max_runs=1,model_configs={'cnn1d':{'epochs':2}})
    for model,params in [('cnn1d',{'epochs':2}),('svm',{}),('cnn1d',{'device':'cuda'})]:
        with pytest.raises(AgentContractError):
            client.submit_ml_experiment(created['session_id'],model_type=model,model_params=params)


def test_resolved_execution_closes_test_and_path_fields():
    from agent_poc.clients.execution_contracts import ResolvedExecution
    for params in ({'test_macro_f1':1},{'device':'/server/private'},{'svm_kernel':'/canary'},
                   {'channels':[1]*100},{'dropout':True}):
        with pytest.raises(ValueError):
            ResolvedExecution.model_validate({'status':'ready','parameters':params})
    assert ResolvedExecution.model_validate({'status':'ready','parameters':{'execution_device':'cpu'}}).status=='ready'
