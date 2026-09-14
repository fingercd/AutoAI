import pytest
from backend.tests.test_agent_v2 import api
from agent_poc.tests.test_review_recovery import components
from agent_poc.orchestration.runtime import start_task


@pytest.mark.parametrize('allowed,configs', [
    (['logistic_regression'], {'misspelled_model':{'epochs':2}}),
    (['logistic_regression'], {'cnn1d':{'epochs':2}}),
    (['misspelled_model'], {'misspelled_model':{}}),
    (['logistic_regression'], {'logistic_regression':{'epochs':2}}),
    (['logistic_regression','cnn1d'], {'cnn1d':{'typo':2}}),
])
def test_invalid_operator_configuration_never_creates_session(api,tmp_path,allowed,configs):
    wire,runtime,llm,client,dataset=components(api,'logistic_regression')
    try:
        state=start_task(runtime,dataset_id=dataset,allowed_models=allowed,model_configs=configs,
            storage=tmp_path/'checkpoint',thread_id='invalid-config',client=client(),llm=llm)
    except (ValueError,TypeError):
        state=None
    assert not any(method=='POST' for method,url in wire.requests)
    if state is not None:
        assert state['identity']['session_id'] is None
        assert state['execution']['run_id'] is None
        assert state['lifecycle']['status']=='needs_attention'
    from backend.app.routers.agent import _agent_service
    assert _agent_service('agent-session-v2').runs.list()==[]


def test_known_unavailable_configuration_is_excluded_explicitly(api,tmp_path,monkeypatch):
    from backend.app import model_config
    original=model_config.model_availability
    monkeypatch.setattr(model_config,'model_availability',lambda name: (False,'dependency_missing_torch') if name=='cnn1d' else original(name))
    wire,runtime,llm,client,dataset=components(api,'logistic_regression')
    state=start_task(runtime,dataset_id=dataset,allowed_models=['logistic_regression','cnn1d'],
        model_configs={'cnn1d':{'epochs':2}},storage=tmp_path/'checkpoint',thread_id='unavailable',client=client(),llm=llm)
    assert state['execution']['run_id'] is not None
    assert state['capabilities']['excluded_models']['cnn1d']=='dependency_missing_torch'
    assert state['capabilities']['frozen_snapshot']['model_configs']=={'logistic_regression':{}}
