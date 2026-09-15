from copy import deepcopy
import pytest
from backend.tests.test_agent_model_sessions import api
from agent_poc.clients.autoai_client import AgentContractError
from agent_poc.tools import build_tool_schemas, validate_tool_arguments
from agent_poc.clients.capabilities import CapabilitySnapshot
from agent_poc.orchestration.state import CapabilitiesStateV2 as CapabilitiesState
from backend.tests.test_scheduler_policy_review import old_snapshot


def test_exclusive_maximum_reaches_wire_schema_and_state(api):
    response = api[0].get('/api/agent/v2/health').json()
    schemas = build_tool_schemas(response)
    request = dict(dataset_id=api[2],selection_metric='macro_f1',allowed_models=['cnn1d'],max_runs=1,
        model_configs={'cnn1d':{'scheduler_factor':0.5}})
    validate_tool_arguments('start_ml_session', request, schemas)
    request['model_configs']['cnn1d']['scheduler_factor']=1
    with pytest.raises(AgentContractError): validate_tool_arguments('start_ml_session', request, schemas)
    raw = {k:response[k] for k in ('models','catalog_version','catalog_digest','availability_digest')}
    state = CapabilitiesState(wire_snapshot=raw)
    assert state.wire_snapshot.model_dump(mode='json') == raw
    legacy = old_snapshot()
    assert CapabilitySnapshot.model_validate(legacy).model_dump(mode='json') == legacy
    assert CapabilitiesState(wire_snapshot=legacy).wire_snapshot.model_dump(mode='json') == legacy
