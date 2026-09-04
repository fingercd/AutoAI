import json

from backend.tests.test_agent_observation_contract import _ready_run
from backend.app.agent.observation import build_observation


def test_test_payload_never_flows_into_observation(tmp_path):
    record, run_dir = _ready_run(tmp_path)
    response = build_observation(
        session_id='safe', session_state='open', selection_metric='macro_f1', run_dir=run_dir,
        record=record, attempt=1, effective_action={}, remaining_runs=0,
    )
    text = json.dumps(response, ensure_ascii=False).lower()
    assert '0.99' not in text
    assert 'secret' not in text
    assert 'test' not in text
