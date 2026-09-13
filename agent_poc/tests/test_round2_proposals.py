"""Real adapter parsing with explicit synthetic HTTP replies, plus durable repair."""
import json
import sqlite3
from contextlib import closing

import httpx
import pytest

from agent_poc.orchestration.llm import LLMAdapter
from agent_poc.orchestration.persistence import CallJournal
from agent_poc.orchestration.runtime import read_status, resume_task
from agent_poc.orchestration.state import StateModel
from agent_poc.tests.test_graph import configuration, run


class Replies:
    def __init__(self, invalid=0):
        self.invalid = invalid
        self.requests = []

    def request(self, method, url, *, headers, json, timeout):
        import json as codec
        self.requests.append(json)
        context = codec.loads(json['messages'][1]['content'])
        tool = context['allowed_actions'][0]
        args = dict(context['bindings'])
        if tool == 'submit_ml_experiment':
            args['model_type'] = 'svm'
        rationale = ('Choose SVM with token=synthetic' if len(self.requests) <= self.invalid
                     else 'Choose the permitted candidate using validation evidence.')
        return httpx.Response(200, json={'model':'local-model', 'choices':[{
            'index':0, 'finish_reason':'stop', 'message':{'role':'assistant',
            'content':codec.dumps({'tool_name':tool,'arguments':args,'rationale':rationale})}}],
            'usage':{'total_tokens':30}})


@pytest.mark.parametrize('invalid,expected,calls,submits', [(1,'completed',3,1), (99,'failed',3,0)])
def test_state_unsafe_model_text_uses_bounded_graph_repair(tmp_path, invalid, expected, calls, submits):
    transport = Replies(invalid)
    adapter = LLMAdapter(configuration().llm_config, transport=transport)
    state, backend, _, clock = run(tmp_path, model=adapter)
    assert state['lifecycle']['status'] == expected
    assert backend.calls.count('submit') == submits
    assert len(transport.requests) == calls
    rows = state['budget']['usage']
    llm_rows = [row for row in rows if row['kind']=='llm']
    assert len(llm_rows) == calls
    assert llm_rows[0]['operation_id'] == llm_rows[1]['operation_id']
    assert all(row['total_tokens']==30 and row['token_status']=='partial' for row in llm_rows)
    assert all(row['input_tokens'] is None and row['output_tokens'] is None for row in llm_rows)
    assert state['budget']['input_tokens']['actual'] is None
    assert 'token=synthetic' not in json.dumps(state)
    assert 'token=synthetic' not in json.dumps(transport.requests)
    if expected == 'failed':
        assert state['lifecycle']['reason_code'] == 'llm_repair_exhausted'
        assert state['execution']['run_id'] is None
    # New connection and new runner recover exactly the same settled usage.
    assert read_status(storage=tmp_path, thread_id='thread-a') == state
    assert resume_task(configuration(), storage=tmp_path, thread_id='thread-a',
                       client=backend, llm=adapter, clock=clock) == state
    assert len(transport.requests) == calls
    with closing(CallJournal(tmp_path/'calls.sqlite', 'thread-a')) as journal:
        before = journal.snapshot()
        first = next(row for row in before if row['kind']=='llm')
        journal.finish(first['id'], total_tokens=999, token_status='partial')
        assert journal.snapshot() == before


def test_old_journal_and_checkpoint_remain_compatible(tmp_path):
    state, _, _, _ = run(tmp_path)
    old = json.loads(json.dumps(state))
    for usage in old['budget']['usage']:
        usage.pop('total_tokens')
    restored = StateModel.model_validate(old).model_dump(mode='json')
    assert restored['identity'] == state['identity']
    assert all(row['total_tokens'] is None for row in restored['budget']['usage'])
    path = tmp_path/'calls.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE orchestration_calls_v1 DROP COLUMN total_tokens')
        db.execute('ALTER TABLE orchestration_calls_v1 DROP COLUMN token_status')
        before = db.execute('SELECT id,operation_id,status,input_tokens,output_tokens FROM orchestration_calls_v1').fetchall()
    for _ in range(2):
        with closing(CallJournal(path, 'thread-a')) as journal:
            assert all(row['total_tokens'] is None and row['token_status'] is None for row in journal.snapshot())
            from agent_poc.orchestration.graph import Dependencies, Nodes
            from agent_poc.tests.test_graph import Backend, Clock, Model
            accounted = Nodes(Dependencies(Backend(), Model(), journal, Clock())).account(restored)
            assert accounted['budget']['usage'] == restored['budget']['usage']
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT id,operation_id,status,input_tokens,output_tokens FROM orchestration_calls_v1').fetchall() == before


def test_total_only_usage_survives_active_checkpoint_restart(tmp_path):
    from agent_poc.orchestration.runtime import start_task
    from agent_poc.tests.test_graph import Backend, Clock
    import subprocess
    import sys
    transport=Replies()
    adapter=LLMAdapter(configuration().llm_config,transport=transport)
    backend,clock=Backend(),Clock()
    backend.pending_observations=1
    waiting=start_task(configuration(),dataset_id='ds-a',allowed_models=['svm'],
        storage=tmp_path,thread_id='thread-a',client=backend,llm=adapter,clock=clock)
    assert waiting['lifecycle']['status']=='waiting'
    result=subprocess.run([sys.executable,'-B','-m','agent_poc.orchestration','status',
        '--storage',str(tmp_path),'--thread-id','thread-a','--full'],
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==waiting
    clock.value=waiting['recovery']['next_wake_at']
    completed=resume_task(configuration(),storage=tmp_path,thread_id='thread-a',
        client=backend,llm=adapter,clock=clock)
    assert completed['lifecycle']['status']=='completed'
    llm_rows=[row for row in completed['budget']['usage'] if row['kind']=='llm']
    assert sum(row['total_tokens'] for row in llm_rows)==60
    assert len(llm_rows)==len(transport.requests)==2
    assert backend.calls.count('submit')==1
