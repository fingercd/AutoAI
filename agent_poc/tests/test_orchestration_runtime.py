"""Persistent runner tests use explicit fixture graphs, never pretend real LLM I/O."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import get_tracing_context, tracing_context
import pytest

from agent_poc.orchestration.llm import LLMConfig
from agent_poc.orchestration.persistence import JSONSerializer, PersistenceError, thread_lock
from agent_poc.orchestration.runtime import (
    RuntimeConfig, RuntimeErrorCode, TaskInterrupted, checkpoint_store, main,
    normalize_endpoint, read_status, resume_task, start_task,
)
from agent_poc.orchestration.state import GraphState, apply_patch


ROOT = Path(__file__).resolve().parents[2]


def fixture_config():
    return RuntimeConfig(backend_url='http://127.0.0.1:8000', principal_scope='fixture-local',
                         llm_config=LLMConfig(base_url='http://127.0.0.1:9999/v1', model='fixture-model'))


def build_wait_graph(dependencies, saver):
    """One fixture submission, persisted wait, then explicit fixture failure."""
    def tick(state):
        assert get_tracing_context()['enabled'] is False
        if state['execution']['run_id'] is None:
            call = dependencies.journal.begin(operation_id='fixture-submit', kind='api',
                name='fixture_submit', maximum=60, max_attempts=3,
                deadline=state['budget']['deadline_at'], now=dependencies.clock())
            dependencies.journal.finish(call)
            return apply_patch(state, {
                'execution': {'run_id': 'fixture-run', 'run_status': 'queued'},
                'lifecycle': {'status': 'waiting', 'stage': 'observe', 'next_action': 'observe'},
                'recovery': {'next_wake_at': dependencies.clock() + 2.0},
                'history': {'events': [{'event_id': 'fixture-event', 'kind': 'fixture_submitted',
                    'sequence': 1, 'occurred_at': dependencies.clock(), 'run_id': 'fixture-run'}]}})
        return apply_patch(state, {
            'execution': {'run_status': 'failed'},
            'lifecycle': {'status': 'failed', 'stage': 'ended', 'next_action': None,
                          'reason_code': 'fixture_completed', 'ended_at': dependencies.clock()},
            'finalization': {'status': 'unselected', 'backend_session_state': 'open',
                             'termination_reason': 'fixture_completed'}})
    graph = StateGraph(GraphState)
    graph.add_node('tick', tick)
    graph.add_edge(START, 'tick')
    graph.add_edge('tick', END)
    return graph.compile(checkpointer=saver)


def build_crash_window_graph(dependencies, saver):
    def prepare(state):
        call = dependencies.journal.begin(operation_id='fixture-prepare', kind='api',
            name='fixture_prepare', maximum=60, max_attempts=3,
            deadline=state['budget']['deadline_at'], now=dependencies.clock())
        dependencies.journal.finish(call)
        return apply_patch(state, {'lifecycle': {'status': 'running', 'stage': 'prepared'}})
    def dispatch(state):
        call = dependencies.journal.begin(operation_id='fixture-dispatch', kind='api',
            name='fixture_dispatch', maximum=60, max_attempts=3,
            deadline=state['budget']['deadline_at'], now=dependencies.clock())
        dependencies.journal.finish(call)
        return apply_patch(state, {'lifecycle': {'status': 'failed', 'stage': 'ended',
            'next_action': None, 'reason_code': 'fixture_completed', 'ended_at': dependencies.clock()},
            'finalization': {'status': 'unselected', 'termination_reason': 'fixture_completed'}})
    graph = StateGraph(GraphState)
    graph.add_node('prepare', prepare)
    graph.add_node('dispatch', dispatch)
    graph.add_edge(START, 'prepare')
    graph.add_edge('prepare', 'dispatch')
    graph.add_edge('dispatch', END)
    return graph.compile(checkpointer=saver, interrupt_before=['dispatch'])


def build_interrupted_graph(dependencies, saver):
    graph = build_crash_window_graph(dependencies, saver)
    class Interrupted:
        get_state = graph.get_state
        update_state = graph.update_state
        def invoke(self, *args, **kwargs):
            graph.invoke(*args, **kwargs)
            raise KeyboardInterrupt
    return Interrupted()


def test_json_serializer_has_no_object_or_pickle_fallback():
    serializer = JSONSerializer()
    assert serializer.loads_typed(serializer.dumps_typed({'value': [1, None, 'ok']})) == {'value': [1, None, 'ok']}
    for encoding in ('pickle', 'msgpack', 'jsonplus'):
        with pytest.raises(PersistenceError, match='encoding_unsupported'):
            serializer.loads_typed((encoding, b'forbidden-object-payload'))
    for value in (object(), {'value': float('nan')}, {'value': float('inf')}):
        with pytest.raises(PersistenceError, match='not_json'):
            serializer.dumps_typed(value)
    with pytest.raises(PersistenceError, match='checkpoint_invalid'):
        serializer.loads_typed(('json', b'{"value":NaN}'))
    constructor = {'lc': 1, 'type': 'constructor', 'id': ['pathlib', 'Path'], 'kwargs': {'x': 'opaque'}}
    assert type(serializer.loads_typed(serializer.dumps_typed(constructor))) is dict


def test_checkpoint_connection_uses_full_sync_and_closes(tmp_path):
    with checkpoint_store(tmp_path) as saver:
        connection = saver.conn
        assert connection.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert isinstance(saver.serde, JSONSerializer)
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connection.execute('SELECT 1')


def test_waiting_status_is_persistent_and_no_wait_does_not_dispatch_early(tmp_path):
    config = fixture_config()
    first = start_task(config, dataset_id='fixture-data', allowed_models=['svm'], storage=tmp_path,
                       thread_id='fixture-thread', clock=lambda: 1000.0, graph_factory=build_wait_graph)
    assert first['lifecycle']['status'] == 'waiting'
    assert read_status(storage=tmp_path, thread_id='fixture-thread') == first
    resumed = resume_task(config, storage=tmp_path, thread_id='fixture-thread', clock=lambda: 1001.0,
                          graph_factory=build_wait_graph)
    assert resumed == first
    with sqlite3.connect(tmp_path / 'calls.sqlite') as connection:
        assert connection.execute('SELECT count(*) FROM orchestration_calls_v1').fetchone()[0] == 1
    assert first['budget']['deadline_at'] == 4600.0
    with pytest.raises(RuntimeErrorCode, match='already_exists'):
        start_task(config, dataset_id='fixture-data', allowed_models=['svm'], storage=tmp_path,
                   thread_id='fixture-thread', clock=lambda: 1001.0, graph_factory=build_wait_graph)


def test_wait_runner_owns_timer_and_disables_remote_tracing(tmp_path):
    now = [1000.0]
    sleeps = []
    def sleeper(seconds):
        assert 0 < seconds <= 1.0
        sleeps.append(seconds)
        now[0] += seconds
    with tracing_context(enabled=True):
        final = start_task(fixture_config(), dataset_id='fixture-data', allowed_models=['svm'],
                           storage=tmp_path, thread_id='fixture-thread', wait=True,
                           clock=lambda: now[0], sleep=sleeper, graph_factory=build_wait_graph)
    assert sleeps == [1.0, 1.0]
    assert final['lifecycle']['status'] == 'failed'
    assert final['execution']['run_id'] == 'fixture-run'
    assert len(final['history']['events']) == 1
    assert final['budget']['deadline_at'] == 4600.0


@pytest.mark.parametrize('change', ['backend', 'scope', 'token', 'llm_model', 'llm_limit', 'api_timeout'])
def test_resume_rejects_changed_frozen_runtime_before_graph_or_network(tmp_path, change):
    config = fixture_config()
    start_task(config, dataset_id='fixture-data', allowed_models=['svm'], storage=tmp_path,
               thread_id='fixture-thread', clock=lambda: 1000.0, graph_factory=build_wait_graph)
    replacements = {
        'backend': {'backend_url': 'http://127.0.0.1:8001'},
        'scope': {'principal_scope': 'different-scope'},
        'token': {'backend_token': 'SYNTHETIC_PRIVATE_CREDENTIAL'},
        'llm_model': {'llm_config': replace(config.llm_config, model='different-model')},
        'llm_limit': {'llm_config': replace(config.llm_config, max_tokens=32)},
        'api_timeout': {'api_timeout': 20.0},
    }
    changed = replace(config, **replacements[change])
    def forbidden_build(*_args):
        raise AssertionError('changed configuration must be rejected before graph creation')
    with pytest.raises(RuntimeErrorCode, match='resume_configuration_mismatch'):
        resume_task(changed, storage=tmp_path, thread_id='fixture-thread', graph_factory=forbidden_build)


def test_normalized_endpoint_and_principal_binding_never_store_credentials():
    assert normalize_endpoint('HTTP://LOCALHOST:80/') == 'http://localhost'
    config = replace(fixture_config(), backend_token='SYNTHETIC_PRIVATE_CREDENTIAL')
    assert 'SYNTHETIC_PRIVATE_CREDENTIAL' not in repr(config)
    assert len(config.principal_fingerprint()) == 64
    assert config.principal_fingerprint() != fixture_config().principal_fingerprint()
    assert config.backend_fingerprint() == fixture_config().backend_fingerprint()


def run_child(script, directory):
    completed = subprocess.run([sys.executable, '-B', '-c', script, str(directory)], cwd=ROOT,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_package_disables_remote_trace_flags_before_loading_graph(tmp_path):
    flags = run_child('''
import os,json
names=('LANGSMITH_TRACING','LANGCHAIN_TRACING_V2','LANGCHAIN_TRACING')
for name in names:
 os.environ[name]='true'
import agent_poc.orchestration
print(json.dumps({name:os.environ[name] for name in names}))
''', tmp_path)
    assert set(flags.values()) == {'false'}


def test_new_process_resumes_wait_and_preserves_single_fixture_submission(tmp_path):
    started = run_child('''
import json,sys
from agent_poc.tests.test_orchestration_runtime import fixture_config,build_wait_graph
from agent_poc.orchestration.runtime import start_task
state=start_task(fixture_config(),dataset_id='fixture-data',allowed_models=['svm'],storage=sys.argv[1],
 thread_id='fixture-thread',clock=lambda:1000.0,graph_factory=build_wait_graph)
print(json.dumps(state))
''', tmp_path)
    completed = run_child('''
import json,sys
from agent_poc.tests.test_orchestration_runtime import fixture_config,build_wait_graph
from agent_poc.orchestration.runtime import resume_task
state=resume_task(fixture_config(),storage=sys.argv[1],thread_id='fixture-thread',clock=lambda:1003.0,
 graph_factory=build_wait_graph)
print(json.dumps(state))
''', tmp_path)
    assert started['identity']['task_id'] == completed['identity']['task_id']
    assert started['execution']['run_id'] == completed['execution']['run_id'] == 'fixture-run'
    assert completed['lifecycle']['next_action'] is None
    assert len(completed['history']['events']) == 1
    with sqlite3.connect(tmp_path / 'calls.sqlite') as connection:
        assert connection.execute('SELECT count(*) FROM orchestration_calls_v1').fetchone()[0] == 1
    def forbidden(*_args):
        raise AssertionError('terminal resume must remain a local read')
    assert resume_task(fixture_config(), storage=tmp_path, thread_id='fixture-thread',
                       graph_factory=forbidden) == completed


def test_interrupted_prepared_tick_resumes_pending_nodes_in_new_process(tmp_path):
    with pytest.raises(TaskInterrupted) as interrupted:
        start_task(fixture_config(), dataset_id='fixture-data', allowed_models=['svm'], storage=tmp_path,
                   thread_id='fixture-thread', clock=lambda: 1000.0, graph_factory=build_interrupted_graph)
    assert interrupted.value.thread_id == 'fixture-thread'
    assert read_status(storage=tmp_path, thread_id='fixture-thread')['lifecycle']['stage'] == 'prepared'
    completed = run_child('''
import json,sys
from agent_poc.tests.test_orchestration_runtime import fixture_config,build_crash_window_graph
from agent_poc.orchestration.runtime import resume_task
state=resume_task(fixture_config(),storage=sys.argv[1],thread_id='fixture-thread',clock=lambda:1001.0,
 graph_factory=build_crash_window_graph)
print(json.dumps(state))
''', tmp_path)
    assert completed['lifecycle']['next_action'] is None
    with sqlite3.connect(tmp_path / 'calls.sqlite') as connection:
        names = [row[0] for row in connection.execute('SELECT name FROM orchestration_calls_v1 ORDER BY id')]
    assert names == ['fixture_prepare', 'fixture_dispatch']


def test_os_thread_lock_rejects_other_process_and_releases_after_kill(tmp_path):
    script = '''
import sys
from pathlib import Path
from agent_poc.orchestration.persistence import thread_lock
with thread_lock(Path(sys.argv[1]),'fixture-thread'):
 print('locked',flush=True)
 sys.stdin.readline()
'''
    process = subprocess.Popen([sys.executable, '-B', '-c', script, str(tmp_path)], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        with pytest.raises(PersistenceError, match='thread_already_running'):
            with thread_lock(tmp_path, 'fixture-thread'):
                pass
        process.kill()
        process.communicate(timeout=10)
        with thread_lock(tmp_path, 'fixture-thread'):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_status_cli_reads_without_llm_or_environment_and_snapshot_is_read_only(tmp_path, capsys):
    state = start_task(fixture_config(), dataset_id='fixture-data', allowed_models=['svm'], storage=tmp_path,
                       thread_id='fixture-thread', clock=lambda: 1000.0, graph_factory=build_wait_graph)
    before = (tmp_path / 'checkpoints.sqlite').read_bytes()
    assert main(['status', '--storage', str(tmp_path), '--thread-id', 'fixture-thread'], environ={}) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['run_id'] == state['execution']['run_id']
    assert (tmp_path / 'checkpoints.sqlite').read_bytes() == before
    with checkpoint_store(tmp_path, read_only=True) as saver:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            saver.conn.execute('CREATE TABLE forbidden (id INTEGER)')


def test_cli_rejects_resume_task_overrides_and_hides_bad_inputs(capsys):
    assert main(['resume', '--thread-id', 'fixture-thread', '--dataset-id', 'SYNTHETIC_PRIVATE_CREDENTIAL'], environ={}) == 2
    assert 'SYNTHETIC_PRIVATE_CREDENTIAL' not in capsys.readouterr().err
    assert main(['start', '--dataset-id', 'fixture-data', '--principal-scope', 'fixture-local',
                 '--llm-base-url', 'http://user:SYNTHETIC_PRIVATE_CREDENTIAL@localhost/v1',
                 '--llm-model', 'fixture-model'], environ={}) == 2
    assert 'SYNTHETIC_PRIVATE_CREDENTIAL' not in capsys.readouterr().err


def test_unknown_thread_does_not_create_checkpoint_storage(tmp_path):
    missing = tmp_path / 'missing'
    with pytest.raises(RuntimeErrorCode, match='thread_not_found'):
        read_status(storage=missing, thread_id='fixture-thread')
    assert not missing.exists()
