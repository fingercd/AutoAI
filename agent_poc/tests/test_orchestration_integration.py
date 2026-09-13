"""Synthetic integration evidence using the real API and a separate worker process.

The scripted provider is deliberately labelled synthetic; it exercises the real
LLM HTTP parsing boundary but is not evidence of access to a deployed model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig


def configure_backend_storage(directory: Path, monkeypatch=None) -> None:
    """Configure test-only backend storage in this process before HTTP requests."""
    import backend.app.paths as paths
    import backend.app.routers.agent as agent_router
    import backend.app.routers.datasets as datasets_router
    import backend.app.routers.deps as deps_router
    import backend.app.routers.runs as runs_router
    import backend.app.training as training

    directory.mkdir(parents=True, exist_ok=True)
    for child in ('uploads', 'runs'):
        (directory / child).mkdir(exist_ok=True)
    values = {'STORAGE_DIR': directory, 'UPLOADS_DIR': directory / 'uploads',
              'RUNS_DIR': directory / 'runs', 'DATASETS_DATABASE': directory / 'datasets.sqlite3',
              'RUNS_DATABASE': directory / 'runs.sqlite3', 'AGENT_DATABASE': directory / 'agent.sqlite3'}
    for module in (paths, agent_router, datasets_router, deps_router, runs_router, training):
        for name, value in values.items():
            if hasattr(module, name):
                if monkeypatch is None:
                    setattr(module, name, value)
                else:
                    monkeypatch.setattr(module, name, value)


class BackendTransport:
    def __init__(self, client: TestClient, *, crash_after: str | None = None):
        self.client, self.crash_after = client, crash_after
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method, url, *, headers, json, timeout):
        path = urlsplit(url).path
        self.calls.append((method, path, json))
        response = self.client.request(method, path, headers=headers, json=json)
        kind = None
        if method == 'POST':
            if path == '/api/agent/sessions':
                kind = 'session_response'
            elif path.endswith('/experiments'):
                kind = 'experiment_response'
            elif path.endswith('/finalize'):
                kind = 'finalize_response'
        if self.crash_after is not None and kind == self.crash_after and response.status_code < 300:
            os._exit(91)  # Deliberate process death after backend commit, before checkpoint.
        return response


class ScriptedProvider:
    """Deterministic provider response, labelled synthetic and selecting SVM."""
    def __init__(self):
        self.calls = []

    def request(self, method, url, *, headers, json, timeout):
        import httpx
        from json import dumps, loads
        self.calls.append(json)
        projected = loads(json['messages'][-1]['content'])
        arguments = dict(projected['bindings'])
        if projected['phase'] == 'submit':
            assert 'svm' in projected['capabilities']['models']
            arguments['model_type'] = 'svm'
            rationale = 'Choose SVM for the single classification experiment.'
        else:
            rationale = 'Finalize the sole eligible candidate using validation.'
        proposal = {'tool_name': projected['allowed_actions'][0], 'arguments': arguments,
                    'rationale': rationale}
        return httpx.Response(200, json={
            'id': f'provider-call-{len(self.calls)}', 'model': 'synthetic-provider',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': dumps(proposal)}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
        })


def synthetic_llm() -> LLMAdapter:
    return LLMAdapter(LLMConfig('http://synthetic.invalid/v1', 'synthetic-provider'),
                      transport=ScriptedProvider())


def worker_process(storage: str, *, hold: bool = False) -> None:
    """Called only in a fresh interpreter; executes the real RunWorker training."""
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.status_projection import project_status
    from backend.app.runs.worker import RunWorker

    directory = Path(storage)
    configure_backend_storage(directory)
    repository = RunRepository(directory / 'runs.sqlite3')
    repository.initialize()

    def execute(record):
        (directory / 'worker-running.json').write_text(json.dumps({
            'run_id': record.run_id, 'state': repository.get(record.run_id).state,
            'pid': os.getpid()}), encoding='utf-8')
        if hold:
            deadline = time.monotonic() + 40
            while not (directory / 'worker-release').exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('integration worker release deadline')
                time.sleep(0.05)
        return execute_claimed_run(record, repository=repository)

    worker = RunWorker(repository=repository, worker_id='graph-integration-worker', execute=execute,
        now=lambda: datetime.now(timezone.utc), heartbeat_seconds=1,
        project_status=lambda record: project_status(directory / 'runs' / record.run_id, record))
    assert worker.run_once()
    (directory / 'worker-result.json').write_text(json.dumps({
        'states': [record.state for record in repository.list()], 'pid': os.getpid()}), encoding='utf-8')


def spawn_worker(storage: Path, *, hold: bool = False):
    command = [sys.executable, '-B', '-c',
        'import sys; from agent_poc.tests.test_orchestration_integration import worker_process; '
        'worker_process(sys.argv[1], hold=sys.argv[2] == "hold")', str(storage), 'hold' if hold else 'run']
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def upload_synthetic_dataset(client: TestClient, directory: Path) -> str:
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    source = directory / 'synthetic-grouped.csv'
    write_grouped_classification_csv(source, groups_per_class=6, repeats=2, feature_count=6)
    with source.open('rb') as handle:
        response = client.post('/api/datasets/upload', files={'file': ('synthetic-grouped.csv', handle, 'text/csv')})
    assert response.status_code == 200, response.text
    return response.json()['dataset_id']


def assert_one_scientific_run(storage: Path) -> str:
    from backend.app.runs.repository import RunRepository
    records = RunRepository(storage / 'runs.sqlite3').list()
    assert len(records) == 1
    with sqlite3.connect(storage / 'agent.sqlite3') as connection:
        assert connection.execute('SELECT count(*) FROM agent_sessions_v1').fetchone()[0] == 1
        assert connection.execute('SELECT count(*) FROM agent_experiment_reservations_v1').fetchone()[0] == 1
    return records[0].run_id


class TickClock:
    def __init__(self, value=None):
        self.value = time.time() if value is None else value

    def __call__(self):
        return self.value

    def wake(self, state):
        self.value = max(self.value, state['recovery']['next_wake_at'] or self.value)


@contextmanager
def graph_context(directory: Path, *, thread_id='integration-thread', clock=None, crash_after=None):
    from backend.app.main import app
    from langgraph.checkpoint.sqlite import SqliteSaver
    from agent_poc.orchestration.graph import Dependencies, build_graph
    from agent_poc.orchestration.persistence import CallJournal, JSONSerializer

    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / 'checkpoints.sqlite3', check_same_thread=False)
    saver = SqliteSaver(connection, serde=JSONSerializer())
    journal = CallJournal(directory / 'calls.sqlite3', thread_id)
    transport = BackendTransport(TestClient(app), crash_after=crash_after)
    clock = clock or TickClock()
    try:
        client = AutoAIClient('http://backend.invalid', transport=transport, max_retries=0)
        graph = build_graph(Dependencies(client, synthetic_llm(), journal, clock), saver)
        yield graph, {'configurable': {'thread_id': thread_id}}, journal, transport, clock
    finally:
        journal.close()
        connection.close()


def initial_state(dataset_id: str):
    from agent_poc.orchestration.llm import PROMPT_VERSION, LLM_CONFIG_VERSION
    from agent_poc.orchestration.state import new_state
    return new_state(dataset_id=dataset_id, allowed_models=['logistic_regression', 'svm'],
        backend_fingerprint='a' * 64, principal_fingerprint='b' * 64,
        llm_config_fingerprint=synthetic_llm().config.fingerprint(),
        prompt_version=PROMPT_VERSION, llm_config_version=LLM_CONFIG_VERSION,
        task_id='integration-task', thread_id='integration-thread')


def ticks_until(graph, config, clock, *, initial=None, predicate=None, limit=24):
    state = graph.invoke(initial or {}, config, durability='sync')
    for _ in range(limit):
        if state['lifecycle']['next_action'] is None or (predicate and predicate(state)):
            return state
        clock.wake(state)
        state = graph.invoke({}, config, durability='sync')
    raise AssertionError(f'bounded test ticks exhausted: {state["lifecycle"]}')


def test_graph_uses_six_tools_and_observes_real_separate_worker(monkeypatch, tmp_path):
    from backend.app.main import app
    storage = tmp_path / 'backend'
    configure_backend_storage(storage, monkeypatch)
    dataset_id = upload_synthetic_dataset(TestClient(app), tmp_path)
    with graph_context(tmp_path / 'orchestrator') as (graph, config, journal, transport, clock):
        queued = ticks_until(graph, config, clock, initial=initial_state(dataset_id),
            predicate=lambda state: state['lifecycle']['status'] == 'waiting')
        assert queued['execution']['run_status'] == 'queued', queued['lifecycle']
        run_id = assert_one_scientific_run(storage)
        assert queued['execution']['run_id'] == run_id
        assert queued['execution']['effective_action']['model_type'] == 'svm'
        worker = spawn_worker(storage, hold=True)
        try:
            deadline = time.monotonic() + 35
            while not (storage / 'worker-running.json').exists():
                assert worker.poll() is None, worker.communicate()
                assert time.monotonic() < deadline
                time.sleep(0.05)
            witness = json.loads((storage / 'worker-running.json').read_text(encoding='utf-8'))
            assert witness['state'] == 'running' and witness['pid'] != os.getpid()
            clock.wake(queued)
            running = graph.invoke({}, config, durability='sync')
            assert running['execution']['run_status'] == 'running', running['lifecycle']
            (storage / 'worker-release').write_text('release', encoding='utf-8')
            stdout, stderr = worker.communicate(timeout=90)
            assert worker.returncode == 0, stdout + stderr
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.communicate(timeout=10)
        clock.wake(running)
        completed = ticks_until(graph, config, clock)
        assert completed['lifecycle']['status'] == 'completed', completed['lifecycle']
        assert completed['execution']['run_status'] == 'succeeded'
        assert completed['finalization']['status'] == 'confirmed'
        assert completed['finalization']['selected_run_id'] == run_id
        assert completed['finalization']['backend_session_state'] == 'finalized'
        assert completed['task']['dataset_fingerprint'] == completed['execution']['dataset_fingerprint']
        assert completed['task']['split_fingerprint'] is None
        assert_one_scientific_run(storage)
        names = {row['name'] for row in journal.snapshot() if row['kind'] == 'api'}
        assert names == {'inspect_ml_capabilities', 'start_ml_session', 'inspect_ml_session',
                         'submit_ml_experiment', 'observe_ml_experiment', 'finalize_ml_session'}
        assert len([row for row in journal.snapshot() if row['kind'] == 'llm']) == 2
        assert completed['budget']['input_tokens']['actual'] == 200
        serialized = json.dumps(completed).lower()
        assert 'final_result_url' not in serialized and str(tmp_path).lower() not in serialized
        assert 'test' not in serialized and 'prediction' not in serialized
        prior = len(journal.snapshot())
        graph.invoke({}, config, durability='sync')
        assert len(journal.snapshot()) == prior


class DecisionCrashGraph:
    """Kill only after the verified LLM decision checkpoint has committed."""
    def __init__(self, graph):
        self.graph = graph

    def get_state(self, *args, **kwargs):
        return self.graph.get_state(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        state = self.graph.invoke(*args, **kwargs)
        if state['lifecycle']['next_action'] == 'submit':
            assert state['execution']['submission_content']['model_type'] == 'svm'
            os._exit(92)
        return state


def driver_process(root: str, dataset_id: str, method: str, fault: str) -> None:
    """Run the production runner in a genuinely new process with ASGI transport."""
    from backend.app.main import app
    from agent_poc.orchestration.graph import build_graph
    from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task, read_status

    directory = Path(root)
    configure_backend_storage(directory / 'backend')
    if fault == 'before_binding':
        from backend.app.agent.repository import AgentSessionRepository

        def crash_before_binding(*_args, **_kwargs):
            os._exit(94)

        AgentSessionRepository.bind_experiment = crash_before_binding
    llm = synthetic_llm()
    configuration = RuntimeConfig(backend_url='http://backend.invalid',
        principal_scope='integration-local', llm_config=llm.config)
    client = AutoAIClient(configuration.backend_url, max_retries=0,
        transport=BackendTransport(TestClient(app), crash_after=fault if fault.endswith('_response') else None))
    clock = TickClock()
    kwargs = {'storage': directory / 'orchestrator', 'thread_id': 'integration-thread',
              'wait': False, 'client': client, 'llm': llm, 'clock': clock}
    if fault == 'decision_checkpoint':
        kwargs['graph_factory'] = lambda deps, saver: DecisionCrashGraph(build_graph(deps, saver))
    if method == 'start':
        state = start_task(configuration, dataset_id=dataset_id,
            allowed_models=['logistic_regression', 'svm'], task_id='integration-task', **kwargs)
    else:
        prior = read_status(storage=directory / 'orchestrator', thread_id='integration-thread')
        clock.wake(prior)
        state = resume_task(configuration, **kwargs)
    (directory / 'driver-state.json').write_text(json.dumps(state), encoding='utf-8')
    if fault == 'waiting_shutdown' and state['lifecycle']['status'] == 'waiting':
        os._exit(93)
    print(json.dumps({'pid': os.getpid(), 'status': state['lifecycle']['status'],
                      'run_id': state['execution']['run_id']}))


def run_driver(directory: Path, dataset_id: str, method: str, fault: str = ''):
    command = [sys.executable, '-B', '-c',
        'import sys; from agent_poc.tests.test_orchestration_integration import driver_process; '
        'driver_process(*sys.argv[1:])', str(directory), dataset_id, method, fault]
    return subprocess.run(command, capture_output=True, text=True, timeout=80,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


@pytest.mark.parametrize('fault', ['session_response', 'experiment_response', 'before_binding', 'decision_checkpoint',
                                  'waiting_shutdown', 'finalize_response', 'running_shutdown'])
def test_new_process_recovery_preserves_one_run_and_one_lock(monkeypatch, tmp_path, fault):
    from backend.app.main import app
    from agent_poc.orchestration.runtime import read_status

    configure_backend_storage(tmp_path / 'backend', monkeypatch)
    dataset_id = upload_synthetic_dataset(TestClient(app), tmp_path)
    first_fault = '' if fault in ('finalize_response', 'running_shutdown') else fault
    result = run_driver(tmp_path, dataset_id, 'start', first_fault)
    if first_fault:
        assert result.returncode in (91, 92, 93, 94), result.stdout + result.stderr
        if fault == 'before_binding':
            assert_one_scientific_run(tmp_path / 'backend')
            with sqlite3.connect(tmp_path / 'backend' / 'agent.sqlite3') as connection:
                assert connection.execute('SELECT state FROM agent_experiment_reservations_v1').fetchone()[0] == 'reserved'
        result = run_driver(tmp_path, dataset_id, 'resume')
    assert result.returncode == 0, result.stdout + result.stderr
    queued = read_status(storage=tmp_path / 'orchestrator', thread_id='integration-thread')
    assert queued['lifecycle']['status'] == 'waiting', queued['lifecycle']
    assert queued['execution']['run_status'] == 'queued'
    assert queued['execution']['effective_action']['model_type'] == 'svm'
    run_id = assert_one_scientific_run(tmp_path / 'backend')
    worker = spawn_worker(tmp_path / 'backend', hold=fault == 'running_shutdown')
    try:
        if fault == 'running_shutdown':
            deadline = time.monotonic() + 35
            while not (tmp_path / 'backend' / 'worker-running.json').exists():
                assert worker.poll() is None, worker.communicate()
                assert time.monotonic() < deadline
                time.sleep(0.05)
            result = run_driver(tmp_path, dataset_id, 'resume', 'waiting_shutdown')
            assert result.returncode == 93, result.stdout + result.stderr
            running = read_status(storage=tmp_path / 'orchestrator', thread_id='integration-thread')
            assert running['execution']['run_status'] == 'running'
            (tmp_path / 'backend' / 'worker-release').write_text('release', encoding='utf-8')
        stdout, stderr = worker.communicate(timeout=90)
        assert worker.returncode == 0, stdout + stderr
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.communicate(timeout=10)
    result = run_driver(tmp_path, dataset_id, 'resume', 'finalize_response' if fault == 'finalize_response' else '')
    if fault == 'finalize_response':
        assert result.returncode == 91, result.stdout + result.stderr
        result = run_driver(tmp_path, dataset_id, 'resume')
    assert result.returncode == 0, result.stdout + result.stderr
    completed = read_status(storage=tmp_path / 'orchestrator', thread_id='integration-thread')
    assert completed['lifecycle']['status'] == 'completed', completed['lifecycle']
    assert completed['finalization']['status'] == 'confirmed'
    assert completed['finalization']['selected_run_id'] == run_id
    assert_one_scientific_run(tmp_path / 'backend')
    with sqlite3.connect(tmp_path / 'orchestrator' / 'calls.sqlite') as connection:
        rows = connection.execute('SELECT kind,name,status FROM orchestration_calls_v1 ORDER BY id').fetchall()
    assert len([row for row in rows if row[0] == 'llm']) == 2
    assert len([row for row in rows if row[1] == 'finalize_ml_session']) == 1
    events = completed['history']['events']
    assert len({event['event_id'] for event in events}) == len(events)
    previous_count = len(rows)
    result = run_driver(tmp_path, dataset_id, 'resume')
    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(tmp_path / 'orchestrator' / 'calls.sqlite') as connection:
        assert connection.execute('SELECT count(*) FROM orchestration_calls_v1').fetchone()[0] == previous_count
