"""Process-level status visibility and CLI outcome semantics."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from agent_poc.orchestration.persistence import PersistenceError
from agent_poc.orchestration.runtime import RuntimeErrorCode, read_status, resume_task
from agent_poc.tests.test_orchestration_runtime import ROOT, fixture_config
from agent_poc.tests.test_graph import Backend, run


def database_facts(directory):
    facts = {}
    for name in ('checkpoints.sqlite','calls.sqlite'):
        with sqlite3.connect(f'file:{(directory/name).as_posix()}?mode=ro', uri=True) as db:
            facts[name] = list(db.iterdump())
    return facts


def test_status_corrupt_checkpoint_is_safe_and_read_only(tmp_path):
    path=tmp_path/'checkpoints.sqlite'
    payload=b'corrupt SYNTHETIC_PRIVATE_CREDENTIAL'
    path.write_bytes(payload)
    with pytest.raises(RuntimeErrorCode,match='^checkpoint_invalid$'):
        read_status(storage=tmp_path,thread_id='fixture-thread')
    assert path.read_bytes()==payload


def test_waiting_runner_allows_other_process_status_but_excludes_resume(tmp_path):
    script = '''
import sys
from agent_poc.orchestration import runtime
from agent_poc.tests.test_orchestration_runtime import fixture_config,build_wait_graph
original=runtime.start_task
def sleeper(seconds):
 print('waiting',flush=True)
 sys.stdin.readline()
 raise KeyboardInterrupt
def start(config,**kwargs):
 return original(config,**kwargs,clock=lambda:1000.0,sleep=sleeper,graph_factory=build_wait_graph)
runtime.start_task=start
runtime._config_from_args=lambda *args:fixture_config()
raise SystemExit(runtime.main(['start','--storage',sys.argv[1],'--thread-id','fixture-thread',
 '--dataset-id','fixture-data','--allowed-models','svm','--wait']))
'''
    runner = subprocess.Popen([sys.executable,'-B','-c',script,str(tmp_path)], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    try:
        assert runner.stdout.readline().strip() == 'waiting'
        before = database_facts(tmp_path)
        state = read_status(storage=tmp_path,thread_id='fixture-thread')
        reader = subprocess.run([sys.executable,'-B','-m','agent_poc.orchestration','status',
            '--storage',str(tmp_path),'--thread-id','fixture-thread','--full'],cwd=ROOT,
            capture_output=True,text=True,timeout=20)
        assert reader.returncode == 0, reader.stderr
        assert json.loads(reader.stdout) == state
        assert state['execution']['run_id']=='fixture-run'
        assert state['lifecycle']['status']=='waiting'
        with pytest.raises(PersistenceError,match='thread_already_running'):
            resume_task(fixture_config(),storage=tmp_path,thread_id='fixture-thread')
        assert database_facts(tmp_path) == before
        out, err = runner.communicate('\n',timeout=20)
        assert runner.returncode == 130, err
        assert json.loads(out)['status']=='interrupted'
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate(timeout=10)


@pytest.mark.parametrize('status,expected', [('completed',0),('failed',1),('timed_out',1),
    ('needs_attention',1),('cancelled',1),('waiting',0)])
@pytest.mark.parametrize('command', ['start','resume','status'])
def test_cli_json_and_process_exit_agree(tmp_path,status,expected,command):
    backend = Backend()
    kwargs = {}
    if status in ('failed','cancelled'):
        backend = Backend(terminal=status,valid=False)
    elif status == 'timed_out':
        backend.pending_observations=100
        kwargs['timeout_seconds']=3
    elif status == 'needs_attention':
        backend.session_response=lambda: {}  # genuine backend contract failure
    elif status == 'waiting':
        backend.pending_observations=1
        # Obtain a real graph's saved waiting state at its first observation.
        from agent_poc.tests.test_graph import Clock, Model, configuration
        from agent_poc.orchestration.runtime import start_task
        state=start_task(configuration(),dataset_id='ds-a',allowed_models=['svm'],
            storage=tmp_path/'graph',thread_id='thread-a',client=backend,llm=Model(),clock=Clock())
    if status != 'waiting':
        state, _, _, _ = run(tmp_path/'graph',backend=backend,**kwargs)
    assert state['lifecycle']['status']==status
    snapshot=tmp_path/'state.json'
    snapshot.write_text(json.dumps(state),encoding='utf-8')
    # Feed actual graph outcomes into CLI dispatch; no HTTP or provider is mocked
    # as successful here. The separate wait-process test exercises the real driver.
    script='''
import json,sys
from pathlib import Path
from agent_poc.orchestration import runtime
from agent_poc.tests.test_orchestration_runtime import fixture_config
state=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
runtime._config_from_args=lambda *args:fixture_config()
runtime.start_task=lambda *args,**kwargs:state
runtime.resume_task=lambda *args,**kwargs:state
runtime.read_status=lambda **kwargs:state
args=[sys.argv[2],'--thread-id','thread-a']
if sys.argv[2]=='start':args+=['--dataset-id','ds-a']
raise SystemExit(runtime.main(args,environ={}))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(snapshot),command],cwd=ROOT,
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==(0 if command=='status' else expected),result.stderr
    assert json.loads(result.stdout)['lifecycle']['status']==status
