"""Stop must clear the current UI, discard artifacts and interrupt native work."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from backend.app.runs.repository import RunRepository
from backend.app.runs.contracts import Principal


def test_batch_stop_removes_all_child_artifacts(tmp_path, monkeypatch):
    from backend.app.routers import batches
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    batch, records = repo.create_batch_queued(dataset_id='fixture', test_dataset_id=None, legacy_data_path=None, config={}, model_types=['pls_da','svm'], repeat_count=1, base_seed=42, dataset_snapshot={}, principal=Principal())
    now = datetime.now(timezone.utc)
    running = repo.claim_next(worker_id='test', now=now)
    repo.finish_success(running.run_id, claim_token=running.claim_token, now=now, manifest_name='manifest.json')
    for record in records:
        directory = tmp_path / record.run_id
        directory.mkdir()
        (directory / 'model.pkl').write_bytes(b'fixture')
        (directory / 'metrics.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(batches, 'get_run_repository', lambda: repo)
    monkeypatch.setattr(batches, 'get_run_dir', lambda rid: tmp_path / rid)
    result = batches.stop_batch(batch.batch_id, Principal())
    assert result['state'] == 'cancelled'
    assert all(repo.get(record.run_id).state == 'cancelled' for record in records)
    assert all(not (tmp_path / record.run_id).exists() for record in records)
    assert batches.stop_batch(batch.batch_id, Principal())['state'] == 'cancelled'


def test_stop_exits_training_process_without_waiting_for_training(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='fixture', config={'model_type':'svm'})
    script = '''
import os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker
repo=RunRepository(Path(sys.argv[1]))
def execute(run):
    time.sleep(60)
    return {'manifest_name':'manifest.json'}
worker=RunWorker(repository=repo,worker_id='stop-process-test',execute=execute,now=lambda:datetime.now(timezone.utc),heartbeat_seconds=.25,on_lease_lost=lambda rid:os._exit(75))
worker.run_once()
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path / 'runs.sqlite3')])
    try:
        deadline = time.monotonic() + 10
        while repo.get(created.run_id).state != 'running' and time.monotonic() < deadline:
            time.sleep(.02)
        assert repo.get(created.run_id).state == 'running'
        start = time.monotonic()
        repo.cancel(created.run_id, now=datetime.now(timezone.utc))
        assert process.wait(timeout=5) == 75
        assert time.monotonic() - start < 3
        assert repo.get(created.run_id).state == 'cancelled'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_stop_resets_panel_and_forgets_current_training():
    html = Path('static/index.html').read_text(encoding='utf-8')
    start = html.index('      function resetStoppedTrainingPanel()')
    end = html.index('\n      }', start) + len('\n      }')
    function = html[start:end]
    script = '''
const assert = require('node:assert/strict');
const nodes = new Map();
const $ = id => { if (!nodes.has(id)) nodes.set(id, {textContent:'old',className:'old',disabled:true,classList:{add(){}},replaceChildren(){this.empty=true;}});return nodes.get(id); };
const memory = new Map(); let currentRunId='run', currentBatchId='batch', stopped=false;
const window={sessionStorage:{setItem(k,v){memory.set(k,v);}},SpecAutoAIResults:{cancelAutoRedirect(){},stopTrainingWatch(){stopped=true;}}};
function stopBatchWatch(){currentBatchId=null;}
function rememberCurrentRun(value){currentRunId=value;}
function clearCurrentRunDetails(){$('metrics').replaceChildren();}
''' + function + '''
resetStoppedTrainingPanel();
assert.equal(currentRunId,null);assert.equal(currentBatchId,null);assert.ok(stopped);
assert.equal($('runStatus').textContent,'尚未开始');assert.equal($('runStatus').className,'status');
assert.ok($('trainingProgress').empty);assert.ok($('batchComparison').empty);
assert.equal(memory.get('specautoai.training.idleAfterStop'),'1');
'''
    subprocess.run(['node','-'],input=script,text=True,check=True,capture_output=True)
