"""T29 acceptance on synthetic databases only; never a production restore tool.

Run with pytest. AUTOAI_T29_EVIDENCE may name an absent output directory.
Rollback is inspection-only into a fresh directory, never replacement in place.
"""
from __future__ import annotations

from contextlib import ExitStack, closing
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import zipfile

import pytest

from agent_poc.orchestration import runtime
from agent_poc.orchestration.diagnosis import report_for, store_diagnosis, verify_report_proposal
from agent_poc.orchestration.llm import Proposal, TokenUsage
from agent_poc.orchestration.persistence import CallJournal, PersistenceError
from agent_poc.orchestration.state import new_state, validate_state
from agent_poc.tests.test_step9_contracts import context
from backend.app.agent.budget import BudgetPolicy
from backend.app.agent.diagnosis import DiagnosisInput, DiagnosisReport, digest
from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.contracts import Principal
from scripts.export_diagnosis import build_diagnosis
from scripts.task_cost_report import readonly_snapshot

ROOT = Path(__file__).resolve().parents[2]
PRE_STEP9 = '39743e455cdfa5f7ba2c8f1f44f17553f38dfe4e'
PRE_ASSESSMENT = '7502267a2e673bcdb8d4583e5a9dea68ce48796c'
DATABASES = ('calls.sqlite', 'checkpoints.sqlite', 'agent.sqlite3')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf8')


def logical(db):
    """Compare every table, column and value, including checkpoint BLOBs."""
    result = {}
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        quoted = '"' + table.replace('"', '""') + '"'
        columns = [row[1] for row in db.execute('PRAGMA table_info(' + quoted + ')')]
        rows = [[{'blob_hex': v.hex()} if isinstance(v, bytes) else v for v in row]
                for row in db.execute('SELECT * FROM ' + quoted)]
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
        result[table] = dict(columns=columns, rows=rows)
    return result


def image(path):
    with closing(readonly_snapshot(path)) as db:
        assert db.execute('PRAGMA integrity_check').fetchall()[0][0] == 'ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        return logical(db)


def summary(value):
    return {name: dict(rows=len(table['rows']), sha256=digest(table)) for name, table in value.items()}


def checkpoint(directory, state):
    from langgraph.checkpoint.base import empty_checkpoint
    with runtime.checkpoint_store(directory) as saver:
        cp = empty_checkpoint()
        cp['channel_values'] = {'__start__': state}
        cp['channel_versions'] = {'__start__': '1'}
        saver.put(runtime._graph_config(state['identity']['thread_id']), cp,
                  {'source': 'input', 'step': -1, 'parents': {}}, {'__start__': '1'})


def snapshot_copy(source, target):
    """The existing WAL-aware reader, materialized to an exclusively new file."""
    with target.open('xb'):
        pass
    with closing(readonly_snapshot(source)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)


def restore_for_inspection(backup, destination):
    # No overwrite flag and no canonical binding rewrite. Existing destinations fail first.
    destination.mkdir(exist_ok=False)
    for name in DATABASES:
        snapshot_copy(backup / name, destination / name)


@pytest.fixture
def no_external_effects(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append('forbidden_dispatch')
        raise AssertionError('T29 must not dispatch network, graph or training work')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(runtime, '_build', forbidden)
    yield calls
    assert calls == []


@pytest.fixture(scope='module')
def old_programs(tmp_path_factory):
    programs = {}
    for revision in (PRE_STEP9, PRE_ASSESSMENT):
        target = tmp_path_factory.mktemp('t29-program-' + revision[:7])
        commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
        archive = subprocess.check_output(['git', 'archive', '--format=zip', commit,
                                           'backend', 'agent_poc', 'scripts'], cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for name in bundle.namelist():
                assert (target / name).resolve().is_relative_to(target.resolve())
            bundle.extractall(target)
        programs[revision] = (target, commit)
    return programs


def old_python(program, code, *args):
    directory, commit = program
    env = dict(os.environ)
    deps = [p for p in env.get('PYTHONPATH', '').split(os.pathsep)
            if p and Path(p).resolve() != ROOT.resolve()]
    env['PYTHONPATH'] = os.pathsep.join([str(directory), *deps])
    env['PYTHONUTF8'] = '1'
    guard = """import sys
external=[]
def audit(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):
  external.append(event)
  raise AssertionError('No external calls in legacy T29 probe')
sys.addaudithook(audit)
"""
    result = subprocess.run([sys.executable, '-c', guard + code, *map(str, args)],
                            cwd=directory, env=env, text=True, encoding='utf8',
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(commit=commit, exit_code=result.returncode, stdout=result.stdout,
                stderr=result.stderr)


LEGACY_SEED = '''
from pathlib import Path
import json,time
from langgraph.checkpoint.base import empty_checkpoint
from agent_poc.orchestration.persistence import CallJournal
from agent_poc.orchestration.runtime import _default_budget_policy,checkpoint_store,_graph_config,_journal_path_digest
from agent_poc.orchestration.state import new_state
from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.contracts import Principal
p=Path(sys.argv[1]);p.mkdir()
policy=_default_budget_policy(task_id='old-task',started_at=time.time(),timeout_seconds=600,
 allowed_models=['logistic_regression'],model_configs={},max_trials=1,max_llm_calls=6,max_api_calls=60,
 max_repair_attempts=2,max_operation_attempts=3,max_output_tokens=1024)
j=CallJournal(p/'calls.sqlite','old-thread');j.bind_budget(policy)
c=j.begin(operation_id='old-choice',kind='llm',name='submit',maximum=6,max_attempts=3,
 deadline=policy.deadline_at,task_id=policy.task_id)
j.mark_dispatched(c)
j.finish(c,input_tokens=17,output_tokens=3,total_tokens=20,token_status='known',cached_tokens=0,
 proposal={'synthetic_legacy_proposal':'preserve_exactly'})
j.close()
a=AgentSessionRepository(p/'agent.sqlite3');a.initialize()
s,_=a.create_session(dataset_id='synthetic-old-data',selection_metric='macro_f1',
 allowed_models=['logistic_regression'],max_runs=1,seed=42,evaluation_config={},modules=[],
 context_policy={},client_request_id='synthetic-old-request',payload_hash='old-hash',
 principal=Principal(owner_id='test-owner',tenant_id='test-tenant'),budget_policy=policy)
state=new_state(dataset_id='synthetic-old-data',allowed_models=['logistic_regression'],
 backend_fingerprint='a'*64,principal_fingerprint='b'*64,llm_config_fingerprint='c'*64,
 task_id='old-task',thread_id='old-thread',wire_version='agent-state-v8',
 processing_mode='fixed',search_mode='fixed',max_trials=1,budget_policy=policy.as_dict(),
 budget_awareness='off',canonical_journal_sha256=_journal_path_digest(p),
 now=policy.started_at,timeout_seconds=600)
state['identity']['session_id']=s.session_id
with checkpoint_store(p) as saver:
 cp=empty_checkpoint();cp['channel_values']={'__start__':state};cp['channel_versions']={'__start__':'1'}
 saver.put(_graph_config('old-thread'),cp,{'source':'input','step':-1,'parents':{}},{'__start__':'1'})
(p/'seed.json').write_text(json.dumps(dict(policy=policy.as_dict(),state=state,session_id=s.session_id)))
print(json.dumps(dict(external_calls=external,session_id=s.session_id,confirmed_calls=1,tokens=20)))
'''


def test_old_schema_double_migration_and_program_rollback(tmp_path, old_programs, no_external_effects):
    live = tmp_path / 'synthetic-legacy'
    seed_receipt = old_python(old_programs[PRE_STEP9], LEGACY_SEED, live)
    before = {name: image(live / name) for name in DATABASES}
    assert 'diagnosis_inputs_v1' not in before['calls.sqlite']
    seed = json.loads((live / 'seed.json').read_text())
    policy = BudgetPolicy.from_dict(seed['policy'])
    for _ in range(2):
        with closing(CallJournal(live / 'calls.sqlite', 'old-thread')) as journal:
            journal.bind_budget(policy)
            assert journal.budget_summary()['llm_calls']['known_actual'] == 1
        AgentSessionRepository(live / 'agent.sqlite3').initialize()
        with runtime.checkpoint_store(live) as saver:
            saver.setup()
        after = {name: image(live / name) for name in DATABASES}
        for name, tables in before.items():
            for table, contents in tables.items():
                assert after[name][table] == contents, (name, table)
    assert runtime.read_status(storage=live, thread_id='old-thread') == seed['state']
    # Rollback executable is the actual pre-Step9 tree. It may READ supported v8 only.
    frozen = {name: image(live / name) for name in DATABASES}
    probe = old_python(old_programs[PRE_STEP9], '''
from pathlib import Path
import json
from agent_poc.orchestration.runtime import read_status
from agent_poc.orchestration.persistence import CallJournal
from backend.app.agent.budget import BudgetPolicy
p=Path(sys.argv[1]);seed=json.loads((p/'seed.json').read_text())
s=read_status(storage=p,thread_id='old-thread');assert s==seed['state']
j=CallJournal(p/'calls.sqlite','old-thread');j.bind_budget(BudgetPolicy.from_dict(seed['policy']))
assert j.budget_summary()['llm_calls']['known_actual']==1
assert json.loads(j.proposal('old-choice'))=={'synthetic_legacy_proposal':'preserve_exactly'}
j.close();print(json.dumps(dict(supported_state=s['versions']['state'],external_calls=external)))
''', live)
    assert {name: image(live / name) for name in DATABASES} == frozen
    record('legacy-migration', dict(seed=seed_receipt, rollback=probe,
           before={n: summary(v) for n, v in before.items()},
           after={n: summary(v) for n, v in frozen.items()},
           allowed='Read supported v8 history using pre-Step9 code with additive schema retained; no resume approval.'))


def test_pre_assessment_report_keeps_original_proposal_and_hash(tmp_path, old_programs, no_external_effects):
    live = tmp_path / 'synthetic-old-report';live.mkdir()
    receipt = old_python(old_programs[PRE_ASSESSMENT], '''
from pathlib import Path
import json
from types import SimpleNamespace
from agent_poc.tests.test_step9_journal import setup
from agent_poc.orchestration.diagnosis import report_for,store_diagnosis
p=Path(sys.argv[1]);j,policy,snapshot=setup(p);j.freeze_diagnosis_input(snapshot)
proposal=SimpleNamespace(arguments=dict(schema_version='diagnosis-proposal-v1',
 input_digest=snapshot['context']['input_digest'],assessment='insufficient_evidence',
 hypotheses=[],suggestions=[]),response_id=None,tool_call_id=None)
c=j.begin(operation_id='old-diagnose',kind='llm',name='diagnose',maximum=6,max_attempts=3,
 deadline=policy.deadline_at,task_id=policy.task_id);j.mark_dispatched(c)
j.finish(c,input_tokens=23,output_tokens=7,total_tokens=30,token_status='known',cached_tokens=0,
 proposal=store_diagnosis(proposal,snapshot['context']))
report=report_for(snapshot,policy={'rules_digest':'a'*64},calls=[{'id':c}],now=1.,proposal=proposal)
j.save_diagnosis_report(report);j.close()
assert 'assessment' not in report
(p/'seed.json').write_text(json.dumps(dict(policy=policy.as_dict(),snapshot=snapshot,report=report)))
print(json.dumps(dict(external_calls=external,original_report_assessment='absent',confirmed_calls=1,tokens=30)))
''', live)
    seed = json.loads((live / 'seed.json').read_text())
    before = image(live / 'calls.sqlite')
    for _ in range(2):
        with closing(CallJournal(live / 'calls.sqlite', 'thread-1')) as journal:
            journal.bind_budget(BudgetPolicy.from_dict(seed['policy']))
            report = journal.diagnosis_report(seed['snapshot']['context']['input_digest'])
            assert report == seed['report']
            assert 'assessment' not in report and 'proposal_digest' not in report
            assert json.loads(journal.proposal('old-diagnose'))['proposal']['assessment'] == 'insufficient_evidence'
            assert journal.budget_summary()['llm_calls']['known_actual'] == 1
        assert image(live / 'calls.sqlite') == before
    assert DiagnosisReport.model_validate(report).model_dump(mode='json') == report
    record('old-report', dict(legacy=receipt, tables=summary(before),
           report_sha=digest(report), original_assessment_absent=True,
           raw_proposal_preserved=True, migrated_twice_without_rewrite=True))


def record(name, value):
    folder = os.environ.get('AUTOAI_T29_EVIDENCE')
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        # Each result is append-only, so a rerun must use a different evidence directory.
        target = path / (name + '.json')
        with target.open('x', encoding='utf8') as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2)


def test_wal_set_restore_canonical_and_post_backup_write_protection(tmp_path, old_programs, no_external_effects):
    live = tmp_path / 'synthetic-live';live.mkdir()
    policy = runtime._default_budget_policy(task_id='new-task', started_at=time.time(),
        timeout_seconds=600, allowed_models=['logistic_regression'], model_configs={}, max_trials=1,
        max_llm_calls=6, max_api_calls=60, max_repair_attempts=2, max_operation_attempts=3,
        max_output_tokens=1024)
    repo = AgentSessionRepository(live / 'agent.sqlite3');repo.initialize()
    with closing(CallJournal(live / 'calls.sqlite', 'new-thread')) as j, ExitStack() as stack:
        j.bind_budget(policy)
        with runtime.checkpoint_store(live) as saver:
            saver.setup()
        keepers = []
        for name in DATABASES:
            db = stack.enter_context(closing(sqlite3.connect(live / name)))
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA wal_autocheckpoint=0')
            keepers.append(db)
        session, _ = repo.create_session(dataset_id='synthetic-new-data', selection_metric='macro_f1',
            allowed_models=['logistic_regression'], max_runs=1, seed=42, evaluation_config={}, modules=[],
            context_policy={}, client_request_id='synthetic-new-request', payload_hash='new-hash',
            principal=Principal(owner_id='test-owner', tenant_id='test-tenant'), budget_policy=policy)
        state = new_state(dataset_id='synthetic-new-data', allowed_models=['logistic_regression'],
            backend_fingerprint='a'*64, principal_fingerprint='b'*64, llm_config_fingerprint='c'*64,
            task_id='new-task', thread_id='new-thread', wire_version='agent-state-v9',
            processing_mode='fixed', search_mode='fixed', max_trials=1,
            budget_policy=policy.as_dict(), budget_awareness='off',
            canonical_journal_sha256=runtime._journal_path_digest(live),
            now=policy.started_at, timeout_seconds=600)
        state['identity']['session_id'] = session.session_id
        state = validate_state(state).model_dump(mode='json')
        checkpoint(live, state)
        bindings = {key: 'a'*64 for key in ('task','thread','session','principal','run','budget_policy',
                    'diagnosis_policy','llm','evidence','effective_config','train_evidence','knowledge','catalog')}
        bindings.update(task=digest('new-task'), thread=digest('new-thread'), session=digest(session.session_id),
                        principal=state['identity']['principal_fingerprint'], run=None,
                        budget_policy=policy.digest, diagnosis_policy=digest(state['task']['diagnosis_policy']))
        ctx = context();ctx['input_digest'] = digest(dict(bindings=bindings, subject='event-1', generation=0))
        snap = DiagnosisInput(subject_event_id='event-1',generation=0,bindings=bindings,context=ctx,
                             displayed_context_digest=digest(ctx),after_diagnosis='terminated').model_dump(mode='json')
        j.freeze_diagnosis_input(snap)
        proposal = Proposal('report_feedback_diagnosis', dict(schema_version='diagnosis-proposal-v1',
            input_digest=ctx['input_digest'],assessment='insufficient_evidence',hypotheses=[],suggestions=[]),
            'Synthetic persistence fixture', None, None, TokenUsage())
        call = j.begin(operation_id='diagnose-fixture',kind='llm',name='diagnose',maximum=6,max_attempts=3,
                       deadline=policy.deadline_at,task_id='new-task',session_id=session.session_id)
        j.mark_dispatched(call)
        j.finish(call,input_tokens=31,output_tokens=9,total_tokens=40,token_status='known',cached_tokens=0,
                 proposal=store_diagnosis(proposal,ctx))
        report = report_for(snap, policy=state['task']['diagnosis_policy'], calls=[{'id':call}],
                            now=1., proposal=proposal)
        j.save_diagnosis_report(report)
        assert build_diagnosis(storage=live,thread_id='new-thread')['report']['assessment']=='insufficient_evidence'
        # All databases are quiescent. Hold write locks across the whole set, not per-file copies.
        for db in keepers:
            db.execute('BEGIN IMMEDIATE')
        blocked = []
        for name in DATABASES:
            with closing(sqlite3.connect(live / name, timeout=0.05)) as contender:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    contender.execute('BEGIN IMMEDIATE')
                blocked.append(name)
        try:
            before = {name: image(live / name) for name in DATABASES}
            wal = {name: dict(bytes=Path(str(live/name)+'-wal').stat().st_size,
                             sha256=hashlib.sha256(Path(str(live/name)+'-wal').read_bytes()).hexdigest())
                   for name in DATABASES}
            assert all(v['bytes'] > 32 for v in wal.values())
            # Prove the relevant diagnosis exists only after committed WAL frames are included.
            with closing(sqlite3.connect((live/'calls.sqlite').as_uri()+'?immutable=1',uri=True)) as main_only:
                has_table = main_only.execute("SELECT 1 FROM sqlite_master WHERE name='diagnosis_reports_v1'").fetchone()
                main_count = main_only.execute('SELECT count(*) FROM diagnosis_reports_v1').fetchone()[0] if has_table else 0
                assert main_count == 0
            backup = tmp_path / 'backup';backup.mkdir()
            for name in DATABASES:
                snapshot_copy(live / name, backup / name)
            assert {name:image(live/name) for name in DATABASES} == before
            assert {name:image(backup/name) for name in DATABASES} == before
        finally:
            for db in keepers:
                db.rollback()
        restored = tmp_path / 'restored-inspection'
        restore_for_inspection(backup, restored)
        assert {name:image(restored/name) for name in DATABASES} == before
        restored_state = runtime.read_status(storage=restored,thread_id='new-thread')
        assert restored_state == state
        with closing(readonly_snapshot(restored/'calls.sqlite')) as db:
            raw = db.execute('SELECT payload_json FROM diagnosis_reports_v1').fetchone()[0]
            restored_input = DiagnosisInput.model_validate_json(db.execute(
                'SELECT payload_json FROM diagnosis_inputs_v1').fetchone()[0]).model_dump(mode='json')
            assert restored_input == snap
            restored_report = DiagnosisReport.model_validate_json(raw).bind(restored_input).model_dump(mode='json')
            assert restored_report == report
            verify_report_proposal(db,'new-thread',restored_report,snap)
            assert db.execute('SELECT count(*) FROM orchestration_calls_v1 WHERE status="confirmed"').fetchone()[0]==1
        with closing(readonly_snapshot(restored/'agent.sqlite3')) as db:
            row=db.execute('SELECT session_id,budget_task_id,owner_id,tenant_id,budget_policy_digest FROM agent_sessions_v1').fetchone()
            assert tuple(row)==(session.session_id,'new-task','test-owner','test-tenant',policy.digest)
        with pytest.raises(ValueError,match='diagnosis_canonical_storage_mismatch'):
            build_diagnosis(storage=restored,thread_id='new-thread')
        with pytest.raises(runtime.RuntimeErrorCode,match='budget_journal_binding_mismatch'):
            runtime._require_existing_budget_journal(restored,restored_state)
        with closing(CallJournal(restored/'calls.sqlite','new-thread')) as copied:
            with pytest.raises(PersistenceError,match='binding_conflict'):
                copied.bind_budget(policy)
            with pytest.raises(PersistenceError,match='binding_conflict'):
                copied.begin(operation_id='forbidden-copy-call',kind='llm',name='diagnose',maximum=6,
                             max_attempts=3,deadline=policy.deadline_at,task_id='new-task')
        assert {name:image(restored/name) for name in DATABASES} == before
        # Older executable must not rewrite v9 to v8 or attempt to resume it.
        rollback = old_python(old_programs[PRE_STEP9], '''
from pathlib import Path
import json
from agent_poc.orchestration.runtime import read_status
try:
 read_status(storage=Path(sys.argv[1]),thread_id='new-thread')
except ValueError as e:
 assert 'unknown State version' in str(e)
 print(json.dumps(dict(rejected='unknown State version',external_calls=external)))
else:raise AssertionError('Old program accepted unsupported v9')
''', restored)
        assert {name:image(restored/name) for name in DATABASES} == before
        # New committed data after the backup must survive an attempted old-backup rollback.
        later = j.begin(operation_id='after-backup',kind='api',name='inspect_ml_session',maximum=60,
                        max_attempts=3,deadline=policy.deadline_at,task_id='new-task',session_id=session.session_id)
        j.mark_dispatched(later);j.finish(later)
        later_session,_=repo.create_session(dataset_id='synthetic-later-data',selection_metric='macro_f1',
            allowed_models=['logistic_regression'],max_runs=1,seed=42,evaluation_config={},modules=[],
            context_policy={},client_request_id='synthetic-after-backup',payload_hash='later-hash',
            principal=Principal(owner_id='test-owner',tenant_id='test-tenant'))
        later_state=deepcopy(state);later_state['identity']['thread_id']='after-backup-thread'
        checkpoint(live,later_state)
        newer={name:image(live/name) for name in DATABASES}
        assert all(newer[name]!=before[name] for name in DATABASES)
        with pytest.raises(FileExistsError):
            restore_for_inspection(backup,live)
        assert {name:image(live/name) for name in DATABASES}==newer
        assert {name:image(backup/name) for name in DATABASES}==before
        assert j.budget_summary()['llm_calls']['known_actual']==1
        assert j.budget_summary()['api_calls']['known_actual']==1
        record('wal-restore',dict(database_set=list(DATABASES),wal=wal,main_file_only_report_count=main_count,
            consistency_cut='No active writers; BEGIN IMMEDIATE held on all three DBs for the entire snapshot interval.',
            blocked_writers=blocked,integrity='ok',foreign_key_violations=[],
            before={n:summary(v) for n,v in before.items()},after_new_writes={n:summary(v) for n,v in newer.items()},
            restored_exactly=True,confirmed_llm_fixture_calls=1,fixture_tokens=40,
            cross_database_session= session.session_id,post_backup_session=later_session.session_id,
            canonical_copy_export='rejected',canonical_copy_write='rejected',old_program=rollback,
            overwrite_newer_canonical='FileExistsError before any file write',
            external_requests=0,training_runs=0,real_llm_calls=0,
            scope='Synthetic committed rows, not physical requests. No Runs DB/artifacts: no Run exists in this fixture.'))
