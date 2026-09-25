from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.runs.guard import GuardPolicy, GuardError, load_bound_dataset, report, project_guard
from backend.app.runs.repository import RunRepository
from backend.app.runs.contracts import Principal
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_bound_parser_uses_verified_bytes_even_if_path_changes(tmp_path, monkeypatch):
    from backend.app import parsers
    source = write_grouped_classification_csv(tmp_path/'data.csv', groups_per_class=3, repeats=1, feature_count=8)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original = parsers.load_modeling_csv
    def replacing(path, *, content):
        source.write_text('changed', encoding='utf-8')
        return original(path, content=content)
    monkeypatch.setattr(parsers, 'load_modeling_csv', replacing)
    data, actual = load_bound_dataset(source, sha)
    assert len(data.labels) == 6 and actual == sha
    with pytest.raises(GuardError) as exc:
        load_bound_dataset(source, sha)
    assert exc.value.code == 'guard_dataset_changed'


@pytest.mark.parametrize('invalid', ['nan', 'overflow', 'axis', 'groups', 'empty'])
def test_invalid_data_rejected_before_any_normalizer_or_model_fit(tmp_path, monkeypatch, invalid):
    from backend.app import training
    source = write_grouped_classification_csv(tmp_path/'data.csv', groups_per_class=3, repeats=1, feature_count=8)
    frame = pd.read_csv(source)
    if invalid == 'nan': frame.iloc[0,3] = np.nan
    if invalid == 'overflow': frame.iloc[0,3] = 1e100
    if invalid == 'axis': frame.columns = [*frame.columns[:3], '1', '0', *frame.columns[5:]]
    if invalid == 'groups': frame = frame.iloc[:2]
    if invalid == 'empty': frame = frame.iloc[:0]
    frame.to_csv(source, index=False)
    fits = []
    def fit(*args, **kwargs):
        fits.append(1)
        raise AssertionError('fit reached')
    monkeypatch.setattr(training, '_fit_x_normalizer', fit)
    monkeypatch.setattr(training, 'build_traditional_model', fit)
    with pytest.raises(ValueError):
        training._run_legacy_training(source, {'model_type':'logistic_regression'}, output_dir=tmp_path/'run')
    assert fits == []


def test_reports_are_deduplicated_scoped_and_atomic_with_success(tmp_path):
    repo = RunRepository(tmp_path/'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='dataset', config={'model_type':'logistic_regression'},
        guard_policy=GuardPolicy().model_dump(), principal=Principal('owner','tenant'))
    now = datetime.now(timezone.utc)
    repo.record_worker_heartbeat(worker_id='w', now=now, contract_version='training-worker-guard-v1')
    run = repo.claim_next(worker_id='w', now=now)
    value = report(run, 'pre_fit')
    repo.save_guard_report(value)
    repo.save_guard_report(value)
    assert repo.get_guard_report(value.report_id, principal=Principal('owner','tenant')) == value
    with pytest.raises(KeyError): repo.get_guard_report(value.report_id, principal=Principal('other','tenant'))
    with pytest.raises(GuardError): repo.finish_success(run.run_id, claim_token=run.claim_token, now=now)
    assert repo.get(run.run_id).state == 'running'
    publication = report(run, 'publication', manifest_digest='a'*64)
    done = repo.finish_success(run.run_id, claim_token=run.claim_token, now=now, publication_report=publication)
    assert done.publication_report_id == publication.report_id
    assert done.state == 'succeeded'
    assert 'owner' not in json.dumps(project_guard(publication))


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), -.1, 1.1])
def test_invalid_selection_metric_cannot_be_filtered_away(value):
    from backend.app.runs.guard import validation_metrics
    with pytest.raises(GuardError):
        validation_metrics({'valid':{'accuracy':.5,'macro_f1':value}}, 'macro_f1')


def test_zero_and_low_scores_are_valid():
    from backend.app.runs.guard import validation_metrics
    assert validation_metrics({'valid':{'macro_f1':0.,'accuracy':.01}}, 'macro_f1') == {'accuracy':.01,'macro_f1':0.}


@pytest.mark.parametrize('fault', ['missing', 'same_size', 'volatile', 'size_bool', 'hash_type', 'wrong_run', 'wrong_schema', 'escape'])
def test_manifest_strictness_does_not_trust_entry_flags(tmp_path, fault):
    from backend.app.runs.artifacts import RunArtifactWriter
    from backend.app.runs.guard import inspect_artifacts
    root = tmp_path/'run'
    writer = RunArtifactWriter(root)
    writer.write_json('model_metadata.json', {'model_type':'cnn1d'})
    manifest = writer.finalize(run_id='run')
    entry = manifest['artifacts']['model_metadata.json']
    entry['required'] = False
    if fault == 'missing':
        (root/'model_metadata.json').unlink()
    elif fault == 'same_size':
        raw = (root/'model_metadata.json').read_bytes()
        (root/'model_metadata.json').write_bytes(b'!'+raw[1:])
    elif fault == 'volatile': entry['volatile'] = True
    elif fault == 'size_bool': entry['size_bytes'] = True
    elif fault == 'hash_type': entry['sha256'] = 7
    elif fault == 'wrong_run': manifest['run_id'] = 'another'
    elif fault == 'wrong_schema': manifest['schema_version'] = 'unknown'
    elif fault == 'escape': manifest['artifacts']['../private.json'] = entry
    writer.write_json('manifest.json', manifest)
    with pytest.raises((GuardError, OSError)):
        inspect_artifacts(root, 'run', required={'model_metadata.json'})


def test_migration_backup_twice_preserves_historical_rows_and_claim_capability(tmp_path):
    import sqlite3
    from backend.app.runs.guard_store import initialize
    repo = RunRepository(tmp_path/'runs.sqlite3'); repo.initialize()
    old = repo.create_queued(dataset_id='legacy', config={'execution_budget_task_id':'old'})
    with sqlite3.connect(repo.database_path) as source, sqlite3.connect(tmp_path/'backup.sqlite3') as backup:
        before = source.execute('SELECT * FROM runs').fetchall()
        source.backup(backup)
        initialize(backup); initialize(backup)
        assert backup.execute('SELECT * FROM runs').fetchall() == before
    restored = RunRepository(tmp_path/'backup.sqlite3')
    now = datetime.now(timezone.utc)
    restored.record_worker_heartbeat(worker_id='new', now=now, contract_version='training-worker-guard-v1')
    assert restored.claim_next(worker_id='new', now=now).run_id == old.run_id


def test_policy_marker_cannot_be_removed_and_bad_claim_rolls_back_report(tmp_path):
    import sqlite3
    from backend.app.runs.repository import InvalidRunTransition
    repo = RunRepository(tmp_path/'runs.sqlite3'); repo.initialize()
    run = repo.create_queued(dataset_id='d',config={'model_type':'logistic_regression'},guard_policy=GuardPolicy().model_dump())
    with sqlite3.connect(repo.database_path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute('UPDATE runs SET guard_policy_json=NULL WHERE run_id=?',(run.run_id,))
    now = datetime.now(timezone.utc)
    repo.record_worker_heartbeat(worker_id='w',now=now,contract_version='training-worker-guard-v1')
    run = repo.claim_next(worker_id='w',now=now)
    value = report(run,'publication',manifest_digest='a'*64)
    with pytest.raises(InvalidRunTransition):
        repo.finish_success(run.run_id,claim_token='bad',now=now,publication_report=value)
    with pytest.raises(KeyError): repo.get_guard_report(value.report_id,principal=Principal())


def test_guard_limits_and_safe_projection(tmp_path, monkeypatch):
    from backend.app.runs import guard
    from backend.app.runs.artifacts import RunArtifactWriter
    writer = RunArtifactWriter(tmp_path/'run')
    writer.write_json('model_metadata.json', {'private_test':'TEST_CANARY', 'path':'PATH_CANARY'})
    writer.finalize(run_id='run')
    ticks = iter([0, 61])
    monkeypatch.setattr(guard.time, 'monotonic', lambda:next(ticks))
    with pytest.raises(GuardError) as exc:
        guard.inspect_artifacts(tmp_path/'run', 'run', required=set())
    assert exc.value.code == 'guard_resource_limit'


def test_required_policy_covers_current_catalog():
    from backend.app.model_catalog import MODEL_DECLARATIONS
    from backend.app.runs.guard import required_artifacts
    for model in MODEL_DECLARATIONS:
        if not model.implemented:
            with pytest.raises(ValueError): required_artifacts({'model_type':model.id})
            continue
        names = required_artifacts({'model_type':model.id,'feature_selection_enabled':False})
        assert {'config.json','metrics.json','split.json','cv_predictions.csv'} <= names
        assert ('model.pkl' if model.execution_family=='traditional_ml' else 'model.pt') in names


@pytest.mark.parametrize('model',['cnn_mamba1d','dscarnet'])
def test_unavailable_or_retired_model_never_fits(tmp_path, monkeypatch, model):
    from backend.app import training
    source = write_grouped_classification_csv(tmp_path/'data.csv',groups_per_class=3,repeats=1,feature_count=8)
    def fit(*a, **kw): raise AssertionError('fit reached')
    monkeypatch.setattr(training,'_fit_x_normalizer',fit)
    with pytest.raises(ValueError): training.prepare_training_inputs(source, {'model_type':model})


@pytest.mark.parametrize('fault',['missing_row','duplicate_row','cross_split'])
def test_frozen_plan_rejects_row_corruption_without_repartition(fault):
    from backend.tests.test_train_evidence_recipes import view, EVAL
    from backend.app.evaluation_plan import build_evaluation_plan, validate_plan
    data=view(); plan=build_evaluation_plan(data,EVAL,42)
    indices={key:list(rows) for key,rows in plan.indices.items()}
    if fault=='missing_row':indices['train'].pop()
    elif fault=='duplicate_row':indices['train'].append(indices['train'][0])
    else:indices['test'][0]=indices['train'][0]
    changed=plan.model_copy(update={'indices':indices})
    with pytest.raises(ValueError):validate_plan(changed,data,EVAL,42)
