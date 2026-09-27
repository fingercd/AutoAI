"""Step 9 provenance tests: real shared training plus adversarial private audits."""
from copy import deepcopy
import json

import pytest

from backend.app import training
from backend.app.diagnostic_evidence import TrainingValidationAudit, paired_metrics
from backend.app.runs.artifacts import ARTIFACT_CATALOG, RunArtifactWriter
from backend.tests.modeling_data_factory import write_grouped_classification_csv


@pytest.fixture
def trained(tmp_path):
    source = write_grouped_classification_csv(tmp_path/'input.csv', groups_per_class=5,
        repeats=2, feature_count=16, labels=('PRIVATE_CLASS_A', 'PRIVATE_CLASS_B'))
    run = tmp_path/'audit-test'
    training._run_legacy_training(source, {'model_type':'logistic_regression',
        'feature_selection_enabled':False}, run_id='audit-test', output_dir=run)
    docs = {p.name:json.loads(p.read_text(encoding='utf8')) for p in run.glob('*.json')}
    return run, docs


def test_private_audit_same_selection_model_pre_refit(trained):
    run, docs = trained
    audit = TrainingValidationAudit.model_validate(docs['training_validation_audit.json'])
    assert audit.status == 'ready'
    fold = audit.folds[0]
    assert fold.train.evaluation_snapshot_id == fold.valid.evaluation_snapshot_id
    assert fold.fit_scope == 'train' and fold.best_epoch is None
    assert paired_metrics(docs)['status'] == 'ready'
    assert 'PRIVATE_CLASS' not in json.dumps(docs['training_validation_audit.json'])
    assert 'test' not in fold.model_dump()
    assert ARTIFACT_CATALOG['training_validation_audit.json']['downloadable'] is False
    assert ARTIFACT_CATALOG['training_validation_audit.json']['required'] is False
    RunArtifactWriter(run).finalize(run_id='audit-test')
    with pytest.raises((FileNotFoundError, PermissionError)):
        RunArtifactWriter(run).resolve_download('training_validation_audit.json')


@pytest.mark.parametrize('fault', ['snapshot','refit','config','split','support','metric','extra','producer'])
def test_semantically_bad_optional_audit_is_unavailable(trained, fault):
    _, original = trained
    docs = deepcopy(original)
    audit = docs['training_validation_audit.json']
    fold = audit['folds'][0]
    if fault == 'snapshot': fold['valid']['evaluation_snapshot_id'] = '0'*64
    elif fault == 'refit': fold['fit_scope'] = 'train+valid'
    elif fault == 'config': audit['config_digest'] = '0'*64
    elif fault == 'split': fold['partition_digest'] = '0'*64
    elif fault == 'support': fold['train']['support']['observation_count'] += 1
    elif fault == 'metric': fold['valid']['metrics']['accuracy'] = .314
    elif fault == 'extra': fold['test'] = {'accuracy': .9}
    elif fault == 'producer': audit['producer_source_digest'] = '0'*64
    assert paired_metrics(docs) == dict(status='unavailable', reason_code='audit_invalid', pairs=[], support=[])


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -.1, 1.1, '0.5'])
def test_metric_type_is_strict(trained, value):
    _, docs = trained
    docs['training_validation_audit.json']['folds'][0]['train']['metrics']['macro_f1'] = value
    assert paired_metrics(docs)['status'] == 'unavailable'


def test_test_canary_does_not_change_projection_and_old_missing_is_honest(trained):
    _, docs = trained
    before = paired_metrics(docs)
    docs['metrics.json']['test'] = {'macro_f1': 'PRIVATE_TEST_CANARY'}
    docs['split.json'][0]['metrics'] = {'macro_f1': 'PRIVATE_TEST_CANARY'}
    assert paired_metrics(docs) == before
    del docs['training_validation_audit.json']
    assert paired_metrics(docs)['reason_code'] == 'audit_missing'


@pytest.mark.parametrize('model', ['logistic_regression', 'random_forest', 'cnn1d', 'pca_mlp'])
def test_audit_collected_before_refit_without_extra_fit(tmp_path, monkeypatch, model):
    from backend.app import diagnostic_evidence
    source = write_grouped_classification_csv(tmp_path/'input.csv', groups_per_class=5,
        repeats=1, feature_count=32)
    events = []
    collect = diagnostic_evidence.collect_fold
    refit = training._fit_final_traditional_model
    def collecting(**kwargs):
        events.append('collect')
        return collect(**kwargs)
    def refitting(**kwargs):
        assert events == ['collect']
        events.append('refit')
        return refit(**kwargs)
    monkeypatch.setattr(diagnostic_evidence, 'collect_fold', collecting)
    monkeypatch.setattr(training, '_fit_final_traditional_model', refitting)
    run=tmp_path/'run'
    training._run_legacy_training(source, {'model_type':model, 'epochs':2,
        'feature_selection_enabled':False}, output_dir=run)
    docs={p.name:json.loads(p.read_text(encoding='utf8')) for p in run.glob('*.json')}
    assert paired_metrics(docs)['status'] == 'ready'
    fold=docs['training_validation_audit.json']['folds'][0]
    if model in ('cnn1d', 'pca_mlp'):
        assert events == ['collect'] and 1 <= fold['best_epoch'] <= 2
    else:
        assert events == ['collect','refit']
    if model == 'random_forest':
        assert fold['selection_criterion'] == 'oob_balanced_accuracy'
        assert fold['selection_reuses_validation'] is False


@pytest.mark.parametrize('groups', [3, 20])
def test_cv_pairs_all_folds_never_pools_independent_support(tmp_path, groups):
    source=write_grouped_classification_csv(tmp_path/'input.csv',groups_per_class=groups,
        repeats=1,feature_count=8)
    run=tmp_path/'run'
    training._run_legacy_training(source, {'model_type':'logistic_regression',
        'split_mode':'leave_one_sample_id_cv','split_train':8,'split_valid':2,'feature_selection_enabled':False}, output_dir=run)
    docs={p.name:json.loads(p.read_text(encoding='utf8')) for p in run.glob('*.json')}
    result=paired_metrics(docs)
    if groups == 3:
        assert result['reason_code'] == 'class_support_incomplete'
        assert result['pairs'] == []
    else:
        assert result['status']=='ready' and len(result['support'])==2*groups and len(result['pairs'])==6*groups
    docs['training_validation_audit.json']['folds'].pop()
    assert paired_metrics(docs)['status']=='unavailable'
