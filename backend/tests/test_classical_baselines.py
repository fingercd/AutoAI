"""New classifiers: actual estimators, fold-local selection, probabilities and contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from backend.app import training as t
from backend.app.contracts import TrainingSpec
from backend.app.models.registry import build_traditional_model
from backend.app.training_experiments import candidate_configs
from backend.tests.test_six_feature_schemes import grouped_frame

NEW_MODELS = ['lasso', 'ridge', 'knn', 'extra_trees', 'shrinkage_lda']


def test_elastic_net_keeps_model_and_declares_numeric_budget():
    config = t.TrainConfig(model_type='logistic_regression', training_profile='quick')
    model = build_traditional_model(config, np.array([0, 1]), 2)
    assert isinstance(model, LogisticRegression)
    assert model.penalty == 'elasticnet' and model.solver == 'saga'
    assert model.max_iter == 10000 and model.tol == 1e-3
    assert [c.logistic_c for c in candidate_configs(config, config.model_type, 20, 20)] == [.1, 1., 10.]


@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('classes', [2, 3])
def test_genuine_classifier_and_thread_budget(monkeypatch, model, classes):
    monkeypatch.setenv('AUTOAI_WORKER_THREADS', '2')
    config = t.TrainConfig(model_type=model)
    estimator = build_traditional_model(config, np.array([0, 1]), 2)
    assert isinstance(estimator, {'lasso': OneVsRestClassifier, 'ridge': OneVsRestClassifier, 'knn': KNeighborsClassifier, 'extra_trees': ExtraTreesClassifier, 'shrinkage_lda': LinearDiscriminantAnalysis}[model])
    if model in {'lasso', 'ridge'}:
        assert isinstance(estimator.estimator, LogisticRegression)
        assert estimator.estimator.penalty == ('l1' if model == 'lasso' else 'l2')
        assert estimator.estimator.solver == 'liblinear'
        assert estimator.estimator.max_iter == 5000
    elif model in {'knn', 'extra_trees'}:
        assert estimator.n_jobs == 2
    else:
        assert estimator.shrinkage == 'auto' and estimator.solver == 'lsqr'
    if model == 'extra_trees':
        assert estimator.bootstrap is False
    rng = np.random.default_rng(42)
    x = rng.normal(size=(classes * 12, 8)); y = np.repeat(np.arange(classes), 12)
    estimator.fit(x, y)
    probabilities = estimator.predict_proba(x)
    assert probabilities.shape == (classes * 12, classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)


@pytest.mark.parametrize('model', NEW_MODELS)
def test_quick_and_full_candidates_are_bounded_by_training_data(model):
    config = t.TrainConfig(model_type=model, training_profile='quick')
    candidates = candidate_configs(config, model, 20, 4)
    assert 1 <= len(candidates) <= 3
    if model == 'knn':
        assert [c.knn_n_neighbors for c in candidates] == [1, 3]
    config.training_profile = 'full'
    candidates = candidate_configs(config, model, 20, 20)
    assert len(candidates) == {'lasso': 5, 'ridge': 5, 'knn': 20, 'extra_trees': 12, 'shrinkage_lda': 1}[model]


@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('mode', ['stratified_holdout', 'leave_one_sample_id_cv', 'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'])
def test_new_models_four_evaluation_modes(tmp_path, monkeypatch, model, mode):
    frame = grouped_frame(count=12, length=120)
    path = tmp_path / 'input.csv'; frame.to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    config = {'model_type': model, 'experiment_version': 'word-0904', 'feature_scheme': 'pca_components_5', 'split_mode': mode}
    if 'external' in mode:
        external = tmp_path / 'external.csv'
        frame.assign(Sample_ID=frame['Sample_ID'].astype(int) + 100).to_csv(external, index=False)
        config['test_data_path'] = str(external)
    result = t.train_model(path, config, run_id=model)
    directory = Path(result['run_dir'])
    experiment = json.loads((directory / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert len(experiment['fold_configurations']) == (12 if 'leave_one' in mode else 1)
    assert (experiment['selected_configuration'] is None) == (mode == 'leave_one_sample_id_cv')
    for fold in json.loads((directory / 'split.json').read_text(encoding='utf-8')):
        assert set(fold['train_sample_ids']).isdisjoint(fold['test_sample_ids'])
        assert set(fold['valid_sample_ids']).isdisjoint(fold['test_sample_ids'])
    assert result['metrics']['test']['accuracy'] is not None
    assert all(row['transform']['output_features'] == 5 for row in experiment['fold_configurations'])


@pytest.mark.parametrize('field,value', [('knn_n_neighbors',0),('knn_p',-1),('knn_weights','bad'),('knn_metric','bad'),('extra_trees_n_estimators',0),('extra_trees_max_depth',0),('extra_trees_min_samples_leaf',0),('extra_trees_max_features',0),('extra_trees_max_features','bad')])
def test_invalid_config_rejected_before_queue(field, value):
    with pytest.raises(ValueError):
        TrainingSpec.from_legacy({'model_type': 'knn' if field.startswith('knn') else 'extra_trees', field: value}).validated(has_external_test=False)


def test_public_thirteen_model_feature_grid():
    from backend.app.contracts import TrainingBatchRequest
    from backend.app.feature_engineering import MODELS, STUDENT_SCHEMES
    from backend.app.routers.batches import _batch_config
    from backend.app.routers.catalog import _MODEL_CATALOG
    public = [m[0] for m in _MODEL_CATALOG if m[3]]
    assert len(public) == 13
    _, config, _ = _batch_config(TrainingBatchRequest(model_types=public,feature_schemes=[s for s,_ in STUDENT_SCHEMES],config={'experiment_version':'word-0904'}),has_external_test=False)
    assert len(config['combinations']) == 63
    assert len(config['inapplicable_combinations']) == 0
