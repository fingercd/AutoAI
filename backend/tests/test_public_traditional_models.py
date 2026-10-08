"""Nine-model capabilities and real fold-local execution of newly public models."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app import training as t
from backend.app.contracts import TrainingConfigValidationError, TrainingSpec
from backend.app.feature_policy import supported_feature_schemes
from backend.app.routers.catalog import get_models_catalog
from backend.app.training_experiments import candidate_configs
from backend.tests.test_word0904_experiments import _small_frame

NEW_MODELS = ['spls_da', 'pca_lda', 'pca_svm']


def test_nine_visible_models_match_versioned_feature_capabilities():
    catalog = get_models_catalog()
    visible = [row for row in catalog['models'] if row['ui_visible']]
    assert len(visible) == 9
    assert set(catalog['training_scheme']['supported_feature_schemes']) == {row['id'] for row in visible}
    for row in catalog['models']:
        assert row['supported_feature_schemes'] == supported_feature_schemes(row['id'])
        assert bool(row['supported_feature_schemes']) == row['ui_visible']
    for model in NEW_MODELS:
        assert TrainingSpec.from_strict({'model_type': model, 'experiment_version': 'word-0904'}).validated(has_external_test=False).values['model_type'] == model


@pytest.mark.parametrize('model', ['pca_lda', 'pca_svm'])
@pytest.mark.parametrize('profile', ['quick', 'full'])
def test_nested_pca_rejected_before_training(model, profile):
    with pytest.raises(TrainingConfigValidationError, match='已内置 PCA'):
        TrainingSpec.from_legacy({'model_type': model, 'experiment_version': 'word-0904',
                                 'training_profile': profile, 'feature_scheme': 'pca_95'}).validated(has_external_test=False)


@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('features,train_count', [(1, 2), (2, 3), (160, 8)])
def test_quick_candidate_count_and_dimension_caps(model, features, train_count):
    config = t.TrainConfig(model_type=model, training_profile='quick')
    candidates = candidate_configs(config, model, features, train_count)
    assert 1 <= len(candidates) <= 3
    for candidate in candidates:
        components = candidate.spls_components if model == 'spls_da' else candidate.pca_components
        assert 1 <= components <= min(features, train_count - 1)
        if model == 'spls_da':
            assert candidate.spls_keepx <= features
        if model == 'pca_svm':
            assert candidate.svm_kernel == 'linear'
    assert candidate_configs(config, model, features, 1) == []


@pytest.mark.parametrize('model', NEW_MODELS)
def test_complete_grid_keeps_dimension_caps_and_stable_candidates(model):
    config = t.TrainConfig(model_type=model, training_profile='full')
    candidates = candidate_configs(config, model, 160, 101)
    expected = {'spls_da': 24, 'pca_lda': 11, 'pca_svm': 55}
    assert len(candidates) == expected[model]
    small = candidate_configs(config, model, 1, 2)
    assert len(small) == (5 if model == 'pca_svm' else 1)


@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('strategy', ['stratified_holdout', 'leave_one_sample_id_cv',
                                     'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'])
def test_quick_new_models_preserve_four_evaluation_contracts(tmp_path, monkeypatch, model, strategy):
    main = tmp_path / 'main.csv'
    _small_frame(length=40).to_csv(main, index=False)
    config = {'model_type': model, 'experiment_version': 'word-0904',
              'training_profile': 'quick', 'feature_scheme': 'full', 'split_mode': strategy}
    if 'external' in strategy:
        external = tmp_path / 'external.csv'
        _small_frame(4, 100, length=40).to_csv(external, index=False)
        config['test_data_path'] = str(external)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(main, config, run_id=model)
    run_dir = Path(result['run_dir'])
    experiment = json.loads((run_dir / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert sum(row['status'] == 'ready' for row in experiment['schemes']) == 1
    assert all(row['selection_metric'] == 'validation_balanced_accuracy' for row in experiment['fold_configurations'])
    expected = 4 if 'external' in strategy else 12 if 'leave_one' in strategy else 2
    assert sum(map(sum, result['metrics']['test']['confusion_matrix'])) == expected
    assert np.isfinite(result['metrics']['test']['balanced_accuracy'])
    assert (experiment['selected_configuration'] is None) == (strategy == 'leave_one_sample_id_cv')
    assert (run_dir / 'all_predictions.csv').is_file()
    assert (run_dir / 'manifest.json').is_file()
    for split in json.loads((run_dir / 'split.json').read_text(encoding='utf-8')):
        assert set(split['train_sample_ids']).isdisjoint(split['test_sample_ids'])
        if split['fold_index'] != 'external_final':
            assert set(split['train_sample_ids']).isdisjoint(split['valid_sample_ids'])


@pytest.mark.parametrize('model', NEW_MODELS)
def test_complete_new_models_execute_all_applicable_schemes(tmp_path, monkeypatch, model):
    main = tmp_path / 'main.csv'
    _small_frame(20, length=40).to_csv(main, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(main, {'model_type': model, 'experiment_version': 'word-0904',
                                 'training_profile': 'full'}, run_id=model)
    experiment = json.loads((Path(result['run_dir']) / 'feature_experiments.json').read_text(encoding='utf-8'))
    expected = set(supported_feature_schemes(model))
    assert {row['scheme_id'] for row in experiment['schemes'] if row['status'] == 'ready'} == expected
    assert all(row['status'] == 'not_applicable' for row in experiment['schemes'] if row['scheme_id'] not in expected)
    assert experiment['selected_configuration']['selection_metric'] == 'mean_balanced_accuracy_grouped_5fold'
    search_path = Path(result['run_dir']) / 'hyperparameter_search.csv'
    with search_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(len(json.loads(row['inner_fold_scores'])) == 5 for row in rows if row['status'] == 'ready')
