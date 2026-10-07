"""Quantity limits, validation selection and isolation for the default quick profile."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app import training as t
from backend.app import training_experiments as e
from backend.app.contracts import TrainingConfigValidationError, TrainingSpec
from backend.app.feature_engineering import FeatureTransform
from backend.tests.test_word0904_experiments import _small_frame


@pytest.mark.parametrize('model', ['pls_da', 'logistic_regression', 'svm', 'random_forest', 'xgboost'])
def test_quick_models_train_three_candidates_and_one_final_model(monkeypatch, model):
    if model == 'xgboost' and importlib.util.find_spec('xgboost') is None:
        pytest.skip('Optional XGBoost unavailable')
    cfg = t.TrainConfig(model_type=model, experiment_version='word-0904')
    candidates = e.candidate_configs(cfg, model, 160, 8)
    assert len(candidates) == 3
    frame = _small_frame()
    x = frame.iloc[:, 4:].to_numpy()
    y = np.tile([0, 1], 6)
    groups = np.arange(12).astype(str)
    splits = {'train': list(range(8)), 'valid': [8, 9], 'test': [10, 11]}
    built, fitted = [], []
    original_build, original_fit = e.build_estimator, FeatureTransform.fit

    def build(config, targets, labels):
        built.append(len(targets))
        return original_build(config, targets, labels)

    def transform_fit(self, values):
        fitted.append(values.copy())
        return original_fit(self, values)

    monkeypatch.setattr(e, 'build_estimator', build)
    monkeypatch.setattr(FeatureTransform, 'fit', transform_fit)
    result = e.fit_experiment_fold(cfg, model, x, y, groups, splits, ['1', '2'], 1, lambda: None, lambda _: None)
    assert built == [8, 8, 8, 10]  # No fifth audit fit.
    assert len(fitted) == 2
    np.testing.assert_array_equal(fitted[0], x[splits['train']])
    np.testing.assert_array_equal(fitted[1], x[splits['train'] + splits['valid']])
    assert result['selection_metric'] == 'validation_balanced_accuracy'
    assert result['selection_score'] == result['evals']['valid']['balanced_accuracy']
    assert len(result['search_rows']) == 3
    assert all(len(row['inner_fold_scores']) == 1 for row in result['search_rows'])
    assert sum(row['is_selected'] for row in result['search_rows']) == 1
    summary = e.summarize_experiments([result], ['1', '2'])
    assert sum(s['status'] == 'ready' for s in summary['schemes']) == 1
    assert sum(s['status'] == 'not_run' for s in summary['schemes']) == 6
    assert all(s['metrics'] is None for s in summary['schemes'] if s['status'] == 'not_run')


def test_quick_selection_uses_validation_and_ignores_test_values():
    frame = _small_frame()
    x = frame.iloc[:, 4:].to_numpy(copy=True)
    y, groups = np.tile([0, 1], 6), np.arange(12).astype(str)
    cfg = t.TrainConfig(model_type='logistic_regression', feature_scheme='pca_95')
    splits = {'train': list(range(8)), 'valid': [8, 9], 'test': [10, 11]}
    first = e.fit_experiment_fold(cfg, cfg.model_type, x, y, groups, splits, ['1', '2'], 1, lambda: None, lambda _: None)
    x[10:] = 1e9
    second = e.fit_experiment_fold(cfg, cfg.model_type, x, y, groups, splits, ['1', '2'], 1, lambda: None, lambda _: None)
    assert first['params'] == second['params']
    assert first['selection_score'] == second['selection_score']
    np.testing.assert_array_equal(first['transformer'].scaler_.mean_, second['transformer'].scaler_.mean_)
    np.testing.assert_array_equal(first['transformer'].pca_.components_, second['transformer'].pca_.components_)


@pytest.mark.parametrize('strategy', ['stratified_holdout', 'leave_one_sample_id_cv',
                                     'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'])
def test_quick_four_evaluation_modes_keep_prediction_contract(tmp_path, monkeypatch, strategy):
    path = tmp_path / 'main.csv'
    _small_frame().to_csv(path, index=False)
    config = {'model_type': 'pls_da', 'experiment_version': 'word-0904', 'split_mode': strategy,
              'feature_scheme': 'bin_5', 'split_train': 7 if strategy != 'stratified_holdout' else 8,
              'split_valid': 3 if strategy != 'stratified_holdout' else 1}
    if 'external' in strategy:
        ext = tmp_path / 'external.csv'
        _small_frame(4, 100).to_csv(ext, index=False)
        config['test_data_path'] = str(ext)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(path, config, run_id='quick')
    root = Path(result['run_dir'])
    experiment = json.loads((root / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert sum(s['status'] == 'ready' for s in experiment['schemes']) == 1
    assert all(f['selection_metric'] == 'validation_balanced_accuracy' for f in experiment['fold_configurations'])
    expected = 4 if 'external' in strategy else 12 if 'leave_one' in strategy else 2
    assert sum(map(sum, result['metrics']['test']['confusion_matrix'])) == expected
    assert (experiment['selected_configuration'] is None) == (strategy == 'leave_one_sample_id_cv')
    assert result['config']['training_profile'] == 'quick'
    assert (root / 'manifest.json').is_file()
    splits = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    for split in splits:
        assert set(split['train_sample_ids']).isdisjoint(split['test_sample_ids'])
        assert split['selection_metric'] == 'validation_balanced_accuracy'
        if split['fold_index'] != 'external_final':
            assert set(split['train_sample_ids']).isdisjoint(split['valid_sample_ids'])


def test_quick_cnn_keeps_scheduler_and_single_scheme(tmp_path, monkeypatch):
    path = tmp_path / 'main.csv'
    _small_frame().to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(path, {'model_type': 'cnn1d', 'experiment_version': 'word-0904',
                                'feature_scheme': 'bin_5', 'epochs': 2}, run_id='cnn')
    experiment = json.loads((Path(result['run_dir']) / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert sum(s['status'] == 'ready' for s in experiment['schemes']) == 1
    params = experiment['selected_configuration']['params']
    assert params['scheduler'] == 'ReduceLROnPlateau'
    assert params['optimizer'] == 'AdamW'
    assert params['loss'] == 'CrossEntropyLoss'


@pytest.mark.parametrize('options', [{'training_profile': 'invalid'}, {'feature_scheme': 'invalid'},
                                    {'model_type': 'cnn1d', 'feature_scheme': 'pca_95'}])
def test_invalid_quick_options_rejected_before_enqueue(options):
    with pytest.raises(TrainingConfigValidationError):
        TrainingSpec.from_legacy({'model_type': 'svm', 'experiment_version': 'word-0904', **options}).validated(has_external_test=False)
