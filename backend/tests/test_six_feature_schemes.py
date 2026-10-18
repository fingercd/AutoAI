"""Six requested schemes: numerical truth, fold-local fitting and legacy compatibility."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.app import training as t
from backend.app.contracts import TrainingSpec
from backend.app.feature_engineering import FeatureTransform, STUDENT_SCHEMES, SCHEMES, bin_mean
from backend.app.feature_policy import training_scheme, validate_training_options
from backend.app.runs import comparison_figures as figures
from backend.tests.test_word0904_experiments import _small_frame


@pytest.mark.parametrize('width', [5, 10, 20])
def test_requested_bins_average_adjacent_points_and_keep_tail(width):
    x = np.arange(1, 30, dtype=float)[None, :]
    expected = [[x[0, start:start + width].mean() for start in range(0, 29, width)]]
    np.testing.assert_array_equal(bin_mean(x, width), expected)
    transform = FeatureTransform(f'bin_{width}').fit(np.vstack([x, x + 5]))
    assert transform.output_features_ == len(expected[0])
    np.testing.assert_allclose(transform.scaler_.mean_, np.array(expected)[0] + 2.5)


@pytest.mark.parametrize('count', [5, 10, 20])
def test_fixed_pca_matches_sklearn_and_never_learns_from_test(count):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 40))
    transform = FeatureTransform(f'pca_components_{count}').fit(x)
    scaled = StandardScaler().fit_transform(x)
    expected = PCA(n_components=count, svd_solver='full').fit_transform(scaled)
    np.testing.assert_allclose(transform.transform(x), expected, rtol=1e-6, atol=1e-6)
    assert transform.output_features_ == count
    assert transform.metadata()['pca_components'] == count
    components, means = transform.pca_.components_.copy(), transform.scaler_.mean_.copy()
    transform.transform(rng.normal(size=(3, 40)) * 1e8)
    np.testing.assert_array_equal(transform.pca_.components_, components)
    np.testing.assert_array_equal(transform.scaler_.mean_, means)


@pytest.mark.parametrize('shape', [(15, 30), (30, 14)])
def test_fixed_pca_insufficient_training_dimension_fails_without_clipping(shape):
    with pytest.raises(ValueError, match='不自动降低主成分数'):
        FeatureTransform('pca_components_15').fit(np.random.default_rng(42).normal(size=shape))


def test_capability_advertises_six_student_options_and_retains_old_api():
    scheme = training_scheme()
    assert scheme['student_default_feature_scheme'] == 'bin_5'
    assert [(s['id'], s['name']) for s in scheme['feature_schemes'] if s['student_visible']] == STUDENT_SCHEMES
    assert scheme['default_feature_scheme'] == 'full'
    for sid, _ in SCHEMES + STUDENT_SCHEMES:
        validate_training_options('quick', sid, 'pls_da')
        config = TrainingSpec.from_legacy({'model_type':'pls_da', 'experiment_version':'word-0904', 'feature_scheme':sid}).validated(has_external_test=False).to_legacy_dict()
        assert config['feature_scheme'] == sid
    for sid, _ in STUDENT_SCHEMES[3:]:
        with pytest.raises(ValueError, match='CNN 不支持 PCA'):
            validate_training_options('quick', sid, 'cnn1d')


def grouped_frame(count=24, offset=0, length=64):
    frame = _small_frame(count=count * 2, offset=offset, length=length)
    frame['Sample_ID'] = np.repeat(np.arange(offset, offset + count).astype(str), 2)
    frame['Label'] = np.repeat(np.tile(['1', '2'], count // 2), 2)
    return frame


@pytest.mark.parametrize('sid', [sid for sid, _ in STUDENT_SCHEMES])
def test_six_quick_schemes_reach_results_and_only_selected_scheme_runs(tmp_path, monkeypatch, sid):
    path = tmp_path / 'input.csv'
    grouped_frame().to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(path, {'model_type':'pls_da', 'experiment_version':'word-0904', 'feature_scheme':sid}, run_id='six')
    experiment = json.loads((Path(result['run_dir']) / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert [s['scheme_id'] for s in experiment['schemes']] == [key for key, _ in STUDENT_SCHEMES]
    assert sum(s['status'] == 'ready' for s in experiment['schemes']) == 1
    assert experiment['selected_configuration']['scheme_id'] == sid
    assert all(s['metrics'] is None for s in experiment['schemes'] if s['status'] == 'not_run')


@pytest.mark.parametrize('strategy', ['stratified_holdout','leave_one_sample_id_cv','external_test_holdout','leave_one_sample_id_cv_with_external_test'])
def test_fixed_pca_four_modes_keep_groups_and_test_scope(tmp_path, monkeypatch, strategy):
    main = tmp_path / 'main.csv'
    grouped_frame().to_csv(main, index=False)
    cfg = {'model_type':'pls_da','experiment_version':'word-0904','feature_scheme':'pca_components_5','split_mode':strategy}
    if 'external' in strategy:
        external = tmp_path / 'external.csv'
        grouped_frame(4, 100).to_csv(external, index=False)
        cfg['test_data_path'] = str(external)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(main, cfg, run_id='four')
    run_dir = Path(result['run_dir'])
    experiment = json.loads((run_dir / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert len(experiment['fold_configurations']) == (24 if 'leave_one' in strategy else 1)
    assert (experiment['selected_configuration'] is None) == (strategy == 'leave_one_sample_id_cv')
    for row in experiment['fold_configurations']:
        assert row['transform']['output_features'] == 5
    splits = json.loads((run_dir / 'split.json').read_text(encoding='utf-8'))
    for split in splits:
        assert set(split['train_sample_ids']).isdisjoint(split['test_sample_ids'])
        assert set(split['valid_sample_ids']).isdisjoint(split['test_sample_ids'])
    assert sum(map(sum, result['metrics']['test']['confusion_matrix'])) == (8 if 'external' in strategy else 48 if 'leave_one' in strategy else 4)


@pytest.mark.parametrize('sid', ['bin_5','bin_10','bin_20'])
def test_cnn_accepts_each_new_bin_without_changing_its_training_policy(tmp_path, monkeypatch, sid):
    path = tmp_path / 'cnn.csv'
    grouped_frame(length=160).to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    result = t.train_model(path, {'model_type':'cnn1d','experiment_version':'word-0904','feature_scheme':sid,'epochs':2}, run_id='cnn')
    exp = json.loads((Path(result['run_dir']) / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert sum(s['status'] == 'ready' for s in exp['schemes']) == 1
    assert sum(s['status'] == 'not_applicable' for s in exp['schemes']) == 3
    selected = exp['selected_configuration']
    assert selected['params']['optimizer'] == 'AdamW'
    assert selected['transform']['output_features'] == int(np.ceil(160 / int(sid.split('_')[1])))


def test_feature_heatmap_supports_six_new_rows_and_mixed_historical_rows():
    base = {'comparable':True,'models':[{'model_type':'pls_da','metrics':{},'experiment':{
        'selected_configuration':{'scheme_id':'bin_5'},'schemes':[
            {'scheme_id':sid,'status':'ready' if sid == 'bin_5' else 'not_run','metrics':{'balanced_accuracy':.8} if sid == 'bin_5' else None}
            for sid, _ in STUDENT_SCHEMES]}}]}
    spec = figures.figure_data(base, kind='features')
    assert spec['rows'] == ['BIN5','BIN10','BIN20','PCA5','PCA10','PCA20']
    assert spec['values'][0] == [.8] and spec['values'][1] == [None]
    assert spec['selected'][0] == [True]
    mixed = copy.deepcopy(base)
    mixed['models'].append({'model_type':'svm','metrics':{},'experiment':{'schemes':[{'scheme_id':'full','status':'ready','metrics':{'balanced_accuracy':.7}}]}})
    assert len(figures.figure_data(mixed, kind='features')['rows']) == 10
