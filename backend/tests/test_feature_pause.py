"""Temporary feature-engineering pause must not disable ordinary training."""
from pathlib import Path

import pytest

from backend.app import feature_policy
from backend.app.contracts import TrainingSpec, TrainingConfigValidationError
from backend.app.runs.comparison_figures import figure_specs, figure_data


def test_feature_pause_blocks_versioned_config_but_keeps_normal_training():
    assert feature_policy.FEATURE_ENGINEERING_ENABLED is False
    for model in ['pls_da', 'logistic_regression', 'svm', 'random_forest', 'xgboost', 'cnn1d']:
        result = TrainingSpec.from_legacy({'model_type': model, 'normalization': 'zscore'}).validated(has_external_test=False)
        assert result.to_legacy_dict()['model_type'] == model
        with pytest.raises(TrainingConfigValidationError, match='特征工程暂时停用'):
            TrainingSpec.from_legacy({'model_type': model, 'experiment_version': 'word-0904'}).validated(has_external_test=False)


def test_direct_training_entry_cannot_bypass_pause(tmp_path):
    from backend.app.training import _run_legacy_training
    with pytest.raises(ValueError, match='特征工程暂时停用'):
        _run_legacy_training(tmp_path / 'not-read.csv', {'model_type': 'svm', 'experiment_version': 'word-0904'})
    assert not (tmp_path / 'not-read.csv').exists()


def test_feature_chart_and_new_archive_inventory_are_disabled():
    from backend.tests.test_comparison_archive import comparison_fixture
    data = comparison_fixture(features=True)
    specs = figure_specs(data)
    assert specs and not any(item['kind'] == 'features' for item in specs)
    assert any(item['kind'] == 'samples' for item in specs)
    with pytest.raises(ValueError, match='特征工程暂时停用'):
        figure_data(data, kind='features')


def test_frontend_feature_block_is_retained_but_gated():
    source = Path('static/js/comparison-page.js').read_text(encoding='utf-8')
    assert 'const FEATURE_ENGINEERING_ENABLED = false;' in source
    assert "if(FEATURE_ENGINEERING_ENABLED){\n    const feature=section('特征工程 × 模型'" in source
    assert 'cmp-matrix-dialog' in source
    css = Path('static/js/comparison-page.css').read_text(encoding='utf-8')
    assert '#view-comparison dialog.cmp-matrix-dialog > button' in css
