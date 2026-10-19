"""Restored traditional models use their real estimators and bounded quick search."""
import json
from pathlib import Path
import pytest
from backend.app import training as t
from backend.app.contracts import TrainingSpec
from backend.app.feature_policy import training_scheme
from backend.app.training_experiments import candidate_configs
from backend.tests.test_six_feature_schemes import grouped_frame


@pytest.mark.parametrize('model', ['spls_da', 'pca_lda', 'pca_svm'])
def test_restored_model_quick_training_and_probabilities(tmp_path, monkeypatch, model):
    path = tmp_path / 'input.csv'
    grouped_frame(length=160).to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    config = {'model_type': model, 'experiment_version': 'word-0904', 'feature_scheme': 'bin_5', 'training_profile': 'quick'}
    spec = TrainingSpec.from_legacy(config).validated(has_external_test=False)
    assert len(candidate_configs(t.TrainConfig(**spec.to_legacy_dict()), model, 32, 30)) == 3
    result = t.train_model(path, config, run_id=model)
    assert result['metrics']['test']['accuracy'] is not None
    exp = json.loads((Path(result['run_dir']) / 'feature_experiments.json').read_text(encoding='utf-8'))
    assert exp['selected_configuration']['scheme_id'] == 'bin_5'
    assert exp['selected_configuration']['params']


@pytest.mark.parametrize('model', ['pca_lda', 'pca_svm'])
def test_internal_pca_rejects_duplicate_outer_pca(model):
    with pytest.raises(ValueError, match='内置 PCA'):
        TrainingSpec.from_legacy({'model_type': model, 'experiment_version': 'word-0904', 'feature_scheme': 'pca_components_20'}).validated(has_external_test=False)
    row = next(s for s in training_scheme()['feature_schemes'] if s['id'] == 'pca_components_20')
    assert model in row['unsupported_models']


@pytest.mark.parametrize('model', ['cnn1d_se', 'resnet1d'])
@pytest.mark.parametrize('mode', ['stratified_holdout', 'leave_one_sample_id_cv', 'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'])
def test_deep_models_all_evaluation_modes(tmp_path, monkeypatch, model, mode):
    frame = grouped_frame(count=12, length=160)
    # Two-class fixtures also exercise the two-logit CrossEntropy contract.
    frame['Label'] = frame['Label'].astype(int) % 2
    path = tmp_path / 'input.csv'
    frame.to_csv(path, index=False)
    monkeypatch.setattr(t, 'RUNS_DIR', tmp_path / 'runs')
    config = {'model_type': model, 'experiment_version': 'word-0904', 'feature_scheme': 'bin_5', 'epochs': 2, 'split_mode': mode}
    if 'external' in mode:
        external = tmp_path / 'external.csv'
        frame.assign(Sample_ID=frame['Sample_ID'].astype(int) + 100).to_csv(external, index=False)
        config['test_data_path'] = str(external)
    result = t.train_model(path, config, run_id=model)
    directory = Path(result['run_dir'])
    metadata = json.loads((directory / 'model_metadata.json').read_text(encoding='utf-8'))
    assert metadata['output_dim'] == 2
    assert metadata['loss_function'] == 'CrossEntropyLoss'
    assert metadata['profile']['model_class'] == {'cnn1d_se': 'CNNSE1DDocumentV2', 'resnet1d': 'ResNet1DDocumentV2'}[model]
    assert (directory / 'feature_transform.joblib').is_file()
    assert result['metrics']['test']['accuracy'] is not None


def test_all_visible_combinations_fit_batch_limit():
    from backend.app.contracts import TrainingBatchRequest
    from backend.app.feature_engineering import MODELS, STUDENT_SCHEMES
    from backend.app.routers.batches import _batch_config
    _, config, _ = _batch_config(TrainingBatchRequest(model_types=sorted(MODELS), feature_schemes=[s for s, _ in STUDENT_SCHEMES], config={'experiment_version': 'word-0904'}), has_external_test=False)
    assert len(config['combinations']) == 75
    assert len(config['inapplicable_combinations']) == 6
