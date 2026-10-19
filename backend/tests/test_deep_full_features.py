"""Deep models run once on full input, independently of ML feature selections."""
from dataclasses import replace
import pytest
from backend.app.contracts import TrainingBatchRequest, TrainingSpec
from backend.app.feature_engineering import DEEP_MODELS, STUDENT_SCHEMES
from backend.app.feature_policy import training_scheme
from backend.app.routers.batches import _batch_config
from backend.app.runs.batch_projection import batch_status_payload, project_model_comparison
from backend.app.runs.comparison_figures import figure_data
from backend.tests.test_model_feature_grid import repo, grid
from backend.tests.test_training_batches import _write_comparison_artifacts


@pytest.mark.parametrize('model', sorted(DEEP_MODELS))
@pytest.mark.parametrize('profile', ['quick', 'full'])
@pytest.mark.parametrize('scheme', [s for s, _ in STUDENT_SCHEMES])
def test_single_run_normalizes_deep_input(model, profile, scheme):
    config = TrainingSpec.from_legacy({'model_type':model,'experiment_version':'word-0904',
                                     'training_profile':profile,'feature_scheme':scheme}).validated(has_external_test=False).to_legacy_dict()
    assert config['feature_scheme'] == 'full'


def test_capability_has_one_deep_scheme_and_forced_map():
    scheme = training_scheme()
    assert scheme['cnn_scheme_count'] == 1
    assert scheme['forced_feature_schemes'] == {m:'full' for m in DEEP_MODELS}
    assert [s['id'] for s in scheme['feature_schemes'] if s['cnn_supported']] == ['full']


def test_real_repository_accepts_all_63_public_combinations(tmp_path):
    from backend.app.routers.catalog import _MODEL_CATALOG
    models = [row[0] for row in _MODEL_CATALOG if row[3]]
    repository = repo(tmp_path)
    batch, runs = grid(repository, models=models, features=[s for s, _ in STUDENT_SCHEMES], mode='leave_one_sample_id_cv')
    assert len(runs) == 63
    assert [(r.config['model_type'],r.config['feature_scheme']) for r in runs if r.config['model_type'] in DEEP_MODELS] == [(m,'full') for m in models if m in DEEP_MODELS]
    work = batch_status_payload(batch.batch_id,runs)['workload']
    assert work['traditional_model_count'] == 10 and work['deep_model_count'] == 3
    assert work['total_outer_folds'] == 1512 and work['inapplicable_count'] == 0


def test_deep_only_pca_choices_create_full_runs_not_failures(tmp_path):
    batch, runs = grid(repo(tmp_path), models=sorted(DEEP_MODELS), features=['pca_components_5','pca_components_10'])
    assert len(runs) == 3 and {r.config['feature_scheme'] for r in runs} == {'full'}
    assert batch.config['feature_schemes'] == ['full']
    assert batch.config['inapplicable_combinations'] == []


def test_implicit_feature_batch_still_has_model_specific_config(tmp_path):
    request = TrainingBatchRequest(model_types=['cnn1d','pls_da'], config={'experiment_version':'word-0904','feature_scheme':'bin_10'})
    _, config, _ = _batch_config(request,has_external_test=False)
    assert config['combinations'] == [{'model_type':'cnn1d','feature_scheme':'full'},{'model_type':'pls_da','feature_scheme':'bin_10'}]


def test_comparison_never_duplicates_deep_predictions_across_schemes(tmp_path):
    batch, runs = grid(repo(tmp_path),models=['pls_da','cnn1d'],features=['bin_5','pca_components_5'])
    finished=[]
    for record in runs:
        _write_comparison_artifacts(tmp_path/record.run_id,record.run_id,score=.8,sample_suffix='same')
        finished.append(replace(record,state='succeeded'))
    data = project_model_comparison(batch_id=batch.batch_id,records=finished,run_dir_for=lambda rid:tmp_path/rid)
    assert data['comparable']
    groups={g['feature_scheme']:g for g in data['scheme_comparisons']}
    assert set(groups) == {'bin_5','pca_components_5','full'}
    assert groups['full']['feature_name'] == '全特征（深度学习）'
    assert [r['model_type'] for r in groups['full']['models']] == ['cnn1d']
    assert all([r['model_type'] for r in groups[s]['models']] == ['pls_da'] for s in ['bin_5','pca_components_5'])
    matrix=figure_data(data,kind='features')
    assert matrix['values']==[[.8,None],[.8,None],[None,.8]]
    assert '全特征' in matrix['reasons'][0][1]
