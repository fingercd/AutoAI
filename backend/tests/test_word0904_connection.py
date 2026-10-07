"""Classic request, feature selection and archive reconnection regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from backend.app import training as t
from backend.app.contracts import TrainingSpec
from backend.app.training_experiments import candidate_configs, grouped_experiment_folds
from backend.app.runs import batch_archive as archive
from backend.tests.test_comparison_archive import repo_batch, comparison_fixture


def test_classic_word_frontend_behaviors():
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    subprocess.run([node,'--test','backend/tests/frontend_word0904.mjs'],check=True)


def test_public_capability_and_four_modes():
    from fastapi.testclient import TestClient
    from backend.app.main import app
    scheme=TestClient(app).get('/api/models').json()['training_scheme']
    assert scheme['enabled'] and scheme['defaults']['epochs']==200
    assert scheme['default_profile']=='quick' and scheme['quick_candidate_count']==3
    assert {item['id'] for item in scheme['profiles']}=={'quick','full'}
    for external,mode in [(False,'stratified_holdout'),(False,'leave_one_sample_id_cv'),(True,'external_test_holdout'),(True,'leave_one_sample_id_cv_with_external_test')]:
        config={'model_type':'svm','experiment_version':scheme['version'],'split_mode':mode,'split_train':6 if mode=='stratified_holdout' else 7,'split_valid':2 if mode=='stratified_holdout' else 3}
        if mode=='stratified_holdout':config['split_test']=2
        actual=TrainingSpec.from_legacy(config).validated(has_external_test=external).to_legacy_dict()
        assert actual['split_mode']==mode and actual['split_train']==config['split_train']


def test_complete_grids_ignore_legacy_manual_search_limits():
    for model,count in [('pls_da',10),('logistic_regression',15),('svm',20),('random_forest',12),('xgboost',12)]:
        cfg=t.TrainConfig(model_type=model,random_forest_search_iterations=1,training_profile='full')
        assert len(candidate_configs(cfg,model,500,100))==count


def test_grouped_five_folds_keep_uneven_measurements_and_classes():
    groups=np.repeat(np.arange(15).astype(str),[20,1,2,3,4]*3)
    y=np.repeat(np.repeat([0,1,2],5),[20,1,2,3,4]*3)
    folds=grouped_experiment_folds(y,groups,list(range(len(y))),42,['1','2','10'])
    held_out=[]
    for train,valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
        assert set(y[train])==set(y[valid])=={0,1,2}
        held_out.extend(valid)
    assert sorted(held_out)==list(range(len(y)))
    with pytest.raises(ValueError,match='每类至少 5'):
        grouped_experiment_folds(y,groups,np.flatnonzero(groups!='0').tolist(),42,['1','2','10'])


def test_archive_rebuilds_missing_feature_figures_atomically(repo_batch,monkeypatch):
    repo,batch,principal=repo_batch
    data=comparison_fixture(n=1,features=True)
    monkeypatch.setattr(archive,'project_model_comparison',lambda **_:data)
    initial=archive.ensure_archive(repo,batch.batch_id,principal,Path)
    old_dir=archive.archive_dir(repo,batch.batch_id)
    manifest_path=old_dir/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']={k:v for k,v in manifest['files'].items() if not k.startswith('features_')}
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    assert archive.archive_status(repo,batch.batch_id,principal)['state']=='outdated'
    rebuilt=archive.ensure_archive(repo,batch.batch_id,principal,Path)
    assert rebuilt['state']=='ready'
    assert sum(name.startswith('features_') for name in rebuilt['files'])==12
    assert archive.archive_dir(repo,batch.batch_id)!=old_dir
    assert old_dir.is_dir()  # Existing readers retain a complete immutable generation.
    assert archive.ensure_archive(repo,batch.batch_id,principal,Path)['created_at']==rebuilt['created_at']


def test_failed_atomic_pointer_switch_preserves_previous_archive(repo_batch, monkeypatch):
    repo,batch,principal=repo_batch
    monkeypatch.setattr(archive,'project_model_comparison',lambda **_:comparison_fixture(n=1,features=False))
    previous=archive.ensure_archive(repo,batch.batch_id,principal,Path)
    old_dir=archive.archive_dir(repo,batch.batch_id)
    replace=archive.os.replace
    def fail_pointer(source,target):
        if Path(target).name=='current.json':raise OSError('simulated publication failure')
        return replace(source,target)
    monkeypatch.setattr(archive.os,'replace',fail_pointer)
    with pytest.raises(OSError):archive.ensure_archive(repo,batch.batch_id,principal,Path,force=True)
    assert archive.archive_dir(repo,batch.batch_id)==old_dir
    assert archive.archive_status(repo,batch.batch_id,principal)['created_at']==previous['created_at']


def test_six_models_create_one_run_each_with_custom_external_loo_ratio(tmp_path):
    from backend.app.contracts import TrainingBatchRequest
    from backend.app.routers.batches import _batch_config
    from backend.app.runs.contracts import Principal
    from backend.app.runs.repository import RunRepository
    models=['pls_da','logistic_regression','svm','random_forest','xgboost','cnn1d']
    types,config,_=_batch_config(TrainingBatchRequest(model_types=models,config={'experiment_version':'word-0904','split_mode':'leave_one_sample_id_cv_with_external_test','split_train':7,'split_valid':3}),has_external_test=True)
    repo=RunRepository(tmp_path/'runs.sqlite3');repo.initialize()
    batch,records=repo.create_batch_queued(dataset_id='fixture',test_dataset_id='external',legacy_data_path=None,config=config,model_types=types,repeat_count=1,base_seed=42,dataset_snapshot={},principal=Principal())
    assert len(records)==6 and batch.repeat_count==1
    assert all(r.config['split_train']==7 and r.config['split_valid']==3 and r.config['split_test']==0 and r.config['experiment_version']=='word-0904' for r in records)
