"""Numerical and end-to-end acceptance for the versioned 0904 experiment."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backend.app.feature_engineering import FeatureTransform, bin_mean
from backend.app.training_experiments import CNN0904, cnn_profile, candidate_configs
from backend.app import training as t


@pytest.fixture(autouse=True)
def enable_retained_experiment_for_regression(monkeypatch):
    """Keep algorithm regression independent of an operator disabling the feature."""
    from backend.app import feature_policy
    monkeypatch.setattr(feature_policy, 'FEATURE_ENGINEERING_ENABLED', True)


def test_binning_preserves_partial_tail():
    np.testing.assert_array_equal(bin_mean(np.array([[1, 3, 5, 7, 9]]), 2), [[2, 6, 9]])


def test_pca_fit_does_not_change_when_test_values_change():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(20, 50))
    transform = FeatureTransform('pca_95').fit(train)
    components = transform.pca_.components_.copy()
    mean = transform.scaler_.mean_.copy()
    transform.transform(rng.normal(size=(5, 50)) * 1e6)
    np.testing.assert_array_equal(components, transform.pca_.components_)
    np.testing.assert_array_equal(mean, transform.scaler_.mean_)
    assert 0 < transform.output_features_ < 50


@pytest.mark.parametrize('n,expected', [(100,[8,16,32]),(300,[16,32,64]),(301,[32,64,128])])
def test_cnn_sample_boundaries(n,expected):
    assert cnn_profile(n,3000)['channels'] == expected


def test_cnn_length_boundary_and_binary_head():
    import torch
    assert cnn_profile(100,3000)['kernels'] == [9,5,3]
    assert cnn_profile(100,3001)['kernels'] == [9,7,5]
    assert cnn_profile(100,3001)['pools'] == [4,2,2]
    model=CNN0904(cnn_profile(60,500),2)
    assert model(torch.zeros(2,1,500)).shape == (2,2)


def test_svm_complete_grid():
    candidates=candidate_configs(t.TrainConfig(model_type='svm',training_profile='full'),'svm',500,40)
    assert len(candidates)==20
    assert sum(c.svm_kernel=='linear' for c in candidates)==5
    assert {c.svm_gamma for c in candidates if c.svm_kernel=='rbf'}=={'scale',.001,.01}


def test_pls_does_not_clip_to_class_count():
    from backend.app.training_experiments import PLS0904
    rng=np.random.default_rng(17)
    model=PLS0904(3).fit(rng.normal(size=(20,10)), np.repeat([0,1],10))
    assert model.model.n_components == 3


def _small_frame(count=12, offset=0, length=160):
    rng=np.random.default_rng(42+offset)
    labels=np.tile(['1','2'],count//2)
    frame=pd.DataFrame(rng.normal(0,.3,(count,length))+np.tile([0,2],count//2)[:,None],columns=[str(i+1) for i in range(length)])
    frame.insert(0,'Name',[f's{i+offset}' for i in range(count)])
    frame.insert(0,'Sample_ID',[str(i+offset) for i in range(count)])
    frame.insert(0,'Label',labels);frame.insert(0,'Index',np.arange(count))
    return frame


@pytest.mark.parametrize('strategy',['stratified_holdout','leave_one_sample_id_cv','external_test_holdout','leave_one_sample_id_cv_with_external_test'])
def test_new_evaluation_paths(tmp_path,monkeypatch,strategy):
    import json
    from backend.app import training_experiments as e
    # Keep the genuine fold-local search and seven schemes; one parameter
    # candidate makes the outer-CV contract test small and deterministic.
    monkeypatch.setattr(e,'candidate_configs',lambda config,*args:[t._clone_config(config,pls_components=1)])
    path=tmp_path/'main.csv';_small_frame().to_csv(path,index=False)
    config={'model_type':'pls_da','experiment_version':'word-0904','split_mode':strategy,'training_profile':'full'}
    if 'external' in strategy:
        ext=tmp_path/'external.csv';_small_frame(4,100).to_csv(ext,index=False);config['test_data_path']=str(ext)
    monkeypatch.setattr(t,'RUNS_DIR',tmp_path/'runs')
    result=t.train_model(path,config,run_id='check')
    experiment=json.loads((Path(result['run_dir'])/'feature_experiments.json').read_text(encoding='utf-8'))
    assert all(s['status']=='ready' for s in experiment['schemes'])
    assert sum(map(sum,result['metrics']['test']['confusion_matrix']))==(4 if 'external' in strategy else 2 if strategy=='stratified_holdout' else 12)
    assert (experiment['selected_configuration'] is None)==(strategy=='leave_one_sample_id_cv')
    assert len(experiment['fold_configurations'])==(12 if 'leave_one' in strategy else 1)
    splits=json.loads((Path(result['run_dir'])/'split.json').read_text(encoding='utf-8'))
    for fold in splits:
        assert set(fold['train_sample_ids']).isdisjoint(fold['test_sample_ids'])
        if fold['fold_index']!='external_final':
            assert set(fold['train_sample_ids']).isdisjoint(fold['valid_sample_ids'])
    # 全量预测明细：holdout 口径由最终模型覆盖全部记录并标注 train/valid/test；
    # 交叉验证口径取逐折 pooled OOF 行，每条记录只出现一次且记为 test。
    import pandas as pd
    all_predictions=pd.read_csv(Path(result['run_dir'])/'all_predictions.csv')
    assert len(all_predictions)==(16 if 'external' in strategy else 12)
    expected_splits={
        'stratified_holdout':{'train','valid','test'},
        'external_test_holdout':{'train','valid','external_test'},
        'leave_one_sample_id_cv':{'test'},
        'leave_one_sample_id_cv_with_external_test':{'test','external_test'},
    }[strategy]
    assert set(all_predictions['split'])==expected_splits
    assert all_predictions['Sample_ID'].nunique()==len(all_predictions)
    assert list(all_predictions.columns[:7])==['dataset','split','fold_index','index','Sample_ID','true_label','pred_label']


def test_selection_ignores_test_values_and_records_partial_scheme(monkeypatch):
    from backend.app import training_experiments as e
    monkeypatch.setattr(e,'candidate_configs',lambda config,*args:[t._clone_config(config,pls_components=1)])
    frame=_small_frame();x=frame.iloc[:,4:].to_numpy(copy=True);y=np.tile([0,1],6);groups=np.arange(12).astype(str)
    splits={'train':list(range(8)),'valid':[8,9],'test':[10,11]}
    cfg=t.TrainConfig(model_type='pls_da',experiment_version='word-0904',training_profile='full')
    a=e.fit_experiment_fold(cfg,'pls_da',x,y,groups,splits,['1','2'],1,lambda:None,lambda _:None)
    x[10:]*=1000
    b=e.fit_experiment_fold(cfg,'pls_da',x,y,groups,splits,['1','2'],1,lambda:None,lambda _:None)
    assert a['scheme_id']==b['scheme_id'] and a['selection_score']==b['selection_score']
    for first,second in zip(a['results'],b['results']):
        assert first['selection_score']==second['selection_score']


def test_failed_feature_scheme_is_not_reported_as_zero(monkeypatch):
    from backend.app import training_experiments as e
    monkeypatch.setattr(e,'candidate_configs',lambda config,*args:[t._clone_config(config,pls_components=1)])
    original=FeatureTransform.fit
    def fail_one(self,x):
        if self.scheme_id=='bin_10':
            raise ValueError('验收用方案失败')
        return original(self,x)
    monkeypatch.setattr(FeatureTransform,'fit',fail_one)
    frame=_small_frame();x=frame.iloc[:,4:].to_numpy();y=np.tile([0,1],6);groups=np.arange(12).astype(str)
    result=e.fit_experiment_fold(t.TrainConfig(model_type='pls_da',training_profile='full'),'pls_da',x,y,groups,{'train':list(range(8)),'valid':[8,9],'test':[10,11]},['1','2'],1,lambda:None,lambda _:None)
    summary=e.summarize_experiments([result],['1','2'])
    failed=next(s for s in summary['schemes'] if s['scheme_id']=='bin_10')
    assert failed['status']=='failed' and failed['metrics'] is None


def test_cnn_external_loso_refits_selected_epochs(tmp_path,monkeypatch):
    import json
    path=tmp_path/'main.csv';_small_frame().to_csv(path,index=False)
    ext=tmp_path/'external.csv';_small_frame(4,100).to_csv(ext,index=False)
    monkeypatch.setattr(t,'RUNS_DIR',tmp_path/'runs')
    result=t.train_model(path,{'model_type':'cnn1d','experiment_version':'word-0904','training_profile':'full','epochs':1,'split_mode':'leave_one_sample_id_cv_with_external_test','split_train':7,'split_valid':3,'test_data_path':str(ext)},run_id='cnn')
    experiment=json.loads((Path(result['run_dir'])/'feature_experiments.json').read_text(encoding='utf-8'))
    assert len(experiment['fold_configurations'])==12
    assert experiment['selected_configuration']['params']['best_epoch']==1
    selected=experiment['selected_configuration']
    assert selected['requested_ratio']=={'train':7,'valid':3,'test':0}
    assert selected['params']['N']==12
    assert selected['params']['selection_sample_count']==selected['split_summary']['train']['sample_count']
    assert selected['split_summary']['train']['sample_count']+selected['split_summary']['valid']['sample_count']==12
    assert sum(s['status']=='ready' for s in experiment['schemes'])==4
    assert sum(map(sum,result['metrics']['test']['confusion_matrix']))==4
    assert result['model_metadata']['loss_function']=='CrossEntropyLoss'


@pytest.mark.skipif(os.environ.get('AUTOAI_FULL_EXPERIMENT_TEST') != '1', reason='Explicit six-model acceptance run')
@pytest.mark.parametrize('model', ['pls_da','logistic_regression','svm','random_forest','xgboost','cnn1d'])
def test_six_model_experiment(tmp_path, monkeypatch, model):
    import json
    rng=np.random.default_rng(42)
    labels=np.repeat(['1','2'],30)
    values=rng.normal(0,.2,(60,500))+np.repeat([0.,2.],30)[:,None]
    frame=pd.DataFrame(values,columns=[str(i+1) for i in range(500)])
    frame.insert(0,'Name',[f's{i}' for i in range(60)])
    frame.insert(0,'Sample_ID',[str(i+1) for i in range(60)])
    frame.insert(0,'Label',labels)
    frame.insert(0,'Index',np.arange(1,61))
    path=tmp_path/'fixture.csv';frame.to_csv(path,index=False)
    monkeypatch.setattr(t,'RUNS_DIR',tmp_path/'runs')
    result=t.train_model(path,{'model_type':model,'experiment_version':'word-0904','training_profile':'full','epochs':200,'seed':42},run_id=model)
    root=Path(result['run_dir'])
    experiment=json.loads((root/'feature_experiments.json').read_text(encoding='utf-8'))
    assert experiment['selected_configuration']
    assert len(experiment['schemes'])==7
    assert sum(s['status']=='ready' for s in experiment['schemes'])==(4 if model=='cnn1d' else 7)
    assert sum(map(sum,result['metrics']['test']['confusion_matrix']))==6
    assert (root/'manifest.json').is_file()
