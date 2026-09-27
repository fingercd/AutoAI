"""Comparison-only rendering/archive contracts. Never submit live training jobs."""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json
import zipfile

import pytest

from backend.app.runs import batch_archive as archive
from backend.app.runs import comparison_figures as figures
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository, RunNotFound


def comparison_fixture(n=4, classes=4, samples=4, features=True):
    ids = ['pls_da','logistic_regression','svm','random_forest','xgboost','cnn1d'][:n]
    labels = ['1','2','10'] + [str(i) for i in range(11, 11 + max(0, classes - 3))]
    labels = labels[:classes]
    models=[]
    for i, mid in enumerate(ids):
        model={'model_type':mid,'run_ids':[mid], 'metrics':{metric:{'mean':.7 + i*.025} for metric in figures.METRICS}}
        if features:
            model['experiment']={'selected_configuration':{'scheme_id':'bin_5','selection_metric':'balanced_accuracy','selection_score':.8}, 'schemes':[
                {'scheme_id':sid,'status':'not_applicable' if mid=='cnn1d' and sid.startswith('pca') else 'ready','metrics':{metric:(.99 if sid=='full' else .8) for metric in figures.METRICS}}
                for sid,_ in figures.SCHEMES]}
        models.append(model)
    return {'schema_version':'model-comparison-v1','batch_id':'fixture','state':'succeeded','comparable':True,'models':models,
            'evaluation':{'primary_aggregation':'direct'},
            'confusion_matrices':[{'model_type':mid,'labels':labels,'confusion_matrix':[[4 if i==j else 0 for j in range(classes)] for i in range(classes)]} for mid in ids],
            'class_recall':{'status':'ready','labels':labels,'rows':[{'model_type':mid,'values':[{'mean':1.,'support':4} for _ in labels]} for mid in ids]},
            'sample_correctness':{'status':'ready','sample_ids':[str(i+1) for i in range(samples)],'values':[{'model_type':mid,'values':[None if i==1 else 0 if i%3==0 else 1 for i in range(samples)]} for mid in ids]}}


@pytest.fixture
def repo_batch(tmp_path):
    repo=RunRepository(tmp_path/'runs.sqlite3');repo.initialize()
    principal=Principal(owner_id='alice',tenant_id='lab')
    batch,records=repo.create_batch_queued(dataset_id='test',test_dataset_id=None,legacy_data_path=None,config={},model_types=['pls_da','svm'],repeat_count=1,base_seed=42,dataset_snapshot={'name':'测试.csv'},principal=principal)
    for _ in records:
        record=repo.claim_next(worker_id='fixture',now=datetime.now(timezone.utc))
        repo.finish_success(record.run_id,claim_token=record.claim_token,now=datetime.now(timezone.utc),manifest_name='manifest.json')
    return repo,batch,principal


def test_pure_figure_contract_missing_labels_selection_and_count_scale(monkeypatch):
    monkeypatch.setattr(figures.feature_policy, 'FEATURE_ENGINEERING_ENABLED', True)
    data=comparison_fixture(n=6,classes=3)
    data['confusion_matrices'][0]['confusion_matrix'][0]=[0,0,0]
    data['confusion_matrices'][1]['confusion_matrix'][0][0]=20
    matrix=figures.figure_data(data,kind='matrix',model='pls_da')
    assert matrix['rows']==['1','2','10']
    assert matrix['values'][0]==[None,None,None]
    assert figures.figure_data(data,kind='matrix',model='pls_da',matrix_mode='count')['vmax']==20
    sample=figures.figure_data(data,kind='samples')
    assert sample['values'][0][1] is None
    assert figures.sample_indices(data,errors=True)==[0,3]
    feature=figures.figure_data(data,kind='features')
    assert not any(feature['selected'][0]) # higher TEST score cannot select full features
    assert all(feature['selected'][1])
    assert feature['values'][4][0] is None # CNN is ranked first; PCA is not applicable


def test_class_precision_figures_preserve_missing_values_and_export(repo_batch, monkeypatch):
    from backend.app.routers.batches import ComparisonFigureRequest

    data = comparison_fixture(n=2, classes=3, features=False)
    data['class_metrics'] = {'status': 'ready', 'labels': ['1', '2', '10'], 'rows': [
        {'model_type': 'pls_da', 'values': [{'precision': .6}, {'precision': None}, {'precision': 0}]},
        {'model_type': 'logistic_regression', 'values': [{'precision': .75}, {'precision': 1}, {'precision': .5}]},
    ]}
    assert ComparisonFigureRequest(kind='precision').kind == 'precision'
    svg, spec = figures.render_figure(data, kind='precision')
    png, png_spec = figures.render_figure(data, kind='precision', format='png')
    assert spec == png_spec
    assert spec['columns'] == ['1', '2', '10']
    assert spec['rows'] == ['Elastic Net', 'PLS-DA']
    assert spec['values'] == [[.75, 1, .5], [.6, None, 0]]
    assert b'Precision' in svg and png.startswith(b'\x89PNG')
    assert figures.figure_data(data, kind='recall')['values'] == [[1., 1., 1.], [1., 1., 1.]]
    repo, batch, principal = repo_batch
    monkeypatch.setattr(archive, 'project_model_comparison', lambda **_: data)
    archive.ensure_archive(repo, batch.batch_id, principal, Path)
    bundle = zipfile.ZipFile(BytesIO(archive.archive_download(repo, batch.batch_id, principal, 'all.zip')))
    assert {'class_precision.svg', 'class_precision.png', 'class_precision.csv'} <= set(bundle.namelist())


@pytest.mark.parametrize('models,classes,samples',[(1,4,4),(4,4,50),(6,10,1000)])
def test_scientific_figures_render_svg_png_and_paginate(models,classes,samples,monkeypatch):
    monkeypatch.setattr(figures.feature_policy, 'FEATURE_ENGINEERING_ENABLED', True)
    data=comparison_fixture(models,classes,samples)
    specs=figures.figure_specs(data)
    assert sum(spec['kind']=='samples' for spec in specs)==(samples+49)//50
    for spec in [specs[0],next(item for item in specs if item['kind']=='matrix'),next(item for item in specs if item['kind']=='samples'),next(item for item in specs if item['kind']=='features')]:
        options={key:value for key,value in spec.items() if key!='id'}
        svg,meta=figures.render_figure(data,width=480,**options)
        png,png_meta=figures.render_figure(data,width=480,format='png',**options)
        assert b'<svg' in svg and png.startswith(b'\x89PNG')
        assert meta==png_meta and meta['height']>0
        if spec['kind']=='samples':assert len(meta['columns'])<=50


def test_archive_roundtrip_scoped_idempotent_integrity_and_stop(repo_batch,monkeypatch):
    repo,batch,principal=repo_batch
    data=comparison_fixture(n=1,features=False)
    monkeypatch.setattr(archive,'project_model_comparison',lambda **_:data)
    status=archive.ensure_archive(repo,batch.batch_id,principal,lambda rid:Path(rid))
    assert status['state']=='ready'
    second=archive.ensure_archive(repo,batch.batch_id,principal,lambda rid:Path(rid))
    assert second['created_at']==status['created_at']
    bundle=zipfile.ZipFile(BytesIO(archive.archive_download(repo,batch.batch_id,principal,'all.zip')))
    assert {'manifest.json','comparison.json','overall_accuracy.svg','overall_accuracy.png','samples_001.csv'}<=set(bundle.namelist())
    assert not any(name.endswith(('.pkl','.pt','.joblib')) for name in bundle.namelist())
    with pytest.raises(RunNotFound):archive.load_archive(repo,batch.batch_id,Principal(owner_id='bob',tenant_id='lab'))
    for name in ['../model.pkl','model.pkl','error.json']:
        with pytest.raises(archive.ArchiveUnavailable):archive.archive_download(repo,batch.batch_id,principal,name)
    target=archive.archive_dir(repo,batch.batch_id)/'overall_accuracy.svg'
    target.write_text('tampered',encoding='utf-8')
    with pytest.raises(archive.ArchiveUnavailable):archive.archive_download(repo,batch.batch_id,principal,target.name)
    repo.cancel_batch_scoped(batch.batch_id,now=datetime.now(timezone.utc),principal=principal)
    with pytest.raises(archive.ArchiveUnavailable):archive.load_archive(repo,batch.batch_id,principal)
    archive.discard_batch_archive(repo,batch.batch_id)
    assert not archive.archive_dir(repo,batch.batch_id).exists()


def test_archive_failure_does_not_change_runs_and_retries(repo_batch,monkeypatch):
    repo,batch,principal=repo_batch
    monkeypatch.setattr(archive,'project_model_comparison',lambda **_:comparison_fixture(n=1,features=False))
    actual=archive.render_figure
    monkeypatch.setattr(archive,'render_figure',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('renderer failed')))
    with pytest.raises(RuntimeError):archive.ensure_archive(repo,batch.batch_id,principal,Path)
    assert archive.archive_status(repo,batch.batch_id,principal)['state']=='failed'
    assert all(record.state=='succeeded' for record in repo.list_batch_runs_scoped(batch.batch_id,principal=principal))
    monkeypatch.setattr(archive,'render_figure',actual)
    assert archive.ensure_archive(repo,batch.batch_id,principal,Path)['state']=='ready'


def test_stop_during_build_prevents_publication(repo_batch,monkeypatch):
    repo,batch,principal=repo_batch
    def interrupted(**_):
        repo.cancel_batch_scoped(batch.batch_id,now=datetime.now(timezone.utc),principal=principal)
        return {'comparable':False,'reason':'fixture'}
    monkeypatch.setattr(archive,'project_model_comparison',interrupted)
    with pytest.raises(archive.ArchiveUnavailable):archive.ensure_archive(repo,batch.batch_id,principal,Path)
    assert not archive.archive_dir(repo,batch.batch_id).exists()


def test_archive_lock_excludes_concurrent_build(repo_batch):
    repo,batch,_=repo_batch
    with archive._lock(repo,batch.batch_id):
        with pytest.raises(archive.ArchiveBusy):
            with archive._lock(repo,batch.batch_id):pass


def test_batch_pagination_and_summary_api(repo_batch,monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.routers import batches
    from backend.app.http.principal import get_principal
    repo,batch,principal=repo_batch
    monkeypatch.setattr(batches,'get_run_repository',lambda:repo)
    app.dependency_overrides[get_principal]=lambda:principal
    try:
        client=TestClient(app)
        response=client.get('/api/training/batches?limit=1')
        assert response.status_code==200 and response.json()['items'][0]['batch_id']==batch.batch_id
        assert client.get('/api/training/batches?limit=0').status_code==422
        assert client.get(f'/api/training/batches/{batch.batch_id}/archive').json()['state']=='missing'
        assert client.post(f'/api/training/batches/{batch.batch_id}/figure',json={'kind':'illegal'}).status_code==422
        app.dependency_overrides[get_principal]=lambda:Principal(owner_id='bob',tenant_id='lab')
        assert client.get(f'/api/training/batches/{batch.batch_id}/archive').status_code==404
    finally:app.dependency_overrides.pop(get_principal,None)


def test_other_page_components_unchanged_and_styles_scoped():
    css=Path('static/js/comparison-page.css').read_text(encoding='utf-8')
    assert ':root' not in css and '#view-results' not in css
    js=Path('static/js/comparison-page.js').read_text(encoding='utf-8')
    assert 'stopRun' not in js and 'createRun' not in js and 'v2/' not in js
    html=Path('static/index.html').read_text(encoding='utf-8')
    start=html.index('      function renderBatchComparison(')
    end=html.index('\n      }',start)
    assert 'createAccuracyRanking' not in html[start:end]


def test_worker_automatically_archives_after_last_model(tmp_path,monkeypatch):
    from backend.app import paths
    from backend.app.runs.worker import RunWorker
    from backend.tests.test_training_batches import _write_comparison_artifacts
    repo=RunRepository(tmp_path/'runs.sqlite3');repo.initialize()
    batch,_=repo.create_batch_queued(dataset_id='fixture',test_dataset_id=None,legacy_data_path=None,config={'split_mode':'stratified_holdout'},model_types=['pls_da','svm'],repeat_count=1,base_seed=42,dataset_snapshot={'sha256':'same'},principal=Principal())
    monkeypatch.setattr(paths,'RUNS_DIR',tmp_path/'runs')
    def execute(run):
        _write_comparison_artifacts(paths.RUNS_DIR/run.run_id,run.run_id,score=.8,sample_suffix='same')
        return {'manifest_name':'manifest.json'}
    worker=RunWorker(repository=repo,worker_id='fixture',execute=execute,now=lambda:datetime.now(timezone.utc))
    assert worker.run_once()
    assert archive.archive_status(repo,batch.batch_id,Principal())['state']=='pending'
    assert worker.run_once()
    assert archive.archive_status(repo,batch.batch_id,Principal())['state']=='ready'
    assert all(run.state=='succeeded' for run in repo.list_batch_runs_scoped(batch.batch_id,principal=Principal()))


def test_incomparable_archive_saves_reason_without_charts(repo_batch,monkeypatch):
    repo,batch,principal=repo_batch
    monkeypatch.setattr(archive,'project_model_comparison',lambda **_:{'comparable':False,'reason':'测试对象不同','models':[]})
    archive.ensure_archive(repo,batch.batch_id,principal,Path)
    saved=archive.load_archive(repo,batch.batch_id,principal)
    assert saved['reason']=='测试对象不同'
    assert archive.archive_status(repo,batch.batch_id,principal)['files']==['comparison.json']
