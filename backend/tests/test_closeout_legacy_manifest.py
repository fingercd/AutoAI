"""Migrated benchmark manifests stay intact, while private/internal data stays closed."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.http.principal import get_principal
from backend.app.runs.artifacts import ArtifactIntegrityError, ManifestCorruptError, RunArtifactWriter
from backend.app.runs.contracts import Principal
from backend.app.runs.guard import GuardError, inspect_artifacts
from backend.tests.test_run_result_contract_v1 import _patch_run_storage, _create_succeeded_run

INTERNAL = [f'details/{kind}/000000.json' for kind in
            ('epochs', 'search_trials', 'candidate_samples', 'samples')] + [
                'weights/fold-0001.pt', 'weights/fold-0001.pkl']


def add_internal(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name in INTERNAL:
        raw = b'{"private_fixture":true}'
        path = folder/name;path.parent.mkdir(exist_ok=True, parents=True);path.write_bytes(raw)
        manifest['artifacts'][name] = dict(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            required=True, downloadable=True, category='internal')
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return (folder/'manifest.json').read_bytes()


@pytest.mark.parametrize('model,family,strategy', [
    ('pls_da','traditional_ml','stratified_holdout'),
    ('cnn1d','deep_learning','stratified_holdout'),
    ('cnn1d','deep_learning','leave_one_sample_id_cv')])
def test_real_legacy_structure_preserves_ready_results_summary_scope_and_private_paths(tmp_path, monkeypatch, model, family, strategy):
    repo,root=_patch_run_storage(monkeypatch,tmp_path)
    principal=Principal(owner_id='owner',tenant_id='tenant')
    record=_create_succeeded_run(repo,root,principal=principal,model_type=model,model_family=family,strategy=strategy,
        cv_summary={'pooled_test':{'macro_f1':.75},'fold_mean':{'test':{'macro_f1':.5}}})
    folder=root/record.run_id;original=add_internal(folder);writer=RunArtifactWriter(folder)
    app.dependency_overrides[get_principal]=lambda:principal
    try:
        client=TestClient(app);url=f'/api/training/runs/{record.run_id}'
        result=client.get(url+'/result').json()
        assert result['run']['result_state']=='ready'
        assert result['metrics']['primary']['macro_f1']==.75
        summary=client.get('/api/training/runs?projection=summary').json()['items'][0]
        assert summary['result_state']=='ready' and summary['test_macro_f1']==.75
        assert client.get(url+'/artifact/metrics.json').status_code==200
        assert client.get(url+'/artifact/config.json').status_code==403
        for name in INTERNAL:
            with pytest.raises(PermissionError):writer.resolve_download(name)
        assert all(not a['downloadable'] and a['download_url'] is None for a in result['artifacts'] if '/' in a['name'])
        app.dependency_overrides[get_principal]=lambda:Principal(owner_id='stranger',tenant_id='tenant')
        assert client.get(url+'/result').status_code==404
        assert client.get(url+'/artifact/metrics.json').status_code==404
        assert (folder/'manifest.json').read_bytes()==original
    finally:app.dependency_overrides.pop(get_principal,None)


@pytest.mark.parametrize('name', ['../escape.json','/tmp/x','C:/x','C:\\x','details\\epochs\\000000.json',
    'details//epochs/000000.json','details/epochs/../000000.json','./metrics.json','details/epochs/%2e%2e.json',
    'details/epochs/000000.json/','unknown/file.json','weights/../model.pt','x\x00y'])
def test_unsafe_or_unknown_manifest_paths_are_rejected(tmp_path,name):
    writer=RunArtifactWriter(tmp_path/'run');writer.write_json('metrics.json',{})
    manifest=writer.finalize(run_id='run');manifest['artifacts'][name]={'downloadable':True}
    (writer.run_dir/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ManifestCorruptError):writer.load_manifest()


def test_symlink_escape_is_not_a_legacy_exception(tmp_path):
    folder=tmp_path/'run';writer=RunArtifactWriter(folder);writer.write_json('metrics.json',{})
    writer.finalize(run_id='run');add_internal(folder)
    target=folder/INTERNAL[0];target.unlink();outside=tmp_path/'outside';outside.write_text('{}')
    try:target.symlink_to(outside)
    except OSError as exc:
        if getattr(exc,'winerror',None)==1314:pytest.skip('Windows lacks symlink privilege; exercised on Linux')
        raise
    with pytest.raises(ManifestCorruptError):writer.load_manifest()


@pytest.mark.parametrize('damage',['same_size','missing','wrong_run','unknown_version'])
def test_corruption_never_becomes_a_ready_result_or_status_fallback(tmp_path,monkeypatch,damage):
    repo,root=_patch_run_storage(monkeypatch,tmp_path);record=_create_succeeded_run(repo,root)
    folder=root/record.run_id;add_internal(folder)
    if damage=='same_size':
        path=folder/'metrics.json';raw=path.read_bytes();path.write_bytes(raw.replace(b'0.75',b'0.99'))
    elif damage=='missing':(folder/'metrics.json').unlink()
    else:
        m=json.loads((folder/'manifest.json').read_text());m['run_id' if damage=='wrong_run' else 'schema_version']='invalid'
        (folder/'manifest.json').write_text(json.dumps(m))
    status=json.loads((folder/'status.json').read_text());status['metrics']={'test':{'macro_f1':.99}}
    (folder/'status.json').write_text(json.dumps(status))
    c=TestClient(app);result=c.get(f'/api/training/runs/{record.run_id}/result').json()
    assert result['run']['result_state'] in ('partial','corrupt_manifest')
    assert result['metrics']['primary']=={}
    summary=c.get('/api/training/runs?projection=summary').json()['items'][0]
    assert summary['test_macro_f1'] is None and summary['result_state']!='ready'


def test_legacy_flags_cannot_publish_config_weights_or_joblib(tmp_path):
    writer=RunArtifactWriter(tmp_path/'run');writer.write_json('config.json',{'data_path':'/private/data'})
    writer.write_bytes('model.pt',b'private');writer.write_bytes('model.pkl',b'private');writer.write_bytes('map.joblib',b'private')
    m=writer.finalize(run_id='run');m.pop('schema_version')
    for entry in m['artifacts'].values():entry['downloadable']=True
    (writer.run_dir/'manifest.json').write_text(json.dumps(m))
    for name in m['artifacts']:
        with pytest.raises(PermissionError):writer.resolve_download(name)


def test_current_guard_and_writer_do_not_accept_legacy_nested_publication(tmp_path):
    writer=RunArtifactWriter(tmp_path/'run');writer.write_json('metrics.json',{})
    writer.finalize(run_id='run');add_internal(writer.run_dir)
    with pytest.raises(ValueError):writer.write_json(INTERNAL[0],{})
    with pytest.raises(GuardError):inspect_artifacts(writer.run_dir,'run',required={'metrics.json'})
