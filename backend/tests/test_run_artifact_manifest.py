import json

import pytest

from backend.app.runs.artifacts import (
    ARTIFACT_CATALOG,
    ArtifactIntegrityError,
    MANIFEST_SCHEMA_VERSION,
    RunArtifactWriter,
)


def test_manifest_is_the_last_committed_public_artifact(tmp_path):
    writer = RunArtifactWriter(tmp_path / 'run-1')
    writer.write_json('metrics.json', {'balanced_accuracy': 0.9})
    writer.write_private_bytes('dscarnet_pca.joblib', b'not-downloadable')
    manifest = writer.finalize(run_id='run-1')

    saved = json.loads((tmp_path / 'run-1' / 'manifest.json').read_text(encoding='utf-8'))
    assert saved == manifest
    assert manifest['artifacts']['metrics.json']['downloadable'] is True
    assert manifest['artifacts']['dscarnet_pca.joblib']['downloadable'] is False


def test_v2_manifest_uses_an_explicit_catalog_and_keeps_model_objects_private(tmp_path):
    writer = RunArtifactWriter(tmp_path / 'run-2')
    writer.write_json('metrics.json', {'test': {'macro_f1': 0.9}})
    writer.write_json('config.json', {'model_type': 'pls_da', 'epochs': 20})
    writer.write_bytes('model.pkl', b'unsafe-pickle')
    writer.write_bytes('unknown.bin', b'internal')

    manifest = writer.finalize(run_id='run-2')

    assert manifest['schema_version'] == MANIFEST_SCHEMA_VERSION
    assert manifest['artifacts']['metrics.json']['downloadable'] is True
    assert manifest['artifacts']['config.json']['downloadable'] is True
    assert manifest['artifacts']['model.pkl']['downloadable'] is False
    assert manifest['artifacts']['unknown.bin']['downloadable'] is False
    assert writer.resolve_download('metrics.json').name == 'metrics.json'
    assert writer.resolve_download('config.json').name == 'config.json'
    with pytest.raises(PermissionError):
        writer.resolve_download('model.pkl')
    with pytest.raises(PermissionError):
        writer.resolve_download('unknown.bin')


def test_new_catalog_exposes_only_sample_level_explainability() -> None:
    assert 'sample_feature_importance.json' in ARTIFACT_CATALOG
    assert 'sample_feature_importance.csv' in ARTIFACT_CATALOG
    assert 'feature_importance.json' not in ARTIFACT_CATALOG
    assert 'feature_importance.csv' not in ARTIFACT_CATALOG


def test_manually_written_global_importance_is_private_for_new_manifests(tmp_path) -> None:
    writer = RunArtifactWriter(tmp_path / 'run-no-global')
    writer.write_json('metrics.json', {'test': {'accuracy': 0.8}})
    writer.write_json('feature_importance.json', {'status': 'ready'})

    manifest = writer.finalize(run_id='run-no-global')

    assert manifest['artifacts']['feature_importance.json']['downloadable'] is False
    with pytest.raises(PermissionError):
        writer.resolve_download('feature_importance.json')


def test_config_with_a_server_path_is_automatically_kept_private(tmp_path):
    writer = RunArtifactWriter(tmp_path / 'run-path-config')
    writer.write_json('config.json', {'model_type': 'pls_da', 'data_path': '/srv/private/data.csv'})

    manifest = writer.finalize(run_id='run-path-config')

    assert manifest['artifacts']['config.json']['downloadable'] is False
    assert manifest['artifacts']['config.json']['download_reason'] == '配置包含服务器路径，仅供内部使用'
    with pytest.raises(PermissionError):
        writer.resolve_download('config.json')


def test_artifact_descriptors_report_missing_and_corrupt_files_without_exposing_paths(tmp_path):
    writer = RunArtifactWriter(tmp_path / 'run-3')
    writer.write_json('metrics.json', {'test': {'accuracy': 0.8}})
    writer.write_json('model_metadata.json', {'model_type': 'pls_da'})
    writer.finalize(run_id='run-3')
    (tmp_path / 'run-3' / 'metrics.json').write_text('{"tampered":true}', encoding='utf-8')

    _, descriptors = writer.descriptors(run_id='run-3')
    by_name = {descriptor['name']: descriptor for descriptor in descriptors}

    assert by_name['metrics.json']['integrity'] == 'corrupt'
    assert by_name['metrics.json']['downloadable'] is False
    assert by_name['metrics.json']['download_url'] is None
    assert by_name['predictions.csv']['integrity'] == 'missing'
    assert by_name['predictions.csv']['required'] is True
    assert by_name['predictions.csv']['reason'] == '必需结果文件未生成'
    assert by_name['model.pkl']['downloadable'] is False
    assert all('path' not in descriptor for descriptor in descriptors)
    assert all(str(tmp_path) not in json.dumps(descriptor, ensure_ascii=False) for descriptor in descriptors)

    with pytest.raises(ArtifactIntegrityError):
        writer.resolve_download('metrics.json')


def test_legacy_manifest_download_flag_remains_read_only_compatible(tmp_path):
    run_dir = tmp_path / 'legacy-run'
    run_dir.mkdir()
    (run_dir / 'feature_importance.json').write_text('{"status":"ready"}', encoding='utf-8')
    (run_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'run_id': 'legacy-run',
                'artifacts': {
                    'feature_importance.json': {'downloadable': True},
                },
            }
        ),
        encoding='utf-8',
    )

    resolved = RunArtifactWriter(run_dir).resolve_download('feature_importance.json')

    assert json.loads(resolved.read_text(encoding='utf-8')) == {'status': 'ready'}
