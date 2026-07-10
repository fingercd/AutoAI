import json

from backend.app.runs.artifacts import RunArtifactWriter


def test_manifest_is_the_last_committed_public_artifact(tmp_path):
    writer = RunArtifactWriter(tmp_path / 'run-1')
    writer.write_json('metrics.json', {'balanced_accuracy': 0.9})
    writer.write_private_bytes('dscarnet_pca.joblib', b'not-downloadable')
    manifest = writer.finalize(run_id='run-1')

    saved = json.loads((tmp_path / 'run-1' / 'manifest.json').read_text(encoding='utf-8'))
    assert saved == manifest
    assert manifest['artifacts']['metrics.json']['downloadable'] is True
    assert manifest['artifacts']['dscarnet_pca.joblib']['downloadable'] is False
