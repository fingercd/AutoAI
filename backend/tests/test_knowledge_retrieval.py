"""Engineering fixtures only; synthetic vectors are never a model quality test."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app import knowledge_retrieval as r


class FixtureEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = np.zeros((len(texts), 512), dtype=np.float32)
        result[:, 0] = 1
        return result


def library(count=50):
    source = dict(source_id='fixture', title='Synthetic test', locator='fixture', publisher='tests',
                  supported_claim='not scientific evidence', applicability_limit='not published real knowledge')
    return r.CardLibrary.model_validate_json(json.dumps(dict(knowledge_set_version='fixture', entries=[
        dict(entry_id=f'card_{i:03}', entry_version='1', title=f'title {i}', body='Same fixture body',
             sources=[source], status='published') for i in range(count)])))


def test_body_only_reuse_and_version_binding(tmp_path):
    encoder = FixtureEncoder()
    first = r.publish(library(), tmp_path, encoder)
    assert encoder.calls == [['Same fixture body']]
    assert len(first.cards.entries) == 50
    changed = library().model_dump(mode='json')
    changed['entries'][0].update(entry_version='2', title='Changed title')
    second = r.publish(r.CardLibrary.model_validate_json(json.dumps(changed)), tmp_path, encoder, previous=first)
    assert encoder.calls[-1] == []
    assert second.manifest['rows'][0]['entry_version'] == '2'
    assert second.manifest_digest != first.manifest_digest
    assert np.array_equal(first.vectors, second.vectors)
    assert r.current_bundle(tmp_path).manifest_digest == second.manifest_digest


@pytest.mark.parametrize('kind', ['nan', 'zero', 'dimension', 'dtype', 'norm'])
def test_bad_vectors(kind):
    vector = np.zeros((1, 512), dtype=np.float32); vector[0, 0] = 1
    if kind == 'nan': vector[0, 0] = np.nan
    if kind == 'zero': vector[:] = 0
    if kind == 'dimension': vector = vector[:, :10]
    if kind == 'dtype': vector = vector.astype(np.float64)
    if kind == 'norm': vector *= 2
    with pytest.raises(r.RetrievalError): r.validate_vectors(vector, 1)


def test_file_hash_and_row_tampering(tmp_path):
    r.publish(library(), tmp_path, FixtureEncoder())
    pointer = json.loads((tmp_path / 'current.json').read_text())
    folder = tmp_path / pointer['directory']
    manifest = json.loads((folder / 'index_manifest.json').read_text())
    manifest['rows'][0]['entry_version'] = '99'
    (folder / 'index_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(r.RetrievalError): r.current_bundle(tmp_path)
    with pytest.raises(r.RetrievalError): r.Bundle(folder)


def test_single_topk_stable_order_and_applicability(tmp_path):
    body = library(6).model_dump(mode='json')
    body['entries'][0]['domains'] = ['raman']
    body['entries'][1]['scopes'] = ['someone_else']
    body['entries'][2]['related_models'] = ['svm']
    bundle = r.publish(r.CardLibrary.model_validate_json(json.dumps(body)), tmp_path, FixtureEncoder())
    q = bundle.vectors[0]
    results = r.retrieve(bundle, q, models={'logistic_regression'}, scope='me')
    assert [x['card'].entry_id for x in results] == ['card_003', 'card_004', 'card_005']
    assert len(r.retrieve(bundle, q, models={'svm'}, scope='me', domain='raman', k=6)) == 5
    with pytest.raises(r.RetrievalError): r.retrieve(bundle, q, models=set(), scope='me', k=7)


def test_train_template_and_explicit_user_text():
    e = SimpleNamespace(statistics=SimpleNamespace(feature_count=1800, sample_group_count=72,
        observation_count=216, repeated_measurement_group_count=72,
        classes=[SimpleNamespace(class_id=i, sample_group_count=24) for i in range(3)]))
    query = r.build_query(e)
    assert '比25' in query['text'] and '24、24、24' in query['text'] and '未知' in query['text']
    assert r.build_query(e, mode='user_text', user_text='explicit')['text'] == 'explicit'
    with pytest.raises(r.RetrievalError): r.build_query(e, mode='user_text', user_text=' ')
    with pytest.raises(r.RetrievalError): r.build_query(e, user_text='implicit')


def test_failed_build_keeps_current_publication(tmp_path):
    previous = r.publish(library(), tmp_path, FixtureEncoder())
    class Failed:
        def encode(self, texts): raise r.RetrievalError('failure')
    with pytest.raises(r.RetrievalError): r.publish(library(), tmp_path, Failed())
    assert r.current_bundle(tmp_path).manifest_digest == previous.manifest_digest


def fixture_modeling_library(version='fixture-cards'):
    from pathlib import Path
    data = json.loads((Path(__file__).parents[1]/'app/knowledge_data/modeling_cards.json').read_text(encoding='utf-8'))
    data['knowledge_set_version'] = version
    for card in data['entries']:
        card['entry_version'] = '1'
    return r.CardLibrary.model_validate_json(json.dumps(data))


def configure_fixture_rag(monkeypatch, root):
    from backend.app import knowledge
    bundle_root = root/'fixture-rag'
    r.publish(fixture_modeling_library(), bundle_root, FixtureEncoder())
    monkeypatch.setenv('AUTOAI_KNOWLEDGE_BUNDLE', str(bundle_root))
    monkeypatch.setattr(knowledge, '_rag_encoder', lambda path: FixtureEncoder())
