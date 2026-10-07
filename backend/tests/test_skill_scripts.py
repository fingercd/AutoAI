"""Data correspondence, uncertainty and recovery tests for the distributable scripts."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest
from openpyxl import Workbook


SCRIPTS = Path(__file__).resolve().parents[2] / 'skills' / 'autoai-research' / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script('prepare_dataset')
client_module = load_script('autoai_client')


def test_xlsx_transpose_multiheader_and_sort_preserve_correspondence(tmp_path):
    source = tmp_path / 'source.xlsx'
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'spectra'
    # Transposed sample columns, two header rows once transposed.
    for row in [['说明', '批次', 'a', 'b'], ['元信息', 'Label', '01', '02'],
                ['元信息', 'Sample_ID', '001', '002'], ['元信息', 'Name', 'a.txt', 'b.txt'],
                ['强度', '2.5', '11.123456789', '21.123456789'], ['强度', '1.5', '12', '22']]:
        sheet.append(row)
    workbook.save(source)
    mapping = {'sort_coordinates': True, 'tables': [{'source': str(source), 'sheet': 'spectra',
               'transpose': True, 'header_row': 2, 'data_start_row': 3,
               'columns': {'label': 2, 'sample_id': 3, 'name': 4}, 'feature_start_column': 5}]}
    original = source.read_bytes()
    result = prepare.convert(mapping, tmp_path / 'converted')
    with Path(result['output']).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ['Index', 'Label', 'Sample_ID', 'Name', '1.5', '2.5']
    assert rows[1] == ['1', '01', '001', 'a.txt', '12', '11.123456789']
    assert rows[2] == ['2', '02', '002', 'b.txt', '22', '21.123456789']
    assert source.read_bytes() == original
    assert result['summary']['sample_count'] == 2
    from backend.app.parsers import load_modeling_csv
    assert load_modeling_csv(result['output']).labels == ['01', '02']


def test_name_mapping_after_preprocessing_and_same_axis_merge(tmp_path):
    source = tmp_path / 'preprocessed.csv'
    source.write_text('Index,Label,Sample_ID,Name,1,2\n1,,,a.txt,4,5\n2,,,b.txt,6,7\n', encoding='utf-8')
    mapping = {'tables': [{'source': str(source), 'columns': {'name': 4}, 'feature_start_column': 5}],
               'metadata_by_name': {'a.txt': {'Label': '甲', 'Sample_ID': '001'}, 'b.txt': {'Label': '乙', 'Sample_ID': '002'}}}
    output = prepare.convert(mapping, tmp_path / 'output')['output']
    assert '001' in Path(output).read_text(encoding='utf-8-sig')
    bad = tmp_path / 'bad.csv'
    bad.write_text('Index,Label,Sample_ID,Name,1,3\n1,甲,003,c.txt,4,5\n', encoding='utf-8')
    mapping['tables'].append({'source': str(bad), 'columns': {'label': 2, 'sample_id': 3, 'name': 4}, 'feature_start_column': 5})
    with pytest.raises(ValueError, match='同轴'):
        prepare.convert(mapping, tmp_path / 'bad-output')
    assert not (tmp_path / 'bad-output').exists()


@pytest.mark.parametrize('data,match', [
    ('1,甲,,a,4,5\n', '不得为空'),
    ('1,甲,001,a,,5\n', '不是数值'),
    ('1,甲,001,a,NaN,5\n', '有限'),
    ('1,甲,001,a,4,5\n2,乙,001,b,6,7\n', '多个 Label'),
    ('1,甲,001,a,4,5\n2,甲,001,b,6,7\n3,乙,002,c,8,9\n', '次数不一致'),
])
def test_invalid_measurements_are_rejected_without_writes(tmp_path, data, match):
    source = tmp_path / 'data.csv'
    source.write_text('Index,Label,Sample_ID,Name,1,2\n' + data, encoding='utf-8')
    mapping = {'tables': [{'source': str(source), 'columns': {'index': 1, 'label': 2, 'sample_id': 3, 'name': 4}, 'feature_start_column': 5}]}
    with pytest.raises(ValueError, match=match):
        prepare.convert(mapping, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_only_explicit_independence_allows_generated_sample_ids(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_text('类别,1,2\n甲,4,5\n乙,6,7\n', encoding='utf-8')
    mapping = {'independent_rows_confirmed': True, 'tables': [{'source': str(source), 'columns': {'label': 1}, 'feature_start_column': 2}]}
    assert prepare.convert(mapping, tmp_path / 'output')['summary']['sample_count'] == 2


def test_formula_is_data_not_code_and_requires_values(tmp_path):
    source = tmp_path / 'formula.xlsx'
    workbook = Workbook()
    workbook.active.append(['=1+2'])
    workbook.save(source)
    with pytest.raises(ValueError, match='公式'):
        prepare.inspect(source)


def test_submit_timeout_restores_saved_key_and_payload(tmp_path, monkeypatch):
    task = tmp_path / 'experiment'
    seen = []
    def handler(request):
        seen.append((request.url.path, request.headers.get('Idempotency-Key'), json.loads(request.content)))
        if request.url.path.endswith('preflight'):
            return httpx.Response(200, json={'runnable': True, 'worker': {'available': True, 'compatible': True},
                                             'submit_payload': {'dataset_id': 'd', 'config': {'model_type': 'svm'}, 'strict_config': True}})
        if len([item for item in seen if item[0].endswith('/runs')]) < 4:
            raise httpx.ReadTimeout('response lost', request=request)
        return httpx.Response(202, json={'run_id': 'r', 'state': 'queued'})
    monkeypatch.setattr(client_module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(client_module.AutoAIClient, 'sync_cloud', lambda self, **kwargs: {'status': 'fixture'})
    client = client_module.AutoAIClient('https://autoai.example', task, transport=httpx.MockTransport(handler))
    client.state['plan'] = {'dataset_id': 'd', 'config': {'model_type': 'svm'}}
    with pytest.raises(httpx.ReadTimeout):
        client.submit()
    restored = client_module.AutoAIClient('https://autoai.example', task, transport=httpx.MockTransport(handler))
    assert restored.submit()['run_id'] == 'r'
    calls = [item for item in seen if item[0].endswith('/runs')]
    assert len({item[1] for item in calls}) == 1 and all(item[2] == calls[0][2] for item in calls)
    assert len([item for item in seen if item[0].endswith('preflight')]) == 1


def test_upload_timeout_is_not_blindly_retried_and_credentials_stay_out_of_state(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOAI_SERVICE_TOKEN', 'test-secret-only')
    source = tmp_path / 'data.csv'
    source.write_text('source', encoding='utf-8')
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers['Authorization'] == 'Bearer test-secret-only'
        raise httpx.ReadTimeout('lost', request=request)
    client = client_module.AutoAIClient('https://autoai.example', tmp_path / 'task', transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.upload(source, 'primary')
    with pytest.raises(ValueError, match='未知'):
        client.upload(source, 'primary')
    assert len(calls) == 1
    assert 'test-secret-only' not in (tmp_path / 'task' / 'state.json').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='其他服务'):
        client.request('GET', 'https://other.example/artifact')


def test_cloud_failure_preserves_journal_then_verifies_replayed_metrics(tmp_path, monkeypatch):
    import types
    client = client_module.AutoAIClient('https://autoai.example', tmp_path / 'task')
    client.state['cloud_id'] = 'fake-id'
    client.log({'test/accuracy': 0.75})
    fake = types.SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('failure')))
    monkeypatch.setitem(sys.modules, 'swanlab', fake)
    assert client.sync_cloud()['cloud_status'] == 'pending_upload_or_verification'
    logged = []
    uploaded_config = {}
    def init(**kwargs):
        uploaded_config.update(kwargs['config'])
        return types.SimpleNamespace(url='https://swanlab.cn/@fixture/AutoAI-Skill/runs/fake-id')
    fake.init = init
    fake.log = lambda metrics, step: logged.append((metrics, step))
    fake.finish = lambda: None
    fake.Api = lambda: types.SimpleNamespace(run=lambda path: types.SimpleNamespace(
        summary=lambda keys: {'test/accuracy': {'value': 0.75}},
        profile={'config': {key: {'value': value} for key, value in uploaded_config.items()}}))
    assert client.sync_cloud()['cloud_status'] == 'verified'
    assert logged == [({'test/accuracy': 0.75}, 0)]
    client.sync_cloud()
    assert len(logged) == 1
    fake.Api = lambda: types.SimpleNamespace(run=lambda path: types.SimpleNamespace(
        summary=lambda keys: {'test/accuracy': {'value': 0.75}}, profile={'config': {}}))
    assert client.sync_cloud()['cloud_status'] == 'uploaded_unverified'
