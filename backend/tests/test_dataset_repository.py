from backend.app.datasets.repository import DatasetRepository
from backend.app.runs.contracts import Principal
import pytest


def test_dataset_id_resolves_only_for_its_server_derived_principal(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text('Index,Name,XXX,Intensity,Label,Sample_ID\n', encoding='utf-8')
    repo = DatasetRepository(tmp_path / 'datasets.sqlite3', storage_root=tmp_path)
    repo.initialize()
    dataset = repo.register(source, original_name='source.csv', principal=Principal(owner_id='owner-a'))

    assert repo.resolve(dataset.dataset_id, principal=Principal(owner_id='owner-a')).path == source.resolve()
    with pytest.raises(PermissionError):
        repo.resolve(dataset.dataset_id, principal=Principal(owner_id=None))


def test_local_principal_resolves_a_local_dataset(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text('Index,Name,XXX,Intensity,Label,Sample_ID\n', encoding='utf-8')
    repo = DatasetRepository(tmp_path / 'datasets.sqlite3', storage_root=tmp_path)
    repo.initialize()
    dataset = repo.register(source, original_name='source.csv', principal=Principal())
    assert repo.resolve(dataset.dataset_id, principal=Principal()).path == source.resolve()
