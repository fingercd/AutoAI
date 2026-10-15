import numpy as np
import pytest
from backend.app.benchmark_data import audit_arrays, DEVELOPMENT_TASKS, _family


def test_group_count_is_independent_sample_count():
    got = audit_arrays([[1, 2], [2, 3], [4, 5], [5, 6]], ['a', 'a', 'b', 'b'], [100, 101], ['patient1', 'patient1', 'patient2', 'patient2'])
    assert got['sample_group_count'] == 2
    assert got['class_counts'] == {'a': 1, 'b': 1}
    assert got['repeats_per_group'] == 2


@pytest.mark.parametrize('x,y,axis,groups,message', [
    ([[1, 2], [np.nan, 3]], ['a','b'], [100,101], ['p','q'], '非有限'),
    ([[1, 2], [2, 3]], ['a','b'], [100,101], ['p','p'], '多个标签'),
    ([[1, 2], [2, 3], [4,5]], ['a','a','b'], [100,101], ['p','p','q'], '次数不一致'),
    ([[1, 2], [2, 3]], ['a','b'], [101,100], ['p','q'], '不严格递增'),
])
def test_rejects_data_without_changing_it(x,y,axis,groups,message):
    with pytest.raises(ValueError,match=message): audit_arrays(x,y,axis,groups)


def test_fixed_development_ownership_and_related_versions():
    assert len(DEVELOPMENT_TASKS) == 15
    assert _family('ucr85','SonyAIBORobotSurface') == _family('ucr85','SonyAIBORobotSurfaceII')
    assert _family('raman34','diabetes_skin_vein') == _family('raman34','diabetes_skin_thumbnail')


def _fixture_assets(tmp_path, monkeypatch, *, valid_hash=True, proven=True):
    import hashlib
    import json
    import sqlite3
    from backend.app import benchmark_data as module
    root = tmp_path / 'Pan'
    raw = tmp_path / 'source.csv'
    raw.write_text('Index,Label,Sample_ID,Name,1,2\n0,a,specimen-A,raw.txt,1,2\n1,b,specimen-B,raw.txt,3,4\n', encoding='utf-8')
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    database = tmp_path / 'step78/storage/datasets.sqlite3'
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE datasets(dataset_id TEXT, path TEXT, sha256 TEXT)')
        db.execute('INSERT INTO datasets VALUES(?,?,?)', ('dataset1', str(raw), digest))
    (root / 'outputs').mkdir(parents=True)
    (root / 'outputs/benchmark-task-manifest.json').write_text(json.dumps({'tasks': [{
        'task_id': 'fixture', 'suite': 'tabmini25', 'samples': 2, 'source': {'url': 'official-collection'},
        'splits': {'main': {'dataset_id': 'dataset1', 'sha256': digest if valid_hash else 'changed'}}}]}))
    audit = root / 'work/benchmark-20261010/audit'
    audit.mkdir(parents=True)
    (audit / 'independence.json').write_text(json.dumps({'fixture': {
        'status': 'verified_sample_ids' if proven else 'archive_case_only',
        'evidence': ['Source labels specimen-A/B as separate collected specimens'] if proven else []}}))
    monkeypatch.setattr(module, 'BIOLOGY_SHAPES', {})
    return root, module


def test_manifest_preserves_unproven_exclusion_and_hashes(tmp_path, monkeypatch):
    root, module = _fixture_assets(tmp_path, monkeypatch, proven=False)
    task = module.prepare_assets(root)['tasks'][0]
    assert task['role'] == 'excluded'
    assert task['independence_status'] == 'archive_case_only'
    assert '独立单位证据' in task['exclusion_reason']
    assert module.sha256(__import__('pathlib').Path(task['main_path'])) == task['main_sha256']


def test_manifest_rejects_changed_immutable_source(tmp_path, monkeypatch):
    root, module = _fixture_assets(tmp_path, monkeypatch, valid_hash=False)
    task = module.prepare_assets(root)['tasks'][0]
    assert task['role'] == 'excluded'
    assert 'SHA256' in task['exclusion_reason']
    assert task['main_path'] is None


def test_manifest_rebuild_is_repeatable(tmp_path, monkeypatch):
    root, module = _fixture_assets(tmp_path, monkeypatch)
    first = module.prepare_assets(root)
    assert first == module.prepare_assets(root)
    assert first['tasks'][0]['role'] == 'locked'


def test_unverified_archive_rows_do_not_become_independent_sample_count():
    from backend.app.benchmark_data import _task, _apply_summary
    task = _task('unknown', 'raman34', 'spectral', {})
    _apply_summary(task, {'row_count': 1500, 'feature_count': 100,
                         'sample_group_count': 1500, 'class_counts': {'a': 750, 'b': 750},
                         'repeats_per_group': 1})
    assert task['sample_group_count'] is None
    assert task['class_counts'] == {}
    assert task['archive_group_count'] == 1500
    assert task['row_class_counts'] == {'a': 750, 'b': 750}


def test_same_acquisition_views_share_family_without_grouping_unrelated_meat_studies():
    assert _family('ucr85', 'UWaveGestureLibraryAll') == _family('ucr85', 'uWaveGestureLibrary_X')
    assert _family('ucr85', 'Meat') != _family('ucr85', 'Beef')



def _full_fixture(tmp_path):
    import json
    from backend.app import benchmark_data as module
    repo = tmp_path / 'Pan'
    old, new = repo / 'work/old', repo / 'work/full-v4'
    (old / 'data').mkdir(parents=True)
    (new / 'data/raw-arrays').mkdir(parents=True)
    return module, repo, old, new


def _full_catalog(module, old, tasks):
    module._save(old / 'data/manifest.json', {'schema_version': 'benchmark-data-v1', 'tasks': tasks})


def _full_csv_task(module, old, name='fixture', main_labels=None, test_labels=None, physical=False):
    task = module._task(name, 'ucr85' if test_labels is not None else 'tabmini25',
                        'feature_index', {'original_feature_names': ['original_feature']})
    task['independence_status'] = 'verified_sample_ids' if physical else 'archive_case_only'
    task['independence_evidence'] = ['Source physical specimens'] if physical else []
    for split, labels in (('main', main_labels or ['a'] * 3 + ['b'] * 3), ('test', test_labels)):
        if labels is None:
            continue
        path = old / 'data' / name / f'{split}.csv'
        module._wide(path, np.arange(len(labels) * 2).reshape(-1, 2), labels, [1, 2],
                     [f'{name}:{split}:{i}' for i in range(len(labels))], ['raw'] * len(labels))
        task.update({f'{split}_path': str(path), f'{split}_sha256': module.sha256(path)})
    return task


def _full_npz(new, name, labels, groups=(), axis=None):
    np.savez(new / 'data/raw-arrays' / f'{name}.npz', x=np.arange(len(labels) * 2).reshape(-1, 2),
             y=np.asarray(labels), axis=np.asarray([1, 2] if axis is None else axis), groups=np.asarray(groups))


def test_full_source_records_retain_more_than_500_and_old_evidence(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('large', 'raman34', 'spectral', {'url': 'source', 'license': 'CC BY 4.0'})
    _full_catalog(module, old, [task])
    original = (old / 'data/manifest.json').read_bytes()
    _full_npz(new, 'large', ['a'] * 301 + ['b'] * 301)
    module._save(new / 'audit/large-source-load.json', {'rows': 602, 'features': 2,
                   'class_counts': {'a': 301, 'b': 301}, 'axis_representation': 'author_tensor_channel_index'})
    first = module.prepare_full_assets(repo, new, old)
    got = first['tasks'][0]
    assert got['role'] == 'locked'
    assert got['independence_status'] == 'approved_archive_record'
    assert got['physical_independence_verified'] is False
    assert '新受试者' in got['generalization_limit']
    assert got['source_row_count'] == got['output_row_count'] == got['sample_group_count'] == 602
    assert got['all_samples_retained'] is True
    assert got['axis_kind'] == 'spectral'
    assert got['axis_unit'] == 'channel_index'
    assert got['axis_label'] == 'feature/channel index'
    assert got['source']['license'] == 'CC BY 4.0'
    assert module.prepare_full_assets(repo, new, old) == first
    assert (old / 'data/manifest.json').read_bytes() == original


def test_full_exact_csv_copy_and_hash_integrity(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = _full_csv_task(module, old)
    _full_catalog(module, old, [task])
    original = __import__('pathlib').Path(task['main_path']).read_bytes()
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'locked'
    assert __import__('pathlib').Path(got['main_path']).read_bytes() == original
    assert got['source']['original_feature_names'] == ['original_feature']
    assert got['input_files'][0]['sha256'] == got['main_sha256']
    __import__('pathlib').Path(task['main_path']).write_bytes(original + b'\n')
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'excluded'
    assert 'SHA256' in got['exclusion_reason']
    assert got['main_path'] is None


def test_full_official_train_singleton_excluded_without_relaxing_valid(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = _full_csv_task(module, old, main_labels=['a', 'a', 'b'], test_labels=['a'] * 3 + ['b'] * 3)
    _full_catalog(module, old, [task])
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'excluded'
    assert '官方Train有单例类别' in got['exclusion_reason']
    assert got['split_coverage_evidence']['insufficient_classes'] == {'b': 1}
    assert got['source_row_count'] == got['output_row_count'] == 9
    assert got['all_samples_retained'] is True


def test_full_true_groups_are_preserved_and_variable_repeats_not_coerced(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('patients', 'raman34', 'spectral', {})
    _full_catalog(module, old, [task])
    labels = ['a'] * 6 + ['b'] * 6
    groups = np.repeat(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'], 2)
    _full_npz(new, 'patients', labels, groups)
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'locked'
    assert got['split_unit'] == 'source_physical_group'
    assert got['sample_group_count'] == 6
    assert got['class_counts'] == {'a': 3, 'b': 3}
    parsed = module.load_modeling_csv(got['main_path'])
    assert parsed.sample_id == list(groups)
    _full_npz(new, 'patients', labels + ['b'], list(groups) + ['p6'])
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'excluded'
    assert '次数不一致' in got['exclusion_reason']
    assert got['source_row_count'] == 13
    assert got['source_repeat_count_histogram'] == {2: 5, 3: 1}


def test_full_missing_source_is_pending_and_next_run_admits_new_source(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('arriving', 'raman34', 'spectral', {})
    task['exclusion_reason'] = 'old HTTP 403'
    _full_catalog(module, old, [task])
    waiting = module.prepare_full_assets(repo, new, old)
    assert waiting['status'] == 'awaiting_sources'
    assert waiting['tasks'][0]['acquisition_status'] == 'awaiting_source'
    _full_npz(new, 'arriving', ['a'] * 3 + ['b'] * 3)
    ready = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert ready['role'] == 'locked'
    assert ready['acquisition_status'] == 'ready'
    assert ready['previous_exclusion_reason'] == 'old HTTP 403'


def test_full_whole_data_singleton_and_duplicate_tasks_have_explicit_evidence(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    singleton = _full_csv_task(module, old, 'singleton', ['a'] * 4 + ['b'])
    locked = _full_csv_task(module, old, 'locked-copy')
    development = _full_csv_task(module, old, 'Beef')
    _full_catalog(module, old, [singleton, locked, development])
    tasks = {t['task_id']: t for t in module.prepare_full_assets(repo, new, old)['tasks']}
    assert tasks['singleton']['role'] == 'excluded'
    assert tasks['singleton']['split_coverage_evidence']['insufficient_classes'] == {'b': 1}
    assert tasks['Beef']['role'] == 'development'
    assert tasks['locked-copy']['duplicate_of'] == 'Beef'
    assert tasks['locked-copy']['role'] == 'excluded'


def test_full_disallows_overwriting_old_root(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    with pytest.raises(ValueError, match='禁止覆盖'):
        module.prepare_full_assets(repo, old, old)



def test_full_missing_new_groups_cannot_override_known_physical_repeats(tmp_path):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('known', 'raman34', 'spectral', {})
    task.update(independence_status='verified_sample_ids', sample_group_count=6,
                independence_evidence=['Original source has six physical specimens with two repeats each'])
    _full_catalog(module, old, [task])
    _full_npz(new, 'known', ['a'] * 6 + ['b'] * 6)
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'excluded'
    assert '不能按记录拆开已知物理组' in got['exclusion_reason']
    assert got['source_row_count'] == 12



@pytest.mark.parametrize('source_kind', ['csv', 'arrays'])
def test_full_reuses_unchanged_prepared_task_without_parsing_writing_or_semantic_hash(tmp_path, monkeypatch, source_kind):
    module, repo, old, new = _full_fixture(tmp_path)
    if source_kind == 'csv':
        task = _full_csv_task(module, old, test_labels=['a'] * 3 + ['b'] * 3)
    else:
        task = module._task('fixture', 'raman34', 'spectral', {})
        _full_npz(new, 'fixture', ['a'] * 3 + ['b'] * 3)
    _full_catalog(module, old, [task])
    first = module.prepare_full_assets(repo, new, old)

    def forbidden(*args, **kwargs):
        raise AssertionError('unchanged verified task should use its previous audit and semantic digest')

    monkeypatch.setattr(module, 'load_modeling_csv', forbidden)
    monkeypatch.setattr(module.np, 'load', forbidden)
    monkeypatch.setattr(module, '_full_write', forbidden)
    monkeypatch.setattr(module, '_full_digest', forbidden)
    assert module.prepare_full_assets(repo, new, old) == first


@pytest.mark.parametrize('changed', ['source', 'receipt', 'output'])
def test_full_cache_invalidates_source_receipt_or_output_changes(tmp_path, monkeypatch, changed):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('fixture', 'raman34', 'spectral', {})
    _full_catalog(module, old, [task])
    _full_npz(new, 'fixture', ['a'] * 3 + ['b'] * 3)
    receipt = new / 'audit/fixture-source-load.json'
    module._save(receipt, {'axis_representation': 'author_spectral_coordinates'})
    first = module.prepare_full_assets(repo, new, old)['tasks'][0]
    if changed == 'source':
        _full_npz(new, 'fixture', ['a'] * 4 + ['b'] * 4)
    elif changed == 'receipt':
        module._save(receipt, {'axis_representation': 'author_tensor_channel_index'})
    else:
        output = __import__('pathlib').Path(first['main_path'])
        output.write_bytes(output.read_bytes() + b'\n')
    calls = []
    original = module._full_write

    def write(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, '_full_write', write)
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert len(calls) == 1
    assert got['role'] == 'locked'
    if changed == 'source':
        assert got['source_row_count'] == 8
        assert got['input_files'] != first['input_files']
    elif changed == 'receipt':
        assert got['axis_unit'] == 'channel_index'
        assert got['acquisition_receipt'] != first['acquisition_receipt']
    else:
        assert got['main_sha256'] == first['main_sha256']


@pytest.mark.parametrize('prior_source', ['csv', 'legacy_array'])
def test_full_fresh_array_arrival_invalidates_old_source_cache(tmp_path, monkeypatch, prior_source):
    module, repo, old, new = _full_fixture(tmp_path)
    if prior_source == 'csv':
        task = _full_csv_task(module, old)
        task.update(suite='raman34', axis_kind='spectral')
    else:
        task = module._task('fixture', 'raman34', 'spectral', {})
        legacy = old / 'data/raw/raman-arrays/fixture.npz'
        legacy.parent.mkdir(parents=True)
        np.savez(legacy, x=np.arange(12).reshape(6, 2), y=['a'] * 3 + ['b'] * 3,
                 axis=[1, 2], groups=[])
    _full_catalog(module, old, [task])
    first = module.prepare_full_assets(repo, new, old)['tasks'][0]
    _full_npz(new, 'fixture', ['a'] * 3 + ['b'] * 3)
    calls = []
    original = module._full_write

    def write(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, '_full_write', write)
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert len(calls) == 1
    assert got['input_files'][0]['path'] == str(new / 'data/raw-arrays/fixture.npz')
    assert got['input_files'] != first['input_files']


def test_full_reuses_audit_but_derives_missing_semantic_fingerprint_once(tmp_path, monkeypatch):
    module, repo, old, new = _full_fixture(tmp_path)
    task = _full_csv_task(module, old)
    _full_catalog(module, old, [task])
    first = module.prepare_full_assets(repo, new, old)
    first['tasks'][0].pop('content_sha256')
    module._save(new / 'data/manifest.json', first)
    calls = []
    original = module._full_digest

    def digest(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('verified unchanged audit does not need full CSV parsing')

    monkeypatch.setattr(module, 'load_modeling_csv', forbidden)
    monkeypatch.setattr(module, '_full_digest', digest)
    second = module.prepare_full_assets(repo, new, old)
    assert len(calls) == 1
    assert second['tasks'][0]['content_sha256'][0]
    assert module.prepare_full_assets(repo, new, old) == second
    assert len(calls) == 1



def _full_decisions(module, new, tasks, **overrides):
    document = {'user_authorized': True, 'authorization_quote': '不是所有的任务都要用，我觉得不合理的任务和数据可以删掉',
                'tasks': tasks, **overrides}
    module._save(new / 'audit/task-decisions.json', document)
    return document


def test_full_explicit_decision_excludes_verified_ready_task_without_reparsing_and_preserves_evidence(tmp_path, monkeypatch):
    module, repo, old, new = _full_fixture(tmp_path)
    excluded = module._task('chosen', 'raman34', 'spectral', {'url': 'official-source'})
    unaffected = _full_csv_task(module, old, 'other', main_labels=['c'] * 3 + ['d'] * 3)
    _full_catalog(module, old, [excluded, unaffected])
    _full_npz(new, 'chosen', ['a'] * 3 + ['b'] * 3)
    module._save(new / 'audit/chosen-source-load.json', {'rows': 6, 'features': 2,
                  'class_counts': {'a': 3, 'b': 3}, 'axis_representation': 'author_spectral_coordinates'})
    first = module.prepare_full_assets(repo, new, old)
    previous = first['tasks'][0]
    raw = new / 'data/raw-arrays/chosen.npz'
    raw_bytes = raw.read_bytes()
    _full_decisions(module, new, {'chosen': {'include': False, 'reason': 'User-reviewed unsuitable source lineage',
                     'reason_code': 'unverified_original_measurement_lineage', 'evidence': ['author dataset documentation']}})

    def forbidden(*args, **kwargs):
        raise AssertionError('explicit exclusion and unchanged other task must not parse or convert')

    monkeypatch.setattr(module, 'load_modeling_csv', forbidden)
    monkeypatch.setattr(module.np, 'load', forbidden)
    monkeypatch.setattr(module, '_full_write', forbidden)
    got = module.prepare_full_assets(repo, new, old)
    chosen, other = got['tasks']
    assert got['candidate_count'] == 2
    assert chosen['role'] == 'excluded'
    assert chosen['acquisition_status'] == 'scientifically_excluded'
    assert chosen['source_row_count'] == 6
    for key in ('input_files', 'acquisition_receipt', 'source', 'main_path', 'main_sha256', 'main_audit'):
        assert chosen[key] == previous[key]
    assert chosen['decision_evidence']['reason_code'] == 'unverified_original_measurement_lineage'
    assert chosen['decision_evidence']['decision_receipt']['sha256'] == module.sha256(new / 'audit/task-decisions.json')
    assert other == first['tasks'][1]
    assert raw.read_bytes() == raw_bytes
    assert module.prepare_full_assets(repo, new, old) == got


def test_full_explicit_decision_preserves_unconverted_source_receipt_counts_and_raw_paths(tmp_path, monkeypatch):
    module, repo, old, new = _full_fixture(tmp_path)
    task = module._task('large-source', 'raman34', 'spectral', {'url': 'source'})
    _full_catalog(module, old, [task])
    _full_npz(new, 'large-source', ['a'] * 3 + ['b'] * 3)
    receipt = {'rows': 6, 'features': 2, 'class_counts': {'a': 3, 'b': 3}, 'explicit_group_count': 4}
    module._save(new / 'audit/large-source-source-load.json', receipt)
    _full_decisions(module, new, {'large-source': {'include': False, 'reason': 'Outside human-approved primary scope',
                                               'reason_code': 'outside_primary_scope', 'evidence': []}})

    def forbidden(*args, **kwargs):
        raise AssertionError('explicit source exclusion must not load/converge the matrix')

    monkeypatch.setattr(module.np, 'load', forbidden)
    monkeypatch.setattr(module, '_full_write', forbidden)
    got = module.prepare_full_assets(repo, new, old)['tasks'][0]
    assert got['role'] == 'excluded'
    assert got['source_row_count'] == 6
    assert got['source_feature_count'] == 2
    assert got['source_row_class_counts'] == {'a': 3, 'b': 3}
    assert got['source_sample_group_count'] == 4
    assert got['acquisition'] == receipt
    assert got['source_count_evidence'] == got['acquisition_receipt']
    assert got['input_files'][0]['path'] == str(new / 'data/raw-arrays/large-source.npz')
    assert got['source']['raw_array_path'] == got['input_files'][0]['path']
    assert got['main_path'] is None
    assert not (new / 'data/raman34/large-source/main.csv').exists()


@pytest.mark.parametrize('document', [
    {'user_authorized': False, 'authorization_quote': 'human quote', 'tasks': {}},
    {'user_authorized': True, 'tasks': {}},
    {'user_authorized': True, 'authorization_quote': 'human quote', 'tasks': []},
    {'user_authorized': True, 'authorization_quote': 'human quote', 'tasks': {'unknown': {'include': False, 'reason': 'reason'}}},
    {'user_authorized': True, 'authorization_quote': 'human quote', 'tasks': {'fixture': {'include': False, 'reason': '  '}}},
    {'user_authorized': True, 'authorization_quote': 'human quote', 'tasks': {'fixture': {'include': True, 'reason': 'reason'}}},
    {'user_authorized': True, 'authorization_quote': 'human quote', 'tasks': {'fixture': {'include': False, 'reason': 'reason', 'evidence': 'not a list'}}},
])
def test_full_rejects_unauthorized_unknown_or_malformed_task_decisions(tmp_path, document):
    module, repo, old, new = _full_fixture(tmp_path)
    task = _full_csv_task(module, old)
    _full_catalog(module, old, [task])
    module._save(new / 'audit/task-decisions.json', document)
    with pytest.raises(ValueError, match='任务排除决定'):
        module.prepare_full_assets(repo, new, old)
    assert not (new / 'data/manifest.json').exists()
