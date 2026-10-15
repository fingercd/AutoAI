"""Prepare immutable benchmark inputs under separately versioned admission rules.

This module does no fitting, imputation, row subsampling, or spectral resampling.
The network acquisition receipts and source evidence live outside Git in work/.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import csv
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .parsers import load_modeling_csv

RAMAN_REVISION = 'c92289ec9a786d2dde4c35f7e14c6701a10fcc68'
BIOLOGY_REVISION = '48cffad4e88ff4b9d2f1c7baffb314d1b3303792'
DEVELOPMENT_TASKS = frozenset(('Beef', 'BeetleFly', 'BirdChicken', 'Coffee', 'ECGFiveDays',
    'Meat', 'OliveOil', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Wine',
    'analcatdata_aids', 'appendicitis', 'parity5', 'prnn_crabs', 'glass2'))
BIOLOGY_SHAPES = {'ALLAML': (72, 7129), 'colon': (62, 2000), 'GLIOMA': (50, 4434),
    'CLL_SUB_111': (111, 11340), 'Carcinom': (174, 9182)}
SUPPORTED_INDEPENDENCE = frozenset(('verified_sample_ids', 'justified_independent_cases'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def audit_arrays(x, y, axis, groups) -> dict:
    """Validate the existing wide-table contract without altering observations."""
    x, y, axis, groups = np.asarray(x), np.asarray(y), np.asarray(axis), np.asarray(groups)
    if x.ndim != 2 or y.ndim != 1 or groups.ndim != 1 or axis.ndim != 1:
        raise ValueError('矩阵、默认目标、公共轴、独立组的维度不符合宽表合同')
    if len(x) != len(y) or len(x) != len(groups) or x.shape[1] != len(axis) or not len(x):
        raise ValueError('行数或公共轴长度不一致，或数据为空')
    if not np.isfinite(x.astype(float)).all() or not np.isfinite(axis.astype(float)).all():
        raise ValueError('包含非有限强度或坐标，不能删行或补值后入组')
    if not np.all(np.diff(axis.astype(float)) > 0):
        raise ValueError('原始公共轴不严格递增，不能重采样后入组')
    frame = pd.DataFrame({'Label': y, 'Sample_ID': groups})
    if frame.isna().any().any() or frame.astype(str).apply(lambda s: s.str.strip().eq('')).any().any():
        raise ValueError('默认目标或独立组缺失')
    grouped = frame.groupby('Sample_ID', sort=False)
    if (grouped.Label.nunique() != 1).any():
        raise ValueError('同一个独立样品出现多个标签')
    repeats = grouped.size()
    if repeats.nunique() != 1:
        raise ValueError('独立样品重复测量次数不一致；不能删行凑齐')
    units = grouped.Label.first()
    return {'sample_group_count': len(units), 'class_counts': {str(k): int(v) for k, v in units.value_counts().sort_index().items()},
            'feature_count': x.shape[1], 'row_count': len(x), 'repeats_per_group': int(repeats.iloc[0])}


def _wide(path: Path, x, y, axis, groups, names) -> dict:
    summary = audit_arrays(x, y, axis, groups)
    frame = pd.DataFrame(x, columns=[format(float(a), '.15g') for a in axis])
    frame.insert(0, 'Name', names)
    frame.insert(0, 'Sample_ID', groups)
    frame.insert(0, 'Label', y)
    frame.insert(0, 'Index', np.arange(len(x)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    load_modeling_csv(path)
    return summary


def _task(task_id, suite, axis_kind, source, family_id=None) -> dict:
    return {'task_id': task_id, 'suite': suite, 'family_id': family_id or f'{suite}:{task_id}',
            'axis_kind': axis_kind, 'role': 'excluded', 'requested_role': 'development' if task_id in DEVELOPMENT_TASKS else 'locked',
            'main_path': None, 'test_path': None, 'main_sha256': None, 'test_sha256': None,
            'independence_status': 'unknown', 'independence_evidence': [], 'sample_group_count': None,
            'class_counts': {}, 'feature_count': None, 'source': {'license': 'unknown', **source}, 'exclusion_reason': None}


def _apply_summary(task, summary):
    if task['independence_status'] in SUPPORTED_INDEPENDENCE:
        task.update(summary)
    else:
        # Unverified archive row IDs describe file structure, not physical samples.
        task.update(row_count=summary['row_count'], feature_count=summary['feature_count'],
                    archive_group_count=summary['sample_group_count'],
                    row_class_counts=summary['class_counts'])


def _admit(task):
    if task['independence_status'] not in SUPPORTED_INDEPENDENCE or not task['independence_evidence']:
        raise ValueError('没有独立单位证据；归档行号不能代替独立样品标识')
    task['role'] = task['requested_role']
    task['exclusion_reason'] = None


def _family(suite, name):
    if suite == 'ucr85':
        if name.lower().startswith('uwavegesturelibrary'): return 'ucr85:uWaveGestureLibrary'
        for prefix in ('Cricket', 'uWaveGestureLibrary', 'NonInvasiveFetalECGThorax', 'NonInvasiveFatalECG_Thorax', 'SonyAIBORobotSurface', 'UWaveGestureLibrary'):
            if name.startswith(prefix): return f'ucr85:{prefix}'
    if suite == 'raman34':
        if name.startswith('diabetes_skin_'): return 'raman34:diabetes_skin'
        if name in ('alzheimer', 'parkinson'): return 'raman34:saliva_neurodegeneration'
        if name.startswith('cancer_cell_'): return 'raman34:cancer_cell_media'
        if name.startswith('rruff_mineral_'): return 'raman34:rruff_mineral'
        if name.startswith('organic_compounds_'): return 'raman34:organic_compounds'
        if name.startswith('marine_pathogens'): return 'raman34:marine_pathogens'
        if name.startswith('illicit_adulterants'): return 'raman34:illicit_adulterants'
    return f'{suite}:{name}'


def _semantic_digest(path: Path) -> str:
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))
    headers = rows[0]
    offset = 4 if headers[3] == 'Name' else 3
    groups = {}
    for row in rows[1:]:
        # Label is part of the observation, while Index/Name/group-name are identifiers.
        groups.setdefault(row[2], []).append((row[1], tuple(row[offset:])))
    payload = (headers[offset:], sorted(sorted(values) for values in groups.values()))
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def prepare_assets(root: Path) -> dict:
    """Rebuild the 149-candidate terminal manifest from acquired source evidence.

    root is the Pan repository root. This is repeatable and downloads no data;
    acquisition is performed separately so running on node3 remains offline.
    Every candidate receives an explicit included/excluded result.
    """
    root = Path(root).resolve()
    base = root / 'work/benchmark-20261010'
    dest, audit = base / 'data', base / 'audit'
    dest.mkdir(parents=True, exist_ok=True)
    evidence = _json(audit / 'independence.json', {})
    tasks = []
    catalog = _json(root / 'outputs/benchmark-task-manifest.json')['tasks']
    database = root.parent / 'step78/storage/datasets.sqlite3'
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        for previous in catalog:
            name, suite = previous['task_id'], previous['suite']
            task = _task(name, suite, 'time_series' if suite == 'ucr85' else 'feature_index', previous['source'], _family(suite, name))
            task['legacy_catalog_sample_count'] = previous['samples']
            proof = evidence.get(name, {})
            task['independence_status'] = proof.get('status', 'archive_case_only')
            task['independence_evidence'] = proof.get('evidence', ['原catalog仅给每归档行稳定ID，未证明独立采集单位'])
            try:
                for split, ref in previous['splits'].items():
                    row = db.execute('select path,sha256 from datasets where dataset_id=?', (ref['dataset_id'],)).fetchone()
                    if row is None: raise ValueError(f'登记数据不存在: {ref["dataset_id"]}')
                    original = Path(row[0]); original = original if original.is_absolute() else database.parent / original
                    if row[1] != ref['sha256'] or sha256(original) != ref['sha256']:
                        raise ValueError(f'{split}文件SHA256与冻结登记不一致')
                    output = dest / suite / name / f'{split}.csv'
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, output)
                    mapping = proof.get('group_map', {})
                    if mapping:
                        # Change metadata only: retain the exact original feature and label text.
                        with output.open(encoding='utf-8', newline='') as stream:
                            rows = list(csv.reader(stream))
                        group_column = rows[0].index('Sample_ID')
                        if any(row[group_column] not in mapping for row in rows[1:]):
                            raise ValueError(f'{split}原始样品映射覆盖不完整')
                        for row in rows[1:]: row[group_column] = mapping[row[group_column]]
                        with output.open('w', encoding='utf-8', newline='') as stream:
                            csv.writer(stream).writerows(rows)
                    parsed = load_modeling_csv(output)
                    metadata = parsed.frame
                    summary = audit_arrays(parsed.intensity, metadata.Label.to_numpy(), parsed.x_axis[0], metadata.Sample_ID.to_numpy())
                    task[f'{split}_path'] = str(output)
                    task[f'{split}_sha256'] = sha256(output)
                    task[f'{split}_audit'] = summary
                    if split == 'main': _apply_summary(task, summary)
                if task['test_path']:
                    a, b = load_modeling_csv(task['main_path']), load_modeling_csv(task['test_path'])
                    if a.x_axis[0] != b.x_axis[0]: raise ValueError('主数据和独立测试公共轴不一致')
                    if set(a.sample_id) & set(b.sample_id): raise ValueError('主数据和独立测试包含相同物理样品；原划分不能用')
                _admit(task)
            except (ValueError, OSError) as ex:
                task['exclusion_reason'] = str(ex)
            tasks.append(task)
            _save(dest / 'manifest.json', {'schema_version': 'benchmark-data-v1', 'status': 'preparing', 'tasks': tasks})
    for name, shape in BIOLOGY_SHAPES.items():
        path = dest / 'raw/biology' / f'{name}.mat'
        source = {'url': 'https://jundongl.github.io/scikit-feature/datasets.html', 'revision': BIOLOGY_REVISION,
                  'license': 'unknown', 'license_evidence': 'ASU source page and repository code license retained; individual dataset reuse terms not established', 'raw_path': str(path)}
        task = _task(name, 'biology5', 'feature_index', source)
        proof = evidence.get(name, {})
        task['independence_status'] = proof.get('status', 'unknown')
        task['independence_evidence'] = proof.get('evidence', [])
        try:
            source['sha256'] = sha256(path)
            mat = loadmat(path)
            x, y = np.asarray(mat['X']), np.asarray(mat['Y']).reshape(-1)
            if x.shape != shape: raise ValueError(f'ASU原矩阵形状{x.shape}与批准{shape}不符')
            output = dest / 'biology5' / name / 'main.csv'
            _apply_summary(task, _wide(output, x, y, np.arange(1, shape[1]+1), [f'{name}:sample:{i}' for i in range(len(x))], [path.name]*len(x)))
            task.update(main_path=str(output), main_sha256=sha256(output))
            _admit(task)
        except (ValueError, OSError, KeyError) as ex: task['exclusion_reason'] = str(ex)
        tasks.append(task)
    raman_catalog = _json(audit / 'raman-catalog.json', {})
    for name, info in raman_catalog.items():
        source = {**info['metadata'], 'license': info.get('license', 'unknown'), 'raman_data_revision': RAMAN_REVISION,
                  'loader_source': info['loader_source']}
        task = _task(name, 'raman34', 'spectral', source, _family('raman34', name))
        receipt = _json(audit / 'raman-results' / f'{name}.json', {})
        task['independence_status'] = receipt.get('independence_status', 'unknown')
        task['independence_evidence'] = receipt.get('independence_evidence', [])
        task['acquisition'] = receipt
        task['sample_group_count'] = receipt.get('sample_group_count')
        task['feature_count'] = receipt.get('feature_count')
        try:
            if receipt.get('error'): raise ValueError(receipt['error'])
            arrays = dest / 'raw/raman-arrays' / f'{name}.npz'
            if not arrays.exists(): raise ValueError('源数据下载/核验尚未完成，不能纳入主评测')
            source['array_sha256'] = sha256(arrays)
            with np.load(arrays, allow_pickle=False) as d:
                x, y, axis, groups, names = (d[k] for k in ('x', 'y', 'axis', 'groups', 'names'))
                _apply_summary(task, audit_arrays(x, y, axis, groups))
                _admit(task)
                if task['sample_group_count'] > 500: raise ValueError('实际独立样品数超过500，不允许从大数据临时抽500')
                if receipt.get('axis_kind') != 'spectral': raise ValueError('源文件未提供可靠的真实Raman位移公共轴')
                _admit(task)
                output = dest / 'raman34' / name / 'main.csv'
                _wide(output, x, y, axis, groups, names)
                task.update(main_path=str(output), main_sha256=sha256(output))
        except (ValueError, OSError, KeyError) as ex:
            task['role'] = 'excluded'; task['exclusion_reason'] = str(ex)
        tasks.append(task)
    # Versions sharing a family inherit development ownership; excluded entries stay excluded.
    development_families = {t['family_id'] for t in tasks if t['requested_role'] == 'development'}
    for task in tasks:
        if task['role'] != 'excluded' and task['family_id'] in development_families: task['role'] = 'development'
    fingerprints = {}
    for task in tasks:
        if task['role'] == 'excluded': continue
        digest = tuple(_semantic_digest(Path(task[k])) if task[k] else None for k in ('main_path','test_path'))
        task['content_sha256'] = list(digest)
        if digest in fingerprints:
            task['role'] = 'excluded'; task['exclusion_reason'] = f'与{fingerprints[digest]}完全重复的任务版本'
        else: fingerprints[digest] = task['task_id']
    result = {'schema_version': 'benchmark-data-v1', 'status': 'complete' if len(tasks) == 149 else 'incomplete_catalog',
              'candidate_count': len(tasks), 'tasks': tasks, 'summary': dict(Counter(t['role'] for t in tasks)),
              'legacy_tabmini19': _json(audit / 'legacy-tabmini19.json', {'status': 'not_in_approved_revision', 'reason': '旧额外19不属于固定新版TabMini25，本轮不纳入'})}
    _save(dest / 'manifest.json', result)
    _save(audit / 'summary.json', {k: v for k, v in result.items() if k != 'tasks'})
    return result


ARCHIVE_RECORD_STATUS = 'approved_archive_record'
ARCHIVE_RECORD_LIMIT = ('划分单位为来源记录；真实受试者、标本或采集批次标识缺失，'
                        '不能据此声称对新受试者、新标本或新批次的泛化。')


def _full_digest(path: Path) -> str:
    """Hash observations and group structure with memory proportional to row count."""
    groups = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        offset = 4 if header[3] == 'Name' else 3
        axis = [float(v).hex() for v in header[offset:]]
        for row in reader:
            payload = (row[1], [float(v).hex() for v in row[offset:]])
            digest = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
            groups.setdefault(row[2], []).append(digest)
    grouped = [hashlib.sha256(''.join(sorted(rows)).encode()).hexdigest() for rows in groups.values()]
    return hashlib.sha256(json.dumps((axis, sorted(grouped)), separators=(',', ':')).encode()).hexdigest()


def _full_write(path, x, y, axis, groups, names):
    # The shared header validator is also used by the public modeling CSV reader.
    # Do not instantiate its per-row axis lists for large, complete source arrays.
    from .parsers import _parse_wide_axis_headers
    headers = [format(float(v), '.17g') for v in axis]
    _parse_wide_axis_headers(headers, metadata_column_count=4)
    if len(names) != len(x) or any(not str(v).strip() for v in names):
        raise ValueError('来源名称缺失或长度与原始记录不一致')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Index', 'Label', 'Sample_ID', 'Name', *headers])
        for i, row in enumerate(x):
            writer.writerow([i, y[i], groups[i], names[i], *row])
    temporary.replace(path)


def _full_source_summary(task, x, y, groups):
    task.update(source_row_count=int(len(x)), source_feature_count=int(x.shape[1]) if x.ndim == 2 else None,
                source_row_class_counts=dict(sorted(Counter(str(v) for v in y).items())))
    if len(groups):
        task['source_sample_group_count'] = len(set(str(v) for v in groups))
        task['source_repeat_count_histogram'] = dict(sorted(Counter(Counter(str(v) for v in groups).values()).items()))


def _full_split_coverage(task):
    main = task['main_audit']['class_counts']
    if len(main) < 2:
        raise ValueError(f'默认分类目标不足两类，实际类别组数: {main}')
    minimum = 2 if task['test_path'] else 3
    insufficient = {k: v for k, v in main.items() if v < minimum}
    if insufficient:
        task['split_coverage_evidence'] = {'required_main_groups_per_class': minimum,
                                          'insufficient_classes': insufficient,
                                          'main_class_counts': main}
        if task['test_path']:
            raise ValueError(f'官方Train有单例类别，不能同时覆盖Train与Valid: {insufficient}')
        raise ValueError(f'完整数据类别组数不足以覆盖Train、Valid、Test: {insufficient}')
    if task['test_path']:
        test = task['test_audit']['class_counts']
        if set(main) != set(test):
            task['split_coverage_evidence'] = {'main_class_counts': main, 'test_class_counts': test}
            raise ValueError('官方Train与Test类别集合不一致，不能满足完整类别覆盖')



def _reuse_full_task(cached, previous, old_digest, repo_root, destination, arrays, receipt_path, *, allow_decision_exclusion=False):
    """Reuse one admitted task only when every selected input and output is unchanged."""
    eligible = cached and (cached.get('acquisition_status') == 'ready' or
                           (allow_decision_exclusion and cached.get('decision_evidence')
                            and cached.get('acquisition_status') == 'scientifically_excluded'))
    if (not eligible
            or cached.get('previous_manifest_sha256') != old_digest
            or not cached.get('all_samples_retained')):
        return None
    name, suite = previous['task_id'], previous['suite']
    if cached.get('task_id') != name or cached.get('suite') != suite:
        return None
    try:
        receipt = ({'path': str(receipt_path), 'sha256': sha256(receipt_path)}
                   if receipt_path.exists() else None)
        if cached.get('acquisition_receipt') != receipt:
            return None
        if suite == 'raman34' and arrays.exists():
            inputs = [{'path': str(arrays), 'sha256': sha256(arrays), 'kind': 'raw_arrays'}]
            splits = ['main']
        else:
            inputs, splits = [], []
            for split in ('main', 'test'):
                source = previous.get(f'{split}_path')
                if not source:
                    continue
                source = Path(source)
                if not source.is_absolute():
                    source = repo_root / source
                digest = sha256(source)
                if digest != previous.get(f'{split}_sha256'):
                    return None
                inputs.append({'path': str(source), 'sha256': digest, 'kind': split})
                splits.append(split)
        if not inputs or cached.get('input_files') != inputs:
            return None
        if bool(cached.get('test_path')) != ('test' in splits):
            return None
        for split in splits:
            output = destination / suite / name / f'{split}.csv'
            if (cached.get(f'{split}_path') != str(output)
                    or not cached.get(f'{split}_audit')
                    or sha256(output) != cached.get(f'{split}_sha256')):
                return None
    except OSError:
        return None
    task = deepcopy(cached)
    # Family ownership and duplicate exclusions are rebuilt for the whole catalog.
    task['role'] = task['requested_role']
    task['exclusion_reason'] = None
    task.pop('duplicate_of', None)
    return task

def prepare_full_assets(repo_root: Path, benchmark_root: Path, source_root: Path | None = None) -> dict:
    """Prepare all source records under the human-approved full-scope revision.

    The old 149-candidate manifest is read-only provenance. Newly acquired Raman
    arrays take precedence, and absent sources remain pending until an explicit
    terminal acquisition receipt exists. No fitting or network access occurs.
    """
    repo_root, benchmark_root = Path(repo_root).resolve(), Path(benchmark_root).resolve()
    source_root = Path(source_root or repo_root / 'work/benchmark-20261010').resolve()
    if (benchmark_root == source_root or benchmark_root in source_root.parents
            or source_root in benchmark_root.parents):
        raise ValueError('新轮次目录必须与旧来源目录分离，禁止覆盖旧轮次证据')
    old_path = source_root / 'data/manifest.json'
    old = _json(old_path)
    if old is None:
        raise ValueError('旧149候选来源manifest不存在')
    old_digest = sha256(old_path)
    destination, audit = benchmark_root / 'data', benchmark_root / 'audit'
    decisions_path = audit / 'task-decisions.json'
    decision_document = _json(decisions_path)
    decisions, decision_receipt = {}, None
    if decision_document is not None:
        if (not isinstance(decision_document, dict)
                or decision_document.get('user_authorized') is not True
                or not isinstance(decision_document.get('authorization_quote'), str)
                or not decision_document['authorization_quote'].strip()
                or not isinstance(decision_document.get('tasks'), dict)):
            raise ValueError('任务排除决定必须有明确人类授权、批准原句和tasks映射')
        decisions = decision_document['tasks']
        unknown = set(decisions) - {t['task_id'] for t in old['tasks']}
        if unknown:
            raise ValueError(f'任务排除决定含未知任务ID: {sorted(unknown)}')
        for name, decision in decisions.items():
            if (not isinstance(decision, dict) or decision.get('include') is not False
                    or not isinstance(decision.get('reason'), str) or not decision['reason'].strip()
                    or not isinstance(decision.get('evidence', []), list)):
                raise ValueError(f'任务排除决定{name}必须明确include=false、非空reason和evidence列表')
        decision_receipt = {'path': str(decisions_path), 'sha256': sha256(decisions_path)}
    prepared = _json(destination / 'manifest.json', {})
    cached_tasks = ({t['task_id']: t for t in prepared.get('tasks', [])}
                    if prepared.get('schema_version') == 'benchmark-data-full-v4' else {})
    tasks, reused_ids = [], set()
    for previous in old['tasks']:
        name, suite = previous['task_id'], previous['suite']
        task = _task(name, suite, previous['axis_kind'], dict(previous['source']),
                     previous.get('family_id', _family(suite, name)))
        task.update(previous_role=previous['role'], previous_exclusion_reason=previous.get('exclusion_reason'),
                    previous_manifest_sha256=old_digest, acquisition_status='awaiting_source',
                    all_samples_retained=False, input_files=[])
        if 'legacy_catalog_sample_count' in previous:
            task['legacy_catalog_sample_count'] = previous['legacy_catalog_sample_count']
        receipt_path = audit / f'{name}-source-load.json'
        receipt = _json(receipt_path, {})
        if receipt:
            task['acquisition'] = receipt
            task['acquisition_receipt'] = {'path': str(receipt_path), 'sha256': sha256(receipt_path)}
        fresh = destination / 'raw-arrays' / f'{name}.npz'
        legacy_array = source_root / 'data/raw/raman-arrays' / f'{name}.npz'
        arrays = fresh if fresh.exists() else legacy_array
        if name in decisions:
            # This exclusion is an explicit human decision, applied before the
            # normal reuse path. Preserve verified audits or acquisition evidence
            # without loading/converting the matrix or deleting any source asset.
            verified = _reuse_full_task(cached_tasks.get(name), previous, old_digest, repo_root,
                                       destination, arrays, receipt_path, allow_decision_exclusion=True)
            if verified is not None:
                task = verified
            else:
                for key, source_key in (('source_row_count', 'rows'), ('source_feature_count', 'features'),
                                        ('source_row_class_counts', 'class_counts'),
                                        ('source_sample_group_count', 'explicit_group_count')):
                    if receipt.get(source_key) is not None:
                        task[key] = deepcopy(receipt[source_key])
                if receipt:
                    task['source_count_evidence'] = deepcopy(task['acquisition_receipt'])
                if suite == 'raman34' and arrays.exists():
                    digest = sha256(arrays)
                    task['input_files'] = [{'path': str(arrays), 'sha256': digest, 'kind': 'raw_arrays'}]
                    task['source'].update(raw_array_path=str(arrays), array_sha256=digest)
                else:
                    for split in ('main', 'test'):
                        source = previous.get(f'{split}_path')
                        if source:
                            source = Path(source)
                            if not source.is_absolute():
                                source = repo_root / source
                            if source.exists():
                                task['input_files'].append({'path': str(source), 'sha256': sha256(source),
                                                            'kind': split})
            decision = decisions[name]
            task.update(role='excluded', acquisition_status='scientifically_excluded',
                        exclusion_reason=decision['reason'],
                        decision_evidence={'user_authorized': True,
                                           'authorization_quote': decision_document['authorization_quote'],
                                           'reason': decision['reason'], 'reason_code': decision.get('reason_code'),
                                           'evidence': deepcopy(decision.get('evidence', [])),
                                           'decision_receipt': decision_receipt})
            tasks.append(task)
            continue
        reused = _reuse_full_task(cached_tasks.get(name), previous, old_digest, repo_root,
                                  destination, arrays, receipt_path)
        if reused is not None:
            tasks.append(reused)
            reused_ids.add(name)
            continue
        try:
            if suite == 'raman34' and arrays.exists():
                digest = sha256(arrays)
                expected = (receipt.get('array_sha256') if arrays == fresh
                            else previous['source'].get('array_sha256'))
                if expected and digest != expected:
                    raise ValueError('来源NPZ SHA256与登记不一致')
                task['input_files'].append({'path': str(arrays), 'sha256': digest, 'kind': 'raw_arrays'})
                with np.load(arrays, allow_pickle=False) as data:
                    x, y, axis, groups = (data[k] for k in ('x', 'y', 'axis', 'groups'))
                    names = data['names'] if 'names' in data else np.repeat(arrays.name, len(x))
                    _full_source_summary(task, x, y, groups)
                    if receipt.get('rows') is not None and receipt['rows'] != len(x):
                        raise ValueError('来源记录数与获取回执不一致')
                    if receipt.get('features') is not None and receipt['features'] != x.shape[1]:
                        raise ValueError('来源特征数与获取回执不一致')
                    if receipt.get('class_counts') is not None and receipt['class_counts'] != task['source_row_class_counts']:
                        raise ValueError('来源标签计数与获取回执不一致')
                    physical_groups = len(groups) and (arrays == fresh or previous.get('independence_status') in SUPPORTED_INDEPENDENCE)
                    if physical_groups:
                        task.update(independence_status='verified_sample_ids',
                                    independence_evidence=previous.get('independence_evidence') or
                                    ['原始数据loader返回的显式样品/受试者分组；保留每组全部记录'],
                                    split_unit='source_physical_group')
                    else:
                        if (previous.get('independence_status') in SUPPORTED_INDEPENDENCE
                                and previous.get('sample_group_count') is not None
                                and previous['sample_group_count'] < len(x)):
                            raise ValueError('已知来源含真实重复分组，但新数组缺少分组；不能按记录拆开已知物理组')
                        if len(groups) and len(set(str(v) for v in groups)) != len(groups):
                            raise ValueError('旧来源重复归档组缺少真实分组证据，不能拆开或猜测物理组')
                        groups = np.asarray([f'{name}:source-record:{i}' for i in range(len(x))])
                        task.update(independence_status=ARCHIVE_RECORD_STATUS,
                                    independence_evidence=['用户批准缺少真实分组时按原始来源记录划分；编号只标识原记录'],
                                    split_unit='source_record', generalization_limit=ARCHIVE_RECORD_LIMIT)
                    representation = receipt.get('axis_representation')
                    task['axis_representation'] = representation or previous.get('acquisition', {}).get('axis_kind') or previous['axis_kind']
                    # Ordered Raman channels retain their modality/model scope;
                    # coordinate units are tracked independently of that scope.
                    task['axis_kind'] = previous['axis_kind']
                    task['axis_unit'] = ('channel_index' if representation and 'index' in representation else
                                         'unverified' if representation and 'unverified' in representation else
                                         'source_spectral_coordinate')
                    task['axis_calibration_verified'] = representation == 'author_spectral_coordinates'
                    task['axis_label'] = ('feature/channel index' if task['axis_unit'] == 'channel_index' else
                                          'author-provided coordinate (unit unverified)' if representation and 'unverified' in representation else
                                          'source spectral coordinate')
                    summary = audit_arrays(x, y, axis, groups)
                    output = destination / suite / name / 'main.csv'
                    _full_write(output, x, y, axis, groups, names)
                if sha256(arrays) != digest:
                    raise ValueError('准备期间来源NPZ发生变化；本轮输入不能冻结')
                task.update(main_path=str(output), main_sha256=sha256(output), main_audit=summary)
                task['source']['array_sha256'] = digest
            else:
                status = previous.get('independence_status', 'unknown')
                physical = status in SUPPORTED_INDEPENDENCE
                task.update(independence_status=status if physical else ARCHIVE_RECORD_STATUS,
                            independence_evidence=previous.get('independence_evidence', []) if physical else
                            ['用户批准缺少真实分组时按原始来源记录划分；归档ID只标识原记录'],
                            split_unit='source_physical_group' if physical else 'source_record')
                if not physical:
                    task['generalization_limit'] = ARCHIVE_RECORD_LIMIT
                for split in ('main', 'test'):
                    source_path = previous.get(f'{split}_path')
                    if not source_path:
                        if split == 'main':
                            raise FileNotFoundError('来源数组/CSV尚未可用，保留待获取状态')
                        continue
                    original = Path(source_path)
                    if not original.is_absolute():
                        original = repo_root / original
                    expected = previous.get(f'{split}_sha256')
                    if not expected or sha256(original) != expected:
                        raise ValueError(f'{split}来源CSV SHA256与旧登记不一致')
                    parsed = load_modeling_csv(original)
                    groups = parsed.frame.Sample_ID.to_numpy()
                    if not physical and len(set(groups)) != len(groups):
                        raise ValueError('来源CSV含重复归档组但没有真实分组证据；不能拆组或猜测分组')
                    summary = audit_arrays(parsed.intensity, parsed.frame.Label.to_numpy(), parsed.x_axis[0], groups)
                    task['input_files'].append({'path': str(original), 'sha256': expected, 'kind': split})
                    output = destination / suite / name / f'{split}.csv'
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, output)
                    if sha256(output) != expected:
                        raise ValueError(f'{split}复制后的CSV SHA256与来源不一致')
                    task.update({f'{split}_path': str(output), f'{split}_sha256': expected, f'{split}_audit': summary})
                task['source_row_count'] = sum(task.get(f'{s}_audit', {}).get('row_count', 0) for s in ('main', 'test'))
                counts = Counter()
                for split in ('main', 'test'):
                    summary = task.get(f'{split}_audit', {})
                    counts.update({k: v * summary['repeats_per_group'] for k, v in summary.get('class_counts', {}).items()})
                task['source_row_class_counts'] = dict(sorted(counts.items()))
                task['source_feature_count'] = task['main_audit']['feature_count']
            task.update(task['main_audit'])
            if task['independence_status'] == ARCHIVE_RECORD_STATUS:
                task['physical_independence_verified'] = False
                task['archive_group_count'] = task['sample_group_count']
            else:
                task['physical_independence_verified'] = True
            if task['test_path']:
                a, b = load_modeling_csv(task['main_path']), load_modeling_csv(task['test_path'])
                if a.x_axis[0] != b.x_axis[0]:
                    raise ValueError('官方Train与Test公共轴不一致')
                if set(a.sample_id) & set(b.sample_id):
                    raise ValueError('官方Train与Test包含相同分组标识，不能拆开物理组')
            task['output_row_count'] = sum(task.get(f'{s}_audit', {}).get('row_count', 0) for s in ('main', 'test'))
            task['all_samples_retained'] = task['output_row_count'] == task['source_row_count']
            if not task['all_samples_retained']:
                raise ValueError('来源与输出记录数不一致')
            _full_split_coverage(task)
            task.update(role=task['requested_role'], acquisition_status='ready', exclusion_reason=None)
        except FileNotFoundError as ex:
            terminal = receipt.get('terminal') is True
            task.update(acquisition_status='source_unavailable' if terminal else 'awaiting_source',
                        exclusion_reason=receipt.get('error') or str(ex))
        except (ValueError, OSError, KeyError, IndexError) as ex:
            task.update(role='excluded', acquisition_status='scientifically_excluded', exclusion_reason=str(ex))
        tasks.append(task)
    # Preserve the approved development family ownership across related views.
    dev_families = {t['family_id'] for t in tasks if t['requested_role'] == 'development'}
    for task in tasks:
        if task['role'] != 'excluded' and task['family_id'] in dev_families:
            task['role'] = 'development'
    # Prefer development representatives if identical content occurs across roles.
    fingerprints = {}
    for task in sorted(tasks, key=lambda t: (t['role'] != 'development', t['task_id'])):
        if task['role'] == 'excluded':
            continue
        cached_digest = task.get('content_sha256') if task['task_id'] in reused_ids else None
        digest = (tuple(cached_digest) if cached_digest is not None else
                  tuple(_full_digest(Path(task[k])) if task[k] else None for k in ('main_path', 'test_path')))
        task['content_sha256'] = list(digest)
        if digest in fingerprints:
            task.update(role='excluded', acquisition_status='scientifically_excluded',
                        duplicate_of=fingerprints[digest], exclusion_reason=f'与{fingerprints[digest]}完全重复的任务版本')
        else:
            fingerprints[digest] = task['task_id']
    pending = sum(t['acquisition_status'] == 'awaiting_source' for t in tasks)
    result = {'schema_version': 'benchmark-data-full-v4',
              'status': 'awaiting_sources' if pending else ('complete' if len(tasks) == 149 else 'incomplete_catalog'),
              'candidate_count': len(tasks), 'pending_source_count': pending,
              'source_manifest': {'path': str(old_path), 'sha256': old_digest},
              'tasks': tasks, 'summary': dict(Counter(t['role'] for t in tasks)),
              'acquisition_summary': dict(Counter(t['acquisition_status'] for t in tasks)),
              'source_record_split_limitation': ARCHIVE_RECORD_LIMIT,
              'legacy_tabmini19': old.get('legacy_tabmini19', [])}
    if sha256(old_path) != old_digest:
        raise ValueError('旧来源manifest在准备期间发生变化；不能冻结新轮次')
    _save(destination / 'manifest.json', result)
    _save(audit / 'data-summary.json', {k: v for k, v in result.items() if k != 'tasks'})
    return result
