"""Inspect research tables and convert an explicit mapping to wide-feature-v2.

All mapping positions are 1-based, applied AFTER an optional transpose.
No model preprocessing, imputation or sample inference is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tables(path, *, encoding='utf-8-sig'):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        text = path.read_text(encoding=encoding)
        try:
            dialect = csv.Sniffer().sniff(text[:65536], delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel
        return {'csv': list(csv.reader(io.StringIO(text), dialect))}
    if path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=False)
        try:
            result = {}
            for sheet in workbook:
                rows = []
                for row in sheet.iter_rows():
                    if any(cell.data_type == 'f' for cell in row):
                        raise ValueError(f'{sheet.title} 含公式；请先另存为数值表，避免使用过期公式缓存')
                    rows.append(['' if cell.value is None else str(cell.value) for cell in row])
                result[sheet.title] = rows
            return result
        finally:
            workbook.close()
    raise ValueError('只支持 .csv 和 .xlsx；旧式 .xls 请先另存为 .xlsx')


def finite(value, context):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context} 不是数值：{value!r}') from exc
    if not math.isfinite(number):
        raise ValueError(f'{context} 必须是有限数值')
    return number


def inspect(path, *, encoding='utf-8-sig'):
    tables = read_tables(path, encoding=encoding)
    sheets = []
    aliases = {'label': {'label', '类别', '分类', '标签'},
               'sample_id': {'sample_id', '样品编号', '样品id', '样本编号'},
               'index': {'index', '序号'}, 'name': {'name', '文件名', '原文件名'}}
    for name, rows in tables.items():
        candidates = defaultdict(list)
        for row_index, row in enumerate(rows[:10], 1):
            for col_index, value in enumerate(row, 1):
                for key, words in aliases.items():
                    if value.strip().lower() in words:
                        candidates[key].append({'row': row_index, 'column': col_index, 'text': value})
        diagnostics = None
        if rows and rows[0][:4] == ['Index', 'Label', 'Sample_ID', 'Name']:
            try:
                axis = [finite(value, '坐标') for value in rows[0][4:]]
                measurements = [row for row in rows[1:] if any(value.strip() for value in row)]
                for row_index, row in enumerate(measurements, 2):
                    if len(row) != len(rows[0]):
                        raise ValueError(f'第 {row_index} 行列数与表头不一致')
                    for col_index, value in enumerate(row[4:], 5):
                        finite(value, f'第 {row_index} 行第 {col_index} 列')
                diagnostics = {'valid': True, 'summary': validate_rows(measurements, axis)}
            except ValueError as exc:
                diagnostics = {'valid': False, 'reason': str(exc)}
        sheets.append({'sheet': name, 'row_count': len(rows),
                       'column_count': max(map(len, rows), default=0),
                       'preview': [row[:12] for row in rows[:8]],
                       'metadata_candidates': dict(candidates), 'canonical_diagnostics': diagnostics})
    return {'source': str(Path(path).resolve()), 'sha256': fingerprint(path), 'sheets': sheets,
            'note': '候选只用于提出映射；不会自动认定标签、样品分组或独立性'}


def validate_rows(rows, axis):
    if not rows:
        raise ValueError('映射没有生成测量记录')
    if not 1 <= len(axis) <= 16380:
        raise ValueError('特征数必须在 1..16380')
    if any(left >= right for left, right in zip(axis, axis[1:])):
        raise ValueError('坐标必须唯一且严格递增')
    indexes = [row[0] for row in rows]
    if len(set(indexes)) != len(indexes):
        raise ValueError('Index 必须唯一；合并工作表可省略 index 映射以生成连续编号')
    grouped = defaultdict(list)
    for row in rows:
        if not row[1].strip() or not row[2].strip():
            raise ValueError('Label 和 Sample_ID 不得为空；请提供明确映射')
        if not row[0].strip():
            raise ValueError('Index 不得为空')
        grouped[row[2]].append(row[1])
    conflicts = [sid for sid, labels in grouped.items() if len(set(labels)) != 1]
    if conflicts:
        raise ValueError(f'同一个 Sample_ID 内出现多个 Label：{conflicts}')
    repeats = {sid: len(labels) for sid, labels in grouped.items()}
    if len(set(repeats.values())) != 1:
        raise ValueError(f'Sample_ID 重复测量次数不一致，当前后端不接受：{repeats}')
    return {'measurement_count': len(rows), 'sample_count': len(grouped),
            'feature_count': len(axis), 'repeats_per_sample': next(iter(repeats.values())),
            'class_sample_counts': dict(Counter(labels[0] for labels in grouped.values()))}


def convert(mapping, output_dir):
    """Return provenance after all tables pass validation, before any output is written."""
    allowed = {'tables', 'independent_rows_confirmed', 'sort_coordinates', 'metadata_by_name'}
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f'未知映射字段：{sorted(unknown)}')
    rows, provenance, axis, sources = [], [], None, {}
    known_table = {'source', 'sha256', 'encoding', 'sheet', 'transpose', 'header_row',
                   'data_start_row', 'data_end_row', 'columns', 'feature_start_column',
                   'feature_end_column', 'coordinates', 'label_value'}
    for table in mapping['tables']:
        unknown = set(table) - known_table
        if unknown:
            raise ValueError(f'未知工作表映射字段：{sorted(unknown)}')
        source = Path(table['source']).resolve()
        sha = fingerprint(source)
        if table.get('sha256') and table['sha256'] != sha:
            raise ValueError(f'{source.name} 已变化，请重新检查映射')
        sources[str(source)] = sha
        tables = read_tables(source, encoding=table.get('encoding', 'utf-8-sig'))
        sheet = table.get('sheet', 'csv')
        matrix = tables[sheet]
        width = max(map(len, matrix), default=0)
        matrix = [row + [''] * (width - len(row)) for row in matrix]
        if table.get('transpose'):
            matrix = [list(row) for row in zip(*matrix)]
            width = len(matrix[0]) if matrix else 0
        header_row = table.get('header_row', 1)
        start = table.get('data_start_row', header_row + 1)
        end = table.get('data_end_row', len(matrix))
        if not 1 <= header_row <= len(matrix) or not 1 <= start <= end <= len(matrix):
            raise ValueError(f'{sheet} 行号越界；行号为 1 基，首尾包含')
        first = table['feature_start_column']
        last = table.get('feature_end_column', width)
        if not 1 <= first <= last <= width:
            raise ValueError(f'{sheet} 特征列范围越界')
        coordinates = table.get('coordinates', matrix[header_row - 1][first - 1:last])
        if len(coordinates) != last - first + 1:
            raise ValueError('显式坐标数量与特征列数不一致')
        numeric_axis = [finite(value, f'{sheet} 坐标') for value in coordinates]
        order = list(range(len(numeric_axis)))
        if mapping.get('sort_coordinates'):
            order.sort(key=numeric_axis.__getitem__)
        current_axis = [numeric_axis[index] for index in order]
        if axis is not None and current_axis != axis:
            raise ValueError('合并工作表必须逐点同轴，不能自动插值或取第一条轴')
        axis = current_axis
        columns = table.get('columns', {})
        if set(columns) - {'index', 'label', 'sample_id', 'name'}:
            raise ValueError('columns 只允许 index、label、sample_id、name')
        if any(not isinstance(col, int) or isinstance(col, bool) or not 1 <= col <= width for col in columns.values()):
            raise ValueError('元数据列号必须位于工作表范围内')
        for row_number in range(start, end + 1):
            cells = matrix[row_number - 1]
            if all(not value.strip() for value in cells):
                continue
            def field(name, default=''):
                return cells[columns[name] - 1] if name in columns else default
            name = field('name', source.name)
            metadata = mapping.get('metadata_by_name', {}).get(name, {})
            label = metadata.get('Label', field('label', table.get('label_value', '')))
            sid = metadata.get('Sample_ID', field('sample_id'))
            if not sid and mapping.get('independent_rows_confirmed') is True:
                sid = f'sample-{len(rows) + 1:06d}'
            values = cells[first - 1:last]
            for col, value in enumerate(values, first):
                finite(value, f'{source.name}/{sheet} 第 {row_number} 行第 {col} 列')
            rows.append([field('index', str(len(rows) + 1)), str(label), str(sid), str(name), *[values[index] for index in order]])
            provenance.append({'source': str(source), 'sheet': sheet, 'mapped_row': row_number,
                               'transposed': bool(table.get('transpose')), 'output_index': rows[-1][0]})
    summary = validate_rows(rows, axis or [])
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('输出目录必须为空，避免覆盖已有任务记录')
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / 'originals'
    raw_dir.mkdir()
    copies = {}
    for source, sha in sources.items():
        target = raw_dir / f'{sha[:12]}-{Path(source).name}'
        shutil.copy2(source, target)
        copies[source] = str(target)
    output = output_dir / 'dataset.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Index', 'Label', 'Sample_ID', 'Name', *[repr(value) for value in axis]])
        writer.writerows(rows)
    report = {'format': 'wide-feature-v2', 'sources': sources, 'original_copies': copies,
              'output': str(output), 'output_sha256': fingerprint(output), 'summary': summary,
              'row_mapping': provenance,
              'operations': {'transpose': any(item.get('transpose') for item in mapping['tables']),
                             'sort_coordinates': bool(mapping.get('sort_coordinates')),
                             'generated_sample_ids': mapping.get('independent_rows_confirmed') is True}}
    (output_dir / 'mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'conversion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('inspect')
    check.add_argument('source', type=Path)
    check.add_argument('--encoding', default='utf-8-sig')
    prepare = sub.add_parser('convert')
    prepare.add_argument('--mapping', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            result = inspect(args.source, encoding=args.encoding)
        else:
            mapping = json.loads(args.mapping.read_text(encoding='utf-8-sig'))
            # Relative source paths are resolved against the mapping file.
            for table in mapping['tables']:
                table['source'] = str((args.mapping.parent / table['source']).resolve())
            result = convert(mapping, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError, csv.Error) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
