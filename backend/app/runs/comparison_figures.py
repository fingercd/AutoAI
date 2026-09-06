"""Comparison-only figures. One Matplotlib document produces both SVG and PNG.

No pyplot/global figure registry, training imports or browser runtime required.
All metric values are fractions; None always remains missing, never zero.
"""
from __future__ import annotations

from io import BytesIO
from math import ceil
from threading import RLock
from typing import Any

import matplotlib
from matplotlib import colors, font_manager
from matplotlib.figure import Figure
import numpy as np
from .. import feature_policy

DRAWING_VERSION = 'comparison-figures-v1'
METRICS = {'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced Accuracy', 'macro_f1': 'Macro-F1', 'weighted_f1': 'Weighted-F1'}
NAMES = {'pls_da': 'PLS-DA', 'logistic_regression': 'Elastic Net', 'svm': 'SVM', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'cnn1d': '1D-CNN', 'spls_da': 'sPLS-DA', 'pca_svm': 'PCA-SVM'}
PALETTE = ['#42756a', '#825c80', '#b48139', '#bb7159', '#647775', '#88924f']
MODEL_COLORS = dict(zip(['pls_da', 'logistic_regression', 'svm', 'random_forest', 'xgboost', 'cnn1d'], PALETTE))
SCHEMES = [('full', 'Full features'), ('bin_5', 'Binning 5'), ('bin_10', 'Binning 10'), ('bin_20', 'Binning 20'), ('pca_90', 'PCA 90%'), ('pca_95', 'PCA 95%'), ('pca_99', 'PCA 99%')]
_LOCK = RLock()
_FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}
_FONTS = [name for name in ['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'] if name in _FONT_NAMES] + ['DejaVu Sans']
_RC = {'font.family': _FONTS, 'font.size': 9.75, 'axes.labelsize': 9.75, 'xtick.labelsize': 9.75, 'ytick.labelsize': 9.75,
       'svg.fonttype': 'path', 'svg.hashsalt': DRAWING_VERSION, 'text.parse_math': False, 'axes.unicode_minus': False}


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def ordered_models(data, sort='balanced_accuracy'):
    if sort not in METRICS:
        raise ValueError('未知排序指标')
    models = list(data.get('models', []))
    def key(row):
        value = finite(row.get('metrics', {}).get(sort, {}).get('mean'))
        return (value is None, -(value or 0), row['model_type'])
    return sorted(models, key=key)


def sample_indices(data, search='', errors=False):
    samples = data.get('sample_correctness', {})
    rows = samples.get('values', [])
    return [i for i, sid in enumerate(samples.get('sample_ids', []))
            if search.casefold() in str(sid).casefold()
            and (not errors or any(i < len(row.get('values', [])) and row['values'][i] == 0 for row in rows))]


def figure_specs(data):
    """Default archive inventory; every sample page and feature metric is included."""
    if not data.get('comparable'):
        return []
    result = [{'kind': 'overall', 'metric': metric, 'id': f'overall_{metric}'} for metric in METRICS]
    result += [{'kind': 'matrix', 'model': row['model_type'], 'id': f'matrix_{row["model_type"]}'} for row in data.get('confusion_matrices', []) if row.get('confusion_matrix')]
    if data.get('class_recall', {}).get('status') == 'ready':
        result.append({'kind': 'recall', 'id': 'class_recall'})
    count = len(data.get('sample_correctness', {}).get('sample_ids', []))
    result += [{'kind': 'samples', 'page': page, 'id': f'samples_{page + 1:03d}'} for page in range(ceil(count / 50))]
    if feature_policy.FEATURE_ENGINEERING_ENABLED and any(row.get('experiment') for row in data.get('models', [])):
        result += [{'kind': 'features', 'metric': metric, 'id': f'features_{metric}'} for metric in METRICS]
    return result


def figure_data(data, *, kind, metric='balanced_accuracy', model='', sort='balanced_accuracy', page=0, search='', errors=False, matrix_mode='percent'):
    """Pure normalization used by renderer, CSV export and contract tests."""
    if kind == 'features' and not feature_policy.FEATURE_ENGINEERING_ENABLED:
        raise ValueError('特征工程暂时停用')
    if not data.get('comparable'):
        raise ValueError(data.get('reason') or '当前结果不可比较')
    if metric not in METRICS or matrix_mode not in {'percent', 'count'} or page < 0:
        raise ValueError('无效图像参数')
    models = ordered_models(data, sort)
    ids = [row['model_type'] for row in models]
    names = [NAMES.get(mid, mid) for mid in ids]
    if kind == 'overall':
        return {'rows': names, 'columns': [METRICS[metric]], 'values': [[finite(row.get('metrics', {}).get(metric, {}).get('mean'))] for row in models], 'model_ids': ids}
    if kind == 'matrix':
        entry = next((row for row in data.get('confusion_matrices', []) if row['model_type'] == model), None)
        if not entry or not entry.get('confusion_matrix'):
            raise ValueError('该模型没有混淆矩阵')
        labels = list(map(str, entry['labels']))
        raw = entry['confusion_matrix']
        if len(raw) != len(labels) or any(len(row) != len(labels) for row in raw):
            raise ValueError('混淆矩阵与类别数量不一致')
        values = [[finite(value) for value in row] for row in raw]
        if matrix_mode == 'percent':
            values = [[None if value is None or not sum(v or 0 for v in row) else value / sum(v or 0 for v in row) for value in row] for row in values]
        vmax = 1 if matrix_mode == 'percent' else max([1] + [finite(v) or 0 for item in data.get('confusion_matrices', []) for row in (item.get('confusion_matrix') or []) for v in row])
        return {'rows': labels, 'columns': labels, 'values': values, 'vmax': vmax}
    if kind == 'recall':
        entries = {row['model_type']: row for row in data.get('class_recall', {}).get('rows', [])}
        labels = list(map(str, data.get('class_recall', {}).get('labels', [])))
        values = [[finite((item or {}).get('mean')) for item in entries.get(mid, {}).get('values', [None] * len(labels))] for mid in ids]
        return {'rows': names, 'columns': labels, 'values': values}
    if kind == 'samples':
        source = data.get('sample_correctness', {})
        entries = {row['model_type']: row for row in source.get('values', [])}
        selected = sample_indices(data, search, errors)
        indices = selected[page * 50:(page + 1) * 50]
        if not indices:
            raise ValueError('没有符合筛选条件的样品')
        values = [[entries.get(mid, {}).get('values', [None] * len(source['sample_ids']))[i] for i in indices] for mid in ids]
        return {'rows': names, 'columns': [str(source['sample_ids'][i]) for i in indices], 'values': values, 'total': len(selected)}
    if kind == 'features':
        values, marks, reasons = [], [], []
        for scheme_id, _ in SCHEMES:
            line, stars, notes = [], [], []
            for row in models:
                experiment = row.get('experiment') or {}
                scheme = next((item for item in experiment.get('schemes', []) if item.get('scheme_id') == scheme_id), {})
                line.append(finite((scheme.get('metrics') or {}).get(metric)) if scheme.get('status') == 'ready' else None)
                stars.append((experiment.get('selected_configuration') or {}).get('scheme_id') == scheme_id)
                notes.append(scheme.get('reason') or ('无历史特征工程数据' if not experiment else scheme.get('status', 'missing')))
            values.append(line); marks.append(stars); reasons.append(notes)
        return {'rows': [name for _, name in SCHEMES], 'columns': names, 'values': values, 'selected': marks, 'reasons': reasons}
    raise ValueError('未知图像类型')


def render_figure(data, *, format='svg', width=360, **options):
    if format not in {'svg', 'png'}:
        raise ValueError('只支持 SVG 和 PNG')
    width = min(1800, max(260, int(width)))
    spec = figure_data(data, **options)
    kind = options['kind']
    rows, columns, values = spec['rows'], spec['columns'], spec['values']
    count_view = kind == 'matrix' and options.get('matrix_mode') == 'count'
    with _LOCK, matplotlib.rc_context(_RC):
        if kind == 'overall':
            height = 68 + max(1, len(rows)) * 32
            fig = Figure(figsize=(width / 96, height / 96), dpi=96, facecolor='white')
            ax = fig.add_axes([108 / width, 30 / height, (width - 153) / width, (height - 62) / height])
            numeric = [row[0] for row in values]
            ax.barh(range(len(rows)), [v * 100 if v is not None else 0 for v in numeric], height=.56,
                    color=[MODEL_COLORS.get(mid, PALETTE[sum(map(ord, mid)) % len(PALETTE)]) for mid in spec['model_ids']])
            for i, value in enumerate(numeric):
                ax.text(103, i, '—' if value is None else f'{value * 100:.1f}%', va='center', clip_on=False)
            ax.set_yticks(range(len(rows)), rows); ax.invert_yaxis(); ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100], ['0%', '50%', '100%'])
            ax.grid(axis='x', color='#e6eae7', linewidth=.6); ax.set_axisbelow(True)
            ax.tick_params(length=0, pad=6)
            for spine in ax.spines.values(): spine.set_visible(False)
        else:
            if not columns or not rows:
                raise ValueError('图像没有可显示数据')
            is_matrix = kind == 'matrix'
            left = 68 if is_matrix else 145
            bottom = 62 if kind == 'samples' else 48
            if kind == 'samples': width = max(320, left + len(columns) * 36 + 20)
            if kind in {'recall', 'features'}: width = max(520, left + len(columns) * 110 + 70)
            plot_width = width - left - (64 if kind != 'samples' else 12)
            plot_height = plot_width if is_matrix else len(rows) * 32
            height = plot_height + bottom + 36
            fig = Figure(figsize=(width / 96, height / 96), dpi=96, facecolor='white')
            ax = fig.add_axes([left / width, bottom / height, plot_width / width, plot_height / height])
            array = np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)
            cmap = colors.ListedColormap(['#f4d5cb', '#d7eadf']) if kind == 'samples' else colors.LinearSegmentedColormap.from_list('forest', ['#ffffff', '#c9ded2', '#719982', '#254e3f'])
            cmap.set_bad('#e8e8e8')
            vmax = spec.get('vmax', 1)
            image = ax.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=0, vmax=vmax, aspect='equal' if is_matrix else 'auto', interpolation='nearest')
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    if value is None: label = '—'
                    elif kind == 'samples': label = '✓' if value == 1 else '×'
                    elif count_view: label = f'{value:g}'
                    else: label = f'{value * 100:.0f}%' if is_matrix else f'{value * 100:.1f}%'
                    if spec.get('selected', [[False] * len(columns)] * len(rows))[i][j]: label += ' *'
                    color = '#ffffff' if kind != 'samples' and value is not None and value / vmax > .62 else '#24392f'
                    # Keep numbers visibly smaller than each square, not merely
                    # smaller than the axis labels (dense four-column layouts).
                    cell_font = min(8.25, max(6, (plot_width / len(columns)) * .19)) if is_matrix else 9.75
                    ax.text(j, i, label, ha='center', va='center', color=color, fontsize=cell_font)
            ax.set_yticks(range(len(rows)), rows)
            ax.set_xticks(range(len(columns)), columns, rotation=90 if kind == 'samples' else 0)
            ax.tick_params(length=0, pad=7)
            ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
            ax.grid(which='minor', color='#e1e7e2', linewidth=.6); ax.tick_params(which='minor', length=0)
            for spine in ax.spines.values(): spine.set_visible(False)
            if is_matrix:
                ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
            if kind != 'samples':
                cax = fig.add_axes([(width - 55) / width, bottom / height, 8 / width, plot_height / height])
                bar = fig.colorbar(image, cax=cax, ticks=[0, vmax / 2, vmax])
                bar.ax.set_yticklabels([f'{v:g}' if count_view else f'{v * 100:.0f}%' for v in [0, vmax / 2, vmax]])
                bar.ax.tick_params(length=0, labelsize=9.75); bar.outline.set_visible(False)
        title = METRICS[options.get('metric', 'balanced_accuracy')] if kind == 'overall' else NAMES.get(options.get('model'), options.get('model')) if kind == 'matrix' else 'Recall / Sensitivity' if kind == 'recall' else 'Sample_ID × Models' if kind == 'samples' else 'Feature schemes · ' + METRICS[options.get('metric', 'balanced_accuracy')]
        fig.text(10 / width, 1 - 8 / height, title, fontsize=12, weight='semibold', va='top', color='#25382f')
        output = BytesIO()
        fig.savefig(output, format=format, dpi=192 if format == 'png' else 96, metadata={'Creator': DRAWING_VERSION} if format == 'svg' else {'Software': DRAWING_VERSION})
        return output.getvalue(), {'width': width, 'height': round(height), **spec}
