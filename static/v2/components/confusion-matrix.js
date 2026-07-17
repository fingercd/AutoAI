/** 混淆矩阵：表格化呈现，数字为文本，颜色只做辅助强调。 */
import { el } from '../lib/dom.js';

export function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix)) return [];
  return matrix
    .filter((row) => Array.isArray(row))
    .map((row) => row.map((value) => Number(value) || 0));
}

export function matrixMax(matrix) {
  return Math.max(0, ...normalizeMatrix(matrix).flat());
}

/** 单元格强度 0..1，用于背景透明度；对角线（正确预测）使用另一色相。 */
export function cellIntensity(value, max) {
  if (!max || max <= 0) return 0;
  return Math.min(1, Math.max(0, Number(value) || 0) / max);
}

export function matrixTotals(matrix) {
  const normalized = normalizeMatrix(matrix);
  const total = normalized.flat().reduce((sum, value) => sum + value, 0);
  const correct = normalized.reduce((sum, row, index) => sum + (row[index] || 0), 0);
  return { total, correct };
}

/**
 * 白底配色的单元格底色：对角线用主色 teal，误测用柔和红；
 * 透明度上限压低，保证深色数字文本在白底上对比度充足。
 */
function cellBackground(diagonal, value, intensity) {
  if (diagonal) return `rgba(15, 118, 110, ${0.08 + intensity * 0.42})`;
  if (value > 0) return `rgba(220, 38, 38, ${0.05 + intensity * 0.35})`;
  return 'transparent';
}

export function renderConfusionMatrix({ matrix, labels = [], caption = '' }) {
  const normalized = normalizeMatrix(matrix);
  if (!normalized.length) {
    return el('p', { className: 'hint', text: '该分区没有混淆矩阵数据。' });
  }
  const resolvedLabels = labels.length === normalized.length
    ? labels.map(String)
    : normalized.map((_, index) => String(index));
  const max = matrixMax(normalized);
  const { total, correct } = matrixTotals(normalized);
  const wrap = el('div', { className: 'table-wrap' });
  const table = el('table', { className: 'data-table confusion-table' }, [
    caption ? el('caption', { text: caption }) : null,
    el('thead', {}, el('tr', {}, [
      el('th', { text: '真实 \\ 预测', attrs: { scope: 'col' } }),
      resolvedLabels.map((label) => el('th', { text: label, attrs: { scope: 'col' } })),
    ])),
    el('tbody', {}, normalized.map((row, rowIndex) => el('tr', {}, [
      el('th', { text: resolvedLabels[rowIndex], attrs: { scope: 'row' } }),
      row.map((value, colIndex) => {
        const diagonal = rowIndex === colIndex;
        const intensity = cellIntensity(value, max);
        const cell = el('td', {
          className: diagonal ? 'confusion-cell confusion-correct' : 'confusion-cell',
          text: String(value),
        });
        cell.style.backgroundColor = cellBackground(diagonal, value, intensity);
        cell.style.textAlign = 'center';
        cell.style.fontVariantNumeric = 'tabular-nums';
        cell.style.minWidth = '52px';
        return cell;
      }),
    ]))),
  ]);
  wrap.append(table);
  wrap.append(el('p', {
    className: 'hint',
    text: `共 ${total} 个样本，预测正确 ${correct} 个。行是真实类别，列是预测类别；绿色为预测正确，红色为误测。`,
  }));
  return wrap;
}
