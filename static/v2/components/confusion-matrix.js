/**
 * 模块：混淆矩阵渲染组件（v2 工作台）。
 *
 * 职责：
 * - 把后端返回的分类混淆矩阵（二维计数数组）渲染成 HTML 表格；
 * - 提供矩阵归一化、最大值、行列合计等纯函数，供本组件和其他结果视图复用。
 *
 * 在系统中的位置：
 * - 属于 `static/v2/components/` 下的展示型组件，被结果页（`run-result-v1` 契约的
 *   confusion_matrix 数据）调用；
 * - 数据来源是训练评估产物（stratified_holdout / leave_one_sample_id_cv /
 *   external_test_holdout 三种口径下的 test 分区混淆矩阵），本文件不计算指标，
 *   只做"防御性归一化 + 表格呈现"。
 *
 * 关键设计约束：
 * - 数字一律以文本呈现，颜色只做辅助强调（无障碍友好，不依赖色觉传达信息）；
 * - 对角线（预测正确）与非对角线（误测）使用不同色相，透明度随计数强度变化；
 * - 所有入口函数对脏数据宽容：非数组、非数字一律降级为空矩阵或 0，绝不抛异常，
 *   因为结果页需要在 Manifest 不完整等边缘情况下仍然可渲染。
 */
import { el } from '../lib/dom.js';

/**
 * 把任意输入归一化为"数字构成的二维矩阵"。
 *
 * 逻辑原理：
 * - 第一层 `Array.isArray` 拦截 null / undefined / 对象等非法输入；
 * - `filter` 丢弃非数组行（后端数据被截断或手工篡改时可能出现）；
 * - `Number(value) || 0` 把字符串数字转为数值，把 NaN / null / 负值以外的
 *   非法项折叠为 0；注意负数会保留，但混淆矩阵计数语义上不会为负。
 *
 * @param {*} matrix 后端返回的原始矩阵（可能是任何脏数据）。
 * @returns {number[][]} 归一化后的二维数字数组；输入非法时返回空数组。
 */
export function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix)) return [];
  return matrix
    .filter((row) => Array.isArray(row))
    .map((row) => row.map((value) => Number(value) || 0));
}

/**
 * 取矩阵中的最大计数，用于把单元格计数归一化到 0..1 的强度。
 *
 * 设计意图：
 * - `Math.max(0, ...)` 的兜底 0 保证空矩阵（无展开元素）时返回 0 而不是 -Infinity，
 *   这样下游 `cellIntensity` 的 `max <= 0` 分支可以统一处理"没有数据"的情况。
 *
 * @param {*} matrix 原始矩阵（内部会再归一化一次，调用方无需先处理）。
 * @returns {number} 全矩阵最大计数，最小为 0。
 */
export function matrixMax(matrix) {
  return Math.max(0, ...normalizeMatrix(matrix).flat());
}

/**
 * 把单个单元格计数映射为 0..1 的相对强度，用于背景透明度计算。
 *
 * 为什么这么做：
 * - 用全矩阵最大值做归一化（而非按行/列归一化），让读者能横向比较
 *   "哪一格的计数绝对最高"，符合混淆矩阵"找最大误测对"的阅读习惯；
 * - `max <= 0` 时直接返回 0，避免除零产生 NaN 污染样式字符串。
 *
 * @param {*} value 单元格计数（脏值会被折叠为 0）。
 * @param {number} max 全矩阵最大值（来自 {@link matrixMax}）。
 * @returns {number} 夹在 [0, 1] 区间的强度值。
 */
export function cellIntensity(value, max) {
  if (!max || max <= 0) return 0;
  return Math.min(1, Math.max(0, Number(value) || 0) / max);
}

/**
 * 计算矩阵的样本总数与预测正确数（对角线之和）。
 *
 * 逻辑原理：
 * - total = 所有单元格求和；
 * - correct = 对角线 `row[index]` 求和；`|| 0` 容忍非方阵（行短于索引时
 *   取到 undefined），避免结果页在异常数据下显示 NaN。
 *
 * @param {*} matrix 原始矩阵。
 * @returns {{total: number, correct: number}} 样本总数与正确数。
 */
export function matrixTotals(matrix) {
  const normalized = normalizeMatrix(matrix);
  const total = normalized.flat().reduce((sum, value) => sum + value, 0);
  const correct = normalized.reduce((sum, row, index) => sum + (row[index] || 0), 0);
  return { total, correct };
}

/**
 * 白底配色的单元格底色：对角线用主色 teal，误测用柔和红；
 * 透明度上限压低，保证深色数字文本在白底上对比度充足。
 *
 * 设计意图：
 * - 正确（teal）与误测（红）分色是约定俗成的语义色；透明度随强度线性增长，
 *   但上限分别压到 0.50 / 0.40，防止高计数单元格的背景色吞掉黑色数字；
 * - 计数为 0 的非对角格返回 transparent，减少视觉噪声——零误测不需要被看见。
 *
 * @param {boolean} diagonal 是否位于对角线（真实类别 === 预测类别）。
 * @param {number} value 单元格计数。
 * @param {number} intensity {@link cellIntensity} 输出的 0..1 强度。
 * @returns {string} CSS 颜色值（rgba 或 'transparent'）。
 */
function cellBackground(diagonal, value, intensity) {
  if (diagonal) return `rgba(15, 118, 110, ${0.08 + intensity * 0.42})`;
  if (value > 0) return `rgba(220, 38, 38, ${0.05 + intensity * 0.35})`;
  return 'transparent';
}

/**
 * 渲染混淆矩阵为完整表格区块（含表头、单元格和底部汇总说明）。
 *
 * 渲染逻辑分段说明：
 * 1. 归一化输入；空矩阵直接返回提示段落，不渲染空表（结果页可能出现
 *    某分区没有混淆矩阵的合法情况，例如训练失败但 Run 记录仍在）。
 * 2. 标签对齐：后端 labels 数量与矩阵阶数一致才使用真实类别名，否则退化为
 *    数字索引——这是防御旧版/异常契约数据，宁可显示 "0/1/2" 也不错位。
 * 3. 表头第一格 "真实 \ 预测" 标明行=真实、列=预测，避免读者把方向看反
 *    （混淆矩阵最常见的误读）。
 * 4. 每个单元格：对角线加 `confusion-correct` 类；`tabular-nums` 等宽数字
 *    便于逐列比较；`minWidth` 防止单位数与多位数列宽跳动。
 * 5. 表格下方追加汇总句，再次用文字复述颜色语义（颜色只做辅助的兜底）。
 *
 * @param {object} options
 * @param {*} options.matrix 混淆矩阵二维数组（脏数据会被归一化）。
 * @param {string[]} [options.labels] 类别名列表，长度需与矩阵阶数一致才生效。
 * @param {string} [options.caption] 表格标题（如 "Test 集（pooled OOF）"），空串则不渲染 caption。
 * @returns {HTMLElement} `.table-wrap` 容器，或空数据时的提示段落。
 */
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
