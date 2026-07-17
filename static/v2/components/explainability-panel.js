/** 单样品解释面板：摘要先行，随后自动加载完整 sample_feature_importance。 */
import { el, clear, svgEl } from '../lib/dom.js';
import { formatMetric } from '../lib/format.js';
import { artifactActionState } from './artifacts.js';
import { niceTicks, formatTick } from '../lib/charts.js';

const METHOD_LABELS = {
  sample_occlusion_log_loss: '窗口遮挡 Log-loss',
  sample_occlusion_importance: '窗口遮挡重要性',
  gradcam_1d: '1D Grad-CAM',
  dscarnet_dual_2d_gradcam: 'DSCARNet 双通路 2D Grad-CAM 回投',
};

/** 白底图表装饰色：与 lib/charts.js 的浅色主题保持一致。 */
const GRID_STROKE = '#e5e9f0';
const AXIS_STROKE = '#94a3b8';
const TICK_FILL = '#64748b';
const CURVE_STROKE = '#2563eb';

function methodLabel(method) {
  return METHOD_LABELS[method] || method || '—';
}

function sampleLabel(sample, position) {
  const mark = sample.correct === false ? '✗' : '✓';
  const name = sample.name || `样品 ${position + 1}`;
  return `${mark} ${name}（真实 ${sample.true_label ?? '—'} → 预测 ${sample.pred_label ?? '—'}）`;
}

const W = 760;
const H = 300;
const M = { top: 16, right: 16, bottom: 40, left: 60 };

/** 优先使用当前样品自己的谱线与 X 轴；旧产物缺字段时才回退到全局轴和基线谱线。 */
export function resolveSampleSeries(sample, payload) {
  const sampleCurve = Array.isArray(sample?.curve) ? sample.curve.map(Number) : [];
  const baselineCurve = Array.isArray(payload?.baseline_curve) ? payload.baseline_curve.map(Number) : [];
  const curve = sampleCurve.length ? sampleCurve : baselineCurve;
  const sampleXAxis = Array.isArray(sample?.sample_x_axis) ? sample.sample_x_axis.map(Number) : [];
  const payloadXAxis = Array.isArray(payload?.x_axis) ? payload.x_axis.map(Number) : [];
  const xAxis = sampleXAxis.length === curve.length ? sampleXAxis : payloadXAxis;
  return { curve, xAxis };
}

function isPrimaryWindow(win, primary) {
  if (!primary) return false;
  if (win.rank !== undefined && primary.rank !== undefined) return Number(win.rank) === Number(primary.rank);
  return Number(win.start_index) === Number(primary.start_index)
    && Number(win.end_index) === Number(primary.end_index);
}

/** 谱线 + 重要性窗口叠加：第一重要区间使用红色，其余窗口按重要性使用橙色。 */
function sampleChart(sample, payload) {
  const { xAxis, curve } = resolveSampleSeries(sample, payload);
  const windows = Array.isArray(sample?.windows) ? sample.windows : [];
  const primary = sample?.primary_segment || (Array.isArray(sample?.top_segments) ? sample.top_segments[0] : null);
  const svg = svgEl('svg', {
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    class: 'chart',
    'aria-label': `样品 ${sample.name ?? ''} 的谱线与重要性窗口`,
    preserveAspectRatio: 'xMidYMid meet',
    style: 'width:100%;height:auto;display:block;',
  });
  svg.append(svgEl('title', {}, `样品 ${sample.name ?? ''} 解释图`));
  svg.append(svgEl('desc', {}, '当前样品谱线叠加重要性窗口；红色为第一重要区间，窗口数值见下方表格。'));

  const n = curve.length;
  const xs = xAxis.length === n && n > 0 ? xAxis : Array.from({ length: n }, (_, index) => index);
  const xMin = windows.length && !n
    ? Math.min(...windows.map((w) => Number(w.start_x ?? w.start_index ?? 0)))
    : Math.min(...xs);
  const xMax = windows.length && !n
    ? Math.max(...windows.map((w) => Number(w.end_x ?? w.end_index ?? 1)))
    : Math.max(...xs);
  let yMin = n ? Math.min(...curve) : 0;
  let yMax = n ? Math.max(...curve) : 1;
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMin === xMax) {
    svg.append(svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'chart-empty', fill: TICK_FILL, 'font-size': 13 }, '没有可绘制的解释数据'));
    return svg;
  }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const sx = (value) => M.left + ((value - xMin) / (xMax - xMin)) * (W - M.left - M.right);
  const sy = (value) => H - M.bottom - ((value - yMin) / (yMax - yMin)) * (H - M.top - M.bottom);

  for (const tick of niceTicks(yMin, yMax, 5)) {
    svg.append(svgEl('line', { x1: M.left, x2: W - M.right, y1: sy(tick), y2: sy(tick), class: 'chart-grid', stroke: GRID_STROKE, 'stroke-width': 1 }));
    svg.append(svgEl('text', { x: M.left - 8, y: sy(tick) + 4, 'text-anchor': 'end', class: 'chart-tick', fill: TICK_FILL, 'font-size': 11 }, formatTick(tick)));
  }
  for (const tick of niceTicks(xMin, xMax, 7)) {
    svg.append(svgEl('text', { x: sx(tick), y: H - M.bottom + 18, 'text-anchor': 'middle', class: 'chart-tick', fill: TICK_FILL, 'font-size': 11 }, formatTick(tick)));
  }

  for (const win of windows) {
    const start = Number(win.start_x ?? xs[Number(win.start_index) || 0] ?? xMin);
    const end = Number(win.end_x ?? xs[Number(win.end_index) || 0] ?? start);
    const importance = Math.min(1, Math.max(0, Number(win.normalized_importance ?? 0)));
    svg.append(svgEl('rect', {
      x: sx(Math.min(start, end)),
      y: M.top,
      width: Math.max(1, sx(Math.max(start, end)) - sx(Math.min(start, end))),
      height: H - M.top - M.bottom,
      fill: isPrimaryWindow(win, primary)
        ? `rgba(220, 38, 38, ${0.18 + importance * 0.35})`
        : `rgba(217, 119, 6, ${0.10 + importance * 0.40})`,
    }));
  }
  if (n > 1) {
    const points = xs.map((value, index) => `${sx(value).toFixed(2)},${sy(curve[index]).toFixed(2)}`).join(' ');
    svg.append(svgEl('polyline', { points, fill: 'none', stroke: CURVE_STROKE, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }));
  }
  svg.append(svgEl('line', { x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom, class: 'chart-axis', stroke: AXIS_STROKE, 'stroke-width': 1.2 }));
  svg.append(svgEl('line', { x1: M.left, x2: M.left, y1: M.top, y2: H - M.bottom, class: 'chart-axis', stroke: AXIS_STROKE, 'stroke-width': 1.2 }));
  return svg;
}

function windowTable(sample) {
  const windows = (Array.isArray(sample?.windows) ? sample.windows : [])
    .slice()
    .sort((a, b) => (Number(a.rank) || 0) - (Number(b.rank) || 0));
  if (!windows.length) return el('p', { className: 'hint', text: '该样品没有窗口级重要性数据。' });
  const wrap = el('div', { className: 'table-wrap' });
  const body = el('tbody');
  const table = el('table', { className: 'data-table' }, [
    el('caption', { text: `样品 ${sample.name ?? ''} 的重要性窗口（按 rank 排序）` }),
    el('thead', {}, el('tr', {}, ['rank', 'X 区间', 'importance', '归一化'].map((head) => el('th', { text: head, attrs: { scope: 'col' } })))),
    body,
  ]);
  const renderRows = (expanded) => {
    clear(body);
    const visible = expanded ? windows : windows.slice(0, 10);
    body.append(...visible.map((win) => el('tr', {}, [
      el('td', { text: String(win.rank ?? '—') }),
      el('td', { text: `${win.start_x ?? win.start_index ?? '—'} ~ ${win.end_x ?? win.end_index ?? '—'}` }),
      el('td', { text: formatMetric(win.importance) }),
      el('td', { text: formatMetric(win.normalized_importance) }),
    ])));
  };
  renderRows(false);
  wrap.append(table);
  if (windows.length > 10) {
    let expanded = false;
    const toggle = el('button', {
      className: 'btn btn-ghost btn-sm',
      text: `展开其余 ${windows.length - 10} 个窗口`,
      attrs: { type: 'button', 'aria-expanded': 'false' },
      on: { click: () => {
        expanded = !expanded;
        renderRows(expanded);
        toggle.textContent = expanded ? '收起窗口明细' : `展开其余 ${windows.length - 10} 个窗口`;
        toggle.setAttribute('aria-expanded', String(expanded));
      } },
    });
    wrap.append(toggle);
  }
  return wrap;
}

/** 小标签 + 值的元信息块，使用共享 grid 布局。 */
function metaBlocks(items) {
  return el('div', { className: 'grid grid-3' }, items.map(([key, value]) =>
    el('div', {}, [
      el('p', { className: 'hint', text: key }),
      el('p', { text: String(value) }),
    ])));
}

function sampleMeta(sample, payload) {
  const items = [
    ['方法', methodLabel(payload?.method)],
    ['真实类别', sample.true_label ?? '—'],
    ['预测类别', sample.pred_label ?? '—'],
    ['预测正确', sample.correct === true ? '是' : sample.correct === false ? '否' : '—'],
    ['真实类别概率', formatMetric(sample.true_probability)],
    ['预测类别概率', formatMetric(sample.pred_probability)],
  ];
  if (sample.fold_index !== null && sample.fold_index !== undefined) items.push(['折', String(sample.fold_index)]);
  if (sample.sample_id) items.push(['Sample_ID', String(sample.sample_id)]);
  return metaBlocks(items);
}

/**
 * 渲染解释性面板。
 * options: { summary, artifacts, onLoadJson(downloadUrl) }
 */
export function renderExplainabilityPanel(container, { summary, artifacts = [], onLoadJson }) {
  clear(container);
  if (!summary) {
    container.append(el('p', { className: 'hint', text: '本次训练未生成单样品解释结果。' }));
    return;
  }
  if (summary.status !== 'ready') {
    container.append(el('p', { className: 'hint', text: `单样品解释不可用：${summary.reason || summary.status || '未知原因'}` }));
    return;
  }
  container.append(metaBlocks([
    ['方法', methodLabel(summary.method)],
    ['样品数', String(summary.sample_count ?? '—')],
    ['数据文件', summary.artifact || 'sample_feature_importance.json'],
  ]));
  const artifactName = summary.artifact || 'sample_feature_importance.json';
  const descriptor = artifacts.find((item) => item?.name === artifactName);
  const action = descriptor ? artifactActionState(descriptor) : { enabled: false, reason: '结果中未登记解释数据文件' };
  const status = el('p', { className: 'hint', attrs: { role: 'status' } });
  const retryHost = el('div', { className: 'row' });
  container.append(el('p', { className: 'hint', text: '完整数据通过已登记的 artifact 自动加载；加载完成后可在本地切换样品，不会重复请求。' }));
  if (!action.enabled) container.append(el('p', { className: 'error-text', text: `无法加载：${action.reason}` }));
  container.append(status, retryHost);
  const explorer = el('div', { className: 'explain-explorer' });
  container.append(explorer);

  const load = async () => {
    clear(retryHost);
    status.textContent = '正在下载并解析解释数据…';
    try {
      const payload = await onLoadJson(descriptor.download_url);
      status.textContent = '';
      renderExplorer(explorer, payload);
    } catch (error) {
      status.textContent = '';
      retryHost.append(
        el('p', { className: 'error-text', attrs: { role: 'alert' }, text: `解释数据加载失败：${error?.message || '未知错误'}` }),
        el('button', { className: 'btn btn-primary', text: '重新加载解释数据', attrs: { type: 'button' }, on: { click: load } }),
      );
    }
  };
  if (action.enabled) load();
}

function renderExplorer(container, payload) {
  clear(container);
  const samples = Array.isArray(payload?.samples) ? payload.samples : [];
  if (!samples.length) {
    container.append(el('p', { className: 'hint', text: '解释数据中没有样品。' }));
    return;
  }
  if (payload?.x_axis_warning?.message) {
    container.append(el('p', { className: 'hint', text: String(payload.x_axis_warning.message) }));
  }
  const select = el('select', { className: 'select', attrs: { id: 'v2-explain-sample', 'aria-label': '选择样品' } },
    samples.map((sample, index) => el('option', { text: sampleLabel(sample, index), attrs: { value: String(index) } })));
  const detail = el('div', { className: 'stack explain-detail' });
  const show = (index) => {
    const sample = samples[index];
    if (!sample) return;
    clear(detail);
    detail.append(sampleMeta(sample, payload));
    detail.append(el('div', { className: 'chart-card' }, [
      el('div', { className: 'scroll-x' }, [sampleChart(sample, payload)]),
      el('p', { className: 'hint', text: '蓝色曲线为当前样品谱线；红色为第一重要区间，橙色为其他重要区间。' }),
    ]));
    detail.append(windowTable(sample));
  };
  select.addEventListener('change', () => show(Number(select.value)));
  container.append(el('div', { className: 'field' }, [
    el('label', { className: 'label', attrs: { for: 'v2-explain-sample' }, text: '切换样品（本地切换，不重新下载）' }),
    select,
  ]));
  container.append(detail);
  show(0);
}
