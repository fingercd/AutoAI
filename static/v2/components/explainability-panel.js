/**
 * 模块说明：单样品可解释性面板（static/v2/components/explainability-panel.js）
 * =================================================================
 * 职责：在 v2 结果页渲染"单样品解释"区块——先展示 run-result-v1 摘要中的
 * 方法/样品数等元信息，再通过已登记的 artifact 自动下载完整的
 * `sample_feature_importance.json`，下载后支持在本地切换样品，查看
 * 谱线 + 重要性窗口叠加图（SVG）与窗口明细表格。
 *
 * 在系统中的位置：
 *   - 属于并行新前端 `static/v2/index.html` 的结果页组件，由结果页主控脚本调用
 *     `renderExplainabilityPanel(container, { summary, artifacts, onLoadJson })`。
 *   - `summary` 来自 `GET /api/training/runs/{run_id}/result`（run-result-v1 契约）
 *     的 explainability 摘要；`artifacts` 是同一份结果的 artifacts[] 描述数组；
 *     `onLoadJson(downloadUrl)` 由主控注入，负责带鉴权地下载并解析 JSON。
 *
 * 协作模块：
 *   - `../lib/dom.js`：`el`/`clear`/`svgEl`（创建 SVG 元素，需走 XML 命名空间）。
 *   - `../lib/format.js`：`formatMetric` 统一数值显示（空值占位、小数位数）。
 *   - `./artifacts.js`：复用 `artifactActionState` 的下载可行性判断（黑名单 +
 *     安全 URL + 后端 reason），保证解释数据下载与 artifact 列表的口径一致。
 *   - `../lib/charts.js`：`niceTicks`/`formatTick` 生成坐标轴刻度。
 *
 * 关键设计约束（对应后端可解释性规则）：
 *   - 后端为不同模型族生成不同方法的重要性数据：传统模型与 pca_mlp、
 *     cnn_transformer1d 使用"窗口遮挡后的真实类别 Log-loss 增量"
 *     （masked_loss - original_loss = log(p_before/p_after)，正值表示遮挡后真实
 *     类别置信度受损）；cnn1d 等五个 1D CNN 使用 1D Grad-CAM；dscarnet 使用
 *     AggMap/PCA 双通路 2D Grad-CAM 回投到 1D 特征。METHOD_LABELS 即这些方法
 *     标识到中文名的映射，未知方法原样显示标识符，不臆造名称。
 *   - 重要性窗口按 rank 排序展示；primary_segment（第一重要区间）在图上用红色、
 *     其余窗口按归一化重要性用不同透明度的橙色。
 *   - 兼容性：旧产物可能缺少样品级 `curve`/`sample_x_axis` 字段，
 *     resolveSampleSeries 会回退到全局 `x_axis`/`baseline_curve`，不直接报错。
 *   - 组件不自己发请求：下载经 onLoadJson 注入，失败时给出可重试按钮。
 */
import { el, clear, svgEl } from '../lib/dom.js';
import { formatMetric } from '../lib/format.js';
import { artifactActionState } from './artifacts.js';
import { niceTicks, formatTick } from '../lib/charts.js';

/**
 * 后端解释方法标识 → 中文展示名。
 * - sample_occlusion_*：窗口遮挡 Log-loss 增量（传统模型 / pca_mlp / cnn_transformer1d）。
 * - gradcam_1d：1D Grad-CAM（cnn1d/cnn1d_se/resnet1d/inception1d/tcn1d）。
 * - dscarnet_dual_2d_gradcam：DSCARNet 的 SAR/CAR 双通路 2D Grad-CAM 回投 1D。
 */
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

/**
 * 方法标识转中文名；未收录的标识原样返回（不臆造名称），空值显示占位符。
 * @param {string} method - payload.method / summary.method。
 * @returns {string} 展示用方法名。
 */
function methodLabel(method) {
  return METHOD_LABELS[method] || method || '—';
}

/**
 * 样品下拉框选项文案：预测对错符号 + 样品名 + 真实/预测类别。
 * `correct === false` 才打 ✗（其余含 undefined 一律打 ✓ 由后端字段语义决定，
 * 摘要中 correct 缺省视为非错误样品）；无名样品用"样品 N"占位（position 为 0 基）。
 * @param {Object} sample - 单样品解释记录。
 * @param {number} position - 样品在 samples 数组中的下标。
 * @returns {string} 选项文本。
 */
function sampleLabel(sample, position) {
  const mark = sample.correct === false ? '✗' : '✓';
  const name = sample.name || `样品 ${position + 1}`;
  return `${mark} ${name}（真实 ${sample.true_label ?? '—'} → 预测 ${sample.pred_label ?? '—'}）`;
}

/** 画布尺寸与边距（SVG viewBox 坐标系单位，实际显示宽度由 CSS 100% 自适应）。 */
const W = 760;
const H = 300;
const M = { top: 16, right: 16, bottom: 40, left: 60 };

/**
 * 解析"当前样品"的谱线与 X 轴（导出供测试复用）。
 * 优先使用当前样品自己的谱线与 X 轴；旧产物缺字段时才回退到全局轴和基线谱线。
 * 仅当 sample_x_axis 长度与曲线一致时才采用样品级轴，否则回退全局 x_axis——
 * 长度不匹配说明数据来自不同网格（例如独立测试集重采样异常），强行使用会错位。
 * @param {Object} sample - 单样品记录，可含 curve / sample_x_axis。
 * @param {Object} payload - 完整 sample_feature_importance.json，含 x_axis / baseline_curve。
 * @returns {{curve: number[], xAxis: number[]}} 数值化后的曲线与 X 轴（均可为空数组）。
 */
export function resolveSampleSeries(sample, payload) {
  const sampleCurve = Array.isArray(sample?.curve) ? sample.curve.map(Number) : [];
  const baselineCurve = Array.isArray(payload?.baseline_curve) ? payload.baseline_curve.map(Number) : [];
  const curve = sampleCurve.length ? sampleCurve : baselineCurve;
  const sampleXAxis = Array.isArray(sample?.sample_x_axis) ? sample.sample_x_axis.map(Number) : [];
  const payloadXAxis = Array.isArray(payload?.x_axis) ? payload.x_axis.map(Number) : [];
  const xAxis = sampleXAxis.length === curve.length ? sampleXAxis : payloadXAxis;
  return { curve, xAxis };
}

/**
 * 判断某个窗口是否为"第一重要区间"（primary segment）。
 * 优先按 rank 相等判断（新产物带 rank 字段）；旧产物没有 rank 时退化为
 * start_index/end_index 坐标相等。两端字段都缺失时返回 false，避免误标红。
 * @param {Object} win - 窗口对象（rank/start_index/end_index/start_x/end_x/...）。
 * @param {Object|null} primary - primary_segment 或 top_segments[0]。
 * @returns {boolean}
 */
function isPrimaryWindow(win, primary) {
  if (!primary) return false;
  if (win.rank !== undefined && primary.rank !== undefined) return Number(win.rank) === Number(primary.rank);
  return Number(win.start_index) === Number(primary.start_index)
    && Number(win.end_index) === Number(primary.end_index);
}

/** 谱线 + 重要性窗口叠加：第一重要区间使用红色，其余窗口按重要性使用橙色。
 *
 * 绘制逻辑（纯手写 SVG，不依赖图表库）：
 *   1. resolveSampleSeries 取曲线与 X 轴；X 轴缺失/长度不符时用点序号 0..n-1 兜底。
 *   2. 极端情况：曲线为空但有窗口时，X 范围改由窗口坐标决定，保证窗口仍可见。
 *   3. X 范围非法（非有限值或零宽度）时渲染"没有可绘制的解释数据"提示并返回，
 *      避免后续除零产生 NaN 坐标。
 *   4. Y 为零高度（水平线曲线）时向上下各扩 1 个单位，防止除零。
 *   5. sx/sy 为数据坐标 → 画布坐标的线性映射；窗口矩形透明度随
 *      normalized_importance（钳制到 0..1）线性加深，primary 用红色系、其余橙色系。
 *   6. 曲线以 polyline 逐点连接；n<=1 时不画线（单点无法成线）。
 *   可访问性：svg 带 role="img"、aria-label 与 <title>/<desc>，供屏幕阅读器朗读。
 * @param {Object} sample - 当前选中样品记录。
 * @param {Object} payload - 完整解释数据。
 * @returns {SVGSVGElement}
 */
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

/**
 * 窗口明细表格：按 rank 升序展示 rank / X 区间 / importance / 归一化值。
 * 窗口超过 10 个时默认只显示前 10 个，附"展开/收起"按钮（aria-expanded 同步），
 * 避免某些模型输出大量窗口时页面过长；无窗口时显示提示文案。
 * X 区间优先显示真实坐标 start_x/end_x，旧产物缺字段时回退 start_index/end_index。
 * @param {Object} sample - 当前选中样品记录。
 * @returns {HTMLElement} 表格外层容器（含可能的展开按钮）。
 */
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

/** 小标签 + 值的元信息块，使用共享 grid 布局。
 * @param {Array<[string, *]>} items - [标签, 值] 二元组列表，值统一 String 化。
 * @returns {HTMLElement}
 */
function metaBlocks(items) {
  return el('div', { className: 'grid grid-3' }, items.map(([key, value]) =>
    el('div', {}, [
      el('p', { className: 'hint', text: key }),
      el('p', { text: String(value) }),
    ])));
}

/**
 * 当前样品的元信息块：方法、真实/预测类别、是否预测正确、两类概率；
 * CV 折次（fold_index）与 Sample_ID 仅在存在时追加——holdout 口径没有 fold_index，
 * 旧产物可能没有 sample_id，动态拼接避免显示无意义的占位行。
 * @param {Object} sample - 当前选中样品记录。
 * @param {Object} payload - 完整解释数据（取 method）。
 * @returns {HTMLElement}
 */
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
 * 渲染解释性面板（本组件唯一入口）。
 *
 * 流程：
 *   1. summary 缺失 → 提示"未生成单样品解释结果"（传统模型之外的旧 Run 或解释未启用）。
 *   2. summary.status !== 'ready' → 透传后端 reason/status 作为不可用原因，不猜原因。
 *   3. 展示方法/样品数/数据文件元信息；按 summary.artifact（默认
 *      sample_feature_importance.json）在 artifacts[] 中找到登记描述，
 *      用 artifactActionState 判定可否下载——与产物列表同一条白名单/安全规则。
 *   4. 可下载则立即自动 load()：经主控注入的 onLoadJson 下载并解析 JSON，
 *      成功后交给 renderExplorer；失败时渲染错误与"重新加载"按钮（重试复用 load）。
 *   说明文案强调"加载完成后本地切换样品，不会重复请求"——完整数据一次下载、
 *   前端内存中切换，符合结果页减少重复请求的设计。
 *
 * @param {HTMLElement} container - 挂载点。
 * @param {Object} options
 * @param {Object|null} options.summary - run-result-v1 的 explainability 摘要。
 * @param {Array} [options.artifacts=[]] - 同一结果的 artifacts[] 描述。
 * @param {Function} options.onLoadJson - (downloadUrl) => Promise<Object>，由主控注入。
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

/**
 * 数据加载完成后的样品浏览器：样品下拉框 + 元信息 + 谱线窗口图 + 窗口明细表。
 * 切换样品只重渲染 detail 区域（本地数据，不再请求网络）；初始展示第 0 个样品。
 * 若 payload 带 x_axis_warning.message（例如独立测试集轴与主数据不一致的告警），
 * 在下拉框之前原样透传展示。
 * @param {HTMLElement} container - 面板内的 explorer 容器。
 * @param {Object} payload - 解析后的 sample_feature_importance.json。
 */
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
