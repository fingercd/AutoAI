/**
 * TEMPORARILY_HIDDEN: retained renderer only.  No current product route imports
 * this module or requests its payload until it is explicitly restored.
 * Native SVG renderer for model-feature-visualization-v1 payloads.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const KNOWN_PLOT_TYPES = new Set(['scatter', 'permutation', 'bar', 'dendrogram']);
const COLORS = ['#16a34a', '#3155b7', '#c1121f', '#f2c80f', '#7c3aed', '#0891b2', '#d97706', '#be185d'];
const SPLIT_LABELS = { train: 'Train', valid: 'Valid', test: 'Test', unknown: '未标注' };

function node(tag, options = {}, ...children) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined && options.text !== null) element.textContent = String(options.text);
  for (const [key, value] of Object.entries(options.attrs || {})) {
    if (value !== null && value !== undefined && value !== false) element.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child !== null && child !== undefined && child !== false) element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function svgNode(tag, attrs = {}, ...children) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) element.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child !== null && child !== undefined) element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function normalizeVisualizationPayload(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const status = ['ready', 'unsupported', 'unavailable'].includes(payload.status) ? payload.status : 'unavailable';
  const plots = Array.isArray(payload.plots)
    ? payload.plots.filter((plot) => plot && typeof plot === 'object' && KNOWN_PLOT_TYPES.has(plot.type))
    : [];
  return {
    schema_version: String(payload.schema_version || ''),
    status,
    reason: String(payload.reason || ''),
    warning: String(payload.warning || ''),
    model_type: String(payload.model_type || ''),
    plots,
  };
}

function scale(domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMin === domainMax) return () => (rangeMin + rangeMax) / 2;
  const factor = (rangeMax - rangeMin) / (domainMax - domainMin);
  return (value) => rangeMin + (value - domainMin) * factor;
}

function ticks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rough = span / Math.max(1, count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-12)));
  const step = [1, 2, 2.5, 5, 10].map((value) => value * magnitude).find((value) => value >= rough) || 10 * magnitude;
  const output = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) output.push(Number(value.toPrecision(12)));
  return output;
}

function format(value) {
  if (!Number.isFinite(value)) return '';
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  return String(Number(value.toFixed(3)));
}

function chartSvg(width, height, title, description) {
  const svg = svgNode('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: 'img',
    class: 'model-feature-chart',
    'aria-label': description || title,
    preserveAspectRatio: 'xMidYMid meet',
  });
  svg.append(svgNode('title', {}, title || '模型特征图'));
  if (description) svg.append(svgNode('desc', {}, description));
  return svg;
}

function axes(svg, { width, height, margin, xMin, xMax, yMin, yMax, xLabel, yLabel }) {
  const sx = scale(xMin, xMax, margin.left, width - margin.right);
  const sy = scale(yMin, yMax, height - margin.bottom, margin.top);
  for (const value of ticks(yMin, yMax, 5)) {
    const y = sy(value);
    svg.append(svgNode('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: '#e5e7eb', 'stroke-width': 1 }));
    svg.append(svgNode('text', { x: margin.left - 8, y: y + 4, 'text-anchor': 'end', fill: '#64748b', 'font-size': 11 }, format(value)));
  }
  for (const value of ticks(xMin, xMax, 6)) {
    const x = sx(value);
    svg.append(svgNode('line', { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, stroke: '#eef2f7', 'stroke-width': 1, 'stroke-dasharray': '3 4' }));
    svg.append(svgNode('text', { x, y: height - margin.bottom + 18, 'text-anchor': 'middle', fill: '#64748b', 'font-size': 11 }, format(value)));
  }
  if (xMin < 0 && xMax > 0) svg.append(svgNode('line', { x1: sx(0), x2: sx(0), y1: margin.top, y2: height - margin.bottom, stroke: '#94a3b8' }));
  if (yMin < 0 && yMax > 0) svg.append(svgNode('line', { x1: margin.left, x2: width - margin.right, y1: sy(0), y2: sy(0), stroke: '#94a3b8' }));
  svg.append(svgNode('line', { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, stroke: '#94a3b8', 'stroke-width': 1.2 }));
  svg.append(svgNode('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, stroke: '#94a3b8', 'stroke-width': 1.2 }));
  if (xLabel) svg.append(svgNode('text', { x: (margin.left + width - margin.right) / 2, y: height - 8, 'text-anchor': 'middle', fill: '#475569', 'font-size': 12 }, xLabel));
  if (yLabel) svg.append(svgNode('text', { x: 15, y: (margin.top + height - margin.bottom) / 2, 'text-anchor': 'middle', fill: '#475569', 'font-size': 12, transform: `rotate(-90 15 ${(margin.top + height - margin.bottom) / 2})` }, yLabel));
  return { sx, sy };
}

function paddedDomain(values) {
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const padding = (max - min) * 0.08;
  return [min - padding, max + padding];
}

export function scatterPlotDomain(plot) {
  const points = (Array.isArray(plot?.points) ? plot.points : [])
    .map((item) => ({ x: finite(item?.x), y: finite(item?.y) }))
    .filter((item) => item.x !== null && item.y !== null);
  const ellipse = (Array.isArray(plot?.ellipse?.points) ? plot.ellipse.points : [])
    .map((item) => ({ x: finite(item?.x), y: finite(item?.y) }))
    .filter((item) => item.x !== null && item.y !== null);
  const coordinates = [...points, ...ellipse];
  if (!coordinates.length) return null;
  const [xMin, xMax] = paddedDomain(coordinates.map((item) => item.x));
  const [yMin, yMax] = paddedDomain(coordinates.map((item) => item.y));
  return { xMin, xMax, yMin, yMax, points, ellipse };
}

function classColors(labels) {
  const unique = [...new Set(labels.map(String))].sort((a, b) => a.localeCompare(b, 'zh-CN', { numeric: true }));
  return new Map(unique.map((label, index) => [label, COLORS[index % COLORS.length]]));
}

function pointShape({ x, y, color, split, size = 5 }) {
  if (split === 'valid') return svgNode('rect', { x: x - size, y: y - size, width: size * 2, height: size * 2, rx: 1, fill: color, stroke: '#ffffff', 'stroke-width': 1 });
  if (split === 'test') return svgNode('path', { d: `M ${x} ${y - size - 1} L ${x + size + 1} ${y + size} L ${x - size - 1} ${y + size} Z`, fill: color, stroke: '#ffffff', 'stroke-width': 1 });
  return svgNode('circle', { cx: x, cy: y, r: size, fill: color, stroke: '#ffffff', 'stroke-width': 1 });
}

function legend(colors, splits = []) {
  const host = node('ul', { className: 'model-feature-legend' });
  for (const [label, color] of colors.entries()) {
    host.append(node('li', {}, node('span', { className: 'model-feature-swatch', attrs: { style: `background:${color}` } }), node('span', { text: label })));
  }
  for (const split of [...new Set(splits)].filter(Boolean)) {
    host.append(node('li', {}, node('span', { className: `model-feature-shape shape-${split}` }), node('span', { text: SPLIT_LABELS[split] || split })));
  }
  return host;
}

function scatterPlot(plot) {
  const points = (Array.isArray(plot.points) ? plot.points : []).map((item) => ({ ...item, x: finite(item.x), y: finite(item.y) })).filter((item) => item.x !== null && item.y !== null);
  if (!points.length) return node('p', { className: 'section-note', text: '没有可绘制的二维坐标。' });
  const width = 600; const height = 330; const margin = { top: 18, right: 18, bottom: 50, left: 58 };
  const domain = scatterPlotDomain(plot);
  const { xMin, xMax, yMin, yMax } = domain;
  const svg = chartSvg(width, height, plot.title, plot.description);
  const { sx, sy } = axes(svg, { width, height, margin, xMin, xMax, yMin, yMax, xLabel: plot.x_label, yLabel: plot.y_label });
  if (plot.ellipse?.points?.length) {
    const ellipsePoints = plot.ellipse.points.map((item) => `${sx(Number(item.x)).toFixed(2)},${sy(Number(item.y)).toFixed(2)}`).join(' ');
    svg.append(svgNode('polyline', { points: ellipsePoints, fill: 'none', stroke: '#94a3b8', 'stroke-width': 1.4, 'stroke-dasharray': '5 4' }));
  }
  const colors = classColors(points.map((item) => item.label));
  for (const item of points) {
    const mark = pointShape({ x: sx(item.x), y: sy(item.y), color: colors.get(String(item.label)), split: item.split });
    mark.append(svgNode('title', {}, `${item.name || `S${Number(item.index) + 1}`} · ${item.label} · ${SPLIT_LABELS[item.split] || item.split} · (${format(item.x)}, ${format(item.y)})`));
    svg.append(mark);
  }
  return node('div', { className: 'model-feature-chart-body' }, node('div', { className: 'model-feature-svg-scroll' }, svg), legend(colors, points.map((item) => item.split)));
}

function permutationPlot(plot) {
  const points = (Array.isArray(plot.points) ? plot.points : []).map((item) => ({ ...item, correlation: finite(item.correlation), r2: finite(item.r2), q2: finite(item.q2) })).filter((item) => item.correlation !== null && item.r2 !== null && item.q2 !== null);
  if (!points.length) return node('p', { className: 'section-note', text: '没有可绘制的置换检验数据。' });
  const width = 600; const height = 330; const margin = { top: 18, right: 18, bottom: 50, left: 58 };
  const [xMin, xMax] = paddedDomain(points.map((item) => item.correlation));
  const [yMin, yMax] = paddedDomain(points.flatMap((item) => [item.r2, item.q2]));
  const svg = chartSvg(width, height, plot.title, plot.description);
  const { sx, sy } = axes(svg, { width, height, margin, xMin, xMax, yMin, yMax, xLabel: plot.x_label, yLabel: plot.y_label });
  for (const item of points) {
    for (const series of [{ key: 'r2', color: '#16a34a', label: 'R²Y' }, { key: 'q2', color: '#3155b7', label: 'Q²' }]) {
      const mark = svgNode(series.key === 'r2' ? 'circle' : 'rect', series.key === 'r2'
        ? { cx: sx(item.correlation), cy: sy(item[series.key]), r: item.original ? 7 : 4, fill: series.color, stroke: item.original ? '#111827' : '#ffffff', 'stroke-width': item.original ? 2 : 1 }
        : { x: sx(item.correlation) - (item.original ? 7 : 4), y: sy(item[series.key]) - (item.original ? 7 : 4), width: item.original ? 14 : 8, height: item.original ? 14 : 8, fill: series.color, stroke: item.original ? '#111827' : '#ffffff', 'stroke-width': item.original ? 2 : 1 });
      mark.append(svgNode('title', {}, `${item.original ? '原模型' : '置换模型'} · ${series.label} ${format(item[series.key])}`));
      svg.append(mark);
    }
  }
  const info = Number.isFinite(Number(plot.q2_p_value)) ? `经验 Q² p=${Number(plot.q2_p_value).toFixed(4)}；${plot.permutation_count || points.length - 1} 次置换` : `${plot.permutation_count || points.length - 1} 次置换`;
  const colors = new Map([['R²Y', '#16a34a'], ['Q²', '#3155b7']]);
  return node('div', { className: 'model-feature-chart-body' }, node('div', { className: 'model-feature-svg-scroll' }, svg), legend(colors), node('p', { className: 'section-note', text: info }));
}

function barPlot(plot) {
  const limit = Math.max(1, Math.min(12, Number(plot.display_limit) || 12));
  const items = (Array.isArray(plot.items) ? plot.items : []).map((item) => ({ ...item, value: finite(item.value) })).filter((item) => item.value !== null).slice(0, limit);
  if (!items.length) return node('p', { className: 'section-note', text: '没有可绘制的 VIP 数据。' });
  const width = 600; const height = 330; const margin = { top: 18, right: 18, bottom: 92, left: 58 };
  const maxValue = Math.max(1, ...items.map((item) => item.value), Number(plot.threshold) || 0) * 1.1;
  const svg = chartSvg(width, height, plot.title, plot.description);
  const { sy } = axes(svg, { width, height, margin, xMin: 0, xMax: items.length, yMin: 0, yMax: maxValue * 1.08, xLabel: plot.x_label, yLabel: plot.y_label });
  const groupWidth = (width - margin.left - margin.right) / items.length;
  items.forEach((item, index) => {
    const x = margin.left + index * groupWidth + groupWidth * 0.14;
    const barWidth = groupWidth * 0.72;
    const y = sy(item.value);
    svg.append(svgNode('rect', { x, y, width: barWidth, height: height - margin.bottom - y, fill: '#16a34a', rx: 2 }, svgNode('title', {}, `${item.label}: ${format(item.value)}`)));
    const labelX = x + barWidth / 2;
    const labelY = height - margin.bottom + 12;
    svg.append(svgNode('text', { x: labelX, y: labelY, fill: '#475569', 'font-size': 10, 'text-anchor': 'end', transform: `rotate(-55 ${labelX} ${labelY})` }, String(item.label)));
  });
  const threshold = finite(plot.threshold);
  if (threshold !== null) {
    svg.append(svgNode('line', { x1: margin.left, x2: width - margin.right, y1: sy(threshold), y2: sy(threshold), stroke: '#d97706', 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }));
  }
  return node('div', { className: 'model-feature-chart-body' }, node('div', { className: 'model-feature-svg-scroll' }, svg), node('p', { className: 'section-note', text: `显示 VIP 最高的 ${items.length} 个特征${threshold !== null ? `；虚线为 VIP=${threshold}` : ''}。` }));
}

function dendrogramPlot(plot) {
  const leaves = Array.isArray(plot.leaves) ? plot.leaves : [];
  const segments = Array.isArray(plot.segments) ? plot.segments : [];
  if (!leaves.length || !segments.length) return node('p', { className: 'section-note', text: '没有可绘制的聚类树数据。' });
  const width = Math.max(760, 100 + leaves.length * 42); const height = 420; const margin = { top: 24, right: 20, bottom: 135, left: 66 };
  const allX = segments.flatMap((segment) => segment.x || []).map(Number).filter(Number.isFinite);
  const allY = segments.flatMap((segment) => segment.y || []).map(Number).filter(Number.isFinite);
  const xMin = Math.min(...allX); const xMax = Math.max(...allX); const yMin = 0; const yMax = Math.max(1e-9, ...allY) * 1.06;
  const svg = chartSvg(width, height, plot.title, plot.description);
  const { sx, sy } = axes(svg, { width, height, margin, xMin, xMax, yMin, yMax, xLabel: '', yLabel: plot.y_label });
  for (const segment of segments) {
    const xs = (segment.x || []).map(Number); const ys = (segment.y || []).map(Number);
    if (xs.length !== ys.length || xs.some((value) => !Number.isFinite(value)) || ys.some((value) => !Number.isFinite(value))) continue;
    svg.append(svgNode('polyline', { points: xs.map((value, index) => `${sx(value)},${sy(ys[index])}`).join(' '), fill: 'none', stroke: '#3155b7', 'stroke-width': 1.8 }));
  }
  const colors = classColors(leaves.map((leaf) => leaf.label));
  for (const leaf of leaves) {
    const x = sx(Number(leaf.x)); const y = height - margin.bottom + 10;
    svg.append(svgNode('text', { x, y, fill: colors.get(String(leaf.label)), 'font-size': 10, 'font-weight': 600, 'text-anchor': 'end', transform: `rotate(-60 ${x} ${y})` }, String(leaf.name || `S${Number(leaf.index) + 1}`)));
  }
  return node('div', { className: 'model-feature-chart-body' }, node('div', { className: 'model-feature-svg-scroll' }, svg), legend(colors), plot.sampled ? node('p', { className: 'section-note', text: `为控制 O(n²) 计算和标签可读性，已从 ${plot.source_sample_count} 条曲线中按类别与分区确定性抽样。` }) : null);
}

function renderPlot(plot) {
  const renderer = { scatter: scatterPlot, permutation: permutationPlot, bar: barPlot, dendrogram: dendrogramPlot }[plot.type];
  const description = plot.type === 'bar'
    ? `展示 VIP 最高的 ${Math.max(1, Math.min(12, Array.isArray(plot.items) ? plot.items.length : 12))} 个特征；VIP>1 是常用启发式，不等于统计显著性。`
    : plot.description;
  const card = node('article', { className: 'result-card model-feature-card' },
    node('h3', { text: plot.title || '模型特征图' }),
    description ? node('p', { className: 'section-note', text: description }) : null,
    renderer(plot),
  );
  return card;
}

export function renderModelFeatureVisualization(payload) {
  const normalized = normalizeVisualizationPayload(payload);
  const host = node('div', { className: 'model-feature-visualization' });
  if (!normalized) {
    host.append(node('p', { className: 'section-note', text: '本次 Run 没有模型特征可视化产物。' }));
    return host;
  }
  if (normalized.status !== 'ready' || !normalized.plots.length) {
    host.append(node('div', { className: 'result-card wide-card model-feature-empty' },
      node('h3', { text: normalized.status === 'unsupported' ? '当前模型/口径暂不支持' : '模型特征图不可用' }),
      node('p', { className: 'section-note', text: normalized.reason || '没有可绘制的数据。' }),
    ));
    return host;
  }
  host.append(node('div', { className: 'model-feature-grid' }, normalized.plots.map(renderPlot)));
  return host;
}
