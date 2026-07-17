/** SVG 图表：折线、分组柱状与刻度计算。渲染仅用于展示，可抽稀；下载数据不受影响。 */
import { svgEl, el } from './dom.js';

/** 浅色主题图表色板：白底上高对比、不过度荧光的蓝绿/蓝系与状态色。 */
export const CHART_COLORS = ['#2563eb', '#0f766e', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#65a30d', '#be185d'];

/** 图表内部装饰色：以 presentation attribute 内联，无 CSS 时也能在白色主题下正确显示。 */
const GRID_STROKE = '#e5e9f0';
const AXIS_STROKE = '#94a3b8';
const TICK_FILL = '#64748b';
const AXIS_LABEL_FILL = '#475569';
const SVG_BASE_STYLE = 'width:100%;height:auto;display:block;';

/** 展示抽稀：桶平均，保留首尾；不修改入参。 */
export function decimateSeries(xs, ys, maxPoints = 1200) {
  const x = Array.from(xs || []);
  const y = Array.from(ys || []);
  if (x.length !== y.length) throw new Error('xs 与 ys 长度不一致');
  if (x.length <= maxPoints || maxPoints < 3) return { xs: x, ys: y };
  const bucketCount = maxPoints - 2;
  const bucketSize = (x.length - 2) / bucketCount;
  const outX = [x[0]];
  const outY = [y[0]];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = 1 + Math.floor(bucket * bucketSize);
    const end = Math.min(x.length - 1, 1 + Math.floor((bucket + 1) * bucketSize));
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    for (let index = start; index < end; index += 1) {
      sumX += x[index];
      sumY += y[index];
      count += 1;
    }
    if (count > 0) {
      outX.push(sumX / count);
      outY.push(sumY / count);
    }
  }
  outX.push(x[x.length - 1]);
  outY.push(y[y.length - 1]);
  return { xs: outX, ys: outY };
}

/** 计算“好看”的刻度值，始终包含端点附近的整步骤值。 */
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  if (min > max) [min, max] = [max, min];
  const span = max - min;
  const roughStep = span / Math.max(1, count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const candidates = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude);
  const step = candidates.find((value) => value >= roughStep) ?? candidates[candidates.length - 1];
  const ticks = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toPrecision(12)));
  }
  return ticks.length ? ticks : [min, max];
}

export function formatTick(value) {
  if (!Number.isFinite(value)) return '';
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  return String(Number(value.toFixed(3)));
}

const CHART_WIDTH = 760;
const CHART_HEIGHT = 320;
const MARGIN = { top: 18, right: 16, bottom: 42, left: 64 };

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMin === domainMax) return () => (rangeMin + rangeMax) / 2;
  const factor = (rangeMax - rangeMin) / (domainMax - domainMin);
  return (value) => rangeMin + (value - domainMin) * factor;
}

/** 统一创建带 viewBox 与内联尺寸的 svg 画布：无 CSS 时也能在卡片内正确缩放。 */
function chartSvg({ title, description, fallbackLabel }) {
  const svg = svgEl('svg', {
    viewBox: `0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`,
    role: 'img',
    class: 'chart',
    'aria-label': description || title || fallbackLabel,
    preserveAspectRatio: 'xMidYMid meet',
    style: SVG_BASE_STYLE,
  });
  if (title) svg.append(svgEl('title', {}, title));
  if (description) svg.append(svgEl('desc', {}, description));
  return svg;
}

function gridLine(attrs) {
  return svgEl('line', { stroke: GRID_STROKE, 'stroke-width': 1, ...attrs });
}

function axisLine(attrs) {
  return svgEl('line', { stroke: AXIS_STROKE, 'stroke-width': 1.2, ...attrs });
}

function tickText(attrs, value) {
  return svgEl('text', { fill: TICK_FILL, 'font-size': 11, ...attrs }, formatTick(value));
}

function axisLabel(attrs, value) {
  return svgEl('text', { fill: AXIS_LABEL_FILL, 'font-size': 12, ...attrs }, value);
}

function emptyText(attrs, value) {
  return svgEl('text', { fill: TICK_FILL, 'font-size': 13, ...attrs }, value);
}

/**
 * 多序列折线图。
 * series: [{ name, xs, ys, color?, dashed? }]
 * options: { title, description, xLabel, yLabel, maxPoints }
 */
export function lineChart({ series, title = '', description = '', xLabel = '', yLabel = '', maxPoints = 1200 }) {
  const clean = (series || [])
    .map((item, index) => {
      const decimated = decimateSeries(item.xs, item.ys, maxPoints);
      return {
        name: item.name || `序列 ${index + 1}`,
        color: item.color || CHART_COLORS[index % CHART_COLORS.length],
        dashed: Boolean(item.dashed),
        xs: decimated.xs.filter((value, i) => Number.isFinite(value) && Number.isFinite(decimated.ys[i])),
        ys: decimated.ys.filter((value, i) => Number.isFinite(value) && Number.isFinite(decimated.xs[i])),
      };
    })
    .filter((item) => item.xs.length > 1);
  const svg = chartSvg({ title, description, fallbackLabel: '折线图' });
  if (!clean.length) {
    svg.append(emptyText({ x: CHART_WIDTH / 2, y: CHART_HEIGHT / 2, 'text-anchor': 'middle', class: 'chart-empty' }, '没有可绘制的数据'));
    return svg;
  }
  const allX = clean.flatMap((item) => item.xs);
  const allY = clean.flatMap((item) => item.ys);
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  let yMin = Math.min(...allY);
  let yMax = Math.max(...allY);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.05;
  yMin -= pad;
  yMax += pad;
  const sx = scaleLinear(xMin, xMax, MARGIN.left, CHART_WIDTH - MARGIN.right);
  const sy = scaleLinear(yMin, yMax, CHART_HEIGHT - MARGIN.bottom, MARGIN.top);

  for (const tick of niceTicks(yMin, yMax, 5)) {
    const y = sy(tick);
    svg.append(gridLine({ class: 'chart-grid', x1: MARGIN.left, x2: CHART_WIDTH - MARGIN.right, y1: y, y2: y }));
    svg.append(tickText({ class: 'chart-tick', x: MARGIN.left - 8, y: y + 4, 'text-anchor': 'end' }, tick));
  }
  for (const tick of niceTicks(xMin, xMax, 7)) {
    const x = sx(tick);
    svg.append(gridLine({ class: 'chart-grid chart-grid-vertical', 'stroke-dasharray': '2 4', x1: x, x2: x, y1: MARGIN.top, y2: CHART_HEIGHT - MARGIN.bottom }));
    svg.append(tickText({ class: 'chart-tick', x, y: CHART_HEIGHT - MARGIN.bottom + 18, 'text-anchor': 'middle' }, tick));
  }
  svg.append(axisLine({ class: 'chart-axis', x1: MARGIN.left, x2: CHART_WIDTH - MARGIN.right, y1: CHART_HEIGHT - MARGIN.bottom, y2: CHART_HEIGHT - MARGIN.bottom }));
  svg.append(axisLine({ class: 'chart-axis', x1: MARGIN.left, x2: MARGIN.left, y1: MARGIN.top, y2: CHART_HEIGHT - MARGIN.bottom }));
  if (xLabel) {
    svg.append(axisLabel({ class: 'chart-axis-label', x: (MARGIN.left + CHART_WIDTH - MARGIN.right) / 2, y: CHART_HEIGHT - 6, 'text-anchor': 'middle' }, xLabel));
  }
  if (yLabel) {
    svg.append(axisLabel({ class: 'chart-axis-label', x: 14, y: (MARGIN.top + CHART_HEIGHT - MARGIN.bottom) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(MARGIN.top + CHART_HEIGHT - MARGIN.bottom) / 2})` }, yLabel));
  }

  for (const item of clean) {
    const points = item.xs.map((value, index) => `${sx(value).toFixed(2)},${sy(item.ys[index]).toFixed(2)}`).join(' ');
    svg.append(svgEl('polyline', {
      points,
      fill: 'none',
      stroke: item.color,
      'stroke-width': 1.8,
      'stroke-linejoin': 'round',
      'stroke-linecap': 'round',
      'stroke-dasharray': item.dashed ? '5 4' : null,
      class: 'chart-line',
    }));
  }
  return svg;
}

export function chartLegend(series) {
  return el('ul', { className: 'legend' },
    (series || []).map((item, index) => el('li', { className: 'legend-item' }, [
      svgEl('svg', { width: 22, height: 10, viewBox: '0 0 22 10', 'aria-hidden': 'true', style: 'flex:none;' }, [
        svgEl('line', {
          x1: 0, x2: 22, y1: 5, y2: 5,
          stroke: item.color || CHART_COLORS[index % CHART_COLORS.length],
          'stroke-width': 2.5,
          'stroke-linecap': 'round',
          'stroke-dasharray': item.dashed ? '5 4' : null,
        }),
      ]),
      el('span', { text: item.name || `序列 ${index + 1}` }),
    ])));
}

/**
 * 分组柱状图：预测分布（真实 vs 预测）。
 * data: { labels, series: [{ name, values, color? }] }
 */
export function groupedBarChart({ labels = [], series = [], title = '', description = '', yLabel = '数量' }) {
  const svg = chartSvg({ title, description, fallbackLabel: '分组柱状图' });
  const maxValue = Math.max(0, ...series.flatMap((item) => item.values.map((value) => Number(value) || 0)));
  if (!labels.length || !series.length || maxValue <= 0) {
    svg.append(emptyText({ x: CHART_WIDTH / 2, y: CHART_HEIGHT / 2, 'text-anchor': 'middle', class: 'chart-empty' }, '没有可绘制的数据'));
    return svg;
  }
  const sy = scaleLinear(0, maxValue * 1.1, CHART_HEIGHT - MARGIN.bottom, MARGIN.top);
  const groupWidth = (CHART_WIDTH - MARGIN.left - MARGIN.right) / labels.length;
  const barWidth = Math.min(38, (groupWidth * 0.7) / series.length);
  for (const tick of niceTicks(0, maxValue * 1.1, 5)) {
    const y = sy(tick);
    svg.append(gridLine({ class: 'chart-grid', x1: MARGIN.left, x2: CHART_WIDTH - MARGIN.right, y1: y, y2: y }));
    svg.append(tickText({ class: 'chart-tick', x: MARGIN.left - 8, y: y + 4, 'text-anchor': 'end' }, tick));
  }
  labels.forEach((label, labelIndex) => {
    const groupX = MARGIN.left + labelIndex * groupWidth + groupWidth / 2;
    series.forEach((item, seriesIndex) => {
      const value = Number(item.values[labelIndex]) || 0;
      const x = groupX - (series.length * barWidth) / 2 + seriesIndex * barWidth;
      const y = sy(value);
      svg.append(svgEl('rect', {
        x: x + 1,
        y,
        width: Math.max(1, barWidth - 2),
        height: Math.max(0, CHART_HEIGHT - MARGIN.bottom - y),
        rx: 2,
        fill: item.color || CHART_COLORS[seriesIndex % CHART_COLORS.length],
      }));
      svg.append(tickText({ class: 'chart-tick', x: x + barWidth / 2, y: y - 4, 'text-anchor': 'middle' }, value));
    });
    svg.append(svgEl('text', { class: 'chart-tick', fill: TICK_FILL, 'font-size': 11, x: groupX, y: CHART_HEIGHT - MARGIN.bottom + 18, 'text-anchor': 'middle' }, String(label)));
  });
  svg.append(axisLine({ class: 'chart-axis', x1: MARGIN.left, x2: CHART_WIDTH - MARGIN.right, y1: CHART_HEIGHT - MARGIN.bottom, y2: CHART_HEIGHT - MARGIN.bottom }));
  svg.append(axisLine({ class: 'chart-axis', x1: MARGIN.left, x2: MARGIN.left, y1: MARGIN.top, y2: CHART_HEIGHT - MARGIN.bottom }));
  if (yLabel) {
    svg.append(axisLabel({ class: 'chart-axis-label', x: 14, y: (MARGIN.top + CHART_HEIGHT - MARGIN.bottom) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(MARGIN.top + CHART_HEIGHT - MARGIN.bottom) / 2})` }, yLabel));
  }
  return svg;
}
