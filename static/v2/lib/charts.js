/**
 * charts.js —— v2 前端的纯 SVG 图表库（折线图、分组柱状图、图例、刻度计算）。
 *
 * 模块职责：
 * - 不依赖任何第三方图表库，直接用 dom.js 的 svgEl/el 手工拼装 SVG，
 *   保持 v2 前端"原生 JS、无构建步骤"的约束；
 * - 结果页用于绘制光谱/色谱曲线、预测分布（真实 vs 预测）等图形。
 *
 * 关键设计约束：
 * - 渲染仅用于展示：允许对长序列做桶平均抽稀（默认 1200 点上限），
 *   避免上万点曲线把 DOM 拖垮；抽稀只发生在显示层，下载数据不受影响；
 * - 所有装饰色以 presentation attribute 内联写入 SVG，即使外部 CSS
 *   未加载，图表在白色主题下仍可读；
 * - 每个 svg 都带 viewBox + 内联 width/height 样式，并写入 <title>/<desc>
 *   与 role="img"，保证缩放正确与基本的可访问性。
 */
import { svgEl, el } from './dom.js';

/** 浅色主题图表色板：白底上高对比、不过度荧光的蓝绿/蓝系与状态色。 */
export const CHART_COLORS = ['#2563eb', '#0f766e', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#65a30d', '#be185d'];

/** 图表内部装饰色：以 presentation attribute 内联，无 CSS 时也能在白色主题下正确显示。 */
const GRID_STROKE = '#e5e9f0';
const AXIS_STROKE = '#94a3b8';
const TICK_FILL = '#64748b';
const AXIS_LABEL_FILL = '#475569';
const SVG_BASE_STYLE = 'width:100%;height:auto;display:block;';

/**
 * 展示抽稀（decimation）：把超长序列压缩到不超过 maxPoints 个点。
 *
 * 算法：保留首、尾两个原始点，中间部分均匀分成 maxPoints-2 个桶，
 * 每个桶输出 x、y 的算术平均值（桶平均法）。
 * 为什么保留首尾：光谱/色谱曲线的端点位置（波数/保留时间范围）是
 * 业务上有意义的信息，抽稀后仍要精确展示。
 * 为什么用桶平均而不是简单隔点取样：平均能抑制尖峰被随机丢弃
 * 造成的视觉误导，展示曲线更平滑、更接近真实包络。
 *
 * 不修改入参（先 Array.from 拷贝），纯函数。
 *
 * @param {Array<number>} xs X 坐标序列（如拉曼位移 / HPLC 时间）。
 * @param {Array<number>} ys 强度序列，与 xs 等长。
 * @param {number} [maxPoints=1200] 抽稀后的最大点数；小于 3 视为不抽稀。
 * @returns {{xs: number[], ys: number[]}} 抽稀后的新数组。
 * @throws {Error} xs 与 ys 长度不一致时抛出（数据契约错误，应尽早暴露）。
 */
export function decimateSeries(xs, ys, maxPoints = 1200) {
  const x = Array.from(xs || []);
  const y = Array.from(ys || []);
  if (x.length !== y.length) throw new Error('xs 与 ys 长度不一致');
  // 未超限（或调用方显式关闭抽稀）时原样返回拷贝
  if (x.length <= maxPoints || maxPoints < 3) return { xs: x, ys: y };
  // 中间可分配的点数 = 总上限减去首尾两个保留点
  const bucketCount = maxPoints - 2;
  const bucketSize = (x.length - 2) / bucketCount;
  const outX = [x[0]];
  const outY = [y[0]];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    // 桶区间 [start, end)：用 floor 切分，最后一个桶 end 顶到倒数第 2 个点
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
      // 桶平均点；空桶（区间退化）直接跳过，避免产生 NaN
      outX.push(sumX / count);
      outY.push(sumY / count);
    }
  }
  outX.push(x[x.length - 1]);
  outY.push(y[y.length - 1]);
  return { xs: outX, ys: outY };
}

/**
 * 计算"好看"的坐标轴刻度值（nice numbers 算法）。
 * 从 {1, 2, 2.5, 5, 10} × 10^n 中选第一个不小于粗略步长的值作为步长，
 * 再从小到大枚举落在 [min, max] 内的整步长刻度，保证刻度值是可读的
 * 整数/有限小数，而不是 0.3333… 这类任意值。
 * toPrecision(12) 用于消除浮点累加产生的 0.30000000000000004 之类尾差。
 *
 * @param {number} min 数据最小值（允许 min > max，内部会交换）。
 * @param {number} max 数据最大值。
 * @param {number} [count=5] 期望刻度数量（近似值，实际可能 ±1）。
 * @returns {number[]} 刻度值数组；非法输入返回 []，min===max 返回单点。
 */
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

/**
 * 刻度文本格式化：大数/小数用科学计数法避免超长字符串，
 * 其余保留最多 3 位小数（经 Number() 去除尾零）。
 * 非法值返回空串——刻度位置画一个空文本比画 "NaN" 更体面。
 */
export function formatTick(value) {
  if (!Number.isFinite(value)) return '';
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  return String(Number(value.toFixed(3)));
}

// 画布固定逻辑尺寸：通过 viewBox 等比缩放，CSS 只控制显示宽度
const CHART_WIDTH = 760;
const CHART_HEIGHT = 320;
// 左边距最大（64）：为 Y 轴刻度文本（可能含科学计数法）预留空间
const MARGIN = { top: 18, right: 16, bottom: 42, left: 64 };

/**
 * 线性映射函数工厂：把 [domainMin, domainMax] 映射到像素区间
 * [rangeMin, rangeMax]。定义域退化为单点时返回中点常函数，
 * 避免除零产生 NaN 坐标。
 */
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
 * 多序列折线图（光谱/色谱曲线、训练曲线等）。
 *
 * 数据清洗流程（clean）：
 * 1. 每条序列先按 maxPoints 抽稀（仅影响显示）；
 * 2. 成对过滤掉 x 或 y 非有限值的点（后端数据可能含 NaN 占位）；
 * 3. 不足 2 个有效点的序列整条丢弃（单点无法成线）。
 * 全部序列为空时返回带"没有可绘制的数据"提示的空图，而不是空 svg。
 *
 * Y 轴处理细节：
 * - 所有序列共用同一坐标系（合并 min/max），便于多曲线直接对比；
 * - yMin===yMax（平线）时人为扩 ±1，避免退化；
 * - 上下各留 5% padding，曲线不贴边。
 *
 * @param {Object} options
 * @param {Array<{name: string, xs: number[], ys: number[], color?: string, dashed?: boolean}>} options.series
 *        序列数组；color 缺省按序取 CHART_COLORS 循环，dashed 画虚线（常用于"参考/原始"对照）。
 * @param {string} [options.title] 图题（写入 <title>，供无障碍与提示）。
 * @param {string} [options.description] 图的描述（写入 <desc>）。
 * @param {string} [options.xLabel] X 轴标题（如 "Raman shift / cm⁻¹"）。
 * @param {string} [options.yLabel] Y 轴标题（如 "Intensity"）。
 * @param {number} [options.maxPoints=1200] 单序列展示抽稀上限。
 * @returns {SVGElement} 可直接 append 的 <svg> 节点。
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

/**
 * 图例：为 lineChart 的序列生成配套图例（色块线 + 名称的 <ul>）。
 * 颜色/虚线规则与 lineChart 完全一致（同一下标同色），保证图图对应；
 * 小色块本身是独立 22×10 的 SVG，aria-hidden 避免读屏器重复朗读。
 * @param {Array<{name?: string, color?: string, dashed?: boolean}>} series
 * @returns {HTMLElement}
 */
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
 * 分组柱状图：预测分布（真实 vs 预测）等"每类多根柱子"的场景。
 *
 * 布局算法：把绘图区按 labels 数量均分为若干组，组内再按 series 数量
 * 并排放柱；柱宽取 38px 上限与组宽 70% 均分值的较小者，标签很多时
 * 自动变窄而不溢出。Y 轴从 0 开始（计数类数据截断会夸大差异），
 * 顶部留 10% 余量给柱顶数值标签。
 * 空数据（无标签/无序列/全零）时返回提示空图。
 *
 * @param {Object} options
 * @param {Array<string|number>} [options.labels] X 轴分组标签（如类别名）。
 * @param {Array<{name: string, values: number[], color?: string}>} [options.series]
 *        每个序列一根柱/组，values 与 labels 等长。
 * @param {string} [options.title]
 * @param {string} [options.description]
 * @param {string} [options.yLabel='数量']
 * @returns {SVGElement}
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
