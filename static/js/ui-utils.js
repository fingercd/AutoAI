/**
 * 模块说明：SpecAutoAI 前端通用 UI 工具集（ui-utils.js）。
 *
 * 职责：
 * - 提供与具体页面无关的纯函数 / 轻量 DOM 工具：自然排序、HPLC 行号范围校验、
 *   曲线编号表达式解析、Sample_ID 折叠表格渲染、指标与时间格式化等。
 * - 被 `static/index.html` 与经典前端模块使用；
 *   前端纯函数有专项契约测试（backend/tests 中的前端纯函数测试），因此函数签名与返回结构属于受保护契约，不能随意变更。
 *
 * 关键设计约束：
 * - 校验规则必须与后端预处理契约一致：例如 HPLC 行号是 1 基、首尾包含，
 *   终止行留空才按检测出的完整点数处理，前端禁止静默截断越界值（与后端 `validateHplcRowRange` 对应接口行为一致）。
 * - 大量 Sample_ID（可能上千）渲染时必须分帧插入，避免一次性阻塞主线程。
 *
 * 协作模块：api-client.js（数据获取）；页面内联脚本通过 window.SpecAutoAIUI 全局桥接调用本模块。
 */

/** 折叠状态下默认展示的 Sample_ID 行数上限。 */
export const DEFAULT_SAMPLE_ID_LIMIT = 10;
/** 曲线编号表达式允许选择的最大曲线条数，防止用户输入超大区间把内存打爆。 */
export const MAX_CURVE_SELECTION = 5000;
/** 展开 Sample_ID 列表时每个动画帧插入的行数，用于分帧渲染。 */
const SAMPLE_ID_BATCH_SIZE = 200;

/**
 * 中文环境下的“自然排序”比较器：
 * numeric: true 让 "sample2" 排在 "sample10" 之前（按数字段比较），
 * sensitivity: 'base' 忽略大小写与重音差异。
 * 单例复用是因为 Intl.Collator 构造相对昂贵，且排序规则全模块一致。
 */
const naturalCollator = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });

/**
 * 自然排序比较函数，供 Array.prototype.sort 使用。
 * 入参允许为 null/undefined，统一归一化成字符串再比较，避免调用方逐个判空。
 * @param {*} first 左值
 * @param {*} second 右值
 * @returns {number} 负数/0/正数，符合 sort 比较器约定
 */
export const naturalCompare = (first, second) => (
  naturalCollator.compare(String(first ?? ''), String(second ?? ''))
);

/**
 * 校验 HPLC 行号范围选择，契约与后端 `/api/preprocess/hplc` 保持一致：
 * - 行号为 1 基、首尾包含（第 1–4000 行 = 4000 个点）；
 * - pointCount 必须来自当前批次文件检测，且不少于 2（少于 2 无法构成有效截取区间）；
 * - 起始行/终止行留空时分别取默认值 1 和完整点数——只有“留空”才用完整点数，禁止静默截断越界值；
 * - 起止都必须落在 `1..pointCount`，起始行不得大于终止行。
 * @param {string|number|null} startValue 用户输入的起始行（可空）
 * @param {string|number|null} endValue 用户输入的终止行（可空）
 * @param {number} pointCount 当前批次检测出的公共点数
 * @returns {{startRow: number, endRow: number, pointCount: number}} 归一化后的 1 基闭区间及实际点数，
 *          可直接用于 multipart 请求参数和前端“实际输出点数”摘要展示。
 * @throws {Error} 任一约束不满足时抛出带中文提示的 Error，由 UI 层展示。
 */
export function validateHplcRowRange(startValue, endValue, pointCount) {
  const limit = Number(pointCount);
  if (!Number.isInteger(limit) || limit < 2) throw new Error('请先完成 HPLC 文件点数检测');
  // 单个输入的解析：空白 → fallback（默认边界）；非整数 → 明确报错，不做四舍五入等隐式修正
  const parseRow = (value, label, fallback) => {
    if (value == null || String(value).trim() === '') return fallback;
    const parsed = Number(value);
    if (!Number.isInteger(parsed)) throw new Error(`HPLC ${label}必须是整数`);
    return parsed;
  };
  const startRow = parseRow(startValue, '起始行', 1);
  const endRow = parseRow(endValue, '终止行', limit);
  if (startRow < 1 || startRow > limit) {
    throw new Error(`HPLC 起始行必须在 1–${limit} 之间；当前为 ${startRow}`);
  }
  if (endRow < 1 || endRow > limit) {
    throw new Error(`HPLC 终止行不能超过 ${limit} 且不能小于 1；当前为 ${endRow}`);
  }
  if (startRow > endRow) throw new Error('HPLC 起始行不能大于终止行');
  return { startRow, endRow, pointCount: endRow - startRow + 1 };
}

/**
 * 解析曲线编号选择表达式，例如 "1,3,5-8,12" → Set {1,3,5,6,7,8,12}。
 * 用于结果页/可解释性页按 Index 挑选要展示的曲线。
 * - 返回 Set 是为了天然去重且查找 O(1)；键统一存字符串形式，与数据行的 Index 字符串比较一致。
 * - 区间写法允许颠倒（"8-5" 等价于 "5-8"），用 min/max 归一化。
 * - 无法解析为有限数字的片段静默跳过（宽容解析），但超出数量上限会显式报错，
 *   防止 "1-999999999" 之类的输入直接撑爆内存。
 * @param {string} input 用户输入的表达式
 * @param {number} [limit] 允许选择的最大条数，默认 MAX_CURVE_SELECTION
 * @returns {Set<string>} 选中的曲线编号集合（字符串键）
 * @throws {Error} 选中数量超过 limit 时抛出
 */
export function parseCurveIndexExpression(input, limit = MAX_CURVE_SELECTION) {
  const maxSelections = Math.max(1, Math.floor(Number(limit) || MAX_CURVE_SELECTION));
  const values = new Set();
  const addValue = (value) => {
    const key = String(value);
    if (!values.has(key) && values.size >= maxSelections) throw new Error(`最多选择${maxSelections}条曲线`);
    values.add(key);
  };
  String(input || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .forEach((part) => {
      if (part.includes('-')) {
        const [start, end] = part.split('-').map((value) => Math.trunc(Number(value.trim())));
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        const lower = Math.min(start, end);
        const upper = Math.max(start, end);
        if (upper - lower + 1 > maxSelections) throw new Error(`最多选择${maxSelections}条曲线`);
        for (let value = lower; value <= upper; value += 1) addValue(value);
      } else {
        const value = Number(part);
        if (Number.isFinite(value)) addValue(value);
      }
    });
  return values;
}

/**
 * 构造 Sample_ID 折叠列表的视图模型（纯函数，不碰 DOM，便于契约测试）。
 * - 先按 sample_id 自然排序（见 naturalCompare），保证展示顺序稳定；
 * - 折叠时只暴露前 limit 条，剩余数量用于生成“展开其余 N 个”按钮文案。
 * @param {Array<{sample_id: *, label: *, count: *}>} groups 后端返回的按 Sample_ID 分组统计
 * @param {boolean} [expanded] 当前是否处于展开状态
 * @param {number} [limit] 折叠时展示条数，默认 DEFAULT_SAMPLE_ID_LIMIT
 * @returns {{total: number, remaining: number, expanded: boolean,
 *            visible: Array, buttonText: string, toggleLabel: string}}
 *          toggleLabel 与 buttonText 同义，是契约字段，供不同 UI 版本取用。
 */
export function buildSampleIdViewModel(groups, expanded = false, limit = DEFAULT_SAMPLE_ID_LIMIT) {
  const safeGroups = Array.isArray(groups) ? [...groups] : [];
  const safeLimit = Math.max(1, Number.isFinite(Number(limit)) ? Math.floor(Number(limit)) : DEFAULT_SAMPLE_ID_LIMIT);
  safeGroups.sort((first, second) => naturalCompare(first?.sample_id, second?.sample_id));
  const remaining = Math.max(0, safeGroups.length - safeLimit);
  const buttonText = expanded ? '收起' : `展开其余 ${remaining} 个`;
  return {
    total: safeGroups.length,
    remaining,
    expanded: Boolean(expanded),
    visible: expanded ? safeGroups : safeGroups.slice(0, safeLimit),
    buttonText,
    toggleLabel: buttonText,
  };
}

/**
 * 轻量 DOM 创建助手（无框架环境下替代 JSX 的最小方案）。
 * 设计要点：
 * - options 支持语义化键：className / text / style / dataset / on(事件表)，
 *   其余键若与元素自身属性同名则走属性赋值（如 value、type），否则退化为 setAttribute；
 *   aria-* 一律走 setAttribute，避免属性/特性混用导致可访问性信息丢失。
 * - value 为 null/undefined/false 时跳过，方便调用方写条件属性；
 * - children 支持嵌套数组（flat 一层）与非 Node 值（自动转为文本节点），
 *   null/false 子节点被跳过，便于 `{cond && element(...)}` 这类条件渲染写法。
 * - 所有文本一律经 textContent/createTextNode 注入，天然免疫 XSS，不使用 innerHTML。
 * @param {string} tagName 标签名
 * @param {object} [options] 属性/事件配置
 * @param {...(Node|string|number|Array|boolean|null)} children 子节点
 * @returns {HTMLElement}
 */
export function element(tagName, options = {}, ...children) {
  const node = document.createElement(tagName);
  Object.entries(options).forEach(([key, value]) => {
    if (value == null || value === false) return;
    if (key === 'className') node.className = String(value);
    else if (key === 'text') node.textContent = String(value);
    else if (key === 'style') node.style.cssText = String(value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'on') {
      Object.entries(value).forEach(([eventName, listener]) => node.addEventListener(eventName, listener));
    } else if (key in node && !key.startsWith('aria-')) {
      node[key] = value;
    } else {
      node.setAttribute(key, String(value));
    }
  });
  children.flat().forEach((child) => {
    if (child == null || child === false) return;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return node;
}

/**
 * 替换目标元素的全部子节点，并返回目标元素便于链式调用。
 * 过滤 null/undefined，兼容条件渲染写法；直接复用原生 Element.replaceChildren。
 */
export function replaceChildren(target, ...children) {
  target.replaceChildren(...children.filter((child) => child != null));
  return target;
}

/**
 * 渲染 Sample_ID 表格的一行（内部函数）。
 * 长 Sample_ID/Label 用 truncate-text 截断显示，同时把完整值放进 title 与 aria-label，
 * 鼠标悬停和屏幕阅读器都能拿到完整内容；tabIndex: 0 让截断单元格可聚焦以触发 title。
 * 缺失字段统一显示 '-'，避免把 undefined/null 字面量渲染进表格。
 */
function sampleIdRow(group) {
  const sampleId = String(group?.sample_id ?? '-');
  const label = String(group?.label ?? '-');
  const count = group?.count ?? '-';
  const idText = element('span', {
    className: 'truncate-text sample-id-text',
    text: sampleId,
    title: sampleId,
    tabIndex: 0,
    'aria-label': `完整 Sample_ID：${sampleId}`,
  });
  return element(
    'tr',
    {},
    element('td', {}, idText),
    element('td', {}, element('span', { className: 'truncate-text', text: label, title: label, tabIndex: 0 })),
    element('td', { text: String(count) }),
  );
}

/**
 * 渲染一个可折叠 Sample_ID 表格。折叠时只创建前 10 行；大量数据展开时分帧插入，
 * 避免一次性占用主线程。返回的 controller 可用于收起或销毁当前渲染任务。
 *
 * 实现要点：
 * - 展开时按 SAMPLE_ID_BATCH_SIZE（200 行/帧）分批 append，通过 requestAnimationFrame
 *   （不可用时退化为 setTimeout）调度，配合 DocumentFragment 减少重排；
 * - renderToken 是“代际令牌”：每次重渲染（展开/收起/数据变化）先自增，
 *   进行中的异步批量插入在下一帧发现 token 过期即放弃，防止旧批次插进新表格；
 * - aria-live 状态区提示“正在展开 x/y”，兼顾可访问性。
 * @param {HTMLElement} target 挂载点（内容会被整体替换）
 * @param {Array} groups 分组数据，见 buildSampleIdViewModel
 * @param {object} [options]
 * @param {number} [options.limit] 折叠行数上限
 * @param {string} [options.listId] 表格容器 id（aria-controls 引用），缺省随机生成
 * @returns {{collapse: Function, destroy: Function}} 控制器：收起列表 / 停止渲染并清空挂载点
 */
export function renderSampleIdList(target, groups, options = {}) {
  const limit = options.limit ?? DEFAULT_SAMPLE_ID_LIMIT;
  const listId = options.listId || `sample-id-list-${Math.random().toString(36).slice(2)}`;
  let expanded = false;
  let frameId = null;
  let renderToken = 0;

  const wrapper = element('div', { className: 'sample-id-list' });
  const tableWrap = element('div', { className: 'table-scroll sample-id-table-wrap', id: listId });
  const body = element('tbody');
  const table = element(
    'table',
    {},
    element('thead', {}, element('tr', {},
      element('th', { text: '样本编号' }),
      element('th', { text: '类别' }),
      element('th', { text: '每样本测量数' }),
    )),
    body,
  );
  tableWrap.append(table);
  const status = element('span', { className: 'sample-id-render-status', role: 'status', 'aria-live': 'polite' });
  const toggle = element('button', {
    className: 'button secondary sample-id-toggle',
    type: 'button',
    'aria-controls': listId,
    'aria-expanded': 'false',
  });
  const actions = element('div', { className: 'sample-id-actions' }, toggle, status);
  wrapper.append(tableWrap, actions);
  replaceChildren(target, wrapper);

  const cancelBatch = () => {
    renderToken += 1;
    if (frameId != null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frameId);
    frameId = null;
  };

  const schedule = (callback) => {
    if (typeof requestAnimationFrame === 'function') frameId = requestAnimationFrame(callback);
    else frameId = setTimeout(callback, 0);
  };

  const render = () => {
    cancelBatch();
    const token = renderToken;
    const model = buildSampleIdViewModel(groups, expanded, limit);
    body.replaceChildren();
    toggle.hidden = model.remaining === 0;
    toggle.textContent = model.buttonText;
    toggle.setAttribute('aria-expanded', String(expanded));
    tableWrap.classList.toggle('expanded', expanded && model.total > limit);
    status.textContent = '';

    const appendRange = (start) => {
      if (token !== renderToken) return;
      const end = Math.min(start + SAMPLE_ID_BATCH_SIZE, model.visible.length);
      const fragment = document.createDocumentFragment();
      for (let index = start; index < end; index += 1) fragment.append(sampleIdRow(model.visible[index]));
      body.append(fragment);
      if (end < model.visible.length) {
        status.textContent = `正在展开 ${end}/${model.visible.length}`;
        schedule(() => appendRange(end));
      } else {
        status.textContent = expanded && model.total > limit ? `已展开全部 ${model.total} 个 Sample_ID` : '';
        frameId = null;
      }
    };

    appendRange(0);
  };

  toggle.addEventListener('click', () => {
    expanded = !expanded;
    render();
  });
  render();

  return {
    collapse() {
      if (!expanded) return;
      expanded = false;
      render();
    },
    destroy() {
      cancelBatch();
      target.replaceChildren();
    },
  };
}

/**
 * 统一格式化评估指标数值：保留 4 位小数；空值/非数值显示 '—'。
 * 结果页所有 macro-F1、accuracy 等指标都经此函数，保证展示精度一致。
 */
export function formatMetric(value) {
  const number = Number(value);
  return value != null && value !== '' && Number.isFinite(number) ? number.toFixed(4) : '—';
}

/**
 * 格式化时间为本地 'zh-CN' 24 小时制字符串。
 * 空值显示 '-'；无法解析的原始值原样返回（宽容展示，不让单条异常时间弄崩整行渲染）。
 */
export function formatTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
}

/**
 * 浏览器环境下的全局桥接：供非 module 的旧版内联脚本使用，
 * 并派发 `specautoai:ui-ready` 事件通知页面“UI 工具已就绪”，解决脚本加载顺序竞争。
 */
if (typeof window !== 'undefined') {
  window.SpecAutoAIUI = {
    DEFAULT_SAMPLE_ID_LIMIT,
    buildSampleIdViewModel,
    element,
    formatMetric,
    formatTime,
    MAX_CURVE_SELECTION,
    naturalCompare,
    parseCurveIndexExpression,
    renderSampleIdList,
    replaceChildren,
    validateHplcRowRange,
  };
  window.dispatchEvent(new CustomEvent('specautoai:ui-ready'));
}
