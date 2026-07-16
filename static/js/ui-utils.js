export const DEFAULT_SAMPLE_ID_LIMIT = 10;
export const MAX_CURVE_SELECTION = 5000;
const SAMPLE_ID_BATCH_SIZE = 200;

const naturalCollator = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });

export const naturalCompare = (first, second) => (
  naturalCollator.compare(String(first ?? ''), String(second ?? ''))
);

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

export function replaceChildren(target, ...children) {
  target.replaceChildren(...children.filter((child) => child != null));
  return target;
}

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
      element('th', { text: 'Sample_ID' }),
      element('th', { text: 'Label' }),
      element('th', { text: '数量' }),
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

export function formatMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : '-';
}

export function formatTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
}

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
  };
  window.dispatchEvent(new CustomEvent('specautoai:ui-ready'));
}
