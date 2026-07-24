/**
 * 模块说明：训练记录列表组件（static/v2/components/run-list.js）
 * =================================================================
 * 职责：渲染 v2 工作台"训练记录"页的 Run 摘要表格——每行展示一个训练 Run 的
 * run_id、模型与数据集、状态徽章、测试集 Macro F1、训练时间、耗时，以及
 * "查看结果 / 复制配置 / 取消 / 删除"四个行内操作按钮。
 *
 * 在系统中的位置：
 *   - 属于并行新前端 `static/v2/index.html` 的展示层组件，由结果/记录页的主控
 *     脚本（v2 的 app 入口）调用 `renderRunList(container, { items, onAction })` 挂载。
 *   - `items` 来自后端 `GET /api/training/runs?projection=summary` 返回的 Run 摘要数组；
 *     其中 `test_macro_f1` 可空——只有成功且 Manifest 完整的 Run 才会带测试主指标
 *     （CV 口径取 pooled test，而非 fold mean），空值交由 formatMetric 显示占位符。
 *
 * 协作模块：
 *   - `../lib/dom.js`：提供 `el`（声明式创建 DOM）与 `clear`（清空容器）两个基础工具。
 *   - `../lib/format.js`：提供状态元信息、状态判定（isActiveState/isTerminalState）
 *     与各类格式化函数，本组件不自己拼状态文案，保证与全局状态机一致。
 *
 * 关键设计约束：
 *   - 组件本身不发起任何 HTTP 请求、不直接路由跳转；所有动作都通过
 *     `onAction(action, item)` 回调上抛给主控脚本，保持"纯渲染 + 事件上抛"的
 *     单向数据流，便于复用与测试。
 *   - 行级可访问性：每行 `<tr tabindex="0">` 可聚焦，支持 ↑/↓/Home/End 键盘巡阅、
 *     Enter 打开结果；点击行内按钮时不触发行点击（stopPropagation + closest 判断）。
 *   - 业务状态约束：仅"排队/运行中"（isActiveState）的 Run 可取消；仅"终态"
 *     （isTerminalState）的 Run 可删除；不可用时按钮禁用并给出原因 title。
 */
import { el, clear } from '../lib/dom.js';
import {
  stateMeta, stateBadgeClass, isTerminalState, isActiveState,
  formatMetric, formatTrainingTime, formatDuration,
} from '../lib/format.js';

/**
 * 生成状态徽章元素。
 * @param {string} state - Run 的状态字符串（如 queued/running/succeeded/failed/canceled）。
 * @returns {HTMLElement} `<span>` 徽章，className 与文案均由 format.js 的
 *   stateMeta/stateBadgeClass 统一决定，保证全站状态配色一致。
 */
export function stateBadge(state) {
  const meta = stateMeta(state);
  return el('span', { className: stateBadgeClass(state), text: meta.label });
}

/**
 * 构造行内操作按钮（私有辅助函数）。
 * @param {Object} item - Run 摘要对象，原样透传给 onAction 回调。
 * @param {Object} options
 * @param {string} options.action - 动作名（'view'/'copy'/'cancel'/'delete'），上抛给主控。
 * @param {string} options.text - 按钮文案。
 * @param {string} options.className - 按钮样式类。
 * @param {boolean} [options.disabled=false] - 是否禁用；disabled=false 时传 null 让 el 不写该属性。
 * @param {string} [options.title] - 悬停提示，用于解释按钮为何禁用。
 * @param {Function} [options.onAction] - 动作回调 onAction(action, item)。
 * 设计要点：click 里先 stopPropagation，避免触发整行的"查看结果"行点击事件。
 */
function rowAction(item, { action, text, className, disabled = false, title, onAction }) {
  return el('button', {
    className,
    text,
    attrs: { type: 'button', disabled: disabled ? true : null, title },
    on: { click: (event) => { event.stopPropagation(); onAction?.(action, item); } },
  });
}

/**
 * 构造单个 Run 的表格行（私有辅助函数）。
 * 行本身是"查看结果"的大点击区：点击行或按 Enter 都上抛 'view'；
 * 但事件源在按钮内时直接 return，防止与按钮动作重复触发。
 * `data-run-id` 与 tabindex 服务于键盘巡阅与测试定位。
 * @param {Object} item - Run 摘要（run_id/model_type/dataset_name/state/
 *   test_macro_f1/duration_seconds/error 等字段，均允许缺失，缺失显示 '—'）。
 * @returns {HTMLElement} `<tr>` 元素。
 */
function runRow(item, { onAction }) {
  return el('tr', {
    attrs: { tabindex: '0', 'data-run-id': item.run_id, title: 'Enter 查看结果' },
    on: {
      click: (event) => {
        if (event.target.closest('button')) return;
        onAction?.('view', item);
      },
      keydown: (event) => {
        if (event.key !== 'Enter' || event.target.closest('button')) return;
        event.preventDefault();
        onAction?.('view', item);
      },
    },
  }, [
    el('td', {}, [
      el('code', { text: item.run_id }),
      item.error ? el('div', { className: 'error-text', text: item.error }) : null,
    ]),
    el('td', { text: `${item.model_type || '—'} · ${item.dataset_name || '—'}` }),
    el('td', {}, stateBadge(item.state)),
    el('td', { text: formatMetric(item.test_macro_f1) }),
    el('td', { text: formatTrainingTime(item) }),
    el('td', { text: formatDuration(item.duration_seconds) }),
    el('td', {}, el('div', { className: 'row' }, [
      rowAction(item, { action: 'view', text: '查看结果', className: 'btn btn-primary btn-sm', title: '打开该 Run 的建模结果', onAction }),
      rowAction(item, { action: 'copy', text: '复制配置', className: 'btn btn-ghost btn-sm', title: '复制训练配置到建模向导', onAction }),
      rowAction(item, {
        action: 'cancel',
        text: '停止',
        className: 'btn btn-ghost btn-sm',
        disabled: !isActiveState(item.state),
        title: isActiveState(item.state) ? '停止训练并丢弃本次产物' : '仅排队或运行中的 Run 可停止',
        onAction,
      }),
      rowAction(item, {
        action: 'delete',
        text: '删除',
        className: 'btn btn-danger btn-sm',
        disabled: !isTerminalState(item.state),
        title: isTerminalState(item.state) ? '删除终态 Run（不可恢复）' : '仅终态 Run 可删除',
        onAction,
      }),
    ])),
  ]);
}

/**
 * 渲染 Run 摘要表格并绑定键盘导航（↑/↓/Home/End 移动焦点，Enter 打开结果）。
 *
 * 逻辑说明：
 *   1. 先清空容器；空列表时渲染空态提示并直接返回。
 *   2. 用 items.map(runRow) 一次性构建 `<tbody>`，避免逐行 append 造成的回流。
 *   3. 键盘导航监听挂在 `.table-wrap` 上（事件委托），仅当焦点在某一行
 *      （document.activeElement 属于 tbody 的 tr）时才拦截方向键；
 *      用 Math.min/Math.max 做边界钳制，不循环回绕。
 *
 * @param {HTMLElement} container - 挂载点，内部内容会被整体替换。
 * @param {Object} options
 * @param {Array<Object>} options.items - Run 摘要数组（见 runRow 字段说明）。
 * @param {Function} [options.onAction] - 动作回调 onAction(action, item)，
 *   action ∈ {'view','copy','cancel','delete'}，由主控脚本决定后续行为
 *   （路由跳转 / 复制配置到建模向导 / 调取消、删除 API）。
 * options: { items, onAction(action, item) }
 */
export function renderRunList(container, { items, onAction }) {
  clear(container);
  if (!items.length) {
    container.append(el('div', { className: 'empty' }, [
      el('p', { text: '当前筛选下没有训练记录。' }),
      el('p', { className: 'hint', text: '提交训练后新的 Run 会出现在这里；也可以调整上方状态筛选。' }),
    ]));
    return;
  }
  const tbody = el('tbody', {}, items.map((item) => runRow(item, { onAction })));
  const wrap = el('div', { className: 'table-wrap' });
  wrap.append(el('table', { className: 'data-table run-summary-table' }, [
    el('caption', { text: '训练 Run 摘要列表；每行可聚焦，Enter 打开结果。' }),
    el('thead', {}, el('tr', {}, ['Run ID', '模型 · 数据集', '状态', '测试集 Macro F1', '训练时间', '耗时', '操作'].map((head) => el('th', { text: head, attrs: { scope: 'col' } })))),
    tbody,
  ]));
  wrap.addEventListener('keydown', (event) => {
    const keys = ['ArrowDown', 'ArrowUp', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const index = rows.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    let next = index;
    if (event.key === 'ArrowDown') next = Math.min(rows.length - 1, index + 1);
    if (event.key === 'ArrowUp') next = Math.max(0, index - 1);
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = rows.length - 1;
    rows[next]?.focus();
  });
  container.append(wrap);
}
