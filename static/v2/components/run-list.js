/** 训练记录列表：摘要表格、状态徽章、每行一个主按钮与键盘巡阅。 */
import { el, clear } from '../lib/dom.js';
import { stateMeta, stateBadgeClass, isTerminalState, isActiveState, formatDateTime, formatDuration } from '../lib/format.js';

export function stateBadge(state) {
  const meta = stateMeta(state);
  return el('span', { className: stateBadgeClass(state), text: meta.label });
}

function rowAction(item, { action, text, className, disabled = false, title, onAction }) {
  return el('button', {
    className,
    text,
    attrs: { type: 'button', disabled: disabled ? true : null, title },
    on: { click: (event) => { event.stopPropagation(); onAction?.(action, item); } },
  });
}

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
    el('td', { text: formatDateTime(item.created_at) }),
    el('td', { text: formatDuration(item.duration_seconds) }),
    el('td', {}, el('div', { className: 'row' }, [
      rowAction(item, { action: 'view', text: '查看结果', className: 'btn btn-primary btn-sm', title: '打开该 Run 的建模结果', onAction }),
      rowAction(item, { action: 'copy', text: '复制配置', className: 'btn btn-ghost btn-sm', title: '复制训练配置到建模向导', onAction }),
      rowAction(item, {
        action: 'cancel',
        text: '取消',
        className: 'btn btn-ghost btn-sm',
        disabled: !isActiveState(item.state),
        title: isActiveState(item.state) ? '取消排队/运行中的 Run' : '仅排队或运行中的 Run 可取消',
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
  wrap.append(el('table', { className: 'data-table' }, [
    el('caption', { text: '训练 Run 摘要列表；每行可聚焦，Enter 打开结果。' }),
    el('thead', {}, el('tr', {}, ['Run ID', '模型 · 数据集', '状态', '创建时间', '耗时', '操作'].map((head) => el('th', { text: head, attrs: { scope: 'col' } })))),
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
