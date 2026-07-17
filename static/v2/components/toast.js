/** 轻量 toast 通知：成功/错误/提示，自动消退，支持手动关闭与可选的“下一步”动作。 */
import { el } from '../lib/dom.js';

const ICONS = { success: '✓', error: '✕', info: 'ℹ' };
const MAX_TOASTS = 4;

let container = null;

export function initToasts(root) {
  container = root;
}

/**
 * 显示一条通知。
 * options: {
 *   type: 'info' | 'success' | 'error',
 *   duration: 毫秒；0 表示不自动消退（错误通知最少 8 秒），
 *   action: { label, onClick } 可选的下一步动作按钮。
 * }
 */
export function showToast(message, { type = 'info', duration = 5000, action = null } = {}) {
  if (!container) return;
  const toast = el('div', { className: `toast toast-${type}`, attrs: { role: type === 'error' ? 'alert' : 'status' } }, [
    el('span', { className: 'toast-icon', text: ICONS[type] || ICONS.info, attrs: { 'aria-hidden': 'true' } }),
    el('span', { className: 'toast-message', text: message }),
    action && typeof action.onClick === 'function'
      ? el('button', {
        className: 'btn btn-sm btn-ghost toast-action',
        text: action.label || '查看',
        attrs: { type: 'button' },
        on: { click: () => { dismiss(); action.onClick(); } },
      })
      : null,
    el('button', {
      className: 'toast-close',
      text: '关闭',
      attrs: { type: 'button', 'aria-label': '关闭通知' },
      on: { click: () => dismiss() },
    }),
  ]);
  container.append(toast);
  const timeout = type === 'error' ? Math.max(duration, 8000) : duration;
  const timer = timeout > 0 ? setTimeout(dismiss, timeout) : null;
  function dismiss() {
    if (timer !== null) clearTimeout(timer);
    toast.remove();
  }
  while (container.children.length > MAX_TOASTS) container.firstChild.remove();
}
