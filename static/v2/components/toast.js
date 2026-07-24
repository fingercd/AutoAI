/**
 * 模块：轻量 toast 通知组件（v2 工作台全局反馈）。
 *
 * 职责：
 * - 在页面角落弹出成功 / 错误 / 提示三类短消息；
 * - 支持自动消退、手动关闭，以及一个可选的"下一步"动作按钮
 *   （例如训练创建成功后附"查看训练记录"跳转）。
 *
 * 在系统中的位置：
 * - 属于 `static/v2/components/` 基础设施型组件，由 v2 入口在启动时
 *   调用 `initToasts(root)` 注入挂载点，之后各页面统一 `showToast(...)`；
 * - 只依赖 `../lib/dom.js` 的 `el`，无状态、无网络、无框架。
 *
 * 关键设计约束：
 * - 错误通知最短展示 8 秒（错误信息需要阅读时间，不能一闪而过）；
 * - 同屏最多 4 条，超出时淘汰最早的一条，防止连续报错刷屏遮挡操作；
 * - `container` 未初始化时静默返回，保证组件在任何加载顺序下都不抛异常。
 */
import { el } from '../lib/dom.js';

/** 三类通知的图标字符；未知 type 回退到 info 图标。 */
const ICONS = { success: '✓', error: '✕', info: 'ℹ' };
/** 同屏通知上限：超过时从最早的开始移除。 */
const MAX_TOASTS = 4;

/** 挂载点引用；由 initToasts 注入，模块级单例（一个页面只需要一个通知栈）。 */
let container = null;

/**
 * 初始化通知挂载点。
 *
 * 为什么需要显式初始化而不自动找 body：
 * - 让调用方决定通知挂在哪个布局节点内（v2 的布局容器，而非 document.body），
 *   避免与页面其他绝对定位元素的层叠上下文打架；
 * - 也使组件在单元测试里可以注入假容器。
 *
 * @param {HTMLElement} root 通知栈的容器元素。
 */
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
 *
 * 逻辑分段：
 * 1. 未初始化直接 return——通知是"锦上添花"，不应因为挂载点缺失让业务流程崩掉。
 * 2. 无障碍语义：error 用 role="alert"（屏幕阅读器立即播报），其余用
 *    role="status"（礼貌播报，不打断当前朗读）；图标 aria-hidden，
 *    因为字符图标对读屏没有信息量。
 * 3. action 按钮先 dismiss 再执行业务回调：防止回调里触发页面跳转后
 *    通知还残留在旧容器上。
 * 4. 计时器与 dismiss 互为守护：dismiss 里 clearTimeout 防止"手动关闭后
 *    定时器又触发一次 remove"（重复 remove 虽不报错但属多余操作）。
 * 5. 末尾的 while 循环是容量淘汰：只删 firstChild（最早的），保持
 *   "新通知永远可见"的直觉。
 *
 * @param {string} message 通知正文（纯文本，el 用 text 赋值，天然防 XSS）。
 * @param {object} [options] 见上方 options 说明。
 * @returns {void} 无返回值；调用方无法也不应持有通知句柄。
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
