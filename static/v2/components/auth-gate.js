/**
 * 模块：服务器模式认证门（v2 工作台）。
 *
 * 职责：
 * - 在 server 部署模式下弹出"输入访问令牌"对话框，阻断未认证的 API 使用；
 * - 两条触发路径：① 启动时 `probeAuth` 主动探测认证配置；② 运行期监听
 *   api-client 派发的 `specautoai:auth-required` 事件（任何接口返回 401 时被动开门）。
 *
 * 在系统中的位置：
 * - 属于 `static/v2/components/` 安全边界组件，由 v2 入口装配；
 * - 协作方：`../api.js` 提供 `getAuthConfig` / `getAuthSession` 两个探测接口，
 *   `window.SpecAutoAIAuth`（由 api-client 挂到全局）负责令牌的实际存取；
 *   本组件只负责 UI 与流程编排，不直接接触存储。
 *
 * 关键设计约束（server 模式安全契约）：
 * - 令牌只写入当前标签页的 sessionStorage（由 api-client 完成），关闭标签页即失效；
 *   绝不写入 URL、localStorage 或日志，防止令牌随链接/持久存储泄漏；
 * - 校验方式是"先存后发真实请求"：`setServerToken` 后立即调 `getAuthSession`，
 *   失败则立刻 `clearServerToken` 回滚，界面上不留无效令牌；
 * - 身份只经服务端 Principal 注入，前端不传 owner_id/tenant_id，本组件也不处理
 *   任何身份信息，只搬运令牌。
 */
import { el } from '../lib/dom.js';
import { getAuthConfig, getAuthSession } from '../api.js';

/**
 * 惰性读取全局认证门面。
 * 用函数包裹而非模块级常量：api-client 的脚本加载顺序不保证先于本模块执行，
 * 每次调用时现取可以避免拿到 undefined 后永久缓存。
 */
const auth = () => window.SpecAutoAIAuth;

/**
 * 初始化认证门并返回控制句柄。
 *
 * 逻辑分段：
 * 1. `dialog` 闭包变量保证单实例：`open()` 里 `if (dialog) return` 防止
 *    多个 401 事件叠加出一叠对话框。
 * 2. 表单语义与文案：type="password" + autocomplete="off" 避免浏览器
 *    把访问令牌当登录密码存进密码管理器；两段说明文案如实告诉用户
 *    令牌的生命周期（仅当前标签页会话存储）。
 * 3. 提交流程（核心安全流程）：
 *    - 空令牌本地拦截，不发请求；
 *    - 先 `setServerToken(token)` 再用 `getAuthSession()` 做一次真实校验——
 *      这样校验请求本身就带着新令牌，成功即证明令牌有效；
 *    - 失败路径必须 `clearServerToken()`：把无效令牌留在 sessionStorage 里
 *      会让后续每个请求都 401，形成"死循环开门"；
 *    - 成功后关闭对话框并回调 `onAuthenticated`，由外部决定下一步
 *      （通常是重试之前失败的加载）。
 * 4. 事件桥：`specautoai:auth-required` 由 api-client 在收到 401 时派发，
 *    这里只做 open()，事件本身不携带令牌等敏感信息。
 *
 * @param {HTMLElement} root 对话框（含遮罩）的挂载容器。
 * @param {object} [options]
 * @param {?Function} options.onAuthenticated 认证成功后的回调。
 * @returns {{open: Function, close: Function}} 供外部（如 probeAuth）调用的句柄。
 */
export function initAuthGate(root, { onAuthenticated } = {}) {
  let dialog = null;

  function close() {
    if (dialog) {
      dialog.remove();
      dialog = null;
    }
  }

  function open(reason = '') {
    if (dialog) return;
    const input = el('input', {
      attrs: { id: 'v2-auth-token', type: 'password', autocomplete: 'off', spellcheck: 'false', required: true },
    });
    const errorBox = el('p', { className: 'form-error', attrs: { role: 'alert' } });
    const form = el('form', { className: 'auth-card' }, [
      el('h2', { text: '服务器访问认证', attrs: { id: 'v2-auth-title' } }),
      el('p', { text: reason || '当前部署为服务器模式，请输入管理员提供的访问令牌。' }),
      el('p', { className: 'muted', text: '令牌只保存在当前标签页的会话存储中，关闭标签页即失效；不会写入 URL 或本地持久存储。' }),
      el('label', { text: '访问令牌', attrs: { for: 'v2-auth-token' } }),
      input,
      errorBox,
      el('div', { className: 'actions' }, [
        el('button', { className: 'primary', text: '验证并继续', attrs: { type: 'submit' } }),
        el('button', {
          className: 'secondary',
          text: '取消',
          attrs: { type: 'button' },
          on: {
            click: () => {
              auth()?.cancelAuthentication();
              close();
            },
          },
        }),
      ]),
    ]);
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const token = input.value.trim();
      if (!token) {
        errorBox.textContent = '请输入访问令牌';
        return;
      }
      auth()?.setServerToken(token);
      try {
        await getAuthSession();
        close();
        onAuthenticated?.();
      } catch (error) {
        auth()?.clearServerToken();
        errorBox.textContent = `令牌校验失败：${error?.message || '未知错误'}`;
      }
    });
    dialog = el('div', {
      className: 'dialog-backdrop',
      attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'v2-auth-title' },
    }, [form]);
    root.append(dialog);
    input.focus();
  }

  window.addEventListener('specautoai:auth-required', () => open());

  return { open, close };
}

/**
 * 启动时探测认证配置：服务器模式且无令牌时主动开门。
 *
 * 为什么启动就要探测：
 * - 不等第一个业务请求 401 才弹窗，而是页面加载后立刻确认"要不要先认证"，
 *   避免用户看到一堆加载失败的报错再被弹窗；
 * - 只有 `auth_required` 且当前标签页没有已存令牌时才开门——同标签页
 *   已经认证过（sessionStorage 仍有令牌）时不打扰；
 * - 整个探测用 try/catch 兜底返回 null：local 模式没有认证接口、或后端
 *   暂时不可达时，探测失败不应阻塞页面正常加载，业务请求会各自处理错误。
 *
 * @param {{open: Function}} gate `initAuthGate` 返回的句柄。
 * @returns {Promise<?object>} 认证配置对象；探测失败返回 null。
 */
export async function probeAuth(gate) {
  try {
    const config = await getAuthConfig();
    if (config?.auth_required && !window.SpecAutoAIAuth?.getServerToken()) {
      gate.open('当前部署为服务器模式，需要先输入访问令牌才能调用 API。');
      return config;
    }
    return config;
  } catch (_) {
    return null;
  }
}
