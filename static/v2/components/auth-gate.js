/**
 * 服务器模式认证门：监听 api-client 派发的 specautoai:auth-required，
 * 令牌只写入当前标签页的 sessionStorage（由 api-client 完成）。
 */
import { el } from '../lib/dom.js';
import { getAuthConfig, getAuthSession } from '../api.js';

const auth = () => window.SpecAutoAIAuth;

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

/** 启动时探测认证配置：服务器模式且无令牌时主动开门。 */
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
