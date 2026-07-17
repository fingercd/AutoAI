/** v2 工作台入口：hash 路由、视图装配、认证门与全局状态区。 */
import { el, clear } from './lib/dom.js';
import { setRoute, setHealth, getState } from './store.js';
import { getHealth } from './api.js';
import { initToasts, showToast } from './components/toast.js';
import { initAuthGate, probeAuth } from './components/auth-gate.js';
import { mountWorkbench } from './views/workbench.js';
import { mountModeling } from './views/modeling.js';
import { mountRuns } from './views/runs.js';
import { mountResult } from './views/result.js';
import { mountManual } from './views/manual.js';

const ROUTES = {
  workbench: {
    title: '预处理工作台',
    sub: '上传原始光谱 / 色谱文件，按固定顺序完成预处理，导出统一建模 CSV。',
    mount: mountWorkbench,
  },
  modeling: {
    title: 'AI 建模向导',
    sub: '四步完成训练配置：数据集 → 评估口径 → 模型选择 → 参数提交；提交后由独立 worker 异步执行。',
    mount: mountModeling,
  },
  runs: {
    title: '训练记录',
    sub: '查看全部训练任务的进度与状态；可复制配置、取消或删除终态任务。',
    mount: mountRuns,
  },
  results: {
    title: '建模结果',
    sub: '指标、曲线、混淆矩阵与可下载产物，全部来自训练真实输出。',
    mount: mountResult,
  },
  manual: {
    title: '使用手册',
    sub: '字段含义、数据契约与常见问题。',
    mount: mountManual,
  },
};
const DEFAULT_VIEW = 'workbench';

export function parseHash(hash) {
  const raw = String(hash || '').replace(/^#/, '');
  const [pathPart, queryPart] = raw.split('?');
  const view = pathPart.replace(/^\//, '') || DEFAULT_VIEW;
  const params = new URLSearchParams(queryPart || '');
  return { view: ROUTES[view] ? view : DEFAULT_VIEW, runId: params.get('run_id') || null };
}

export function buildResultHash(runId) {
  return `#/results?run_id=${encodeURIComponent(runId)}`;
}

function navigate(hash) {
  if (window.location.hash === hash) {
    render();
    return;
  }
  window.location.hash = hash;
}

function announce(message) {
  const region = document.getElementById('v2-status');
  if (region) region.textContent = message;
}

let currentView = null;

function render() {
  const route = parseHash(window.location.hash);
  setRoute(route);
  const definition = ROUTES[route.view];
  currentView?.unmount?.();
  currentView = null;

  const viewRoot = document.getElementById('v2-view-root');
  const viewTitle = document.getElementById('v2-view-title');
  clear(viewRoot);
  viewTitle.textContent = definition.title;
  const viewSub = document.getElementById('v2-view-sub');
  if (viewSub) viewSub.textContent = definition.sub || '';
  document.title = `${definition.title} · SpecAutoAI v2`;

  for (const link of document.querySelectorAll('.v2-nav a')) {
    const active = link.dataset.view === route.view;
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }

  currentView = definition.mount(viewRoot, {
    route,
    announce,
    navigate,
    toast: showToast,
  }) || null;
  announce(`已进入${definition.title}`);
}

async function refreshHealth(badge) {
  if (!badge) return;
  try {
    const health = await getHealth();
    setHealth(health);
    clear(badge);
    const mode = health?.deployment_mode === 'server' ? '服务器模式' : '本机模式';
    const worker = health?.worker;
    const workerOk = Boolean(worker?.available) && worker?.compatible !== false;
    const workerText = worker?.available
      ? worker.compatible === false
        ? 'worker 版本不兼容'
        : `worker 在线 ${worker.live_count ?? 0}`
      : 'worker 不可用';
    const contracts = health?.contracts && typeof health.contracts === 'object' ? health.contracts : {};
    const contractText = contracts.run_result ? `契约 ${contracts.run_result}` : '契约版本未知';
    const contractDetail = Object.entries(contracts)
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n');
    badge.append(
      el('span', { className: 'badge badge-muted', text: mode }),
      el('span', { className: `badge ${workerOk ? 'badge-success' : 'badge-danger'}`, text: workerText }),
      el('span', {
        className: 'badge badge-info',
        text: contractText,
        attrs: { title: contractDetail || '后端未返回契约版本信息' },
      }),
    );
  } catch (_) {
    clear(badge);
    badge.append(el('span', { className: 'badge badge-danger', text: '健康检查失败' }));
  }
}

function boot() {
  initToasts(document.getElementById('v2-toasts'));
  const gate = initAuthGate(document.getElementById('v2-auth-root'), {
    onAuthenticated: () => {
      showToast('认证成功，被中断的请求会自动重试。', { type: 'success' });
      render();
    },
  });
  probeAuth(gate);
  const healthBadge = document.getElementById('v2-health-badge');
  refreshHealth(healthBadge);
  window.setInterval(() => refreshHealth(healthBadge), 30000);
  window.addEventListener('hashchange', render);
  if (!window.location.hash) {
    window.location.replace(`#/${DEFAULT_VIEW}`);
    return;
  }
  render();
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
}
