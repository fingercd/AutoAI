/**
 * 模块：v2 工作台入口（ES module 入口脚本，由 static/v2/index.html 以 type="module" 加载）。
 *
 * 职责与定位：
 * - 负责四件全局性工作：hash 路由解析与视图装配、认证门（auth-gate）初始化、
 *   顶栏健康徽标轮询、以及无障碍状态播报区。
 * - 不实现任何具体业务界面：各视图（预处理工作台、AI 建模、训练记录、建模结果、
 *   使用手册）分别在 views/*.js 中，本文件通过 ROUTES 表把路由名映射到其 mount 函数。
 * - 与之协作：store.js（路由/健康状态写入）、api.js（getHealth 轮询）、
 *   components/toast.js（全局提示）、components/auth-gate.js（认证门）、lib/dom.js（DOM 工具）。
 *
 * 关键设计约束：
 * - 使用原生 Hash 路由（#/workbench 等），不引入第三方前端框架或构建链；
 *   结果页专属 URL 契约是 `#/results?run_id=...`，供外部/通知直接深链。
 * - 侧边导航顺序固定为：预处理工作台 → AI 建模 → 建模结果 → 训练记录 → 使用手册。
 */

/** v2 工作台入口：hash 路由、视图装配、认证门与全局状态区。 */
import { el, clear } from './lib/dom.js';
import { setRoute, setHealth, getState } from './store.js';
import { getHealth } from './api.js';
import { initToasts, showToast } from './components/toast.js';
import { initAuthGate, probeAuth } from './components/auth-gate.js';
import { mountWorkbench } from './views/workbench.js';
import { mountModeling } from './views/modeling.js';
import { mountRuns } from './views/runs.js';
import { mountResult } from './views/result.js?v=20260820-compact-visualization-v3';
import { mountComparison } from './views/comparison.js';
import { mountManual } from './views/manual.js';

/**
 * 路由表：视图名 → { title 页标题, sub 副标题, mount 视图挂载函数 }。
 * mount(viewRoot, ctx) 由各视图导出，返回可选的 { unmount } 句柄供切换时清理。
 */
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
  comparison: {
    title: '模型性能比较',
    sub: '仅汇总经过完整性校验的成功子 Run；缺失结果不会伪造成零值。',
    mount: mountComparison,
  },
  manual: {
    title: '使用手册',
    sub: '字段含义、数据契约与常见问题。',
    mount: mountManual,
  },
};
/** 默认视图：无 hash 或 hash 无法识别时回退到预处理工作台。 */
const DEFAULT_VIEW = 'workbench';

/**
 * 解析 location.hash 为结构化路由。
 *
 * 支持的形态：`#/workbench`、`#/results?run_id=xxx` 等；
 * 容错策略：空 hash、未知视图名一律回退 DEFAULT_VIEW，保证页面永远可渲染。
 *
 * @param {string} hash 原始 hash（可含或不含前导 #）。
 * @returns {{view: string, runId: string|null, batchId: string|null}} 路由参数。
 */
export function parseHash(hash) {
  const raw = String(hash || '').replace(/^#/, '');
  const [pathPart, queryPart] = raw.split('?');
  const view = pathPart.replace(/^\//, '') || DEFAULT_VIEW;
  const params = new URLSearchParams(queryPart || '');
  return { view: ROUTES[view] ? view : DEFAULT_VIEW, runId: params.get('run_id') || null, batchId: params.get('batch_id') || null };
}

/**
 * 构造结果页专属深链：`#/results?run_id=<编码后的 id>`。
 * run_id 经 encodeURIComponent 处理，防止特殊字符破坏 hash 结构。
 * @param {string} runId
 * @returns {string} 可直接赋给 location.hash 的字符串。
 */
export function buildResultHash(runId) {
  return `#/results?run_id=${encodeURIComponent(runId)}`;
}

/**
 * 编程式导航。
 * 注意：若目标 hash 与当前相同，浏览器不会触发 hashchange 事件，
 * 此时必须手动 render()，否则视图不会刷新（这是 hash 路由的常见陷阱）。
 * @param {string} hash 目标 hash，通常由 buildResultHash 生成。
 */
function navigate(hash) {
  if (window.location.hash === hash) {
    render();
    return;
  }
  window.location.hash = hash;
}

/**
 * 向无障碍播报区（#v2-status, aria-live="polite"）写入文本，
 * 供屏幕阅读器感知视图切换等动态变化；视觉用户不可见。
 * @param {string} message
 */
function announce(message) {
  const region = document.getElementById('v2-status');
  if (region) region.textContent = message;
}

/** 当前已挂载视图的句柄（含可选 unmount 方法），切换视图前用于清理。 */
let currentView = null;

/**
 * 核心渲染流程：解析 hash → 写入 store → 卸载旧视图 → 更新标题/导航高亮 → 挂载新视图。
 *
 * 由 hashchange 事件、boot() 与 navigate() 三处触发，是 SPA 的唯一装配入口。
 * 设计要点：
 * - 先调旧视图的 unmount（若提供），让视图注销定时器/事件/订阅，避免泄漏与幽灵更新；
 * - clear(viewRoot) 整体清空容器而不是原地 diff——无框架下最简单可靠的方案；
 * - 导航高亮用 aria-current="page" 表达，兼顾样式与无障碍语义。
 */
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

  // 侧边导航高亮：只在当前项上设置 aria-current="page"，其余移除。
  for (const link of document.querySelectorAll('.v2-nav a')) {
    const active = link.dataset.view === route.view;
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }

  // 挂载新视图；视图可返回 { unmount } 句柄，也可能返回 undefined（无清理需求）。
  currentView = definition.mount(viewRoot, {
    route,
    announce,
    navigate,
    toast: showToast,
  }) || null;
  announce(`已进入${definition.title}`);
}

/**
 * 刷新顶栏健康徽标：调 GET /health 并把结果渲染成三个徽标。
 *
 * 三个徽标分别表达：
 * 1. 部署模式（server → '服务器模式'，其余 → '本机模式'）；
 * 2. worker 状态——available 且 compatible !== false 才算健康；
 *    细分三种文案：不可用 / 版本不兼容 / 在线（附 live_count）；
 * 3. 结果契约版本（contracts.run_result，如 run-result-v1），
 *    完整契约表放进 title 悬浮提示，便于核对前后端契约是否对齐。
 *
 * 失败处理：健康检查是"尽力而为"的旁路信息，任何异常都只显示
 * '健康检查失败' 徽标，绝不让它阻断页面主体功能。
 *
 * @param {HTMLElement} badge #v2-health-badge 容器；不存在时直接返回。
 */
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

/**
 * 启动流程（模块加载且 DOM 就绪后执行一次）：
 * 1. 初始化 toast 通知区；
 * 2. 初始化认证门并探测当前会话（probeAuth 会在需要时弹出登录浮层，
 *    认证成功后回调 render() 重绘——此前被 401 中断的请求由 api-client 自动重试）；
 * 3. 立即刷新一次健康徽标，并每 30 秒轮询；
 * 4. 注册 hashchange 监听；若无 hash 则用 location.replace 写入默认路由
 *    （replace 而非赋值：不在历史记录里留下"空 hash"这条无用条目，后退体验更好），
 *    否则直接 render()。
 */
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

// 浏览器环境守卫：使本模块可被 Node 下的单元测试 import（此时不自动启动）。
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  // DOM 未就绪时等 DOMContentLoaded，否则（脚本在 body 末尾且为 module 延迟执行）直接启动。
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
}
