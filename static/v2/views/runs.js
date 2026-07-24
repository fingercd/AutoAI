/** 训练记录：summary 分页、客户端状态筛选、键盘巡阅、取消/删除/复制配置。 */
/*
 * 模块说明
 * ========
 * 本文件是 v2 工作台“训练记录”视图的挂载入口，对应 Hash 路由 `#/runs`。
 *
 * 在系统中的位置：
 * - 属于 static/v2 原生 JS 前端的 views 层，由路由层调用 `mountRuns(container, deps)` 挂载。
 * - 与 `static/v2/api.js`（HTTP 封装）、`static/v2/components/run-list.js`（列表行渲染）、
 *   `static/v2/store.js`（跨视图草稿共享）、`static/v2/lib/format.js`（状态文案）协作。
 *
 * 关键设计约束：
 * - 列表数据来自 `GET /api/training/runs?projection=summary`（summary 投影），
 *   因此本视图看不到完整 config，只有“复制配置”时才额外调用 `getRun` 拉取详情。
 * - 分页采用后端游标（cursor）而不是页码：每页 PAGE_SIZE 条，`next_cursor` 为空即没有更多。
 * - 状态筛选是纯客户端筛选（只过滤当前已加载的 items），不触发新的服务端请求。
 * - 取消/删除的可用性由后端按 Run 状态约束：取消仅限排队/运行中，删除仅限终态；
 *   本视图只做确认弹窗与错误提示，真正的权限/状态校验（含 Principal 作用域隔离）在服务端。
 * - 复制配置通过 `setModelingDraft` 写入共享草稿，再跳转到 `#/modeling`，由建模向导读取。
 */
import { el, clear } from '../lib/dom.js';
import { listRunsSummary, getRun, stopRun, deleteRun } from '../api.js';
import { renderRunList } from '../components/run-list.js';
import { setModelingDraft } from '../store.js';
import { RUN_STATE_META } from '../lib/format.js';

const PAGE_SIZE = 20;

/**
 * 渲染“加载中”骨架屏。
 *
 * @param {number} [count=3] 生成的骨架条数量，与真实行高近似，减少加载时的布局跳动。
 * @returns {HTMLElement} 包在 `.stack` 容器里的骨架元素；`aria-hidden` 避免读屏软件误读。
 */
function skeletonRows(count = 3) {
  const wrap = el('div', { className: 'stack', attrs: { 'aria-hidden': 'true' } });
  for (let index = 0; index < count; index += 1) {
    const bar = el('div', { className: 'skeleton' });
    bar.style.height = '44px';
    wrap.append(bar);
  }
  return wrap;
}

/**
 * 挂载“训练记录”视图。
 *
 * 流程：构建静态骨架（筛选/刷新/新建按钮 + 列表容器）→ 立即调用 `load(true)` 拉取第一页
 * → 用户通过“加载更多”按游标翻页，通过下拉框做客户端状态筛选，通过行操作按钮执行
 * 查看结果 / 取消 / 删除 / 复制配置。
 *
 * @param {HTMLElement} container 路由层提供的挂载容器。
 * @param {object} deps 视图依赖：
 *   - `announce(text)`：向无障碍 live region 播报（如“训练记录已刷新”）；
 *   - `toast(text, {type})`：全局提示条，用于操作成功/失败反馈；
 *   - `navigate(hash)`：Hash 路由跳转（查看结果 → `#/results`，新建/复制 → `#/modeling`）。
 * @returns {{ unmount(): void }} 卸载句柄；unmount 只递增 loadToken 作废旧请求，
 *   不再触碰 DOM（DOM 由路由层在切换视图时统一清空）。
 */
export function mountRuns(container, { announce, toast, navigate }) {
  // 视图本地状态：items 为已加载的 summary 行，nextCursor 为服务端分页游标，
  // filter 为客户端状态筛选，loading 防止并发请求，error 记录最近一次失败。
  const state = {
    items: [],
    nextCursor: null,
    filter: 'all',
    loading: false,
    error: null,
  };
  // 请求令牌：每次发起加载时 +1，响应返回时比对；不一致说明已发起更新的请求
  // 或视图已卸载，本次响应直接丢弃，避免旧响应覆盖新数据（竞态防护）。
  let loadToken = 0;

  const root = el('section', { className: 'stack', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);

  // 状态筛选下拉框：选项由 RUN_STATE_META 动态生成，保证与后端 Run 状态枚举一致。
  const filterSelect = el('select', { className: 'select', attrs: { id: 'v2-runs-filter' } }, [
    el('option', { text: '全部状态', attrs: { value: 'all' } }),
    ...Object.entries(RUN_STATE_META).map(([key, meta]) => el('option', { text: meta.label, attrs: { value: key } })),
  ]);
  // 筛选只改 state.filter 并重渲染，不重新请求服务端——筛选作用域是当前已加载的条目。
  filterSelect.addEventListener('change', () => {
    state.filter = filterSelect.value;
    renderList();
  });
  const refreshButton = el('button', { className: 'btn btn-ghost btn-sm', text: '刷新', attrs: { type: 'button' }, on: { click: () => load(true) } });
  const createButton = el('button', { className: 'btn btn-primary', text: '新建训练', attrs: { type: 'button' }, on: { click: () => navigate('#/modeling') } });
  const statusLine = el('p', { className: 'hint', attrs: { role: 'status' } });
  const listHost = el('div');
  const moreButton = el('button', {
    className: 'btn btn-ghost',
    text: '加载更多',
    attrs: { type: 'button', hidden: true },
    on: { click: () => loadMore() },
  });

  root.append(
    el('p', { className: 'page-sub', text: '复核每一次训练：查看结果、复用配置、清理无效 Run。' }),
    el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: 'Run 列表' }),
        el('div', { className: 'card-actions' }, [
          el('div', { className: 'field' }, [
            el('label', { className: 'label', attrs: { for: 'v2-runs-filter' }, text: '状态筛选' }),
            filterSelect,
          ]),
          refreshButton,
          createButton,
        ]),
      ]),
      el('p', { className: 'hint' }, [
        '键盘：', el('kbd', { className: 'kbd', text: '↑' }), ' / ', el('kbd', { className: 'kbd', text: '↓' }), ' 巡阅，', el('kbd', { className: 'kbd', text: 'Enter' }),
        ' 查看结果。取消仅适用排队/运行中；删除仅适用终态且不可恢复。',
      ]),
      listHost,
      statusLine,
      el('div', { className: 'row' }, [moreButton]),
    ]),
  );

  /**
   * 计算当前筛选条件下应展示的条目。
   * 'all' 返回全部已加载条目；其余按 item.state 精确匹配。
   */
  function filteredItems() {
    if (state.filter === 'all') return state.items;
    return state.items.filter((item) => item.state === state.filter);
  }

  /**
   * 重渲染列表区域与底部状态行。
   * 首次加载（无数据且 loading）时显示骨架屏；之后交给 renderRunList 渲染真实行。
   * “加载更多”按钮的可见性完全由 nextCursor 决定：无游标即没有下一页。
   */
  function renderList() {
    clear(listHost);
    if (state.loading && !state.items.length) {
      listHost.append(skeletonRows());
    } else {
      renderRunList(listHost, {
        items: filteredItems(),
        onAction: handleAction,
      });
    }
    moreButton.hidden = !state.nextCursor;
    statusLine.textContent = state.loading
      ? '加载中…'
      : `已加载 ${state.items.length} 条${state.filter === 'all' ? '' : `，筛选后 ${filteredItems().length} 条`}${state.nextCursor ? '，还有更多' : ''}`;
  }

  /**
   * 加载第一页（或整体刷新）。
   *
   * @param {boolean} reset 为 true 时清空已有条目与游标，从头加载；
   *   取消/删除/刷新后都走 reset 路径，保证列表与服务端一致。
   * 竞态处理：用 loadToken 比对丢弃过期响应；loading 标志阻止重复进入。
   */
  async function load(reset) {
    if (state.loading) return;
    state.loading = true;
    const token = ++loadToken;
    if (reset) {
      state.items = [];
      state.nextCursor = null;
    }
    renderList();
    try {
      const result = await listRunsSummary({ limit: PAGE_SIZE });
      if (token !== loadToken) return;
      state.items = Array.isArray(result?.items) ? result.items : [];
      state.nextCursor = result?.next_cursor || null;
      state.error = null;
      announce('训练记录已刷新');
    } catch (error) {
      if (token !== loadToken) return;
      state.error = error;
      toast(`训练记录加载失败：${error?.message || '未知错误'}`, { type: 'error' });
    } finally {
      // 只有最新一次请求允许收尾写状态；过期请求连 loading 都不能复位，
      // 否则会提前解除新请求的并发锁。
      if (token === loadToken) {
        state.loading = false;
        renderList();
      }
    }
  }

  /**
   * 按服务端游标加载下一页并去重合并。
   * 去重原因：游标分页期间列表头部可能插入新 Run（例如刚提交的训练），
   * 直接拼接可能出现重复 run_id，这里按 run_id 过滤已存在条目。
   */
  async function loadMore() {
    if (state.loading || !state.nextCursor) return;
    state.loading = true;
    renderList();
    const token = ++loadToken;
    try {
      const result = await listRunsSummary({ limit: PAGE_SIZE, cursor: state.nextCursor });
      if (token !== loadToken) return;
      const fresh = Array.isArray(result?.items) ? result.items : [];
      state.items = [...state.items, ...fresh.filter((item) => !state.items.some((old) => old.run_id === item.run_id))];
      state.nextCursor = result?.next_cursor || null;
    } catch (error) {
      if (token !== loadToken) return;
      toast(`加载更多失败：${error?.message || '未知错误'}`, { type: 'error' });
    } finally {
      if (token === loadToken) {
        state.loading = false;
        renderList();
      }
    }
  }

  /**
   * 统一处理列表行操作。
   *
   * @param {'view'|'cancel'|'delete'|'copy'} action 操作类型（由 run-list 组件发出）。
   * @param {object} item 该行的 summary 条目（至少含 run_id、state）。
   *
   * 语义：
   * - view：跳转专属结果 URL `#/results?run_id=...`；
   * - cancel：二次确认后调用取消接口，随后整表刷新（Run 状态会变）；
   * - delete：二次确认后删除，记录与产物一并移除且不可恢复，随后整表刷新；
   * - copy：summary 投影不含 config，需要先 `getRun` 拉详情，再把配置写入
   *   建模草稿（setModelingDraft）并跳转建模向导；无 config 时明确报错而不是带空配置过去。
   */
  async function handleAction(action, item) {
    if (action === 'view') {
      navigate(`#/results?run_id=${encodeURIComponent(item.run_id)}`);
      return;
    }
    if (action === 'cancel') {
      if (!window.confirm(`确定取消 Run ${item.run_id}？`)) return;
      try {
        await stopRun(item.run_id);
        toast(`Run ${item.run_id} 已 STOP。`, { type: 'success' });
      } catch (error) {
        toast(`停止失败：${error?.message || '未知错误'}`, { type: 'error' });
      }
      await load(true);
      return;
    }
    if (action === 'delete') {
      if (!window.confirm(`确定永久删除 Run ${item.run_id}？记录与产物都会被移除，不可恢复。`)) return;
      try {
        await deleteRun(item.run_id);
        toast(`Run ${item.run_id} 已删除。`, { type: 'success' });
      } catch (error) {
        toast(`删除失败：${error?.message || '未知错误'}`, { type: 'error' });
      }
      await load(true);
      return;
    }
    if (action === 'copy') {
      try {
        const detail = await getRun(item.run_id);
        const config = detail?.config;
        if (!config || typeof config !== 'object' || !Object.keys(config).length) {
          toast('该 Run 没有可取用的训练配置，无法复制。', { type: 'error' });
          return;
        }
        setModelingDraft({
          config,
          datasetId: detail.dataset_id || null,
          datasetName: detail.dataset_name || null,
          testDatasetId: config.test_dataset_id || null,
          testDatasetName: config.test_dataset_name || null,
          sourceRunId: item.run_id,
        });
        toast('已复制配置，正在进入建模向导。', { type: 'success' });
        navigate('#/modeling');
      } catch (error) {
        toast(`无法获取配置：${error?.message || '未知错误'}`, { type: 'error' });
      }
    }
  }

  load(true);

  return {
    // 卸载：递增 loadToken 使所有在途响应失效即可；
    // DOM 清理由路由层负责，这里不做额外操作。
    unmount() {
      loadToken += 1;
    },
  };
}
