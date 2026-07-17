/** 训练记录：summary 分页、客户端状态筛选、键盘巡阅、取消/删除/复制配置。 */
import { el, clear } from '../lib/dom.js';
import { listRunsSummary, getRun, cancelRun, deleteRun } from '../api.js';
import { renderRunList } from '../components/run-list.js';
import { setModelingDraft } from '../store.js';
import { RUN_STATE_META } from '../lib/format.js';

const PAGE_SIZE = 20;

function skeletonRows(count = 3) {
  const wrap = el('div', { className: 'stack', attrs: { 'aria-hidden': 'true' } });
  for (let index = 0; index < count; index += 1) {
    const bar = el('div', { className: 'skeleton' });
    bar.style.height = '44px';
    wrap.append(bar);
  }
  return wrap;
}

export function mountRuns(container, { announce, toast, navigate }) {
  const state = {
    items: [],
    nextCursor: null,
    filter: 'all',
    loading: false,
    error: null,
  };
  let loadToken = 0;

  const root = el('section', { className: 'stack', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);

  const filterSelect = el('select', { className: 'select', attrs: { id: 'v2-runs-filter' } }, [
    el('option', { text: '全部状态', attrs: { value: 'all' } }),
    ...Object.entries(RUN_STATE_META).map(([key, meta]) => el('option', { text: meta.label, attrs: { value: key } })),
  ]);
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

  function filteredItems() {
    if (state.filter === 'all') return state.items;
    return state.items.filter((item) => item.state === state.filter);
  }

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
      if (token === loadToken) {
        state.loading = false;
        renderList();
      }
    }
  }

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

  async function handleAction(action, item) {
    if (action === 'view') {
      navigate(`#/results?run_id=${encodeURIComponent(item.run_id)}`);
      return;
    }
    if (action === 'cancel') {
      if (!window.confirm(`确定取消 Run ${item.run_id}？`)) return;
      try {
        await cancelRun(item.run_id);
        toast(`Run ${item.run_id} 已取消。`, { type: 'success' });
      } catch (error) {
        toast(`取消失败：${error?.message || '未知错误'}`, { type: 'error' });
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
    unmount() {
      loadToken += 1;
    },
  };
}
