/**
 * 模块：模型能力目录组件（v2 工作台 "AI 建模" 页）。
 *
 * 职责：
 * - 渲染 `GET /api/models` 返回的模型能力目录（15 个目标模型，按 family 分组）；
 * - 提供单选交互：可选中一个 `available=true` 的模型并通过回调通知外部；
 * - 不可用模型（如 `cnn_mamba1d` 因 `mamba-ssm` 依赖缺失）折叠展示并说明原因。
 *
 * 在系统中的位置：
 * - 属于 `static/v2/components/` 展示/交互组件，被 v2 建模页装配；
 * - 依赖 `../lib/dom.js` 的 `el/clear` 建 DOM，`../lib/format.js` 的
 *   `MODEL_FAMILY_LABELS` 提供 family 的中文名。
 *
 * 关键设计约束（与后端能力目录契约对应）：
 * - 可用项优先推荐：每组内可训练模型排前，不可用项默认收进 <details> 折叠；
 * - 不做静默替代：之前选中的模型若变得不可用，只显式提示用户改选，
 *   绝不自动换成近似模型（这是 AGENTS.md 的硬性业务约束）；
 * - 纯前端组件，不发起网络请求；models 数据由调用方注入。
 */
import { el, clear } from '../lib/dom.js';
import { MODEL_FAMILY_LABELS } from '../lib/format.js';

/**
 * family 的固定展示顺序：传统机器学习 → 基础深度 → 卷积 → 长序列 → 二维映射。
 * 顺序即业务上的"由浅入深"推荐路径；不在列表中的 family（未来新增）会被
 * `groupModelsByFamily` 追加到末尾，保证前向兼容。
 */
const FAMILY_ORDER = ['traditional_ml', 'basic_deep', 'convolutional', 'long_range', 'two_dimensional_mapping'];

/**
 * 把扁平的模型列表按 family 分组，并保持固定组序。
 *
 * 逻辑原理：
 * - 先按 FAMILY_ORDER 预建空组，保证输出顺序稳定（与接口返回顺序无关）；
 * - 缺失 family 字段的模型兜底归入 'traditional_ml'，未知 family 动态开新组，
 *   这样后端新增 family 时前端不会丢模型；
 * - 最后过滤掉空组，避免渲染只有标题没有卡片的空 section。
 *
 * @param {Array<object>} models `GET /api/models` 返回的模型数组。
 * @returns {Array<[string, Array<object>]>} `[family, items]` 二元组列表，仅含非空组。
 */
export function groupModelsByFamily(models) {
  const groups = new Map();
  for (const family of FAMILY_ORDER) groups.set(family, []);
  for (const model of models || []) {
    const family = model?.family || 'traditional_ml';
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push(model);
  }
  return [...groups.entries()].filter(([, items]) => items.length);
}

/**
 * 构建单张模型卡片（radio + label 组合）。
 *
 * 结构与设计意图：
 * - 外层用 <label for> 包裹 radio，整张卡片都可点击选中，扩大命中区域；
 * - 不可用模型：`radio.disabled=true`、卡片加 `model-card-disabled`、
 *   显示 `unavailable_reason`（如依赖缺失），且 change 回调里再次检查
 *   `available`——双保险，防止禁用状态被脚本绕过后误触发选择；
 * - "已选择" 徽标仅对可用模型渲染（三元表达式处不可用项换成原因文本），
 *   其显隐由外层 `syncSelected` 统一同步。
 *
 * @param {object} model 单个模型目录项（id/display_name/available/
 *   explainability_method/unavailable_reason 等字段）。
 * @param {object} options
 * @param {string} options.name radio 组名，同组互斥；同时用作 id 前缀保证唯一。
 * @param {boolean} options.selected 初始是否选中。
 * @param {?Function} options.onChoose 选中回调，传入模型 id；不可用模型传 null。
 * @returns {{card: HTMLElement, radio: HTMLInputElement, selectedBadge: HTMLElement}}
 *   卡片元素及其内部需要外部同步的子元素引用。
 */
function modelCard(model, { name, selected, onChoose }) {
  const available = model.available === true;
  const selectedBadge = el('span', { className: 'badge status-running', text: '✓ 已选择' });
  selectedBadge.hidden = !selected;
  const radio = el('input', {
    attrs: {
      type: 'radio',
      name,
      value: model.id,
      disabled: available ? null : true,
      checked: selected ? true : null,
      id: `${name}-${model.id}`,
    },
    on: { change: () => { if (available) onChoose?.(model.id); } },
  });
  const card = el('label', {
    className: `card model-card${available ? '' : ' model-card-disabled'}`,
    attrs: { for: `${name}-${model.id}`, 'data-model-id': model.id, 'data-selected': selected ? 'true' : null },
  }, [
    el('span', { className: 'row spread' }, [
      el('span', { className: 'row' }, [radio, el('strong', { text: model.display_name || model.id })]),
      available
        ? el('span', { className: 'badge status-succeeded', text: '可训练' })
        : el('span', { className: 'badge status-failed', text: '不可用' }),
    ]),
    el('span', { className: 'hint', text: `${model.id} · 解释方法：${model.explainability_method || '—'}` }),
    available ? selectedBadge : el('span', { className: 'error-text', text: model.unavailable_reason || '当前环境不可用' }),
  ]);
  return { card, radio, selectedBadge };
}

/**
 * 渲染模型目录为单选列表：每组内可训练模型排前，不可用模型默认折叠。
 * 选择变化在组件内部同步视觉状态（不打断焦点），并通过 onSelect(id) 通知外部。
 * options: { models, selectedId, onSelect(id), name }
 *
 * 详细逻辑分段：
 * 1. 清空容器；空目录显示占位提示并提前返回（接口失败时调用方会传空数组）。
 * 2. `currentId` 是组件内部的选中状态副本：之后用户点击只改它并调用
 *    `syncSelected()` 局部更新类名/徽标，不做整树重渲染——这样既不丢失
 *    焦点，也避免反复重建 radio 组。
 * 3. 每个 family 一个 section：标题行显示中文 family 名与可训练/不可用计数；
 *    可用模型直接铺卡片网格，不可用模型收进 <details class="advanced">，
 *    点开才能看到原因——保持主界面只呈现"现在能用的"。
 * 4. 尾部校验：若外部传入的 `selectedId` 在当前目录里不存在或已不可用，
 *    追加一条 role="alert" 的警告，明确告知"系统不会用其他模型静默替代"
 *    （对应后端能力目录的硬约束：如 cnn_mamba1d 不可用时不允许近似顶替）。
 *
 * @param {HTMLElement} container 挂载点（函数会先 clear）。
 * @param {object} options
 * @param {Array<object>} options.models 模型目录数组。
 * @param {?string} options.selectedId 外部记忆的当前选中模型 id。
 * @param {?Function} options.onSelect 选择变化回调，参数为模型 id。
 * @param {string} [options.name='v2-model'] radio 组名，同页多个实例时需区分。
 */
export function renderModelCatalog(container, { models, selectedId, onSelect, name = 'v2-model' }) {
  clear(container);
  if (!Array.isArray(models) || !models.length) {
    container.append(el('div', { className: 'empty', text: '模型目录为空，请稍后重试。' }));
    return;
  }
  let currentId = selectedId || null;
  const entries = [];

  const syncSelected = () => {
    for (const entry of entries) {
      const active = entry.model.id === currentId;
      entry.card.dataset.selected = active ? 'true' : 'false';
      entry.card.classList.toggle('model-card-selected', active);
      if (entry.selectedBadge) entry.selectedBadge.hidden = !active;
    }
  };

  const list = el('div', { className: 'stack', attrs: { role: 'radiogroup', 'aria-label': '选择模型' } });
  for (const [family, items] of groupModelsByFamily(models)) {
    const available = items.filter((model) => model.available === true);
    const unavailable = items.filter((model) => model.available !== true);
    const block = el('section', { className: 'stack' }, [
      el('div', { className: 'row spread' }, [
        el('h3', { className: 'card-title', text: MODEL_FAMILY_LABELS[family] || family }),
        el('span', { className: 'hint', text: `${available.length} 个可训练${unavailable.length ? ` · ${unavailable.length} 个不可用` : ''}` }),
      ]),
    ]);
    if (available.length) {
      const grid = el('div', { className: 'grid grid-3' });
      for (const model of available) {
        const entry = modelCard(model, { name, selected: model.id === currentId, onChoose: (id) => { currentId = id; syncSelected(); onSelect?.(id); } });
        entries.push({ model, ...entry });
        grid.append(entry.card);
      }
      block.append(grid);
    } else {
      block.append(el('p', { className: 'hint', text: '该组当前没有可训练的模型。' }));
    }
    if (unavailable.length) {
      const grid = el('div', { className: 'grid grid-3' });
      for (const model of unavailable) {
        const entry = modelCard(model, { name, selected: false, onChoose: null });
        grid.append(entry.card);
      }
      block.append(el('details', { className: 'advanced' }, [
        el('summary', { text: `查看 ${unavailable.length} 个当前不可用的模型（含原因）` }),
        grid,
      ]));
    }
    list.append(block);
  }
  container.append(list);
  syncSelected();
  if (currentId && !models.some((model) => model.id === currentId && model.available)) {
    container.append(el('p', {
      className: 'error-text',
      attrs: { role: 'alert' },
      text: '之前选择的模型当前不可用，请改选上方标有“可训练”的模型；系统不会用其他模型静默替代。',
    }));
  }
}

/**
 * 按 id 在目录中查找模型。
 *
 * @param {Array<object>} models 模型目录数组（允许 null/undefined）。
 * @param {string} id 目标模型 id，如 'cnn1d'、'pls_da'。
 * @returns {?object} 命中的模型项；未命中返回 null（调用方需自行判空）。
 */
export function findModel(models, id) {
  return (models || []).find((model) => model.id === id) || null;
}
