/** 模型能力目录：渲染 GET /api/models 的目标；可用项优先推荐，不可用项折叠并显示原因，不做静默替代。 */
import { el, clear } from '../lib/dom.js';
import { MODEL_FAMILY_LABELS } from '../lib/format.js';

const FAMILY_ORDER = ['traditional_ml', 'basic_deep', 'convolutional', 'long_range', 'two_dimensional_mapping'];

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

export function findModel(models, id) {
  return (models || []).find((model) => model.id === id) || null;
}
