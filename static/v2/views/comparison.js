/** Responsive model-comparison-v1 page; keeps the last good render while polling. */
import { el, clear } from '../lib/dom.js';
import { groupedBarChart, chartLegend } from '../lib/charts.js';
import { renderConfusionMatrix } from '../components/confusion-matrix.js';
import { getBatch, getBatchComparison, stopBatch } from '../api.js';

const METRICS = [
  ['accuracy', 'Accuracy'],
  ['balanced_accuracy', 'Balanced Accuracy'],
  ['macro_f1', 'Macro-F1'],
  ['weighted_f1', 'Weighted-F1'],
];

const format = (value) => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—');
const heatColor = (value) => value == null ? 'transparent' : `hsl(${Math.round(value * 120)} 65% 88%)`;
const modelRows = (comparison) => comparison.overall_metrics?.rows || comparison.models || [];

function resultBadge(row) {
  const requested = Number(row.requested_repeats) || 1;
  const successful = Number(row.successful_repeats) || 0;
  return requested > 1 ? '历史结果' : (successful ? '结果完整' : '结果缺失');
}

function metricCard(row) {
  const accuracy = row.metrics?.accuracy || {};
  const accuracyWidth = typeof accuracy.mean === 'number' && Number.isFinite(accuracy.mean)
    ? Math.max(0, Math.min(100, accuracy.mean * 100)) : null;
  return el('article', { className: 'card comparison-model-card' }, [
    el('div', { className: 'row spread' }, [
      el('h3', { className: 'card-title', text: row.model_type }),
      el('span', { className: 'badge', text: resultBadge(row) }),
    ]),
    el('div', { className: 'comparison-bar-track' }, accuracyWidth == null
      ? el('span', { className: 'hint', text: '缺少 Accuracy' })
      : el('div', { className: 'comparison-bar-fill', attrs: { style: `width:${accuracyWidth}%` } })),
    el('p', { className: 'hint', text: `Accuracy ${format(accuracy.mean)}` }),
  ]);
}

function metricTable(rows, { heat = false } = {}) {
  return el('div', { className: 'table-wrap comparison-scroll' }, el('table', { className: `data-table${heat ? ' comparison-heat' : ''}` }, [
    el('thead', {}, el('tr', {}, [el('th', { text: '模型' }), ...METRICS.map(([, name]) => el('th', { text: name }))])),
    el('tbody', {}, rows.map((row) => el('tr', {}, [
      el('th', { text: row.model_type }),
      ...METRICS.map(([key]) => {
        const metric = row.metrics?.[key] || {};
        return el('td', {
          text: format(metric.mean),
          attrs: heat && Number.isFinite(metric.mean) ? { style: `background:${heatColor(metric.mean)}` } : {},
        });
      }),
    ]))),
  ]));
}

function metricVisualization(rows) {
  const complete = rows.length > 0 && rows.every((row) => METRICS.every(([key]) => Number.isFinite(row.metrics?.[key]?.mean)));
  if (rows.length <= 5 && complete) {
    const series = METRICS.map(([key, name]) => ({
      name,
      values: rows.map((row) => Number(row.metrics?.[key]?.mean) || 0),
    }));
    return el('div', { className: 'card comparison-chart-card' }, [
      groupedBarChart({
        labels: rows.map((row) => row.model_type),
        series,
        title: '四项测试指标分组对比',
        description: '按模型比较 Accuracy、Balanced Accuracy、Macro-F1 和 Weighted-F1。',
        yLabel: '得分（0–1）',
      }),
      chartLegend(series),
    ]);
  }
  return el('div', { className: 'card stack' }, [
    el('h3', { className: 'card-title', text: rows.length > 5 ? '四项测试指标热力表' : '四项测试指标明细' }),
    !complete ? el('p', { className: 'hint', text: '部分指标缺失，因此不绘制会误导为 0 的柱状图。' }) : null,
    metricTable(rows, { heat: true }),
  ]);
}

function heatTable(title, firstColumn, columns, rows, valueFor, { description = '' } = {}) {
  return el('section', { className: 'card stack comparison-section' }, [
    el('h2', { className: 'card-title', text: title }),
    description ? el('p', { className: 'hint', text: description }) : null,
    el('div', { className: 'table-wrap comparison-scroll' }, el('table', { className: 'data-table comparison-heat' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: firstColumn }), ...columns.map((column) => el('th', { text: column }))])),
      el('tbody', {}, rows.map((row) => el('tr', {}, [
        el('th', { text: row.model_type }),
        ...columns.map((column, index) => {
          const value = valueFor(row, column, index);
          return el('td', { text: value == null ? '—' : `${Math.round(value * 100)}%`, attrs: { style: `background:${heatColor(value)}` } });
        }),
      ]))),
    ])),
  ]);
}

function correctnessTable(correctness, models) {
  const columns = correctness.sample_ids || [];
  const modelByType = new Map(models.map((model) => [model.model_type, model]));
  return el('section', { className: 'card stack comparison-section' }, [
    el('h2', { className: 'card-title', text: 'Sample_ID × 模型预测正确/错误' }),
    el('p', { className: 'hint', text: '每个单元格表示该模型对一个 Sample_ID 的最终判断：绿色为正确，红色为错误。' }),
    el('div', { className: 'table-wrap comparison-scroll' }, el('table', { className: 'data-table comparison-heat' }, [
      el('thead', {}, el('tr', {}, [el('th', { text: '模型 / Sample_ID' }), ...columns.map((column) => el('th', { text: column }))])),
      el('tbody', {}, (correctness.values || []).map((row) => {
        const historical = Number(modelByType.get(row.model_type)?.requested_repeats || 1) > 1;
        return el('tr', {}, [
          el('th', { text: row.model_type }),
          ...columns.map((_column, index) => {
            const value = row.values?.[index];
            if (!Number.isFinite(Number(value))) {
              return el('td', { className: 'comparison-correctness-cell is-missing', text: '—' });
            }
            if (Number(value) >= 1) {
              return el('td', { className: 'comparison-correctness-cell is-correct', text: '正确' });
            }
            if (Number(value) <= 0) {
              return el('td', { className: 'comparison-correctness-cell is-wrong', text: '错误' });
            }
            if (historical) {
              return el('td', {
                className: 'comparison-correctness-cell',
                text: `${Math.round(Number(value) * 100)}% 正确`,
                attrs: { style: `background:${heatColor(value)}`, title: '历史重复批次中部分运行预测正确' },
              });
            }
            return el('td', { className: 'comparison-correctness-cell is-wrong', text: '错误' });
          }),
        ]);
      })),
    ])),
  ]);
}

function confusionMatrixCard(item, navigate) {
  const runId = item.run_ids?.[0] || '';
  return el('article', { className: 'card stack comparison-matrix-card' }, [
    el('div', { className: 'row spread' }, [
      el('h3', { className: 'card-title', text: item.model_type || '未知模型' }),
      (item.run_ids || []).length > 1 ? el('span', { className: 'badge', text: '历史合并结果' }) : null,
    ]),
    renderConfusionMatrix({
      matrix: item.confusion_matrix,
      labels: item.labels || [],
      runId,
      split: 'test',
      aggregation: 'batch',
      showDownload: false,
    }),
    runId ? el('button', {
      className: 'btn btn-ghost',
      text: '查看该模型详细结果',
      attrs: { type: 'button' },
      on: { click: () => navigate(`#/results?run_id=${encodeURIComponent(runId)}`) },
    }) : null,
  ]);
}

function renderComparisonContent(target, comparison, navigate) {
  clear(target);
  if (!comparison.comparable) {
    target.append(el('div', { className: 'card' }, el('p', { className: 'hint', text: comparison.reason || '等待可比较的成功结果。' })));
    return;
  }
  const models = modelRows(comparison);
  target.append(el('section', { className: 'stack comparison-section' }, [
    el('h2', { className: 'card-title', text: '总体指标比较' }),
    el('div', { className: 'comparison-model-grid' }, models.map(metricCard)),
    metricVisualization(models),
    metricTable(models),
  ]));
  const heatSections = [];
  const correctness = comparison.sample_correctness || {};
  if (correctness.status === 'ready') {
    heatSections.push(correctnessTable(correctness, models));
  }
  const recall = comparison.class_recall || {};
  if (recall.status === 'ready') {
    heatSections.push(heatTable(
      '按类别 Recall / Sensitivity',
      '模型 / 类别',
      recall.labels || [],
      recall.rows || [],
      (row, _label, index) => row.values?.[index]?.mean ?? null,
      { description: 'Recall = TP / (TP + FN)，按测试集预测记录统计。' },
    ));
  }
  if (heatSections.length) target.append(el('div', { className: 'comparison-heat-grid' }, heatSections));
  const matrices = comparison.confusion_matrices || [];
  target.append(el('section', { className: 'card stack comparison-section' }, [
    el('h2', { className: 'card-title', text: '各模型混淆矩阵' }),
    matrices.length
      ? el('div', { className: 'comparison-matrix-grid' }, matrices.map((item) => confusionMatrixCard(item, navigate)))
      : el('p', { className: 'hint', text: '当前没有可显示的混淆矩阵。' }),
  ]));
}

function renderBatchStatus(target, batch, onStop) {
  const counts = batch.counts || {};
  clear(target);
  target.append(el('div', { className: 'card stack' }, [
    el('div', { className: 'row spread' }, [
      el('h2', { className: 'card-title', text: `批次 ${batch.batch_id}` }),
      el('span', { className: 'badge', text: batch.state || 'queued' }),
    ]),
    el('p', { className: 'hint', text: `排队 ${counts.queued || 0} · 运行 ${counts.running || 0} · 成功 ${counts.succeeded || 0} · 失败 ${counts.failed || 0} · 已停止 ${counts.cancelled || 0}` }),
    ['queued', 'running'].includes(batch.state)
      ? el('button', { className: 'btn btn-ghost', text: '停止整个批次', attrs: { type: 'button' }, on: { click: onStop } })
      : null,
  ]));
}

export function mountComparison(container, { route, navigate, toast }) {
  const batchId = route.batchId;
  const root = el('section', { className: 'stack comparison-page' });
  const statusRegion = el('div', { className: 'comparison-status-region' });
  const errorRegion = el('div', { className: 'error-text comparison-error', attrs: { role: 'status' } });
  const contentRegion = el('div', { className: 'stack comparison-content' });
  root.append(statusRegion, errorRegion, contentRegion);
  container.append(root);
  if (!batchId) {
    statusRegion.append(el('div', { className: 'card' }, el('p', { className: 'hint', text: '请从批量训练创建完成页打开模型比较。' })));
    return {};
  }

  let timer = null;
  let controller = null;
  let disposed = false;
  let retryCount = 0;
  let lastComparisonSignature = '';

  const schedule = (delay) => {
    if (!disposed) timer = window.setTimeout(render, delay);
  };

  async function render() {
    if (disposed) return;
    if (timer) clearTimeout(timer);
    controller?.abort();
    controller = new AbortController();
    try {
      const [batch, comparison] = await Promise.all([
        getBatch(batchId, { signal: controller.signal }),
        getBatchComparison(batchId, { signal: controller.signal }),
      ]);
      if (disposed) return;
      retryCount = 0;
      errorRegion.textContent = '';
      renderBatchStatus(statusRegion, batch, async () => {
        if (!window.confirm(`确定停止批次 ${batchId} 吗？`)) return;
        await stopBatch(batchId);
        toast('已请求停止批次。');
        render();
      });
      const signature = JSON.stringify(comparison);
      if (signature !== lastComparisonSignature) {
        renderComparisonContent(contentRegion, comparison, navigate);
        lastComparisonSignature = signature;
      }
      if (['queued', 'running'].includes(batch.state)) schedule(3000);
    } catch (error) {
      if (disposed || error?.name === 'AbortError') return;
      retryCount += 1;
      errorRegion.textContent = `批次状态暂时无法刷新：${error?.message || '网络异常'}。已保留上一次结果，将自动重试。`;
      schedule(Math.min(3000 * (2 ** (retryCount - 1)), 30000));
    }
  }

  statusRegion.append(el('div', { className: 'card' }, el('p', { className: 'hint', text: '正在读取批次状态…' })));
  render();
  return {
    unmount() {
      disposed = true;
      if (timer) clearTimeout(timer);
      controller?.abort();
    },
  };
}
