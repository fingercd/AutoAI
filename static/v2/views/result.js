/** 建模结果 / 解释性视图：只消费 run-result-v1，single-flight 轮询，终态停止。 */
import { el, clear, saveBlob } from '../lib/dom.js';
import {
  stateMeta, resultStateMeta, isActiveState, formatMetric, formatDateTime, formatDuration,
  SCALAR_METRIC_KEYS, METRIC_LABELS, EVALUATION_STRATEGIES, AGGREGATION_LABELS,
  resolveSplitScalars, validateCvAggregation, isCvResult, findServerPaths,
  resultStateBadgeClass,
} from '../lib/format.js';
import { createPoller } from '../lib/poller.js';
import { lineChart, groupedBarChart, chartLegend } from '../lib/charts.js';
import { getRunResult, getRun, getHealth, listRunsSummary, download, downloadArtifactJson } from '../api.js';
import { renderConfusionMatrix } from '../components/confusion-matrix.js';
import { renderArtifacts } from '../components/artifacts.js';
import { renderExplainabilityPanel } from '../components/explainability-panel.js';
import { stateBadge } from '../components/run-list.js';

const POLL_INTERVAL_MS = 3000;
const RESULT_SCHEMA = 'run-result-v1';

const SPLIT_LABELS = { train: 'Train', valid: 'Valid', test: 'Test' };

/** 小标签 + 值的元信息块，使用共享 grid 布局。 */
function metaBlocks(items, gridClass = 'grid grid-3') {
  return el('div', { className: gridClass }, items.map(([key, value]) =>
    el('div', {}, [
      el('p', { className: 'hint', text: key }),
      el('p', { text: String(value ?? '—') }),
    ])));
}

/** 结果完整性徽章：partial / manifest 问题用需要注意的色调，文案承担语义。 */
function resultBadge(resultState) {
  const meta = resultStateMeta(resultState);
  return el('span', { className: resultStateBadgeClass(resultState), text: `结果：${meta.label}` });
}

function metricCards(primary, aggregation) {
  const values = primary && typeof primary === 'object' ? primary : {};
  const cards = SCALAR_METRIC_KEYS
    .filter((key) => values[key] !== null && values[key] !== undefined)
    .map((key) => el('div', { className: 'metric-card' }, [
      el('span', { className: 'metric-label', text: METRIC_LABELS[key] || key }),
      el('span', { className: 'metric-value', text: formatMetric(values[key]) }),
    ]));
  if (!cards.length) return el('p', { className: 'hint', text: '没有可展示的标量指标。' });
  return el('div', {}, [
    aggregation ? el('p', { className: 'hint', text: `主指标口径：${AGGREGATION_LABELS[aggregation] || aggregation}` }) : null,
    el('div', { className: 'metric-grid' }, cards),
  ]);
}

function foldAuditTable(result) {
  const foldMean = result?.metrics?.fold_mean;
  const foldStd = result?.metrics?.fold_std;
  if (!foldMean || typeof foldMean !== 'object' || !Object.keys(foldMean).length) return null;
  const splits = ['train', 'valid', 'test'].filter((name) => foldMean[name] && typeof foldMean[name] === 'object');
  if (!splits.length) return null;
  const wrap = el('div', { className: 'table-wrap' });
  wrap.append(el('table', { className: 'data-table' }, [
    el('caption', { text: '逐折审计指标（fold mean ± fold std）；Test 主指标以 pooled OOF 为准，不与此表混用。' }),
    el('thead', {}, el('tr', {}, [
      el('th', { text: '分区', attrs: { scope: 'col' } }),
      SCALAR_METRIC_KEYS.map((key) => el('th', { text: METRIC_LABELS[key] || key, attrs: { scope: 'col' } })),
    ])),
    el('tbody', {}, splits.map((name) => el('tr', {}, [
      el('th', { text: name, attrs: { scope: 'row' } }),
      SCALAR_METRIC_KEYS.map((key) => {
        const mean = foldMean[name]?.[key];
        const std = foldStd?.[name]?.[key];
        return el('td', { text: mean === null || mean === undefined ? '—' : `${formatMetric(mean)}${std === null || std === undefined ? '' : ` ± ${formatMetric(std)}`}` });
      }),
    ]))),
  ]));
  return wrap;
}

function historyChart(history) {
  if (!history?.available || !Array.isArray(history.rows) || !history.rows.length) return null;
  const columns = Array.isArray(history.columns) ? history.columns : Object.keys(history.rows[0] || {});
  const epochKey = columns.find((key) => key.toLowerCase() === 'epoch');
  const numericKeys = columns
    .filter((key) => key !== epochKey)
    .filter((key) => history.rows.every((row) => row[key] === null || row[key] === undefined || Number.isFinite(Number(row[key]))))
    .slice(0, 4);
  if (!numericKeys.length) return null;
  const xs = history.rows.map((row, index) => (epochKey ? Number(row[epochKey]) : index + 1));
  const series = numericKeys.map((key) => ({
    name: key,
    xs,
    ys: history.rows.map((row) => Number(row[key])),
  }));
  return el('div', {}, [
    el('div', { className: 'chart-card' }, [
      el('div', { className: 'scroll-x' }, [
        lineChart({
          series,
          title: '训练过程曲线',
          description: `深度模型真实 epoch 曲线：${numericKeys.join('、')}。`,
          xLabel: epochKey || 'epoch',
          yLabel: '数值',
        }),
      ]),
      chartLegend(series),
    ]),
    history.truncated ? el('p', { className: 'hint', text: '曲线过长，响应已被后端截断，仅展示前 1000 行。' }) : null,
  ]);
}

function splitPanel(result, split) {
  const analysis = result?.analysis?.splits?.[split];
  const scalars = resolveSplitScalars(result, split);
  const host = el('div', { className: 'stack' });
  const aggregation = analysis?.aggregation || scalars?.aggregation;
  host.append(el('p', { className: 'hint', text: `口径：${AGGREGATION_LABELS[aggregation] || aggregation || '—'}${scalars?.note ? `；${scalars.note}` : ''}` }));
  const values = scalars?.values;
  if (values && typeof values === 'object') {
    host.append(el('div', { className: 'metric-grid' },
      SCALAR_METRIC_KEYS
        .filter((key) => values[key] !== null && values[key] !== undefined)
        .map((key) => el('div', { className: 'metric-card' }, [
          el('span', { className: 'metric-label', text: METRIC_LABELS[key] || key }),
          el('span', { className: 'metric-value', text: formatMetric(values[key]) }),
        ]))));
    if (scalars.foldStd && typeof scalars.foldStd === 'object' && Object.keys(scalars.foldStd).length) {
      host.append(el('p', { className: 'hint', text: `审计 fold std：${SCALAR_METRIC_KEYS.filter((key) => scalars.foldStd[key] != null).map((key) => `${METRIC_LABELS[key] || key} ${formatMetric(scalars.foldStd[key])}`).join('，')}` }));
    }
  } else {
    host.append(el('p', { className: 'hint', text: '该分区没有标量指标。' }));
  }
  const labels = analysis?.prediction_distribution?.labels
    || Object.keys(analysis?.classification_report || {}).filter((key) => !['accuracy', 'macro avg', 'weighted avg'].includes(key));
  host.append(el('h3', { text: '混淆矩阵' }));
  host.append(renderConfusionMatrix({
    matrix: analysis?.confusion_matrix,
    labels: labels || [],
    caption: `${SPLIT_LABELS[split]} 分区混淆矩阵`,
  }));
  const distribution = analysis?.prediction_distribution;
  if (distribution?.labels?.length) {
    const series = [
      { name: '真实数量', values: distribution.true_counts || [], color: '#2563eb' },
      { name: '预测数量', values: distribution.predicted_counts || [], color: '#d97706' },
    ];
    host.append(el('h3', { text: '预测分布' }));
    host.append(el('div', { className: 'chart-card' }, [
      el('div', { className: 'scroll-x' }, [
        groupedBarChart({
          labels: distribution.labels,
          series,
          title: `${SPLIT_LABELS[split]} 分区预测分布`,
          description: '按类别对比真实与预测样本数量。',
        }),
      ]),
      chartLegend(series),
    ]));
  }
  return host;
}

function analysisSection(result) {
  const host = el('div', { className: 'card' });
  host.append(el('h2', { className: 'card-title', text: '分区分析' }));
  host.append(el('p', { className: 'hint', text: '先复核 Test 分区，再对照 Train/Valid 判断过拟合或划分问题。' }));
  const splits = ['train', 'valid', 'test'];
  const tabButtons = [];
  const panelHost = el('div', { className: 'panel', attrs: { role: 'tabpanel' } });
  const show = (split) => {
    for (const button of tabButtons) {
      const active = button.dataset.split === split;
      button.classList.toggle('tab-active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
    }
    clear(panelHost);
    panelHost.append(splitPanel(result, split));
  };
  const tabs = el('div', { className: 'tabs', attrs: { role: 'tablist', 'aria-label': '数据分区' } });
  for (const split of splits) {
    const button = el('button', {
      className: 'tab',
      text: SPLIT_LABELS[split],
      attrs: { type: 'button', role: 'tab', 'aria-selected': 'false', tabindex: '-1' },
      dataset: { split },
      on: { click: () => show(split) },
    });
    tabButtons.push(button);
    tabs.append(button);
  }
  tabs.addEventListener('keydown', (event) => {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    const index = tabButtons.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    let next = index;
    if (event.key === 'ArrowRight') next = Math.min(splits.length - 1, index + 1);
    if (event.key === 'ArrowLeft') next = Math.max(0, index - 1);
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = splits.length - 1;
    if (next !== index) {
      show(splits[next]);
      tabButtons[next].focus();
    }
  });
  host.append(tabs, panelHost);
  show('test');
  return host;
}

function unavailableAnalysisNotes(result) {
  const notes = [];
  for (const [key, label] of [['roc', 'ROC / ROC-AUC'], ['precision_recall', 'Precision-Recall']]) {
    const entry = result?.analysis?.[key];
    if (entry && entry.available === false) {
      notes.push(el('li', { text: `${label}：${entry.reason || '当前训练产物未计算'}` }));
    }
  }
  if (!notes.length) return null;
  return el('div', { className: 'card' }, [
    el('h2', { className: 'card-title', text: '暂不可用的分析' }),
    el('p', { className: 'hint', text: '以下能力当前没有正式计算产物，不绘制空图、不伪造数值：' }),
    el('ul', {}, notes),
  ]);
}

/** 结论卡：主指标 + 结果完整性放在首屏；评估细节与逐折审计默认折叠。 */
function conclusionCard(result) {
  const run = result.run || {};
  const evaluation = result.evaluation || {};
  const strategyMeta = EVALUATION_STRATEGIES[evaluation.strategy];
  const resultMeta = resultStateMeta(run.result_state);
  const modelText = result.model?.type
    ? `${result.model.type}${result.model.family ? `（${result.model.family}）` : ''}`
    : '—';
  const card = el('div', { className: 'card' });
  card.append(el('div', { className: 'row spread' }, [
    el('h2', { className: 'card-title', text: '结论' }),
    resultBadge(run.result_state),
  ]));
  card.append(el('p', { className: 'hint', text: [
    strategyMeta ? strategyMeta.shortLabel : evaluation.strategy,
    evaluation.fold_count !== null && evaluation.fold_count !== undefined ? `${evaluation.fold_count} 折` : null,
    `主分区 ${evaluation.primary_split || 'test'}`,
    `主口径 ${AGGREGATION_LABELS[evaluation.primary_aggregation] || evaluation.primary_aggregation || '—'}`,
    result.model?.type ? `模型 ${modelText}` : null,
  ].filter(Boolean).join(' · ') }));
  card.append(metricCards(result.metrics?.primary, evaluation.primary_aggregation));
  card.append(el('p', { className: 'hint', text: `结果完整性：${resultMeta.label}。${resultMeta.description || ''}` }));
  if (isCvResult(result)) {
    card.append(el('p', { className: 'hint', text: '交叉验证：Test 主指标为全部折合并的 OOF 预测（pooled OOF）；Train/Valid 标量为逐折均值，两者不混用。' }));
  }

  const detailHost = el('div', { className: 'stack', attrs: { id: 'v2-eval-details', hidden: true } });
  detailHost.append(metaBlocks([
    ['评估口径', strategyMeta ? strategyMeta.label : evaluation.strategy || '—'],
    ['折数', evaluation.fold_count ?? '—'],
    ['主分区', evaluation.primary_split || 'test'],
    ['主口径', AGGREGATION_LABELS[evaluation.primary_aggregation] || evaluation.primary_aggregation || '—'],
    ['模型', modelText],
  ]));
  const audit = foldAuditTable(result);
  if (audit) detailHost.append(audit);
  const toggle = el('button', {
    className: 'btn btn-ghost btn-sm',
    text: '评估与审计明细',
    attrs: { type: 'button', 'aria-expanded': 'false', 'aria-controls': 'v2-eval-details' },
    on: {
      click: () => {
        const opening = detailHost.hidden;
        detailHost.hidden = !opening;
        toggle.setAttribute('aria-expanded', opening ? 'true' : 'false');
        toggle.textContent = opening ? '收起评估与审计明细' : '评估与审计明细';
      },
    },
  });
  card.append(el('div', { className: 'row' }, [toggle]), detailHost);
  return card;
}

function datasetCard(result) {
  const dataset = result.dataset || {};
  return el('div', { className: 'card' }, [
    el('h2', { className: 'card-title', text: '数据集快照' }),
    metaBlocks([
      ['名称', dataset.name],
      ['曲线数', dataset.curve_count],
      ['Sample_ID 数', dataset.sample_id_count],
      ['类别数', dataset.class_count],
      ['特征数', dataset.feature_count],
      ['独立测试曲线数', dataset.test_curve_count],
    ]),
  ]);
}

/** 单样品解释：默认展开并加载，仍可由用户收起以减少页面长度。 */
function explainabilityCard(result) {
  const body = el('div', { attrs: { id: 'v2-explain-panel' } });
  const toggle = el('button', {
    className: 'btn btn-ghost',
    text: '收起单样品解释',
    attrs: { type: 'button', 'aria-expanded': 'true', 'aria-controls': 'v2-explain-panel' },
    on: {
      click: () => {
        const opening = body.hidden;
        body.hidden = !opening;
        toggle.setAttribute('aria-expanded', opening ? 'true' : 'false');
        toggle.textContent = opening ? '收起单样品解释' : '展开单样品解释';
      },
    },
  });
  renderExplainabilityPanel(body, {
    summary: result.explainability?.samples,
    artifacts: result.artifacts || [],
    onLoadJson: (url) => downloadArtifactJson(url),
  });
  return el('div', { className: 'card' }, [
    el('h2', { className: 'card-title', text: '单样品解释' }),
    el('p', { className: 'hint', text: '解释数据默认自动加载；加载后切换样品不会重复请求。' }),
    toggle,
    body,
  ]);
}

function artifactsCard(result, { toast }) {
  const host = el('div');
  renderArtifacts(host, {
    artifacts: result.artifacts || [],
    onDownload: async (item) => {
      try {
        const { blob, filename } = await download(item.download_url);
        saveBlob(blob, item.suggested_filename || filename || item.name);
      } catch (error) {
        toast(`下载 ${item.name} 失败：${error?.message || '未知错误'}`, { type: 'error' });
      }
    },
  });
  return el('div', { className: 'card' }, [
    el('h2', { className: 'card-title', text: '产物下载' }),
    el('p', { className: 'hint', text: '逐项下载；禁用项显示原因。模型对象、权重与内部清单不开放下载。' }),
    host,
  ]);
}

function fullResult(result, { toast }) {
  const frag = el('div', { className: 'stack' });
  const cvCheck = validateCvAggregation(result);
  if (!cvCheck.ok) {
    frag.append(el('div', { className: 'card' }, [
      el('h2', { className: 'card-title', text: '口径校验提醒' }),
      el('ul', {}, cvCheck.violations.map((item) => el('li', { className: 'error-text', text: item }))),
    ]));
  }
  frag.append(conclusionCard(result));
  frag.append(analysisSection(result));
  const history = historyChart(result.analysis?.history);
  if (history) {
    frag.append(el('div', { className: 'card' }, [el('h2', { className: 'card-title', text: '训练过程' }), history]));
  }
  frag.append(datasetCard(result));
  const unavailable = unavailableAnalysisNotes(result);
  if (unavailable) frag.append(unavailable);
  frag.append(explainabilityCard(result));
  frag.append(artifactsCard(result, { toast }));
  return frag;
}

function statusView(result) {
  const run = result.run || {};
  const meta = stateMeta(run.state);
  const resultMeta = resultStateMeta(run.result_state);
  const progress = run.progress || {};
  const items = [
    ['Run ID', run.run_id],
    ['执行状态', `${meta.label}（${run.state || '—'}）`],
    ['结果状态', `${resultMeta.label}（${run.result_state || '—'}）`],
    ['创建时间', formatDateTime(run.created_at)],
    ['开始时间', run.started_at ? formatDateTime(run.started_at) : '未开始'],
    ['结束时间', formatDateTime(run.finished_at)],
    ['耗时', formatDuration(run.duration_seconds)],
  ];
  if (progress.fold_progress_text) items.push(['折进度', `${progress.fold_progress_text}${progress.current_fold_sample_id ? `（当前折 Sample_ID：${progress.current_fold_sample_id}）` : ''}`]);
  if (progress.target_epochs) items.push(['目标 epochs', progress.target_epochs]);
  const host = el('div', { className: 'card' }, [
    el('h2', { className: 'card-title', text: '任务状态' }),
    metaBlocks(items),
    resultMeta.description ? el('p', { className: 'hint', text: resultMeta.description }) : null,
  ]);
  if (isActiveState(run.state)) {
    host.append(el('p', { className: 'hint', attrs: { role: 'status' }, text: '正在自动刷新（每 3 秒，串行执行），进入终态后自动停止。' }));
  }
  if (run.state === 'failed' && run.error) {
    host.append(el('div', { className: 'stack' }, [
      el('h3', { text: '失败信息' }),
      el('p', { className: 'error-text', text: run.error.message || '—' }),
      metaBlocks([
        ['错误码', run.error.code || '—'],
        ['阶段', run.error.stage || '—'],
        ['可重试', run.error.retryable ? '是' : '否'],
      ]),
    ]));
  }
  return host;
}

export function mountResult(container, { route, announce, toast, navigate }) {
  const root = el('section', { className: 'stack', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);
  let poller = null;
  let disposed = false;

  async function renderRecent() {
    root.append(el('p', { className: 'hint', attrs: { role: 'status' }, text: '正在加载最近任务…' }));
    try {
      const result = await listRunsSummary({ limit: 20 });
      if (disposed) return;
      clear(root);
      const items = Array.isArray(result?.items) ? result.items : [];
      if (!items.length) {
        root.append(el('div', { className: 'card' }, [
          el('div', { className: 'empty' }, [
            el('p', { text: '还没有训练记录。' }),
            el('p', { className: 'hint', text: '先在 AI 建模页提交一次训练，结果会出现在这里。' }),
            el('button', { className: 'btn btn-primary', text: '去建模', attrs: { type: 'button' }, on: { click: () => navigate('#/modeling') } }),
          ]),
        ]));
        return;
      }
      const wrap = el('div', { className: 'table-wrap' });
      wrap.append(el('table', { className: 'data-table' }, [
        el('caption', { text: '最近训练任务' }),
        el('thead', {}, el('tr', {}, ['Run ID', '模型 · 数据集', '状态', '开始时间', '耗时', '操作'].map((head) => el('th', { text: head, attrs: { scope: 'col' } })))),
        el('tbody', {}, items.map((item) => el('tr', {}, [
          el('td', {}, el('code', { text: item.run_id })),
          el('td', { text: `${item.model_type || '—'} · ${item.dataset_name || '—'}` }),
          el('td', {}, stateBadge(item.state)),
          el('td', { text: item.started_at ? formatDateTime(item.started_at) : `未开始（创建于 ${formatDateTime(item.created_at)}）` }),
          el('td', { text: formatDuration(item.duration_seconds) }),
          el('td', {}, el('button', {
            className: 'btn btn-primary btn-sm',
            text: '查看结果',
            attrs: { type: 'button' },
            on: { click: () => navigate(`#/results?run_id=${encodeURIComponent(item.run_id)}`) },
          })),
        ]))),
      ]));
      root.append(el('div', { className: 'card' }, [
        el('div', { className: 'row spread' }, [
          el('h2', { className: 'card-title', text: '最近任务' }),
          el('div', { className: 'card-actions' }, [
            el('button', { className: 'btn btn-ghost btn-sm', text: '打开训练记录', attrs: { type: 'button' }, on: { click: () => navigate('#/runs') } }),
          ]),
        ]),
        wrap,
      ]));
    } catch (error) {
      if (disposed) return;
      clear(root);
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '最近任务加载失败' }),
        el('p', { className: 'error-text', attrs: { role: 'alert' }, text: error?.message || '未知错误' }),
        el('div', { className: 'row' }, [
          el('button', { className: 'btn btn-primary', text: '重试', attrs: { type: 'button' }, on: { click: () => { clear(root); renderRecent(); } } }),
        ]),
      ]));
    }
  }

  function renderPayload(result) {
    clear(root);
    if (result?.schema_version !== RESULT_SCHEMA) {
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '结果契约不受支持' }),
        el('p', { className: 'error-text', text: `期望 ${RESULT_SCHEMA}，实际为 ${result?.schema_version || '未知'}。请确认 Web 与 Worker 版本一致。` }),
      ]));
      return;
    }
    const leaks = findServerPaths(result, { ignoreKeys: ['download_url'] });
    if (leaks.length) {
      // 不展示原始载荷，只提示存在不合规字段；下载 URL 由后端生成，按整体使用。
      console.warn('[v2] 结果响应包含疑似服务器路径字段：', leaks.join(', '));
    }
    const run = result.run || {};
    const resultMeta = resultStateMeta(run.result_state);
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    root.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('div', { className: 'stack' }, [
          el('h2', { className: 'card-title', text: `Run ${run.run_id || route.runId}` }),
          el('div', { className: 'row' }, [
            stateBadge(run.state),
            resultBadge(run.result_state),
          ]),
        ]),
        el('div', { className: 'card-actions' }, [
          el('button', { className: 'btn btn-ghost btn-sm', text: '返回训练记录', attrs: { type: 'button' }, on: { click: () => navigate('#/runs') } }),
        ]),
      ]),
      metaBlocks([
        ['创建时间', formatDateTime(run.created_at)],
        ['开始时间', run.started_at ? formatDateTime(run.started_at) : '未开始'],
        ['结束时间', formatDateTime(run.finished_at)],
        ['耗时', formatDuration(run.duration_seconds)],
      ], 'grid grid-2'),
      resultMeta.description ? el('p', { className: 'hint', text: resultMeta.description }) : null,
    ]));
    if (warnings.length) {
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '结果提醒' }),
        el('ul', {}, warnings.map((warning) => el('li', { text: warning }))),
      ]));
    }
    if (run.state === 'failed' || run.state === 'cancelled' || isActiveState(run.state)) {
      root.append(statusView(result));
      return;
    }
    root.append(fullResult(result, { toast }));
  }

  async function handleNotFound() {
    let health = null;
    try {
      health = await getHealth();
    } catch (_) {
      health = null;
    }
    if (disposed) return;
    clear(root);
    if (health?.contracts?.run_result === RESULT_SCHEMA) {
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '任务不存在或不可见' }),
        el('p', { className: 'hint', text: `Run ${route.runId} 不存在、已删除，或不属于当前账号作用域。` }),
        el('div', { className: 'row' }, [
          el('button', { className: 'btn btn-ghost', text: '返回训练记录', attrs: { type: 'button' }, on: { click: () => navigate('#/runs') } }),
        ]),
      ]));
      return;
    }
    // 仅在 /health 明确未声明 run-result-v1 时回退旧状态接口。
    try {
      const legacy = await getRun(route.runId);
      if (disposed) return;
      clear(root);
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '后端不支持 run-result-v1' }),
        el('p', { className: 'hint', text: '当前 Web 未声明 run-result-v1 契约，已回退到兼容状态接口；请升级后端以查看完整结果。' }),
        metaBlocks([
          ['Run ID', legacy?.run_id || route.runId],
          ['状态', legacy?.state || legacy?.status || '—'],
          ['模型', legacy?.config?.model_type || '—'],
        ]),
      ]));
    } catch (error) {
      if (disposed) return;
      clear(root);
      root.append(el('div', { className: 'card' }, [
        el('h2', { className: 'card-title', text: '任务不存在或不可见' }),
        el('p', { className: 'error-text', text: error?.message || `Run ${route.runId} 不存在或不可见。` }),
      ]));
    }
  }

  function renderLoadError(error) {
    clear(root);
    root.append(el('div', { className: 'card' }, [
      el('h2', { className: 'card-title', text: '结果加载失败' }),
      el('p', { className: 'error-text', text: error?.message || '网络或服务器错误。' }),
      el('div', { className: 'row' }, [
        el('button', { className: 'btn btn-primary', text: '重试', attrs: { type: 'button' }, on: { click: start } }),
        el('button', { className: 'btn btn-ghost', text: '返回训练记录', attrs: { type: 'button' }, on: { click: () => navigate('#/runs') } }),
      ]),
    ]));
  }

  function start() {
    poller?.stop();
    clear(root);
    const skeleton = el('div', { className: 'skeleton', attrs: { 'aria-hidden': 'true' } });
    skeleton.style.height = '180px';
    root.append(
      el('p', { className: 'hint', attrs: { role: 'status' }, text: '正在加载建模结果…' }),
      el('div', { className: 'card' }, [skeleton]),
    );
    poller = createPoller({
      intervalMs: POLL_INTERVAL_MS,
      task: ({ signal }) => getRunResult(route.runId, { signal }),
      shouldContinue: (data) => isActiveState(data?.run?.state),
      onResult: (data) => {
        if (disposed) return;
        renderPayload(data);
        if (isActiveState(data?.run?.state)) announce(`Run ${route.runId} ${stateMeta(data.run.state).label}`);
      },
      onError: (error) => {
        if (disposed) return false;
        if (error?.status === 404) {
          handleNotFound();
          return false;
        }
        renderLoadError(error);
        return false;
      },
    });
    poller.start();
  }

  if (route.runId) {
    start();
  } else {
    renderRecent();
  }

  return {
    unmount() {
      disposed = true;
      poller?.stop();
    },
  };
}
