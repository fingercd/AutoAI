import { downloadFile, request } from './api-client.js';
import { element, formatMetric, formatTime, replaceChildren } from './ui-utils.js';

export const AUTO_REDIRECT_DELAY_MS = 3000;
const ACTIVE_STATES = new Set(['queued', 'pending', 'running']);
const TERMINAL_STATES = new Set(['succeeded', 'success', 'failed', 'cancelled', 'paused', 'ready', 'partial', 'missing_manifest', 'corrupt_manifest']);
const VIEW_ALIASES = {
  raman: 'raman',
  hplc: 'chromatography',
  chromatography: 'chromatography',
  modeling: 'modeling',
  results: 'results',
  runs: 'runs',
  help: 'manual',
  manual: 'manual',
};
const VIEW_PATHS = {
  raman: 'raman',
  chromatography: 'hplc',
  modeling: 'modeling',
  results: 'results',
  runs: 'runs',
  manual: 'help',
};

export function parseHash(hash = '') {
  const raw = String(hash || '').replace(/^#/, '').replace(/^\//, '');
  const [rawPath = '', rawQuery = ''] = raw.split('?', 2);
  const view = VIEW_ALIASES[rawPath.toLowerCase()] || 'raman';
  const params = new URLSearchParams(rawQuery);
  return { view, runId: params.get('run_id') || null };
}

export function buildViewHash(view) {
  return `#/${VIEW_PATHS[view] || 'raman'}`;
}

export function buildResultHash(runId) {
  const normalized = String(runId || '').trim();
  return normalized ? `#/results?run_id=${encodeURIComponent(normalized)}` : '#/results';
}

export function shouldAutoRedirect({ runId, state, isNewRun }) {
  return Boolean(String(runId || '').trim())
    && Boolean(isNewRun)
    && ['succeeded', 'success', 'ready', 'partial'].includes(String(state || '').toLowerCase());
}

export class RunPollController {
  constructor(fetcher, options = {}) {
    this.fetcher = fetcher;
    this.baseDelay = options.baseDelay ?? 1200;
    this.maxDelay = options.maxDelay ?? 10000;
    this.timer = null;
    this.controller = null;
    this.generation = 0;
    this.failures = 0;
    this.runId = null;
  }

  stop() {
    this.generation += 1;
    if (this.timer != null) clearTimeout(this.timer);
    this.timer = null;
    this.controller?.abort();
    this.controller = null;
    this.runId = null;
  }

  watch(runId, handlers = {}) {
    this.stop();
    this.runId = String(runId || '');
    this.failures = 0;
    const generation = this.generation;
    const schedule = (delay) => {
      if (generation !== this.generation || !this.runId) return;
      const visibilityDelay = typeof document !== 'undefined' && document.hidden ? Math.max(delay, 5000) : delay;
      this.timer = setTimeout(tick, visibilityDelay);
    };
    const tick = async () => {
      if (generation !== this.generation || !this.runId) return;
      this.controller = new AbortController();
      try {
        const payload = await this.fetcher(this.runId, this.controller.signal);
        if (generation !== this.generation) return;
        this.failures = 0;
        await handlers.onData?.(payload, this.runId);
        if (handlers.isTerminal?.(payload) !== false) {
          this.stop();
          return;
        }
        schedule(this.baseDelay);
      } catch (error) {
        if (generation !== this.generation || error?.name === 'AbortError') return;
        this.failures += 1;
        await handlers.onError?.(error, this.runId, this.failures);
        if (handlers.isTerminalError?.(error)) {
          this.stop();
          return;
        }
        schedule(Math.min(this.baseDelay * (2 ** Math.min(this.failures, 4)), this.maxDelay));
      } finally {
        if (generation === this.generation) this.controller = null;
      }
    };
    schedule(0);
  }
}

function byId(id) {
  return document.getElementById(id);
}

function canonicalState(payload) {
  return String(
    payload?.run?.result_state
    || payload?.result_state
    || payload?.run?.state
    || payload?.state
    || payload?.run?.status
    || payload?.status
    || 'pending',
  ).toLowerCase();
}

function isActivePayload(payload) {
  return ACTIVE_STATES.has(canonicalState(payload));
}

function normalizedLegacyResult(payload) {
  const metrics = payload?.metrics || {};
  const state = canonicalState(payload);
  return {
    schema_version: 'legacy-run-projection',
    run: {
      run_id: payload?.run_id,
      state: payload?.state || (payload?.status === 'success' ? 'succeeded' : payload?.status),
      status: payload?.status,
      result_state: state === 'success' || state === 'succeeded' ? 'ready' : state,
      created_at: payload?.created_at,
      started_at: payload?.started_at,
      finished_at: payload?.completed_at || payload?.finished_at,
      duration_seconds: payload?.duration_seconds,
      progress: payload?.progress,
      error: payload?.error,
    },
    dataset: {
      dataset_id: payload?.dataset_id,
      name: payload?.dataset_name || payload?.config?.dataset_name,
      curve_count: payload?.sample_count,
      sample_id_count: payload?.sample_id_count,
      class_count: payload?.class_count,
      feature_count: payload?.feature_count,
    },
    model: {
      type: payload?.model_type || payload?.config?.model_type,
      family: payload?.model_family,
      artifact_note: payload?.model_artifact,
    },
    evaluation: {
      strategy: payload?.evaluation_strategy || payload?.config?.evaluation_strategy,
      fold_count: payload?.fold_count || payload?.config?.fold_count,
      primary_split: 'test',
      primary_aggregation: payload?.evaluation_strategy === 'leave_one_sample_id_cv' ? 'pooled_oof' : 'direct',
    },
    metrics: {
      primary: metrics?.test || metrics,
      splits: { train: metrics?.train || {}, valid: metrics?.valid || {}, test: metrics?.test || metrics },
      cv: {},
    },
    analysis: {
      confusion_matrix: metrics?.test?.confusion_matrix || metrics?.confusion_matrix || [],
      classification_report: metrics?.test?.classification_report || metrics?.classification_report || {},
      prediction_distribution: null,
      history: { available: Array.isArray(payload?.history) && payload.history.length > 0, rows: payload?.history || [] },
    },
    explainability: {
      global: payload?.feature_importance || {},
      samples: payload?.sample_feature_importance || {},
    },
    artifacts: Array.isArray(payload?.artifacts) ? payload.artifacts : [],
    warnings: ['这是历史 Run 的兼容投影；部分时间、数据快照或下载清单可能未记录。'],
    label_names: payload?.label_names || [],
  };
}

export function normalizeResult(payload) {
  if (payload?.schema_version === 'run-result-v1' && payload?.run) return payload;
  return normalizedLegacyResult(payload || {});
}

async function fetchResult(runId, signal) {
  const encoded = encodeURIComponent(runId);
  try {
    return await request(`/api/training/runs/${encoded}/result`, { signal });
  } catch (error) {
    if (error?.status !== 404) throw error;
    return request(`/api/training/runs/${encoded}`, { signal });
  }
}

const trainingPoller = new RunPollController(
  (runId, signal) => request(`/api/training/runs/${encodeURIComponent(runId)}`, { signal }),
);
const resultPoller = new RunPollController(fetchResult);
const newRunIds = new Set();
let currentResultRunId = null;
let resultRenderGeneration = 0;
let lastRenderedResult = null;
let redirectTimer = null;
let redirectTicker = null;
let redirectRunId = null;
let authReturnFocus = null;

function stateMeta(state) {
  const states = {
    queued: ['等待训练', 'pending'],
    pending: ['等待训练', 'pending'],
    running: ['训练中', 'running'],
    succeeded: ['训练完成', 'success'],
    success: ['训练完成', 'success'],
    ready: ['结果可用', 'success'],
    partial: ['部分结果可用', 'warning'],
    missing_manifest: ['结果清单缺失', 'failed'],
    corrupt_manifest: ['结果清单损坏', 'failed'],
    failed: ['训练失败', 'failed'],
    cancelled: ['训练已取消', 'cancelled'],
    paused: ['训练已取消', 'cancelled'],
  };
  return states[state] || ['状态未知', ''];
}

function resultState(result) {
  return String(result?.run?.result_state || result?.run?.state || result?.run?.status || 'pending').toLowerCase();
}

function evaluationLabel(strategy) {
  const labels = {
    stratified_holdout: '分层留出（Train / Valid / Test）',
    leave_one_sample_id_cv: '按 Sample_ID 留一交叉验证',
    external_test_holdout: '独立测试集最终评估',
  };
  return labels[strategy] || strategy || '-';
}

function aggregationLabel(value) {
  const labels = {
    pooled_oof: '合并 OOF 预测',
    fold_mean: '各折均值',
    fold_std: '各折标准差',
    direct: '直接计算',
  };
  return labels[value] || value || '-';
}

function valueOrDash(value) {
  return value == null || value === '' ? '-' : String(value);
}

function durationText(run) {
  const supplied = Number(run?.duration_seconds);
  if (Number.isFinite(supplied)) return supplied < 60 ? `${supplied.toFixed(1)} 秒` : `${(supplied / 60).toFixed(1)} 分钟`;
  const start = new Date(run?.started_at).getTime();
  const end = new Date(run?.finished_at).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return '-';
  const seconds = Math.max(0, (end - start) / 1000);
  return seconds < 60 ? `${seconds.toFixed(1)} 秒` : `${(seconds / 60).toFixed(1)} 分钟`;
}

function notice(kind, title, message, actions = []) {
  const content = element('div', {}, element('strong', { text: title }), element('span', { text: message }));
  const actionWrap = actions.length ? element('div', { className: 'notice-actions' }, ...actions) : null;
  return element('div', { className: `notice notice-${kind}`, role: kind === 'error' ? 'alert' : 'status' }, content, actionWrap);
}

function clearResultSections() {
  ['resultOverview', 'resultMetrics', 'resultSplitMetrics', 'resultAnalysis', 'resultExplainability', 'resultArtifacts']
    .forEach((id) => byId(id)?.replaceChildren());
}

function copyButton(value, label = '复制') {
  const button = element('button', { className: 'button secondary compact', type: 'button', text: label });
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(String(value));
      button.textContent = '已复制';
      setTimeout(() => { button.textContent = label; }, 1500);
    } catch (_) {
      button.textContent = '复制失败';
    }
  });
  return button;
}

function overviewItem(label, value, extra = null) {
  return element('div', { className: 'overview-item' },
    element('dt', { text: label }),
    element('dd', {}, element('span', { className: 'truncate-text', text: valueOrDash(value), title: valueOrDash(value) }), extra),
  );
}

function renderOverview(result) {
  const target = byId('resultOverview');
  if (!target) return;
  const run = result.run || {};
  const dataset = result.dataset || {};
  const model = result.model || {};
  const modelMeta = window.SpecAutoAIModelMeta?.(model.type) || {};
  const modelDisplayName = model.display_name || modelMeta.displayName || model.type || '训练任务';
  const evaluation = result.evaluation || {};
  const state = resultState(result);
  const [stateLabel, stateClass] = stateMeta(state);
  const runId = run.run_id || currentResultRunId || '-';
  const stateBadge = element('span', { className: `status ${stateClass}`, text: stateLabel });
  const titleRow = element('div', { className: 'result-heading' },
    element('div', {}, element('p', { className: 'eyebrow', text: '建模结果' }), element('h2', { text: modelDisplayName })),
    stateBadge,
  );
  const grid = element('dl', { className: 'overview-grid' },
    overviewItem('Run ID', runId, copyButton(runId)),
    overviewItem('模型 ID', model.type),
    overviewItem('数据集', dataset.name || dataset.dataset_id),
    overviewItem('评估方式', evaluationLabel(evaluation.strategy)),
    overviewItem('主指标口径', aggregationLabel(evaluation.primary_aggregation)),
    overviewItem('创建时间', formatTime(run.created_at)),
    overviewItem('开始时间', formatTime(run.started_at)),
    overviewItem('结束时间', formatTime(run.finished_at)),
    overviewItem('训练耗时', durationText(run)),
    overviewItem('曲线数', dataset.curve_count),
    overviewItem('Sample_ID 数', dataset.sample_id_count),
    overviewItem('类别数', dataset.class_count),
    overviewItem('特征数', dataset.feature_count),
    overviewItem('数据指纹', dataset.sha256 ? String(dataset.sha256).slice(0, 16) : null),
    overviewItem('交叉验证折数', evaluation.fold_count),
  );
  const warnings = Array.isArray(result.warnings) ? result.warnings.filter(Boolean) : [];
  const warningList = warnings.length
    ? element('div', { className: 'result-warnings' }, ...warnings.map((warning) => notice('warning', '结果提示', String(warning))))
    : null;
  replaceChildren(target, titleRow, grid, warningList);
}

const METRIC_DEFINITIONS = [
  ['accuracy', 'Accuracy', '总体预测正确比例。'],
  ['balanced_accuracy', 'Balanced Accuracy', '各类别召回率等权平均，类别不均衡时更有参考意义。'],
  ['macro_precision', 'Macro Precision', '先分别计算每类 Precision，再对类别等权平均。'],
  ['macro_recall', 'Macro Recall', '先分别计算每类 Recall，再对类别等权平均。'],
  ['macro_f1', 'Macro F1', '先分别计算每类 F1，再对类别等权平均。'],
];

function metricValue(metrics, key) {
  if (!metrics || typeof metrics !== 'object') return undefined;
  if (metrics[key] != null) return metrics[key];
  if (key.startsWith('macro_')) return metrics[key.replace('macro_', '')];
  return undefined;
}

function renderCoreMetrics(result) {
  const target = byId('resultMetrics');
  if (!target) return;
  const metrics = result.metrics?.primary || result.metrics?.splits?.test || {};
  const available = METRIC_DEFINITIONS.filter(([key]) => Number.isFinite(Number(metricValue(metrics, key))));
  const heading = element('div', { className: 'section-heading' },
    element('div', {}, element('h2', { text: '核心指标' }), element('p', { text: `优先展示 ${aggregationLabel(result.evaluation?.primary_aggregation)} 的测试主指标。` })),
  );
  if (!available.length) {
    replaceChildren(target, heading, notice('empty', '指标尚不可用', '训练完成并生成指标后将在这里展示。'));
    return;
  }
  const cards = element('div', { className: 'result-metric-grid' }, ...available.map(([key, label, help]) => (
    element('article', { className: 'result-metric-card', title: help, tabIndex: 0 },
      element('span', { text: label }),
      element('strong', { text: formatMetric(metricValue(metrics, key)) }),
      element('small', { text: help }),
    )
  )));
  replaceChildren(target, heading, cards);
}

function renderSplitMetrics(result) {
  const target = byId('resultSplitMetrics');
  if (!target) return;
  const splits = result.metrics?.splits || {};
  const rows = ['train', 'valid', 'test'].filter((name) => splits[name] && Object.keys(splits[name]).length);
  if (!rows.length) return;
  const splitLabels = { train: 'Train', valid: 'Valid', test: 'Test' };
  const table = element('table', {},
    element('thead', {}, element('tr', {},
      ...['数据分区', '聚合口径', 'Accuracy', 'Balanced Accuracy', 'Macro P', 'Macro R', 'Macro F1']
        .map((label) => element('th', { text: label })),
    )),
    element('tbody', {}, ...rows.map((name) => {
      const split = splits[name] || {};
      const values = split.values && typeof split.values === 'object' ? split.values : split;
      let aggregation = split.aggregation;
      if (!aggregation && name === 'test') aggregation = result.evaluation?.primary_aggregation;
      if (!aggregation && result.evaluation?.strategy === 'leave_one_sample_id_cv') aggregation = 'fold_mean';
      if (!aggregation) aggregation = 'direct';
      return element('tr', {},
        element('th', { text: splitLabels[name] }),
        element('td', { text: aggregationLabel(aggregation) }),
        ...['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
          .map((key) => element('td', { text: formatMetric(metricValue(values, key)) })),
      );
    })),
  );
  replaceChildren(target,
    element('div', { className: 'section-heading' }, element('div', {},
      element('h2', { text: '训练、验证与测试集' }),
      element('p', { text: '交叉验证中的 OOF 测试指标与折均值分开标注，避免混用。' }),
    )),
    element('div', { className: 'table-scroll' }, table),
  );
}

function confusionData(result) {
  const source = result.analysis?.confusion_matrix;
  const matrix = Array.isArray(source) ? source : source?.matrix;
  const report = result.analysis?.classification_report || {};
  const labels = source?.labels || result.label_names || Object.keys(report).filter((key) => (
    !['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(key)
  ));
  return { matrix: Array.isArray(matrix) ? matrix : [], labels };
}

function renderConfusion(result) {
  const { matrix, labels } = confusionData(result);
  if (!matrix.length) return null;
  const safeLabels = matrix.map((_, index) => valueOrDash(labels[index] ?? `class_${index}`));
  const table = element('table', { className: 'matrix-table' },
    element('thead', {}, element('tr', {}, element('th', { text: '真实 \\ 预测' }), ...safeLabels.map((label) => element('th', { text: label })) )),
    element('tbody', {}, ...matrix.map((row, rowIndex) => element('tr', {},
      element('th', { text: safeLabels[rowIndex] }),
      ...(Array.isArray(row) ? row : []).map((value) => element('td', { text: valueOrDash(value) })),
    ))),
  );
  return element('article', { className: 'result-card' },
    element('h3', { text: '混淆矩阵' }),
    element('p', { className: 'section-note', text: '行是真实类别，列是预测类别。' }),
    element('div', { className: 'matrix-wrap' }, table),
  );
}

function classRows(report) {
  if (!report || typeof report !== 'object') return [];
  return Object.entries(report).filter(([label, values]) => (
    values && typeof values === 'object' && !['macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(label)
  ));
}

function renderClassMetrics(result) {
  const rows = classRows(result.analysis?.classification_report);
  if (!rows.length) return null;
  const table = element('table', {},
    element('thead', {}, element('tr', {}, ...['类别', 'Precision', 'Recall', 'F1', 'Support'].map((label) => element('th', { text: label })))),
    element('tbody', {}, ...rows.map(([label, values]) => element('tr', {},
      element('th', { text: label }),
      element('td', { text: formatMetric(values.precision) }),
      element('td', { text: formatMetric(values.recall) }),
      element('td', { text: formatMetric(values['f1-score'] ?? values.f1_score) }),
      element('td', { text: valueOrDash(values.support) }),
    ))),
  );
  return element('article', { className: 'result-card' },
    element('h3', { text: '各类别指标' }),
    element('div', { className: 'table-scroll' }, table),
  );
}

function distributionRows(result) {
  const supplied = result.analysis?.prediction_distribution;
  if (Array.isArray(supplied)) return supplied;
  if (supplied && typeof supplied === 'object') {
    if (Array.isArray(supplied.labels)) {
      return supplied.labels.map((label, index) => ({
        label,
        actual: supplied.true_counts?.[index] ?? supplied.actual_counts?.[index] ?? 0,
        predicted: supplied.predicted_counts?.[index] ?? 0,
      }));
    }
    return Object.entries(supplied).map(([label, value]) => (
      typeof value === 'object' ? { label, ...value } : { label, predicted: value }
    ));
  }
  const { matrix, labels } = confusionData(result);
  if (!matrix.length) return [];
  return matrix.map((row, index) => ({
    label: labels[index] ?? `class_${index}`,
    actual: row.reduce((sum, value) => sum + Number(value || 0), 0),
    predicted: matrix.reduce((sum, item) => sum + Number(item?.[index] || 0), 0),
  }));
}

function renderDistribution(result) {
  const rows = distributionRows(result);
  if (!rows.length) return null;
  const maxValue = Math.max(1, ...rows.flatMap((row) => [Number(row.actual || row.true_count || 0), Number(row.predicted || row.predicted_count || 0)]));
  return element('article', { className: 'result-card' },
    element('h3', { text: '预测结果分布' }),
    element('div', { className: 'distribution-list' }, ...rows.map((row) => {
      const actual = Number(row.actual ?? row.true_count ?? 0);
      const predicted = Number(row.predicted ?? row.predicted_count ?? 0);
      return element('div', { className: 'distribution-row' },
        element('strong', { className: 'truncate-text', text: valueOrDash(row.label ?? row.class_name), title: valueOrDash(row.label ?? row.class_name) }),
        element('div', { className: 'distribution-bars' },
          element('span', { className: 'distribution-bar actual', title: `真实 ${actual}`, style: `width:${Math.max(2, (actual / maxValue) * 100)}%` }),
          element('span', { className: 'distribution-bar predicted', title: `预测 ${predicted}`, style: `width:${Math.max(2, (predicted / maxValue) * 100)}%` }),
        ),
        element('small', { text: `真实 ${actual} / 预测 ${predicted}` }),
      );
    })),
  );
}

function auditParamsTable(params) {
  const entries = params && typeof params === 'object' ? Object.entries(params) : [];
  if (!entries.length) return notice('empty', '未提供参数明细', '该 Run 没有可展示的参数选择记录。');
  return element('div', { className: 'table-scroll' }, element('table', {},
    element('thead', {}, element('tr', {}, element('th', { text: '参数' }), element('th', { text: '取值' }))),
    element('tbody', {}, ...entries.map(([key, value]) => element('tr', {},
      element('th', { text: key }),
      element('td', { text: typeof value === 'object' ? JSON.stringify(value) : valueOrDash(value) }),
    ))),
  ));
}

function renderTrainingAudit(result) {
  const audit = result.analysis?.training_audit;
  if (!audit || typeof audit !== 'object') return null;
  const traditional = audit.traditional;
  const deep = audit.deep_training;
  if (!traditional && !deep) return null;
  const details = element('details', { className: 'result-card wide-card audit-details' },
    element('summary', { text: '训练与参数审计' }),
  );
  if (traditional && typeof traditional === 'object') {
    details.append(element('h4', { text: '传统模型参数选择' }));
    if (traditional.best_params) {
      const score = traditional.selection_score ?? traditional.valid_balanced_accuracy;
      details.append(element('p', { className: 'section-note', text: `${valueOrDash(traditional.selection_metric || '验证指标')}：${formatMetric(score)}` }));
      details.append(auditParamsTable(traditional.best_params));
    }
    const folds = Array.isArray(traditional.best_params_by_fold) ? traditional.best_params_by_fold : [];
    if (folds.length) {
      details.append(element('div', { className: 'table-scroll' }, element('table', {},
        element('thead', {}, element('tr', {}, ...['折', '选择指标', '得分', '最佳参数'].map((label) => element('th', { text: label })))),
        element('tbody', {}, ...folds.map((entry) => element('tr', {},
          element('td', { text: valueOrDash(entry.fold_index) }),
          element('td', { text: valueOrDash(entry.selection_metric) }),
          element('td', { text: formatMetric(entry.selection_score ?? entry.valid_balanced_accuracy) }),
          element('td', { text: entry.params ? JSON.stringify(entry.params) : '-' }),
        ))),
      )));
    }
    details.append(element('p', { className: 'section-note', text: '完整参数搜索 CSV 如已生成，可在结果下载区逐项下载。' }));
  }
  if (deep && typeof deep === 'object') {
    details.append(element('h4', { text: '深度模型训练摘要' }));
    const grid = element('div', { className: 'result-metric-grid audit-metrics' });
    [
      ['最佳验证 Loss', deep.best_valid_loss],
      ['实际训练 Epoch', deep.actual_epochs],
      ['最低学习率', deep.min_learning_rate],
    ].filter(([, value]) => value != null).forEach(([label, value]) => {
      grid.append(element('article', { className: 'result-metric-card' },
        element('span', { text: label }),
        element('strong', { text: Number.isFinite(Number(value)) ? formatMetric(value) : valueOrDash(value) }),
      ));
    });
    if (grid.childElementCount) details.append(grid);
  }
  return details;
}

function historyRows(result) {
  const history = result.analysis?.history;
  if (Array.isArray(history)) return history;
  return Array.isArray(history?.rows) ? history.rows : [];
}

function drawHistoryChart(canvas, rows, series) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(rect.width || 640, 1);
  const height = Math.max(rect.height || 260, 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 42, right: 16, top: 28, bottom: 32 };
  const values = series.flatMap((item) => rows.map((row) => Number(row[item.key])).filter(Number.isFinite));
  if (!values.length) {
    ctx.fillStyle = '#667085';
    ctx.fillText('暂无可绘制数据', pad.left, pad.top);
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  ctx.strokeStyle = '#d8dee8';
  ctx.strokeRect(pad.left, pad.top, plotWidth, plotHeight);
  series.forEach((item, seriesIndex) => {
    ctx.beginPath();
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    let started = false;
    rows.forEach((row, index) => {
      const value = Number(row[item.key]);
      if (!Number.isFinite(value)) return;
      const x = pad.left + (index / Math.max(rows.length - 1, 1)) * plotWidth;
      const y = pad.top + plotHeight - ((value - min) / span) * plotHeight;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    if (started) ctx.stroke();
    ctx.fillStyle = item.color;
    ctx.fillRect(pad.left + seriesIndex * 150, 8, 12, 12);
    ctx.fillStyle = '#344054';
    ctx.fillText(item.label, pad.left + 18 + seriesIndex * 150, 18);
  });
}

function renderHistory(result) {
  const rows = historyRows(result);
  if (!rows.length) {
    const reason = result.analysis?.history?.reason;
    return element('article', { className: 'result-card' },
      element('h3', { text: '训练过程' }),
      notice('empty', '无训练过程曲线', reason || '该模型不产生逐 Epoch 训练曲线。'),
    );
  }
  const folds = [...new Set(rows.map((row) => Number(row.fold_index || 1)).filter(Number.isFinite))].sort((a, b) => a - b);
  const card = element('article', { className: 'result-card wide-card' }, element('h3', { text: '训练过程曲线' }));
  const toolbar = element('div', { className: 'chart-toolbar' });
  const select = element('select', { id: 'resultHistoryFoldSelect', 'aria-label': '选择交叉验证折' }, ...folds.map((fold) => element('option', { value: String(fold), text: `第 ${fold}/${folds.length} 折` })));
  if (folds.length > 1) toolbar.append(element('label', { htmlFor: 'resultHistoryFoldSelect', text: '查看折数' }), select);
  const lossCanvas = element('canvas', { className: 'result-chart', role: 'img', 'aria-label': '训练损失曲线' });
  const metricCanvas = element('canvas', { className: 'result-chart', role: 'img', 'aria-label': '验证指标曲线' });
  const charts = element('div', { className: 'chart-grid' },
    element('div', {}, element('h4', { text: 'Loss' }), lossCanvas),
    element('div', {}, element('h4', { text: '验证指标' }), metricCanvas),
  );
  card.append(toolbar, charts);
  if (result.analysis?.history?.truncated) {
    card.append(element('p', { className: 'section-note', text: '页面只加载了训练过程摘要；完整记录请在结果下载区下载。' }));
  }
  const draw = () => {
    const fold = Number(select.value || folds[0]);
    const selected = rows.filter((row) => Number(row.fold_index || 1) === fold).sort((a, b) => Number(a.epoch || 0) - Number(b.epoch || 0));
    requestAnimationFrame(() => {
      drawHistoryChart(lossCanvas, selected, [
        { key: 'train_loss', label: 'train loss', color: '#b45309' },
        { key: 'valid_loss', label: 'valid loss', color: '#7c3aed' },
      ]);
      drawHistoryChart(metricCanvas, selected, [
        { key: 'valid_accuracy', label: 'valid accuracy', color: '#0f766e' },
        { key: 'valid_macro_f1', label: 'valid macro F1', color: '#475467' },
      ]);
    });
  };
  select.addEventListener('change', draw);
  draw();
  return card;
}

function renderAnalysis(result) {
  const target = byId('resultAnalysis');
  if (!target) return;
  const cards = [renderConfusion(result), renderClassMetrics(result), renderDistribution(result), renderHistory(result), renderTrainingAudit(result)].filter(Boolean);
  replaceChildren(target,
    element('div', { className: 'section-heading' }, element('div', {},
      element('h2', { text: '图表与分析' }),
      element('p', { text: '这里只展示当前 Run 已真实生成或可由现有指标直接推导的分析。' }),
    )),
    element('div', { className: 'result-analysis-grid' }, ...cards),
  );
}

function segmentLabel(segment) {
  if (segment?.start_x != null || segment?.end_x != null) return `${valueOrDash(segment.start_x)} – ${valueOrDash(segment.end_x)}`;
  if (segment?.start_index != null || segment?.end_index != null) return `特征 ${valueOrDash(segment.start_index)} – ${valueOrDash(segment.end_index)}`;
  return valueOrDash(segment?.label || segment?.name);
}

function importanceBars(segments) {
  const safe = Array.isArray(segments) ? segments.filter(Boolean).slice(0, 8) : [];
  if (!safe.length) return notice('empty', '没有明显正贡献区间', '当前解释结果未返回可排序的重要特征区间。');
  const max = Math.max(1e-12, ...safe.map((item) => Math.abs(Number(item.normalized_importance ?? item.importance ?? 0))));
  return element('div', { className: 'importance-list' }, ...safe.map((item, index) => {
    const value = Number(item.normalized_importance ?? item.importance ?? 0);
    return element('div', { className: 'importance-row' },
      element('span', { text: `${index + 1}. ${segmentLabel(item)}` }),
      element('div', { className: 'importance-track' }, element('span', { style: `width:${Math.max(2, Math.abs(value) / max * 100)}%` })),
      element('strong', { text: formatMetric(item.importance ?? item.normalized_importance) }),
    );
  }));
}

function explainabilityArtifact(result, type) {
  const summary = result.explainability?.[type] || {};
  const preferredName = summary.artifact || summary.name;
  const artifacts = (Array.isArray(result.artifacts) ? result.artifacts : []).filter((artifact) => {
    const category = String(artifact?.category || '').toLowerCase();
    const name = String(artifact?.name || '').toLowerCase();
    return category !== 'model' && !['model.pkl', 'model.pt'].includes(name) && !name.endsWith('.joblib');
  });
  if (preferredName) return artifacts.find((artifact) => artifact.name === preferredName) || null;
  return artifacts.find((artifact) => {
    const name = String(artifact.name || '').toLowerCase();
    const category = String(artifact.category || '').toLowerCase();
    if (category !== 'explainability' || artifact.format !== 'json') return false;
    return type === 'samples' ? name.includes('sample') : !name.includes('sample') && name.includes('importance');
  }) || null;
}

async function explainabilityPayload(result, type, generation) {
  const summary = result.explainability?.[type] || {};
  if (type === 'samples' && Array.isArray(summary.samples)) return summary;
  if (type === 'global' && Array.isArray(summary.top_segments)) return summary;
  const artifact = explainabilityArtifact(result, type);
  if (!artifact?.downloadable || !artifact?.download_url || !safeDownloadUrl(artifact.download_url)) return null;
  const payload = await request(artifact.download_url);
  return generation === resultRenderGeneration ? payload : null;
}

function renderGlobalImportance(target, payload, summary) {
  const method = payload?.method || summary?.method;
  replaceChildren(target,
    element('h3', { text: '全局特征重要性' }),
    element('p', { className: 'section-note', text: method ? `解释方法：${method}` : '按当前模型实际生成的解释产物展示。' }),
    importanceBars(payload?.top_segments || summary?.top_segments),
  );
}

function sampleDisplayLabel(sample) {
  const sampleName = sample?.name || sample?.sample_id || sample?.index || sample?.result_id || '-';
  const fold = sample?.fold_index ? `第 ${sample.fold_index} 折 · ` : '';
  return `${fold}${sampleName} · 真实 ${valueOrDash(sample?.true_label)} · 预测 ${valueOrDash(sample?.pred_label)}`;
}

function renderSampleImportance(target, payload, summary) {
  const samples = Array.isArray(payload?.samples) ? payload.samples : [];
  replaceChildren(target, element('h3', { text: '单样品可解释性' }));
  if (!samples.length) {
    target.append(notice('empty', '暂无单样品结果', summary?.reason || payload?.reason || '当前 Run 未生成单样品解释产物。'));
    return;
  }
  const select = element('select', { id: 'resultSampleExplanationSelect', 'aria-label': '选择要查看解释结果的样品' }, ...samples.map((sample, index) => (
    element('option', { value: String(index), text: sampleDisplayLabel(sample) })
  )));
  const details = element('div', { className: 'sample-explanation-details' });
  const renderSelected = () => {
    const sample = samples[Number(select.value || 0)] || samples[0];
    replaceChildren(details,
      element('dl', { className: 'feature-meta' },
        overviewItem('真实标签', sample.true_label),
        overviewItem('预测标签', sample.pred_label),
        overviewItem('预测状态', sample.correct === true ? '正确' : sample.correct === false ? '不一致' : '-'),
        overviewItem('真实类别概率', Number.isFinite(Number(sample.true_probability)) ? formatMetric(sample.true_probability) : '-'),
      ),
      importanceBars(sample.top_segments),
    );
  };
  select.addEventListener('change', renderSelected);
  target.append(element('div', { className: 'feature-toolbar' }, element('label', { htmlFor: 'resultSampleExplanationSelect', text: '样品' }), select), details);
  renderSelected();
}

async function renderExplainability(result) {
  const target = byId('resultExplainability');
  if (!target) return;
  const generation = resultRenderGeneration;
  const globalCard = element('article', { className: 'result-card' });
  const sampleCard = element('article', { className: 'result-card' });
  replaceChildren(target,
    element('div', { className: 'section-heading' }, element('div', {},
      element('h2', { text: '模型解释' }),
      element('p', { text: '解释方法随模型而异；正贡献表示该区间对当前预测或真实类别置信度更重要。' }),
    )),
    element('div', { className: 'result-analysis-grid' }, globalCard, sampleCard),
  );
  globalCard.append(element('h3', { text: '全局特征重要性' }), notice('loading', '正在读取', '正在加载解释性产物。'));
  const sampleArtifact = explainabilityArtifact(result, 'samples');
  const sampleSummary = result.explainability?.samples || {};
  const canLoadSamples = Array.isArray(sampleSummary.samples) || sampleArtifact?.downloadable === true;
  const sampleSize = Number(sampleArtifact?.size_bytes);
  const sampleSizeLabel = Number.isFinite(sampleSize) ? `（约 ${(sampleSize / 1024 / 1024).toFixed(1)} MB）` : '';
  const loadSamples = element('button', { className: 'button secondary', type: 'button', text: `加载单样品解释${sampleSizeLabel}` });
  if (canLoadSamples) {
    replaceChildren(sampleCard,
      element('h3', { text: '单样品可解释性' }),
      element('p', { className: 'section-note', text: '单样品解释文件可能较大，默认不影响结果页首屏加载。' }),
      loadSamples,
    );
  } else {
    replaceChildren(sampleCard,
      element('h3', { text: '单样品可解释性' }),
      notice('empty', '暂不可用', sampleSummary.reason || sampleArtifact?.reason || '当前 Run 未生成单样品解释产物。'),
    );
  }
  if (canLoadSamples) loadSamples.addEventListener('click', async () => {
    loadSamples.disabled = true;
    replaceChildren(sampleCard, element('h3', { text: '单样品可解释性' }), notice('loading', '正在读取', `正在加载单样品解释${sampleSizeLabel}。`));
    try {
      const samplePayload = await explainabilityPayload(result, 'samples', generation);
      if (generation !== resultRenderGeneration) return;
      if (samplePayload) renderSampleImportance(sampleCard, samplePayload, result.explainability?.samples);
      else replaceChildren(sampleCard, element('h3', { text: '单样品可解释性' }), notice('empty', '暂不可用', result.explainability?.samples?.reason || '当前 Run 未提供可读取的单样品解释产物。'));
    } catch (error) {
      if (generation !== resultRenderGeneration) return;
      const retry = element('button', { className: 'button secondary', type: 'button', text: '重新加载' });
      retry.addEventListener('click', () => renderExplainability(result));
      replaceChildren(sampleCard, element('h3', { text: '单样品可解释性' }), notice('error', '加载失败', error.message, [retry]));
    }
  }, { once: true });
  try {
    const globalPayload = await explainabilityPayload(result, 'global', generation);
    if (generation !== resultRenderGeneration) return;
    if (globalPayload) renderGlobalImportance(globalCard, globalPayload, result.explainability?.global);
    else replaceChildren(globalCard, element('h3', { text: '全局特征重要性' }), notice('empty', '暂不可用', result.explainability?.global?.reason || '当前 Run 未提供可读取的全局解释产物。'));
  } catch (error) {
    if (generation !== resultRenderGeneration) return;
    replaceChildren(globalCard, element('h3', { text: '全局特征重要性' }), notice('error', '加载失败', error.message));
  }
}

function safeDownloadUrl(url) {
  try {
    const parsed = new URL(url, window.location.href);
    return parsed.origin === window.location.origin && parsed.pathname.startsWith('/api/training/runs/');
  } catch (_) {
    return false;
  }
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const safeFilename = String(filename || 'artifact').replace(/[\\/:*?"<>|]+/g, '_');
  const anchor = element('a', { href: url, download: safeFilename });
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function triggerArtifactDownload(artifact, button, message) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = '下载中…';
  message.textContent = '';
  try {
    const response = await downloadFile(artifact.download_url);
    saveBlob(response.blob, artifact.suggested_filename || response.filename || artifact.name || 'artifact');
    message.textContent = `${artifact.label || artifact.name} 已开始保存。`;
  } catch (error) {
    message.textContent = `下载失败：${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function artifactCard(artifact, message) {
  const integrity = String(artifact?.integrity || '').toLowerCase();
  const integrityBlocked = ['mismatch', 'corrupt', 'missing', 'failed'].includes(integrity);
  const available = artifact?.downloadable === true
    && artifact?.applicable !== false
    && artifact?.exists !== false
    && !integrityBlocked
    && safeDownloadUrl(artifact?.download_url);
  const label = artifact?.label || artifact?.name || '未命名产物';
  const meta = [String(artifact?.format || '').toUpperCase(), artifact?.size_bytes != null ? `${Math.ceil(Number(artifact.size_bytes) / 1024)} KB` : '']
    .filter(Boolean).join(' · ');
  const button = element('button', { className: 'button secondary', type: 'button', text: `下载${label}`, disabled: !available });
  if (available) button.addEventListener('click', () => triggerArtifactDownload(artifact, button, message));
  const reason = available ? '可下载' : artifact?.reason || (artifact?.applicable === false ? '当前模型不适用' : '文件未生成或不可下载');
  button.title = reason;
  return element('article', { className: 'artifact-card' },
    element('div', {}, element('strong', { text: label }), element('span', { text: meta || valueOrDash(artifact?.name) }), element('small', { text: reason })),
    button,
  );
}

function renderArtifacts(result) {
  const target = byId('resultArtifacts');
  if (!target) return;
  const artifacts = (Array.isArray(result.artifacts) ? result.artifacts : []).filter((artifact) => {
    const category = String(artifact?.category || '').toLowerCase();
    const name = String(artifact?.name || '').toLowerCase();
    return category !== 'model' && !['model.pkl', 'model.pt'].includes(name) && !name.endsWith('.joblib');
  });
  const message = element('p', { className: 'download-message', role: 'status', 'aria-live': 'polite' });
  const heading = element('div', { className: 'section-heading' }, element('div', {},
    element('h2', { text: '结果下载' }),
    element('p', { text: '每个文件单独说明格式和可用状态；模型二进制文件不会在此页面公开。' }),
  ));
  if (!artifacts.length) {
    replaceChildren(target, heading, notice('empty', '没有可展示的下载清单', '历史 Run 或缺失的 Manifest 可能无法提供逐项下载状态。'));
    return;
  }
  const groups = new Map();
  artifacts.forEach((artifact) => {
    const category = artifact.category || 'other';
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(artifact);
  });
  const categoryLabels = {
    metrics: '指标与评估',
    predictions: '预测结果',
    explainability: '解释性分析',
    config: '配置与元数据',
    metadata: '配置与元数据',
    training: '训练过程',
    model: '模型文件',
    internal: '内部/兼容文件',
    other: '其他产物',
  };
  const groupNodes = [...groups.entries()].map(([category, items]) => {
    const cards = element('div', { className: 'artifact-grid' }, ...items.map((artifact) => artifactCard(artifact, message)));
    if (category === 'internal') {
      return element('details', { className: 'artifact-group' }, element('summary', { text: categoryLabels[category] }), cards);
    }
    return element('section', { className: 'artifact-group' }, element('h3', { text: categoryLabels[category] || category }), cards);
  });
  replaceChildren(target, heading, ...groupNodes, message);
}

function renderTerminalMessage(result) {
  const target = byId('resultPageState');
  const state = resultState(result);
  const run = result.run || {};
  if (ACTIVE_STATES.has(state)) {
    const progress = run.progress;
    const progressParts = progress && typeof progress === 'object'
      ? [progress.message, progress.phase, progress.fold_progress_text, progress.current_fold != null ? `第 ${progress.current_fold} 折` : null, progress.epoch != null ? `Epoch ${progress.epoch}` : null].filter(Boolean)
      : [];
    let detail = progressParts.length
      ? progressParts.join(' · ')
      : progress == null
        ? '任务由独立训练 worker 执行，页面会自动刷新。'
        : `当前进度：${valueOrDash(progress)}`;
    if ((state === 'queued' || state === 'pending') && window.SpecAutoAIHealth?.worker?.available === false) {
      detail = '训练 Worker 未运行，任务会保持排队；请启动 Worker 后再等待页面自动刷新。';
    }
    replaceChildren(target, notice('loading', state === 'running' ? '模型正在训练' : '任务正在排队', detail));
    return false;
  }
  if (state === 'failed') {
    const error = typeof run.error === 'object' ? run.error.message : run.error;
    replaceChildren(target, notice('error', '训练失败', error || '没有可用结果，请根据 Run ID 查看服务器日志。'));
    return false;
  }
  if (state === 'cancelled' || state === 'paused') {
    replaceChildren(target, notice('warning', '训练已取消', '此任务不会继续执行，也不会生成新的结果产物。'));
    return false;
  }
  const warningStates = {
    partial: '部分结果或产物缺失，以下区域只展示已确认可用的内容。',
    missing_manifest: '结果文件清单缺失；概览可用，但下载入口可能不可用。',
    corrupt_manifest: '结果文件清单损坏；请联系管理员检查该 Run 的产物目录。',
  };
  if (warningStates[state]) replaceChildren(target, notice('warning', '结果不完整', warningStates[state]));
  else replaceChildren(target);
  return true;
}

async function renderResultPayload(payload) {
  const result = normalizeResult(payload);
  lastRenderedResult = result;
  resultRenderGeneration += 1;
  clearResultSections();
  renderOverview(result);
  const canShowResult = renderTerminalMessage(result);
  if (!canShowResult) return result;
  renderCoreMetrics(result);
  renderSplitMetrics(result);
  renderAnalysis(result);
  renderArtifacts(result);
  renderExplainability(result);
  return result;
}

function resultErrorActions(runId) {
  return [
    element('button', { className: 'button primary', type: 'button', text: '重试', on: { click: () => loadResult(runId) } }),
    element('a', { className: 'button secondary', href: '#/runs', text: '查看训练记录' }),
  ];
}

function renderResultError(error, runId) {
  clearResultSections();
  const target = byId('resultPageState');
  let title = '结果加载失败';
  let message = error?.message || '无法读取该训练任务。';
  if (error?.status === 404) {
    title = '任务不存在或已删除';
    message = `没有找到 Run ${runId}。请检查链接，或从训练记录重新进入。`;
  } else if (error?.status === 403) {
    title = '没有访问权限';
    message = '当前身份无权查看该训练任务。';
  } else if (error?.status === 401) {
    title = '需要服务器认证';
    message = '输入有效访问令牌后点击重试。';
  }
  replaceChildren(target, notice('error', title, message, resultErrorActions(runId)));
}

async function renderResultLanding() {
  resultPoller.stop();
  currentResultRunId = null;
  lastRenderedResult = null;
  clearResultSections();
  const target = byId('resultPageState');
  const input = element('input', { type: 'text', placeholder: '输入 Run ID', 'aria-label': 'Run ID' });
  const form = element('form', { className: 'result-run-form' }, input, element('button', { className: 'button primary', type: 'submit', text: '查看结果' }));
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (input.value.trim()) navigateToResult(input.value.trim());
  });
  replaceChildren(target, notice('empty', '请选择一次训练任务', '输入 Run ID，或从下面的最近任务与训练记录进入。'), form);
  const overview = byId('resultOverview');
  try {
    let runs;
    try { runs = await request('/api/training/runs?projection=summary&limit=10'); }
    catch (_) { runs = await request('/api/training/runs'); }
    const list = Array.isArray(runs) ? runs.slice(0, 10) : Array.isArray(runs?.items) ? runs.items : [];
    const rows = list.map((run) => {
      const runId = run.run_id || run.run?.run_id;
      const modelId = run.model_type || run.model?.type || run.config?.model_type;
      const modelName = window.SpecAutoAIModelMeta?.(modelId)?.displayName || modelId || '-';
      const link = element('a', { className: 'button secondary compact', href: buildResultHash(runId), text: '查看' });
      return element('tr', {},
        element('td', {}, element('span', { className: 'truncate-text', text: runId, title: runId })),
        element('td', { text: modelName }),
        element('td', { text: stateMeta(canonicalState(run))[0] }),
        element('td', {}, link),
      );
    });
    replaceChildren(overview,
      element('div', { className: 'section-heading' }, element('div', {}, element('h2', { text: '最近任务' }), element('p', { text: '列表只加载摘要，完整结果在进入 Run 后获取。' }))),
      rows.length ? element('div', { className: 'table-scroll' }, element('table', {},
        element('thead', {}, element('tr', {}, ...['Run ID', '模型', '状态', ''].map((label) => element('th', { text: label })))),
        element('tbody', {}, ...rows),
      )) : notice('empty', '暂无训练记录', '完成一次 AI 建模后可在此查看结果。'),
    );
  } catch (error) {
    replaceChildren(overview, notice('error', '最近任务加载失败', error.message));
  }
}

export function loadResult(runId) {
  const normalized = String(runId || '').trim();
  resultRenderGeneration += 1;
  if (!normalized) return renderResultLanding();
  currentResultRunId = normalized;
  lastRenderedResult = null;
  clearResultSections();
  replaceChildren(byId('resultPageState'), notice('loading', '正在加载建模结果', `Run ${normalized}`));
  resultPoller.watch(normalized, {
    onData: renderResultPayload,
    onError: (error, id, failures) => {
      if (failures === 1 || error?.status === 403 || error?.status === 404) renderResultError(error, id);
      else replaceChildren(byId('resultPageState'), notice('warning', '连接暂时中断', `正在进行第 ${failures} 次重试：${error.message}`));
    },
    isTerminal: (payload) => !isActivePayload(normalizeResult(payload)),
    isTerminalError: (error) => [401, 403, 404].includes(error?.status),
  });
}

export function watchTrainingRun(runId, onData, onError) {
  trainingPoller.watch(runId, {
    onData,
    onError,
    isTerminal: (payload) => !isActivePayload(payload),
    isTerminalError: (error) => [401, 403, 404].includes(error?.status),
  });
}

export function stopTrainingWatch() {
  trainingPoller.stop();
}

export function markNewRun(runId) {
  const normalized = String(runId || '').trim();
  if (!normalized) return;
  newRunIds.add(normalized);
  cancelAutoRedirect();
}

export function cancelAutoRedirect() {
  if (redirectTimer != null) clearTimeout(redirectTimer);
  if (redirectTicker != null) clearInterval(redirectTicker);
  redirectTimer = null;
  redirectTicker = null;
  redirectRunId = null;
}

export function showNewRunSuccess(run) {
  const runId = String(run?.run_id || run?.run?.run_id || '');
  const state = canonicalState(run);
  if (!shouldAutoRedirect({ runId, state, isNewRun: newRunIds.has(runId) })) return false;
  if (parseHash(window.location.hash).view !== 'modeling') return false;
  if (redirectRunId === runId) return true;
  cancelAutoRedirect();
  redirectRunId = runId;
  ['metrics', 'trainingAudit', 'trainingCharts', 'featureImportance', 'confusionMatrix', 'downloads']
    .forEach((id) => byId(id)?.replaceChildren());
  const countdown = element('strong', { text: '3 秒后进入建模结果' });
  const link = element('a', { className: 'button primary', href: buildResultHash(runId), text: '立即查看结果' });
  link.addEventListener('click', cancelAutoRedirect);
  replaceChildren(byId('trainingProgress'), notice('success', '训练已完成', '结果已保存，可通过专属 Run 链接随时重新打开。', [link, countdown]));
  const started = Date.now();
  redirectTicker = setInterval(() => {
    const remaining = Math.max(0, Math.ceil((AUTO_REDIRECT_DELAY_MS - (Date.now() - started)) / 1000));
    countdown.textContent = remaining > 0 ? `${remaining} 秒后进入建模结果` : '正在打开建模结果…';
  }, 250);
  redirectTimer = setTimeout(() => {
    const destination = buildResultHash(runId);
    cancelAutoRedirect();
    window.location.hash = destination;
  }, AUTO_REDIRECT_DELAY_MS);
  return true;
}

export function navigateToResult(runId) {
  const destination = buildResultHash(runId);
  if (window.location.hash === destination) loadResult(runId);
  else window.location.hash = destination;
}

export function navigateToView(view) {
  const destination = buildViewHash(view);
  if (window.location.hash === destination) applyCurrentRoute();
  else window.location.hash = destination;
}

function applyCurrentRoute() {
  const route = parseHash(window.location.hash);
  window.SpecAutoAIShowView?.(route.view, route);
  if (route.view === 'results') loadResult(route.runId);
  else resultPoller.stop();
  if (route.view !== 'modeling') cancelAutoRedirect();
}

function showAuthDialog() {
  const dialog = byId('authDialog');
  if (!dialog) return;
  if (dialog.classList.contains('hidden')) authReturnFocus = document.activeElement;
  dialog.classList.remove('hidden');
  dialog.setAttribute('aria-hidden', 'false');
  if (byId('authStatus')) byId('authStatus').textContent = '服务器需要认证';
  const input = byId('serverTokenInput');
  input.value = '';
  setTimeout(() => input.focus(), 0);
}

function hideAuthDialog() {
  const dialog = byId('authDialog');
  if (!dialog) return;
  dialog.classList.add('hidden');
  dialog.setAttribute('aria-hidden', 'true');
  byId('serverTokenInput').value = '';
  byId('authError').textContent = '';
  if (authReturnFocus instanceof HTMLElement) authReturnFocus.focus();
  authReturnFocus = null;
}

async function verifyAuthentication() {
  await request('/api/auth/session', { authRetry: false });
  byId('authStatus')?.classList.remove('hidden');
  byId('authStatus').textContent = '服务器已认证';
  if (byId('authButton')) byId('authButton').textContent = '退出服务器认证';
  hideAuthDialog();
}

async function initializeAuthentication() {
  const authButton = byId('authButton');
  const authStatus = byId('authStatus');
  const form = byId('authForm');
  if (!authButton || !authStatus || !form) return;
  window.addEventListener('specautoai:auth-required', showAuthDialog);
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const token = byId('serverTokenInput').value.trim();
    if (!token) {
      byId('authError').textContent = '请输入服务器访问令牌。';
      return;
    }
    window.SpecAutoAIAuth?.setServerToken(token);
    try {
      await verifyAuthentication();
    } catch (error) {
      window.SpecAutoAIAuth?.clearServerToken();
      byId('authError').textContent = error.message || '令牌无效。';
      showAuthDialog();
    }
  });
  byId('authCancel').addEventListener('click', () => {
    window.SpecAutoAIAuth?.cancelAuthentication();
    hideAuthDialog();
  });
  document.addEventListener('keydown', (event) => {
    const dialog = byId('authDialog');
    if (dialog.classList.contains('hidden')) return;
    if (event.key === 'Escape') {
      window.SpecAutoAIAuth?.cancelAuthentication();
      hideAuthDialog();
      return;
    }
    if (event.key === 'Tab') {
      const focusable = [...dialog.querySelectorAll('button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])')]
        .filter((node) => !node.disabled && !node.hidden);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  authButton.addEventListener('click', () => {
    if (window.SpecAutoAIAuth?.getServerToken()) {
      window.SpecAutoAIAuth.clearServerToken();
      resultPoller.stop();
      trainingPoller.stop();
      lastRenderedResult = null;
      clearResultSections();
      window.SpecAutoAIClearSessionUi?.();
      if (byId('resultPageState')) replaceChildren(byId('resultPageState'), notice('warning', '已退出服务器认证', '再次读取服务器数据时需要重新输入访问令牌。'));
      authStatus.textContent = '服务器未认证';
      authButton.textContent = '输入访问令牌';
    } else showAuthDialog();
  });
  try {
    const config = await request('/api/auth/config', { authRetry: false });
    if (config?.mode !== 'server' && config?.auth_required !== true) return;
    authButton.classList.remove('hidden');
    authStatus.classList.remove('hidden');
    authStatus.textContent = window.SpecAutoAIAuth?.getServerToken() ? '正在验证服务器令牌' : '服务器未认证';
    authButton.textContent = window.SpecAutoAIAuth?.getServerToken() ? '退出服务器认证' : '输入访问令牌';
    if (window.SpecAutoAIAuth?.getServerToken()) {
      try { await verifyAuthentication(); }
      catch (_) {
        window.SpecAutoAIAuth.clearServerToken();
        authStatus.textContent = '服务器未认证';
        authButton.textContent = '输入访问令牌';
        showAuthDialog();
      }
    }
  } catch (_) {
    // 旧后端没有 auth/config 时按本地模式继续，避免影响原有单机使用。
  }
}

function initialize() {
  window.SpecAutoAIResults = {
    AUTO_REDIRECT_DELAY_MS,
    buildResultHash,
    buildViewHash,
    cancelAutoRedirect,
    loadResult,
    markNewRun,
    navigateToResult,
    navigateToView,
    parseHash,
    showNewRunSuccess,
    stopTrainingWatch,
    watchTrainingRun,
  };
  window.addEventListener('hashchange', applyCurrentRoute);
  window.addEventListener('specautoai:model-catalog-ready', () => {
    if (lastRenderedResult && parseHash(window.location.hash).view === 'results') renderOverview(lastRenderedResult);
  });
  window.addEventListener('specautoai:health-ready', () => {
    if (lastRenderedResult && parseHash(window.location.hash).view === 'results' && isActivePayload(lastRenderedResult)) {
      renderTerminalMessage(lastRenderedResult);
    }
  });
  initializeAuthentication();
  if (!window.location.hash) window.history.replaceState(null, '', buildViewHash('raman'));
  applyCurrentRoute();
  window.dispatchEvent(new CustomEvent('specautoai:results-ready'));
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
}
