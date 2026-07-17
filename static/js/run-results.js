import { downloadFile, request } from './api-client.js';
import { element, formatMetric, formatTime, replaceChildren } from './ui-utils.js';

export const AUTO_REDIRECT_DELAY_MS = 3000;
const ACTIVE_STATES = new Set(['queued', 'pending', 'running']);
const TERMINAL_STATES = new Set(['succeeded', 'success', 'failed', 'cancelled', 'paused', 'ready', 'partial', 'missing_manifest', 'corrupt_manifest']);
const TRADITIONAL_MODEL_IDS = new Set(['pls_da', 'pca_lda', 'logistic_regression', 'svm', 'random_forest', 'xgboost']);
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
      splits: {},
      confusion_matrix: metrics?.test?.confusion_matrix || metrics?.confusion_matrix || [],
      classification_report: metrics?.test?.classification_report || metrics?.classification_report || {},
      prediction_distribution: null,
      history: { available: Array.isArray(payload?.history) && payload.history.length > 0, rows: payload?.history || [] },
    },
    explainability: {
      samples: payload?.sample_feature_importance || {},
    },
    artifacts: Array.isArray(payload?.artifacts) ? payload.artifacts : [],
    warnings: ['当前后端使用旧版结果接口；部分时间、数据快照或下载清单可能无法恢复。请升级并同时重启 Web 与 Worker。'],
    label_names: payload?.label_names || [],
  };
}

export function normalizeResult(payload) {
  if (payload?.schema_version === 'run-result-v1' && payload?.run) return payload;
  return normalizedLegacyResult(payload || {});
}

export function backendSupportsResultContract(health) {
  return health?.contracts?.run_result === 'run-result-v1';
}

export function shouldUseLegacyResultFallback(error, health) {
  return error?.status === 404
    && health != null
    && !backendSupportsResultContract(health);
}

async function healthCapabilities(signal, forceRefresh = false) {
  if (!forceRefresh && window.SpecAutoAIHealth && Object.keys(window.SpecAutoAIHealth).length) return window.SpecAutoAIHealth;
  try {
    const health = await request('/health', { signal, authRetry: false });
    window.SpecAutoAIHealth = health || {};
    return window.SpecAutoAIHealth;
  } catch (_) {
    return null;
  }
}

async function fetchResult(runId, signal) {
  const encoded = encodeURIComponent(runId);
  try {
    return await request(`/api/training/runs/${encoded}/result`, { signal });
  } catch (error) {
    if (error?.status !== 404) throw error;
    const health = await healthCapabilities(signal, true);
    if (!shouldUseLegacyResultFallback(error, health)) throw error;
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
let trainingResultReturnFocus = null;
let trainingDialogFocusTimer = null;
let sampleExplanationResizeHandler = null;
let historyResizeHandler = null;
const explainabilityPayloadCache = new Map();

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
  if (sampleExplanationResizeHandler) {
    window.removeEventListener('resize', sampleExplanationResizeHandler);
    sampleExplanationResizeHandler = null;
  }
  if (historyResizeHandler) {
    window.removeEventListener('resize', historyResizeHandler);
    historyResizeHandler = null;
  }
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

export function analysisSplitEntries(result) {
  const supplied = result?.analysis?.splits;
  const labels = { train: 'Train', valid: 'Valid', test: 'Test' };
  const entries = ['train', 'valid', 'test']
    .map((name) => ({ name, label: labels[name], analysis: supplied?.[name] }))
    .filter((entry) => entry.analysis && typeof entry.analysis === 'object');
  if (entries.length) return entries;
  const legacy = result?.analysis;
  if (!legacy || typeof legacy !== 'object') return [];
  const hasLegacyAnalysis = legacy.confusion_matrix || legacy.classification_report || legacy.prediction_distribution;
  return hasLegacyAnalysis ? [{ name: 'test', label: 'Test', analysis: legacy }] : [];
}

function splitAggregationLabel(splitAnalysis, result, splitName) {
  const aggregation = splitAnalysis?.aggregation
    || (splitName === 'test' ? result?.evaluation?.primary_aggregation : null)
    || (result?.evaluation?.strategy === 'leave_one_sample_id_cv' ? 'fold_mean' : 'direct');
  if (aggregation === 'pooled_cross_fold') return '跨折合并预测（样本可能重复）';
  return aggregationLabel(aggregation);
}

function confusionData(splitAnalysis, result) {
  const source = splitAnalysis?.confusion_matrix;
  const matrix = Array.isArray(source) ? source : source?.matrix;
  const report = splitAnalysis?.classification_report || {};
  const labels = source?.labels || result.label_names || Object.keys(report).filter((key) => (
    !['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(key)
  ));
  return { matrix: Array.isArray(matrix) ? matrix : [], labels };
}

function renderSplitConfusion(splitName, splitAnalysis, result) {
  const { matrix, labels } = confusionData(splitAnalysis, result);
  if (!matrix.length) return null;
  const splitLabel = { train: 'Train', valid: 'Valid', test: 'Test' }[splitName] || splitName;
  const safeLabels = matrix.map((_, index) => valueOrDash(labels[index] ?? `class_${index}`));
  const table = element('table', { className: 'matrix-table' },
    element('thead', {}, element('tr', {}, element('th', { text: '真实 \\ 预测' }), ...safeLabels.map((label) => element('th', { text: label })) )),
    element('tbody', {}, ...matrix.map((row, rowIndex) => element('tr', {},
      element('th', { text: safeLabels[rowIndex] }),
      ...(Array.isArray(row) ? row : []).map((value) => element('td', { text: valueOrDash(value) })),
    ))),
  );
  return element('article', { className: 'result-card result-split-card' },
    element('h3', { text: `${splitLabel} 混淆矩阵` }),
    element('p', { className: 'section-note', text: `行是真实类别，列是预测类别 · ${splitAggregationLabel(splitAnalysis, result, splitName)}` }),
    element('div', { className: 'matrix-wrap' }, table),
  );
}

function classRows(report) {
  if (!report || typeof report !== 'object') return [];
  return Object.entries(report).filter(([label, values]) => (
    values && typeof values === 'object' && !['macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(label)
  ));
}

function renderSplitClassMetrics(splitName, splitAnalysis, result) {
  const rows = classRows(splitAnalysis?.classification_report);
  if (!rows.length) return null;
  const splitLabel = { train: 'Train', valid: 'Valid', test: 'Test' }[splitName] || splitName;
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
  return element('article', { className: 'result-card result-split-card' },
    element('h3', { text: `${splitLabel} 各类别指标` }),
    element('p', { className: 'section-note', text: splitAggregationLabel(splitAnalysis, result, splitName) }),
    element('div', { className: 'table-scroll' }, table),
  );
}

function distributionRows(splitAnalysis, result) {
  const supplied = splitAnalysis?.prediction_distribution;
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
  const { matrix, labels } = confusionData(splitAnalysis, result);
  if (!matrix.length) return [];
  return matrix.map((row, index) => ({
    label: labels[index] ?? `class_${index}`,
    actual: row.reduce((sum, value) => sum + Number(value || 0), 0),
    predicted: matrix.reduce((sum, item) => sum + Number(item?.[index] || 0), 0),
  }));
}

export function distributionScaleTicks(maxValue, desiredTicks = 5) {
  const safeMax = Math.max(0, Math.ceil(Number(maxValue) || 0));
  if (safeMax === 0) return [0];
  const target = Math.max(2, Math.floor(Number(desiredTicks) || 5));
  const rawStep = safeMax / (target - 1);
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rawStep, 1)));
  const residual = rawStep / magnitude;
  const niceResidual = residual <= 1 ? 1 : residual <= 2 ? 2 : residual <= 5 ? 5 : 10;
  const step = Math.max(1, niceResidual * magnitude);
  const top = Math.ceil(safeMax / step) * step;
  const ticks = [];
  for (let value = top; value >= 0; value -= step) ticks.push(value);
  return ticks;
}

function renderSplitDistribution(splitName, splitAnalysis, result) {
  const rows = distributionRows(splitAnalysis, result);
  if (!rows.length) return null;
  const maxValue = Math.max(1, ...rows.flatMap((row) => [Number(row.actual || row.true_count || 0), Number(row.predicted || row.predicted_count || 0)]));
  const ticks = distributionScaleTicks(maxValue);
  const scaleTop = Math.max(1, ticks[0] || maxValue);
  const splitLabel = { train: 'Train', valid: 'Valid', test: 'Test' }[splitName] || splitName;
  const columns = rows.map((row) => {
      const actual = Number(row.actual ?? row.true_count ?? 0);
      const predicted = Number(row.predicted ?? row.predicted_count ?? 0);
      const label = valueOrDash(row.label ?? row.class_name);
      return element('div', { className: 'distribution-column' },
        element('div', { className: 'distribution-bar-pair' },
          element('div', { className: 'distribution-vertical-bar actual', title: `${label} · 真实 ${actual}`, style: `height:${(actual / scaleTop) * 100}%` },
            element('strong', { text: actual }),
          ),
          element('div', { className: 'distribution-vertical-bar predicted', title: `${label} · 预测 ${predicted}`, style: `height:${(predicted / scaleTop) * 100}%` },
            element('strong', { text: predicted }),
          ),
        ),
        element('span', { className: 'distribution-label truncate-text', text: label, title: label }),
      );
  });
  const summaryTable = element('table', { className: 'visually-hidden' },
    element('caption', { text: `${splitLabel} 预测结果分布` }),
    element('thead', {}, element('tr', {}, element('th', { text: '类别' }), element('th', { text: '真实数量' }), element('th', { text: '预测数量' }))),
    element('tbody', {}, ...rows.map((row) => element('tr', {},
      element('th', { text: valueOrDash(row.label ?? row.class_name) }),
      element('td', { text: Number(row.actual ?? row.true_count ?? 0) }),
      element('td', { text: Number(row.predicted ?? row.predicted_count ?? 0) }),
    ))),
  );
  return element('article', { className: 'result-card result-split-card' },
    element('h3', { text: `${splitLabel} 预测结果分布` }),
    element('p', { className: 'section-note', text: splitAggregationLabel(splitAnalysis, result, splitName) }),
    element('div', {
      className: 'distribution-chart',
      role: 'img',
      'aria-label': `${splitLabel} 各类别真实数量与预测数量竖向柱状图`,
    },
    element('div', { className: 'distribution-axis' }, ...ticks.map((tick) => element('span', { text: tick }))),
    element('div', { className: 'distribution-plot', style: `min-width:${Math.max(320, rows.length * 72)}px` }, ...columns)),
    element('div', { className: 'distribution-legend' },
      element('span', {}, element('i', { className: 'actual' }), '真实'),
      element('span', {}, element('i', { className: 'predicted' }), '预测'),
    ),
    summaryTable,
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

export function chartDomain(values, options = {}) {
  const finite = (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite);
  if (Number.isFinite(Number(options.min)) && Number.isFinite(Number(options.max))) {
    return [Number(options.min), Number(options.max)];
  }
  if (!finite.length) return [0, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (options.includeZero) min = Math.min(0, min);
  if (min === max) {
    const padding = Math.max(Math.abs(min) * 0.1, 0.1);
    min -= padding;
    max += padding;
  } else {
    const padding = (max - min) * 0.08;
    min -= padding;
    max += padding;
  }
  return [min, max];
}

function linearTicks(min, max, count = 5) {
  const safeCount = Math.max(2, Math.floor(count));
  return Array.from({ length: safeCount }, (_, index) => min + ((max - min) * index) / (safeCount - 1));
}

function drawHistoryChart(canvas, rows, series, options = {}) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(rect.width || 640, 1);
  const height = Math.max(rect.height || 260, 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px system-ui, sans-serif';
  const pad = { left: 58, right: 18, top: 34, bottom: 48 };
  const values = series.flatMap((item) => rows.map((row) => Number(row[item.key])).filter(Number.isFinite));
  if (!values.length) {
    ctx.fillStyle = '#667085';
    ctx.fillText('暂无可绘制数据', pad.left, pad.top);
    return;
  }
  const epochs = rows.map((row, index) => Number(row.epoch ?? index + 1)).filter(Number.isFinite);
  const xMinRaw = Math.min(...epochs);
  const xMaxRaw = Math.max(...epochs);
  const xMin = xMinRaw === xMaxRaw ? xMinRaw - 0.5 : xMinRaw;
  const xMax = xMinRaw === xMaxRaw ? xMaxRaw + 0.5 : xMaxRaw;
  const [min, max] = chartDomain(values, options.domain || {});
  const span = Math.max(max - min, 1e-9);
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const yTicks = linearTicks(min, max, 5);
  const xTickCount = Math.min(6, Math.max(2, epochs.length));
  const xTicks = linearTicks(xMinRaw, xMaxRaw, xTickCount);
  ctx.textBaseline = 'middle';
  yTicks.forEach((tick) => {
    const y = pad.top + plotHeight - ((tick - min) / span) * plotHeight;
    ctx.strokeStyle = '#e4e8ef';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotWidth, y);
    ctx.stroke();
    ctx.fillStyle = '#667085';
    ctx.textAlign = 'right';
    ctx.fillText(Number(tick).toFixed(options.decimals ?? 3), pad.left - 8, y);
  });
  xTicks.forEach((tick) => {
    const x = pad.left + ((tick - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
    ctx.strokeStyle = '#eef1f5';
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotHeight);
    ctx.stroke();
    ctx.fillStyle = '#667085';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(String(Math.round(tick)), x, pad.top + plotHeight + 8);
  });
  ctx.strokeStyle = '#98a2b3';
  ctx.strokeRect(pad.left, pad.top, plotWidth, plotHeight);
  series.forEach((item, seriesIndex) => {
    ctx.beginPath();
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    let started = false;
    rows.forEach((row, index) => {
      const value = Number(row[item.key]);
      if (!Number.isFinite(value)) return;
      const epoch = Number(row.epoch ?? index + 1);
      const x = pad.left + ((epoch - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
      const y = pad.top + plotHeight - ((value - min) / span) * plotHeight;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    if (started) ctx.stroke();
    ctx.fillStyle = item.color;
    ctx.fillRect(pad.left + seriesIndex * 150, 10, 12, 12);
    ctx.fillStyle = '#344054';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.label, pad.left + 18 + seriesIndex * 150, 16);
  });
  ctx.fillStyle = '#475467';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(options.xLabel || 'Epoch', pad.left + plotWidth / 2, height - 4);
  ctx.save();
  ctx.translate(13, pad.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(options.yLabel || '数值', 0, 0);
  ctx.restore();
}

function renderHistory(result) {
  const modelType = String(result.model?.type || '').toLowerCase();
  if (result.model?.family === 'traditional_ml' || result.model?.family === 'traditional' || TRADITIONAL_MODEL_IDS.has(modelType)) return null;
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
      ], { xLabel: 'Epoch', yLabel: 'Loss', decimals: 3 });
      drawHistoryChart(metricCanvas, selected, [
        { key: 'valid_accuracy', label: 'valid accuracy', color: '#0f766e' },
        { key: 'valid_macro_f1', label: 'valid macro F1', color: '#475467' },
      ], { xLabel: 'Epoch', yLabel: '指标值', decimals: 2, domain: { min: 0, max: 1 } });
    });
  };
  select.addEventListener('change', draw);
  if (historyResizeHandler) window.removeEventListener('resize', historyResizeHandler);
  historyResizeHandler = () => requestAnimationFrame(draw);
  window.addEventListener('resize', historyResizeHandler, { passive: true });
  draw();
  return card;
}

function renderAnalysis(result) {
  const target = byId('resultAnalysis');
  if (!target) return;
  const splits = analysisSplitEntries(result);
  const confusionCards = splits.map(({ name, analysis }) => renderSplitConfusion(name, analysis, result)).filter(Boolean);
  const classCards = splits.map(({ name, analysis }) => renderSplitClassMetrics(name, analysis, result)).filter(Boolean);
  const distributionCards = splits.map(({ name, analysis }) => renderSplitDistribution(name, analysis, result)).filter(Boolean);
  const historyCard = renderHistory(result);
  const auditCard = renderTrainingAudit(result);
  const group = (title, note, cards) => cards.length
    ? element('section', { className: 'analysis-group' },
      element('div', { className: 'analysis-group-heading' }, element('h3', { text: title }), element('p', { text: note })),
      element('div', { className: 'analysis-split-grid' }, ...cards),
    )
    : null;
  replaceChildren(target,
    element('div', { className: 'section-heading' }, element('div', {},
      element('h2', { text: '图表与分析' }),
      element('p', { text: '这里只展示当前 Run 已真实生成或可由现有指标直接推导的分析。' }),
    )),
    group('混淆矩阵', '并列比较 Train、Valid 与 Test；行是真实类别，列是预测类别。', confusionCards),
    group('各类别指标', '分别查看每个数据分区中各类别的 Precision、Recall、F1 与样本量。', classCards),
    group('预测结果分布', '每个类别使用竖向分组柱比较真实数量和预测数量。', distributionCards),
    historyCard,
    auditCard,
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

function explainabilityArtifact(result) {
  const summary = result.explainability?.samples || {};
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
    return category === 'explainability' && artifact.format === 'json' && name.includes('sample');
  }) || null;
}

async function explainabilityPayload(result, generation) {
  const summary = result.explainability?.samples || {};
  if (Array.isArray(summary.samples)) return summary;
  const artifact = explainabilityArtifact(result);
  if (!artifact?.downloadable || !artifact?.download_url || !safeDownloadUrl(artifact.download_url)) return null;
  if (!explainabilityPayloadCache.has(artifact.download_url)) {
    explainabilityPayloadCache.set(artifact.download_url, request(artifact.download_url).catch((error) => {
      explainabilityPayloadCache.delete(artifact.download_url);
      throw error;
    }));
    while (explainabilityPayloadCache.size > 4) {
      const oldestKey = explainabilityPayloadCache.keys().next().value;
      if (!oldestKey || oldestKey === artifact.download_url) break;
      explainabilityPayloadCache.delete(oldestKey);
    }
  }
  const payload = await explainabilityPayloadCache.get(artifact.download_url);
  return generation === resultRenderGeneration ? payload : null;
}

function sampleAxis(sample, payload) {
  const curve = Array.isArray(sample?.curve) ? sample.curve.map(Number) : [];
  const supplied = Array.isArray(sample?.sample_x_axis) && sample.sample_x_axis.length === curve.length
    ? sample.sample_x_axis
    : Array.isArray(payload?.x_axis) && payload.x_axis.length === curve.length
      ? payload.x_axis
      : curve.map((_, index) => index);
  return supplied.map(Number);
}

function sampleWindows(sample, curveLength) {
  return (Array.isArray(sample?.windows) ? sample.windows : []).filter((windowItem) => {
    const start = Number(windowItem?.start_index);
    const end = Number(windowItem?.end_index);
    return Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end >= start && end < curveLength;
  });
}

export function validateSampleExplanationPayload(payload) {
  if (payload?.status && payload.status !== 'ready') return { valid: false, reason: payload.reason || '解释结果尚未生成。', samples: [] };
  const samples = (Array.isArray(payload?.samples) ? payload.samples : []).filter((sample) => {
    const curve = Array.isArray(sample?.curve) ? sample.curve : [];
    const axis = sampleAxis(sample, payload);
    return curve.length > 1 && axis.length === curve.length && curve.every((value) => Number.isFinite(Number(value)))
      && axis.every((value) => Number.isFinite(Number(value)));
  });
  return samples.length
    ? { valid: true, reason: null, samples }
    : { valid: false, reason: payload?.reason || '解释文件没有可绘制的样品曲线。', samples: [] };
}

export function importanceHeatColor(value) {
  const normalized = Math.max(0, Math.min(1, Number(value) || 0));
  const start = [255, 247, 237];
  const end = [220, 38, 38];
  const rgb = start.map((channel, index) => Math.round(channel + (end[index] - channel) * normalized));
  return `rgb(${rgb.join(', ')})`;
}

function drawSampleExplanationChart(canvas, sample, payload) {
  const curve = sample.curve.map(Number);
  const axis = sampleAxis(sample, payload);
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(rect.width || 760, 1);
  const height = Math.max(rect.height || 320, 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px system-ui, sans-serif';
  const pad = { left: 58, right: 18, top: 20, bottom: 46 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xMin = Math.min(...axis);
  const xMax = Math.max(...axis);
  const [yMin, yMax] = chartDomain(curve);
  const xSpan = Math.max(xMax - xMin, 1e-9);
  const ySpan = Math.max(yMax - yMin, 1e-9);
  const xAt = (value) => pad.left + ((value - xMin) / xSpan) * plotWidth;
  const yAt = (value) => pad.top + plotHeight - ((value - yMin) / ySpan) * plotHeight;
  const primary = sample?.primary_segment;
  if (primary) {
    const startIndex = Math.max(0, Math.min(curve.length - 1, Number(primary.start_index) || 0));
    const endIndex = Math.max(startIndex, Math.min(curve.length - 1, Number(primary.end_index) || startIndex));
    const startX = Number.isFinite(Number(primary.start_x)) ? Number(primary.start_x) : axis[startIndex];
    const endX = Number.isFinite(Number(primary.end_x)) ? Number(primary.end_x) : axis[endIndex];
    ctx.fillStyle = 'rgba(220, 38, 38, 0.16)';
    ctx.fillRect(xAt(Math.min(startX, endX)), pad.top, Math.max(2, xAt(Math.max(startX, endX)) - xAt(Math.min(startX, endX))), plotHeight);
  }
  linearTicks(yMin, yMax, 5).forEach((tick) => {
    const y = yAt(tick);
    ctx.strokeStyle = '#e4e8ef';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotWidth, y);
    ctx.stroke();
    ctx.fillStyle = '#667085';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(Number(tick).toFixed(3), pad.left - 8, y);
  });
  linearTicks(xMin, xMax, 5).forEach((tick) => {
    const x = xAt(tick);
    ctx.fillStyle = '#667085';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(Number(tick).toFixed(3), x, pad.top + plotHeight + 8);
  });
  ctx.strokeStyle = '#98a2b3';
  ctx.strokeRect(pad.left, pad.top, plotWidth, plotHeight);
  ctx.strokeStyle = '#0f766e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.forEach((value, index) => {
    const x = xAt(axis[index]);
    const y = yAt(value);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#475467';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('X 轴', pad.left + plotWidth / 2, height - 4);
  ctx.save();
  ctx.translate(13, pad.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Intensity', 0, 0);
  ctx.restore();
}

function drawImportanceHeatmap(canvas, sample, payload) {
  const curve = sample.curve.map(Number);
  const windows = sampleWindows(sample, curve.length);
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(rect.width || 760, 1);
  const height = Math.max(rect.height || 72, 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const pad = 8;
  const plotWidth = width - pad * 2;
  windows.forEach((windowItem) => {
    const start = Number(windowItem.start_index);
    const end = Number(windowItem.end_index);
    const x = pad + (start / curve.length) * plotWidth;
    const itemWidth = Math.max(1, ((end - start + 1) / curve.length) * plotWidth);
    ctx.fillStyle = importanceHeatColor(windowItem.normalized_importance ?? windowItem.importance);
    ctx.fillRect(x, 12, itemWidth, height - 24);
  });
  ctx.strokeStyle = '#d0d5dd';
  ctx.strokeRect(pad, 12, plotWidth, height - 24);
  canvas.setAttribute('aria-label', `${sampleDisplayLabel(sample)} 的特征窗口重要性热力条，共 ${windows.length} 个窗口`);
}

function explainabilityErrorMessage(error) {
  if (error?.status === 403) return '当前身份没有读取该解释文件的权限。';
  if (error?.status === 404) return '单样品解释文件不存在或已被移除。';
  if (error?.status === 409) return '解释文件尚未发布完成，请稍后重试。';
  return error?.message || '单样品解释加载失败。';
}

function sampleDisplayLabel(sample) {
  const sampleName = sample?.name || sample?.sample_id || sample?.index || sample?.result_id || '-';
  const fold = sample?.fold_index ? `第 ${sample.fold_index} 折 · ` : '';
  return `${fold}${sampleName} · 真实 ${valueOrDash(sample?.true_label)} · 预测 ${valueOrDash(sample?.pred_label)}`;
}

function renderSampleImportance(target, payload, summary) {
  const validation = validateSampleExplanationPayload(payload);
  const samples = validation.samples;
  replaceChildren(target, element('h3', { text: '单样品可解释性' }));
  if (!validation.valid) {
    target.append(notice('empty', '暂无单样品结果', validation.reason || summary?.reason || '当前 Run 未生成单样品解释产物。'));
    return;
  }
  const select = element('select', { id: 'resultSampleExplanationSelect', 'aria-label': '选择要查看解释结果的样品' }, ...samples.map((sample, index) => (
    element('option', { value: String(index), text: sampleDisplayLabel(sample) })
  )));
  const details = element('div', { className: 'sample-explanation-details' });
  const renderSelected = () => {
    const sample = samples[Number(select.value || 0)] || samples[0];
    const curveCanvas = element('canvas', { className: 'sample-explanation-chart', role: 'img', 'aria-label': `${sampleDisplayLabel(sample)} 的曲线与第一重要区间` });
    const heatCanvas = element('canvas', { className: 'sample-importance-heatmap', role: 'img' });
    const primaryText = sample?.primary_segment
      ? `第一重要区间：${segmentLabel(sample.primary_segment)}`
      : '没有可用的第一重要区间。';
    replaceChildren(details,
      element('dl', { className: 'feature-meta' },
        overviewItem('Sample_ID / Name', sample.name || sample.sample_id || sample.index),
        overviewItem('真实标签', sample.true_label),
        overviewItem('预测标签', sample.pred_label),
        overviewItem('预测状态', sample.correct === true ? '正确' : sample.correct === false ? '不一致' : '-'),
        overviewItem('真实类别概率', Number.isFinite(Number(sample.true_probability)) ? formatMetric(sample.true_probability) : '-'),
      ),
      element('p', { className: 'section-note explanation-method', text: `解释方法：${payload?.method || summary?.method || '-'} · ${primaryText}` }),
      curveCanvas,
      element('div', { className: 'heatmap-heading' },
        element('strong', { text: '特征窗口重要性' }),
        element('span', { className: 'heatmap-legend' },
          element('span', { text: '重要性低' }),
          element('i', { 'aria-hidden': 'true' }),
          element('span', { text: '重要性高' }),
        ),
      ),
      heatCanvas,
      element('h4', { text: '重要区间明细' }),
      importanceBars(sample.top_segments),
    );
    const draw = () => {
      drawSampleExplanationChart(curveCanvas, sample, payload);
      drawImportanceHeatmap(heatCanvas, sample, payload);
    };
    if (sampleExplanationResizeHandler) window.removeEventListener('resize', sampleExplanationResizeHandler);
    sampleExplanationResizeHandler = () => requestAnimationFrame(draw);
    window.addEventListener('resize', sampleExplanationResizeHandler, { passive: true });
    requestAnimationFrame(draw);
  };
  select.addEventListener('change', renderSelected);
  target.append(element('div', { className: 'feature-toolbar' }, element('label', { htmlFor: 'resultSampleExplanationSelect', text: '样品' }), select), details);
  renderSelected();
}

async function renderExplainability(result) {
  const target = byId('resultExplainability');
  if (!target) return;
  const generation = resultRenderGeneration;
  const sampleCard = element('article', { className: 'result-card wide-card' });
  replaceChildren(target,
    element('div', { className: 'section-heading' }, element('div', {},
      element('h2', { text: '模型解释' }),
      element('p', { text: '按测试集单样品展示真实曲线和重要区间；不同模型使用其实际生成的解释方法。' }),
    )),
    element('div', { className: 'result-analysis-grid' }, sampleCard),
  );
  const sampleArtifact = explainabilityArtifact(result);
  const sampleSummary = result.explainability?.samples || {};
  const canLoadSamples = Array.isArray(sampleSummary.samples) || sampleArtifact?.downloadable === true;
  const sampleSize = Number(sampleArtifact?.size_bytes);
  const sampleSizeLabel = Number.isFinite(sampleSize) ? `（约 ${(sampleSize / 1024 / 1024).toFixed(1)} MB）` : '';
  if (!canLoadSamples) {
    replaceChildren(sampleCard,
      element('h3', { text: '单样品可解释性' }),
      notice('empty', '暂不可用', sampleSummary.reason || sampleArtifact?.reason || '当前 Run 未生成单样品解释产物。'),
    );
    return;
  }
  replaceChildren(sampleCard,
    element('h3', { text: '单样品可解释性' }),
    notice('loading', '正在读取', `正在自动加载单样品解释${sampleSizeLabel}。`),
  );
  try {
    const samplePayload = await explainabilityPayload(result, generation);
    if (generation !== resultRenderGeneration) return;
    if (samplePayload) renderSampleImportance(sampleCard, samplePayload, result.explainability?.samples);
    else replaceChildren(sampleCard, element('h3', { text: '单样品可解释性' }), notice('empty', '暂不可用', result.explainability?.samples?.reason || '当前 Run 未提供可读取的单样品解释产物。'));
  } catch (error) {
    if (generation !== resultRenderGeneration) return;
    const retry = element('button', { className: 'button secondary', type: 'button', text: '重新加载' });
    retry.addEventListener('click', () => renderExplainability(result));
    replaceChildren(sampleCard, element('h3', { text: '单样品可解释性' }), notice('error', '加载失败', explainabilityErrorMessage(error), [retry]));
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
  const format = String(artifact?.format || '').toUpperCase();
  const meta = [artifact?.name, format, artifact?.size_bytes != null ? `${Math.ceil(Number(artifact.size_bytes) / 1024)} KB` : '']
    .filter(Boolean).join(' · ');
  const button = element('button', { className: 'button secondary', type: 'button', text: `下载${label}${format ? `（${format}）` : ''}`, disabled: !available });
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
    const legacyGlobalImportance = ['feature_importance.json', 'feature_importance.csv'].includes(name);
    return category !== 'model'
      && !legacyGlobalImportance
      && !['model.pkl', 'model.pt'].includes(name)
      && !name.endsWith('.joblib');
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
      const datasetName = run.dataset_name || run.dataset?.name || run.config?.dataset_name || '-';
      const trainingTime = run.started_at || run.created_at;
      const timeLabel = run.started_at ? formatTime(trainingTime) : `${formatTime(trainingTime)}（创建）`;
      const link = element('a', { className: 'button secondary compact', href: buildResultHash(runId), text: '查看' });
      return element('tr', {},
        element('td', {}, element('span', { className: 'truncate-text', text: runId, title: runId })),
        element('td', {}, element('span', { className: 'truncate-text', text: datasetName, title: datasetName })),
        element('td', { text: modelName }),
        element('td', { text: stateMeta(canonicalState(run))[0] }),
        element('td', { text: timeLabel }),
        element('td', { text: durationText(run) }),
        element('td', {}, link),
      );
    });
    replaceChildren(overview,
      element('div', { className: 'section-heading' }, element('div', {}, element('h2', { text: '最近任务' }), element('p', { text: '列表只加载摘要，完整结果在进入 Run 后获取。' }))),
      rows.length ? element('div', { className: 'table-scroll' }, element('table', {},
        element('thead', {}, element('tr', {}, ...['Run ID', '数据集', '模型', '状态', '训练时间', '耗时', ''].map((label) => element('th', { text: label })))),
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

export function cancelAutoRedirect(options = {}) {
  const restoreFocus = options?.restoreFocus !== false;
  if (redirectTimer != null) clearTimeout(redirectTimer);
  if (redirectTicker != null) clearInterval(redirectTicker);
  redirectTimer = null;
  redirectTicker = null;
  redirectRunId = null;
  hideTrainingResultDialog(restoreFocus);
}

function hideTrainingResultDialog(restoreFocus = true) {
  const dialog = byId('trainingResultDialog');
  if (!dialog || dialog.classList.contains('hidden')) return;
  dialog.classList.add('hidden');
  dialog.setAttribute('aria-hidden', 'true');
  if (trainingDialogFocusTimer != null) clearTimeout(trainingDialogFocusTimer);
  trainingDialogFocusTimer = null;
  if (restoreFocus && trainingResultReturnFocus instanceof HTMLElement) trainingResultReturnFocus.focus();
  trainingResultReturnFocus = null;
}

function showTrainingResultDialog(runId) {
  const dialog = byId('trainingResultDialog');
  const nowButton = byId('trainingResultNow');
  const stayButton = byId('trainingResultStay');
  const countdown = byId('trainingResultCountdown');
  if (!dialog || !nowButton || !stayButton || !countdown) return null;
  trainingResultReturnFocus = document.activeElement;
  byId('trainingResultRunId').textContent = runId;
  countdown.textContent = '3 秒后进入建模结果';
  nowButton.onclick = () => {
    cancelAutoRedirect({ restoreFocus: false });
    navigateToResult(runId);
  };
  stayButton.onclick = () => cancelAutoRedirect();
  dialog.classList.remove('hidden');
  dialog.setAttribute('aria-hidden', 'false');
  trainingDialogFocusTimer = setTimeout(() => {
    trainingDialogFocusTimer = null;
    if (!dialog.classList.contains('hidden')) nowButton.focus();
  }, 0);
  return countdown;
}

export function showNewRunSuccess(run) {
  const runId = String(run?.run_id || run?.run?.run_id || '');
  const state = canonicalState(run);
  if (!shouldAutoRedirect({ runId, state, isNewRun: newRunIds.has(runId) })) return false;
  if (parseHash(window.location.hash).view !== 'modeling') return false;
  if (redirectRunId === runId) return true;
  cancelAutoRedirect();
  redirectRunId = runId;
  ['metrics', 'trainingAudit', 'trainingCharts', 'confusionMatrix', 'downloads']
    .forEach((id) => byId(id)?.replaceChildren());
  const link = element('a', { className: 'button primary', href: buildResultHash(runId), text: '立即查看结果' });
  link.addEventListener('click', () => cancelAutoRedirect({ restoreFocus: false }));
  replaceChildren(byId('trainingProgress'), notice('success', '训练已完成', '结果已保存，可立即进入专属结果页，也可以留在当前页面。', [link]));
  const countdown = showTrainingResultDialog(runId);
  if (!countdown) return true;
  const started = Date.now();
  redirectTicker = setInterval(() => {
    const remaining = Math.max(0, Math.ceil((AUTO_REDIRECT_DELAY_MS - (Date.now() - started)) / 1000));
    countdown.textContent = remaining > 0 ? `${remaining} 秒后进入建模结果` : '正在打开建模结果…';
  }, 250);
  redirectTimer = setTimeout(() => {
    const destination = buildResultHash(runId);
    cancelAutoRedirect({ restoreFocus: false });
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
  if (route.view === 'results') {
    requestAnimationFrame(() => {
      const target = byId('resultPageState');
      if (!target) return;
      target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
    });
  }
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
      explainabilityPayloadCache.clear();
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
  document.addEventListener('keydown', (event) => {
    const dialog = byId('trainingResultDialog');
    if (!dialog || dialog.classList.contains('hidden')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      cancelAutoRedirect();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = [...dialog.querySelectorAll('button, a[href], [tabindex]:not([tabindex="-1"])')]
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
