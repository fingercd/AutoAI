/** 状态映射、指标格式化、CV 口径守卫和服务器路径检查等纯函数。 */

export const RUN_STATE_META = {
  queued: { label: '排队中', tone: 'pending' },
  running: { label: '运行中', tone: 'active' },
  succeeded: { label: '已成功', tone: 'success' },
  failed: { label: '已失败', tone: 'danger' },
  cancelled: { label: '已取消', tone: 'muted' },
};

/** 规范 state 到旧兼容 status 的映射，仅用于展示对照。 */
export const LEGACY_STATUS_BY_STATE = {
  queued: 'pending',
  running: 'running',
  succeeded: 'success',
  failed: 'failed',
  cancelled: 'paused',
};

export const RESULT_STATE_META = {
  pending: { label: '等待执行', tone: 'pending', description: 'Run 已排队，等待 worker 领取。' },
  running: { label: '训练中', tone: 'active', description: 'worker 正在执行训练。' },
  failed: { label: '训练失败', tone: 'danger', description: '训练执行失败，请查看错误信息。' },
  cancelled: { label: '已取消', tone: 'muted', description: '该 Run 已被取消，不再产生结果。' },
  ready: { label: '结果完整', tone: 'success', description: '训练成功，必需结果文件完整。' },
  partial: { label: '结果不完整', tone: 'warning', description: '训练成功，但部分结果文件缺失或完整性校验失败。' },
  missing_manifest: { label: 'Manifest 缺失', tone: 'warning', description: '训练成功，但产物清单缺失，无法验证或下载文件。' },
  corrupt_manifest: { label: 'Manifest 损坏', tone: 'danger', description: '产物清单无法解析，无法验证或下载文件。' },
};

export const TERMINAL_STATES = new Set(['succeeded', 'failed', 'cancelled']);
export const ACTIVE_STATES = new Set(['queued', 'running']);

export function stateMeta(state) {
  return RUN_STATE_META[state] || { label: state || '未知', tone: 'muted' };
}

export function resultStateMeta(resultState) {
  return RESULT_STATE_META[resultState] || { label: resultState || '未知', tone: 'muted', description: '' };
}

export function isTerminalState(state) {
  return TERMINAL_STATES.has(state);
}

export function isActiveState(state) {
  return ACTIVE_STATES.has(state);
}

/** 共享 class API 的 Run 状态徽章类：badge + status-<state>；未知状态退回中性 badge。 */
export function stateBadgeClass(state) {
  return RUN_STATE_META[state] ? `badge status-${state}` : 'badge';
}

/**
 * result_state → 共享徽章类。partial / missing_manifest 用排队色表示“需要注意”，
 * 具体语义由徽章文案承担；未知状态退回中性 badge。
 */
const RESULT_STATE_BADGE_CLASSES = {
  pending: 'badge status-queued',
  running: 'badge status-running',
  ready: 'badge status-succeeded',
  failed: 'badge status-failed',
  corrupt_manifest: 'badge status-failed',
  cancelled: 'badge status-cancelled',
  partial: 'badge status-queued',
  missing_manifest: 'badge status-queued',
};

export function resultStateBadgeClass(resultState) {
  return RESULT_STATE_BADGE_CLASSES[resultState] || 'badge';
}

export const SCALAR_METRIC_KEYS = [
  'accuracy',
  'balanced_accuracy',
  'macro_precision',
  'macro_recall',
  'macro_f1',
  'weighted_f1',
];

export const METRIC_LABELS = {
  accuracy: '准确率 Accuracy',
  balanced_accuracy: '平衡准确率',
  macro_precision: '宏精确率',
  macro_recall: '宏召回率',
  macro_f1: '宏 F1',
  weighted_f1: '加权 F1',
};

export const EVALUATION_STRATEGIES = {
  stratified_holdout: {
    label: '分层留出 stratified_holdout',
    shortLabel: '分层留出 8:1:1',
    description: '按 Sample_ID 整组、按标签比例划分 8:1:1 的 train/valid/test。',
  },
  leave_one_sample_id_cv: {
    label: '留一样本交叉验证 leave_one_sample_id_cv',
    shortLabel: '留一 Sample_ID CV',
    description: '每折留一个 Sample_ID 作 test，其余按 8:2 形成 train/valid；Test 主指标由全部交叉验证折的测试预测合并计算。',
  },
  external_test_holdout: {
    label: '独立测试集 external_test_holdout',
    shortLabel: '独立测试集 8:2',
    description: '主数据按 Sample_ID 整组 8:2 划分 train/valid，独立测试集作为最终 test；与交叉验证互斥。',
  },
};

export const AGGREGATION_LABELS = {
  direct: '直接计算',
  pooled_oof: '合并交叉验证预测',
  fold_mean: '逐折均值（审计）',
  pooled_cross_fold: '跨折预测合并（样本可能重复）',
};

export function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(4);
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '—';
  const total = Number(seconds);
  if (total < 60) return `${total.toFixed(1)} 秒`;
  const minutes = Math.floor(total / 60);
  const rest = Math.round(total - minutes * 60);
  if (minutes < 60) return `${minutes} 分 ${rest} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes - hours * 60} 分`;
}

export function formatBytes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const bytes = Number(value);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
}

/** 训练时间优先取实际开始时间；尚未开始时回退任务创建时间并明确标注。 */
export function formatTrainingTime(run) {
  if (run?.started_at) return formatDateTime(run.started_at);
  if (run?.created_at) return `${formatDateTime(run.created_at)}（任务创建）`;
  return '—';
}

export const MODEL_FAMILY_LABELS = {
  traditional_ml: '传统机器学习',
  basic_deep: '基础深度学习',
  convolutional: '卷积网络',
  long_range: '长程建模',
  two_dimensional_mapping: '二维映射网络',
};

const CV_STRATEGY = 'leave_one_sample_id_cv';

export function isCvResult(result) {
  return result?.evaluation?.strategy === CV_STRATEGY;
}

/**
 * CV 口径守卫：解析某个 split 的标量指标。
 * - CV 的 test 只允许 pooled_oof；train/valid 标量只允许 fold mean（fold_std 为审计值）。
 * - 非 CV（holdout）全部为 direct。
 * 返回 { split, aggregation, values, foldStd, note } 或 null。
 */
export function resolveSplitScalars(result, split) {
  const entry = result?.metrics?.splits?.[split];
  if (!entry || typeof entry !== 'object') return null;
  if (isCvResult(result)) {
    if (split === 'test') {
      const values = entry.aggregation === 'pooled_oof'
        ? entry.values ?? null
        : result?.metrics?.pooled_oof ?? null;
      return {
        split,
        aggregation: 'pooled_oof',
        values,
        foldStd: null,
        note: 'Test 主指标由全部交叉验证折的测试预测合并计算',
      };
    }
    return {
      split,
      aggregation: 'fold_mean',
      values: entry.values ?? null,
      foldStd: entry.fold_std ?? null,
      note: 'Train/Valid 标量为逐折均值，标准差仅作审计；图表分析使用跨折合并预测',
    };
  }
  return {
    split,
    aggregation: 'direct',
    values: entry.values ?? null,
    foldStd: null,
    note: '',
  };
}

/**
 * 校验 CV 结果没有把 fold mean、pooled cross-fold 与 pooled_oof 混在同一口径。
 * 返回 { ok, violations: string[] }。
 */
export function validateCvAggregation(result) {
  const violations = [];
  if (!isCvResult(result)) return { ok: true, violations };
  const splits = result?.metrics?.splits || {};
  const testAggregation = splits.test?.aggregation;
  if (testAggregation && testAggregation !== 'pooled_oof') {
    violations.push(`CV test split 口径必须为 pooled_oof，实际为 ${testAggregation}`);
  }
  for (const split of ['train', 'valid']) {
    const aggregation = splits[split]?.aggregation;
    if (aggregation && aggregation !== 'fold_mean') {
      violations.push(`CV ${split} split 标量口径必须为 fold_mean，实际为 ${aggregation}`);
    }
  }
  const analysisSplits = result?.analysis?.splits || {};
  const testChart = analysisSplits.test?.aggregation;
  if (testChart && testChart !== 'pooled_oof') {
    violations.push(`CV test 图表口径必须为 pooled_oof，实际为 ${testChart}`);
  }
  for (const split of ['train', 'valid']) {
    const chart = analysisSplits[split]?.aggregation;
    if (chart && chart !== 'pooled_cross_fold') {
      violations.push(`CV ${split} 图表口径必须为 pooled_cross_fold，实际为 ${chart}`);
    }
  }
  return { ok: violations.length === 0, violations };
}

/** 契约允许下载/展示的名称黑名单之外的检查放在 artifacts 组件；这里只做路径形态判断。 */
const WINDOWS_ABS_PATH = /^[A-Za-z]:[\\/]/;
const UNC_PATH = /^\\\\/;
const POSIX_ABS_PATH = /^\/(home|Users|var|opt|srv|tmp|mnt|data|root|etc|usr)\//;
const SERVER_PATH_KEYS = new Set(['path', 'data_path', 'test_data_path', 'run_dir', 'output_path', 'common_time_path', 'dataset_path']);

export function looksLikeServerPath(value) {
  if (typeof value !== 'string' || !value) return false;
  return WINDOWS_ABS_PATH.test(value) || UNC_PATH.test(value) || POSIX_ABS_PATH.test(value);
}

/**
 * 递归检查响应中是否夹带服务器路径字段。
 * options.ignoreKeys：允许出现的键名（例如 download_url，其查询串由后端生成、作为整体 URL 使用）。
 * 返回违规键路径数组，空数组表示干净。
 */
export function findServerPaths(value, options = {}, keyPath = '') {
  const ignoreKeys = new Set(options.ignoreKeys || ['download_url']);
  const violations = [];
  const walk = (current, path) => {
    if (Array.isArray(current)) {
      current.forEach((item, index) => walk(item, `${path}[${index}]`));
      return;
    }
    if (current && typeof current === 'object') {
      for (const [key, item] of Object.entries(current)) {
        const childPath = path ? `${path}.${key}` : key;
        if (ignoreKeys.has(key)) continue;
        if (SERVER_PATH_KEYS.has(key) || key.endsWith('_path')) {
          violations.push(childPath);
          continue;
        }
        walk(item, childPath);
      }
      return;
    }
    if (looksLikeServerPath(current)) {
      violations.push(path || '(root)');
    }
  };
  walk(value, keyPath);
  return violations;
}
