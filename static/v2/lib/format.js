/**
 * format.js —— v2 前端的"格式化与口径"纯函数库。
 *
 * 模块职责：
 * - Run / result_state 状态到中文文案、色调（tone）、徽章 class 的映射；
 * - 指标（accuracy、macro_f1 等）、时长、字节数、日期时间的展示格式化；
 * - 交叉验证（leave_one_sample_id_cv）评估口径守卫：保证结果页不会把
 *   pooled_oof、fold_mean、pooled_cross_fold 三种口径混在一起展示
 *   （契约见 docs/run_result_contract.md，接口为 run-result-v1）；
 * - 服务器路径泄漏检查：server 部署模式下，前端响应里不应夹带服务器
 *   绝对路径（安全要求：config.json 含服务器路径时不可下载）。
 *
 * 在系统中的位置：被 static/v2 各视图组件（建模结果、训练记录等）import，
 * 本身不依赖任何 DOM / 网络 API，是纯函数模块，因此可在 Node 下直接做单元测试。
 *
 * 关键设计约束：
 * - 全部函数对 null / undefined / NaN 健壮，非法输入统一回退为 '—' 或
 *   中性样式，绝不在 UI 上抛出异常或渲染 "undefined"；
 * - CV 主指标口径以后端返回为准：CV 的 test 标量只认 pooled_oof，
 *   train/valid 标量只认 fold_mean（fold_std 仅作审计展示）。
 */

/**
 * Run 生命周期状态 → 展示元数据。
 * label：中文文案；tone：色调语义（对应 styles.css 中 .badge / run-row 的
 * data-tone 配色：pending/active/success/danger/muted）。
 * 状态机来源：后端训练 Run（queued → running → succeeded/failed/cancelled）。
 */
export const RUN_STATE_META = {
  queued: { label: '排队中', tone: 'pending' },
  running: { label: '运行中', tone: 'active' },
  succeeded: { label: '已成功', tone: 'success' },
  failed: { label: '已失败', tone: 'danger' },
  cancelled: { label: 'STOP', tone: 'muted' },
};

/** 规范 state 到旧兼容 status 的映射，仅用于展示对照。 */
export const LEGACY_STATUS_BY_STATE = {
  queued: 'pending',
  running: 'running',
  succeeded: 'success',
  failed: 'failed',
  cancelled: 'paused',
};

/**
 * result_state（结果页视图状态）→ 展示元数据。
 * 与 RUN_STATE_META 的区别：RUN_STATE_META 描述 Run 本身的执行状态；
 * result_state 是后端在 run-result-v1 契约里给出的"结果可用性"判断——
 * 训练成功（succeeded）之后还要区分结果文件是否完整、Manifest 是否可校验，
 * 因此多出 ready / partial / missing_manifest / corrupt_manifest 四种状态。
 * description 用于向用户解释当前为什么不能下载或查看完整结果。
 */
export const RESULT_STATE_META = {
  pending: { label: '等待执行', tone: 'pending', description: 'Run 已排队，等待 worker 领取。' },
  running: { label: '训练中', tone: 'active', description: 'worker 正在执行训练。' },
  failed: { label: '训练失败', tone: 'danger', description: '训练执行失败，请查看错误信息。' },
  cancelled: { label: 'STOP', tone: 'muted', description: '该 Run 已停止，本次训练产物已丢弃。' },
  ready: { label: '结果完整', tone: 'success', description: '训练成功，必需结果文件完整。' },
  partial: { label: '结果不完整', tone: 'warning', description: '训练成功，但部分结果文件缺失或完整性校验失败。' },
  missing_manifest: { label: 'Manifest 缺失', tone: 'warning', description: '训练成功，但产物清单缺失，无法验证或下载文件。' },
  corrupt_manifest: { label: 'Manifest 损坏', tone: 'danger', description: '产物清单无法解析，无法验证或下载文件。' },
};

export const TERMINAL_STATES = new Set(['succeeded', 'failed', 'cancelled']);
export const ACTIVE_STATES = new Set(['queued', 'running']);

/**
 * Run 状态 → 展示元数据；未知状态回退为"原样显示 + muted 中性色"，
 * 保证后端新增状态时前端不崩、只是样式朴素。
 */
export function stateMeta(state) {
  return RUN_STATE_META[state] || { label: state || '未知', tone: 'muted' };
}

/** result_state 版本的状态元数据查询，回退策略同 stateMeta。 */
export function resultStateMeta(resultState) {
  return RESULT_STATE_META[resultState] || { label: resultState || '未知', tone: 'muted', description: '' };
}

/** 是否为终态：终态意味着不再变化，轮询可以停止。 */
export function isTerminalState(state) {
  return TERMINAL_STATES.has(state);
}

/** 是否为活跃态：活跃态需要持续轮询进度。 */
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

/** 结果页要展示的标量指标键（与 run-result-v1 契约中的 metrics 字段对应）。 */
export const SCALAR_METRIC_KEYS = [
  'accuracy',
  'balanced_accuracy',
  'macro_precision',
  'macro_recall',
  'macro_f1',
  'weighted_f1',
];

/** 指标键 → 中文展示名（宏平均 = macro，各类等权；加权 = weighted，按支持数加权）。 */
export const METRIC_LABELS = {
  accuracy: '准确率 Accuracy',
  balanced_accuracy: '平衡准确率',
  macro_precision: '宏精确率',
  macro_recall: '宏召回率',
  macro_f1: '宏 F1',
  weighted_f1: '加权 F1',
};

/**
 * 三种分类评估口径的展示文案（与后端训练评估策略一一对应）：
 * - stratified_holdout：无独立测试集时按 Sample_ID 整组 8:1:1 划分；
 * - leave_one_sample_id_cv：每折留 1 个 Sample_ID 做 test 的交叉验证；
 * - external_test_holdout：有独立测试集时主数据 8:2 划分，独立集做最终 test。
 * label 用于表单/结果页全称，shortLabel 用于空间受限处，description 解释口径约束。
 */
export const EVALUATION_STRATEGIES = {
  stratified_holdout: {
    label: '分层留出 stratified_holdout',
    shortLabel: '分层留出 8:1:1',
    description: '按 Sample_ID 整组、以 8:1:1 为目标划分；Valid/Test 至少各包含每类 1 个样品组。',
  },
  leave_one_sample_id_cv: {
    label: '留一样本交叉验证 leave_one_sample_id_cv',
    shortLabel: '留一 Sample_ID CV',
    description: '每折留一个 Sample_ID 作 test，其余按 8:2 形成 train/valid；Test 主指标由全部交叉验证折的测试预测合并计算。',
  },
  external_test_holdout: {
    label: '独立测试集 external_test_holdout',
    shortLabel: '独立测试集 8:2',
    description: '主数据按 Sample_ID 整组 8:2 划分 train/valid，独立测试集作为最终 test。',
  },
  leave_one_sample_id_cv_with_external_test: {
    label: '独立测试集 + 留一审计',
    shortLabel: '独立测试集 + 留一审计',
    description: '主数据按 Sample_ID 留一生成合并预测审计；最终在全部主数据重训，只以独立测试集作为主指标。',
  },
};

/**
 * 指标聚合口径 → 中文名。
 * direct：非 CV 直接计算；pooled_oof：CV 全部折的 out-of-fold 预测合并后
 * 计算（这是 CV 测试集主指标的唯一合法口径）；fold_mean：逐折求均值，
 * 仅作审计参考，不能当主指标；pooled_cross_fold：图表分析用，把各折
 * train/valid 预测跨折合并，样本可能重复出现。
 */
export const AGGREGATION_LABELS = {
  direct: '直接计算',
  direct_external_test: '独立测试集直接计算',
  pooled_oof: '合并交叉验证预测',
  fold_mean: '逐折均值（审计）',
  pooled_cross_fold: '跨折预测合并（样本可能重复）',
};

/**
 * 指标数值格式化：统一 4 位小数；null/undefined/NaN 显示占位符 '—'。
 * Number(value) 先做一次宽容转换，兼容后端可能发来的字符串数字。
 */
export function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(4);
}

/**
 * 秒数 → 人性化时长：不足 1 分钟保留 1 位小数（"12.3 秒"），
 * 之后逐级进位为 "X 分 Y 秒"、"X 小时 Y 分"。
 * 注意分钟进位后对余数四舍五入（Math.round），避免 "1 分 59.6 秒"。
 */
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

/**
 * 字节数 → 人性化体积：B / KB / MB 三档，二进制进制（1024）。
 * 结果文件体积通常 < 1 GB，因此没有 TB 档。
 */
export function formatBytes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const bytes = Number(value);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * ISO 时间串 → 本地中文格式（24 小时制）。
 * 无法解析时原样返回字符串而不是 '—'：保留原始信息比隐藏更有助于排查。
 */
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

/** 模型家族 → 中文分组名（对应能力目录 15 个目标模型的分组展示）。 */
export const MODEL_FAMILY_LABELS = {
  traditional_ml: '传统机器学习',
  basic_deep: '基础深度学习',
  convolutional: '卷积网络',
  long_range: '长程建模',
  two_dimensional_mapping: '二维映射网络',
};

/** CV 策略标识：只有这一种策略走交叉验证口径规则。 */
const CV_STRATEGY = 'leave_one_sample_id_cv';

/** 判断 run-result-v1 结果对象是否采用交叉验证口径。 */
export function isCvResult(result) {
  return result?.evaluation?.strategy === CV_STRATEGY;
}

/**
 * CV 口径守卫：解析某个 split（train/valid/test）的标量指标及其聚合口径。
 * - CV 的 test 只允许 pooled_oof；train/valid 标量只允许 fold mean（fold_std 为审计值）。
 * - 非 CV（holdout）全部为 direct。
 *
 * 为什么 test 分支要在 entry.values 缺失时回退 result.metrics.pooled_oof：
 * 兼容旧版结果文件——早期契约把 pooled_oof 指标放在 metrics.pooled_oof
 * 顶层而不是 splits.test.values 里，两处任一存在都视为合法。
 *
 * 返回 { split, aggregation, values, foldStd, note } 或 null（split 不存在时）。
 * note 是给用户看的口径说明，防止把 fold mean 误当 CV 主指标。
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
 * 规则（与 run-result-v1 契约一致）：
 * - metrics.splits.test 必须是 pooled_oof（主指标 = 全部折 out-of-fold 预测合并）；
 * - metrics.splits.train/valid 标量必须是 fold_mean（仅审计）；
 * - analysis.splits.test 图表必须是 pooled_oof；
 * - analysis.splits.train/valid 图表必须是 pooled_cross_fold（跨折合并，样本可重复）。
 * 校验宽松点：aggregation 字段缺失时不视为违规（交给后端/其他守卫兜底），
 * 只拦"明确写错口径"的情况。
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
// 三种"服务器绝对路径"形态：Windows 盘符、UNC 网络路径、POSIX 常见根目录。
// POSIX 用目录白名单而不是简单 /^\\//，避免把前端相对路径 '/static/...' 误判为泄漏。
const WINDOWS_ABS_PATH = /^[A-Za-z]:[\\/]/;
const UNC_PATH = /^\\\\/;
const POSIX_ABS_PATH = /^\/(home|Users|var|opt|srv|tmp|mnt|data|root|etc|usr)\//;
// 高危键名：server 模式下这些键不应出现在给浏览器的响应里（身份与路径只由服务端注入）
const SERVER_PATH_KEYS = new Set(['path', 'data_path', 'test_data_path', 'run_dir', 'output_path', 'common_time_path', 'dataset_path']);

/**
 * 粗判一个字符串是否像服务器绝对路径。
 * 只接受非空字符串；命中三种形态之一即返回 true。
 * 这是启发式检查（宁宽勿漏），用于安全提示，不作为精确解析。
 */
export function looksLikeServerPath(value) {
  if (typeof value !== 'string' || !value) return false;
  return WINDOWS_ABS_PATH.test(value) || UNC_PATH.test(value) || POSIX_ABS_PATH.test(value);
}

/**
 * 递归检查响应中是否夹带服务器路径字段。
 * 两条判定路径：
 * 1. 键名命中 SERVER_PATH_KEYS 或以 '_path' 结尾 → 直接记违规（不问值）；
 * 2. 叶子字符串值命中 looksLikeServerPath → 记违规。
 * options.ignoreKeys：允许出现的键名（例如 download_url，其查询串由后端生成、作为整体 URL 使用）。
 * 返回违规键路径数组（如 "config.data_path"、"files[2].path"），空数组表示干净。
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
