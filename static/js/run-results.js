/**
 * run-results.js —— 建模结果页与 Hash 路由的核心模块（旧版主前端 `static/index.html`）。
 *
 * 系统位置与职责：
 *   1. Hash 路由：维护 `#/results?run_id=...`、`#/runs` 等原生 Hash 路由的解析与跳转，
 *      导航顺序固定为"AI 建模 → 建模结果 → 训练记录"。
 *   2. Run 状态轮询：`RunPollController` 提供带指数退避、页面隐藏降频、代际（generation）
 *      防串扰的通用轮询器，分别用于训练进度与结果加载。
 *   3. 结果渲染：以 `run-result-v1` 契约为准渲染概览、核心指标、分区分指标、混淆矩阵、
 *      各类别指标、预测分布、训练曲线（canvas 手绘）、参数审计、单样品可解释性与产物下载。
 *      对旧后端提供 legacy 结果投影的兼容归一化。
 *   4. 训练完成引导：新建 Run 成功后弹出 3 秒倒计时对话框，自动跳转到专属结果页。
 *   5. server 模式认证：访问令牌的输入/验证/退出，令牌只存于 sessionStorage（由
 *      `SpecAutoAIAuth` 管理），本模块只负责对话框与状态 UI。
 *
 * 协作模块：
 *   - `api-client.js`：`request()` / `downloadFile()`，统一鉴权与错误处理。
 *   - `ui-utils.js`：`element()`（安全的 DOM 构建，全部走 textContent，杜绝 XSS）、
 *     `formatMetric` / `formatTime` / `replaceChildren`。
 *   - 全局桥接：`window.SpecAutoAIModelMeta`（模型目录元数据）、`window.SpecAutoAIHealth`
 *     （后端能力/契约探测）、`window.SpecAutoAIShowView`（视图切换）、`window.SpecAutoAIAuth`。
 *
 * 关键业务约束（按代码实际行为）：
 *   - 结果页以 `GET /api/training/runs/{run_id}/result` 返回的 `run-result-v1` 为准；
 *     仅当后端不支持该契约（404 且 health 未声明）时才回退旧版 Run 详情接口。
 *   - CV（leave_one_sample_id_cv）场景的主指标口径必须区分 pooled OOF、fold mean、
 *     fold std，页面上分开标注，不混用。
 *   - artifact 下载白名单：`model.pkl`、`model.pt`、`.joblib` 及旧版全局
 *     `feature_importance.json/csv` 不在结果页公开；下载 URL 必须同源且以
 *     `/api/training/runs/` 开头（防止被投毒成跨域/任意地址）。
 *   - 所有文本节点均通过 `element()` 的 text 属性写入，用户/后端数据不进入 HTML。
 */
import { downloadFile, request } from './api-client.js';
import { element, formatMetric, formatTime, replaceChildren } from './ui-utils.js';

// ---- 路由与状态常量 ----

/** 训练完成后自动跳转结果页前的倒计时毫秒数（同时是对话框倒计时的总时长）。 */
export const AUTO_REDIRECT_DELAY_MS = 3000;

/**
 * 仍在"活跃"中的 Run 状态集合：轮询遇到这些状态会继续，否则停止。
 * queued/pending 是排队中，running 是训练中。
 */
const ACTIVE_STATES = new Set(['queued', 'pending', 'running']);

/**
 * 终态（或准终态）集合：到达这些状态后轮询停止。
 * 除了常规 succeeded/failed/cancelled，还包含结果侧状态 ready（结果可用）、
 * partial（部分产物缺失）、missing_manifest / corrupt_manifest（清单缺失/损坏），
 * 因为结果页轮询读取的是 result 接口，状态字段可能来自 run.result_state。
 */
const TERMINAL_STATES = new Set(['succeeded', 'success', 'failed', 'cancelled', 'paused', 'ready', 'partial', 'missing_manifest', 'corrupt_manifest']);

/**
 * 六个传统机器学习模型 ID。它们不产生逐 Epoch 训练曲线（无 history），
 * 渲染"训练过程"卡片时据此直接跳过。
 */
const TRADITIONAL_MODEL_IDS = new Set(['pls_da', 'pca_lda', 'logistic_regression', 'svm', 'random_forest', 'xgboost']);

/**
 * URL path → 内部视图名的别名表。
 * 兼容多种历史/口语写法：如 `#/hplc` 与 `#/chromatography` 都指向色谱视图，
 * `#/help` 与 `#/manual` 都指向手册视图。不在表中的 path 回退为 'raman'。
 */
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

/**
 * 内部视图名 → 规范 URL path 的反查表（buildViewHash 使用）。
 * 注意方向与 VIEW_ALIASES 相反：内部固定用 'chromatography'，对外 URL 用 'hplc'；
 * 内部 'manual' 对外用 'help'。这是为了对外链接稳定、内部命名统一。
 */
const VIEW_PATHS = {
  raman: 'raman',
  chromatography: 'hplc',
  modeling: 'modeling',
  results: 'results',
  runs: 'runs',
  manual: 'help',
};

/**
 * 解析当前 location.hash 为结构化路由。
 *
 * 输入形如 `#/results?run_id=abc` 或 `#/hplc`；解析步骤：
 *   1. 去掉开头的 `#` 与 `/`；
 *   2. 按第一个 `?` 拆出 path 与 query；
 *   3. path 小写后经 VIEW_ALIASES 归一化为内部视图名，未知 path 回退 'raman'；
 *   4. query 用 URLSearchParams 解析，只提取 `run_id`。
 *
 * @param {string} [hash=''] 原始 hash 字符串。
 * @returns {{view: string, runId: string|null}} 视图名与可选的 Run ID。
 */
export function parseHash(hash = '') {
  const raw = String(hash || '').replace(/^#/, '').replace(/^\//, '');
  const [rawPath = '', rawQuery = ''] = raw.split('?', 2);
  const view = VIEW_ALIASES[rawPath.toLowerCase()] || 'raman';
  const params = new URLSearchParams(rawQuery);
  return { view, runId: params.get('run_id') || null };
}

/**
 * 构造某个视图的规范 hash，如 `#/hplc`、`#/modeling`。
 * 未知视图回退 'raman'（首页预处理视图）。
 * @param {string} view 内部视图名。
 * @returns {string} 以 `#/` 开头的 hash。
 */
export function buildViewHash(view) {
  return `#/${VIEW_PATHS[view] || 'raman'}`;
}

/**
 * 构造某个 Run 的专属结果页 hash：`#/results?run_id=<id>`。
 * runId 先 trim 再 encodeURIComponent；为空时返回不带参数的 `#/results`（结果落地页）。
 * 专属 URL 是项目契约之一：结果页可刷新、可复制链接直达。
 */
export function buildResultHash(runId) {
  const normalized = String(runId || '').trim();
  return normalized ? `#/results?run_id=${encodeURIComponent(normalized)}` : '#/results';
}

/**
 * 判断训练完成后是否应该自动跳转结果页。
 *
 * 三个条件缺一不可：
 *   1. runId 非空（跳转需要目标）；
 *   2. isNewRun 为真——只有"本会话刚由用户发起"的 Run 才自动跳，避免用户手动
 *      浏览历史 Run 时被突然跳转打断；
 *   3. 状态为成功类终态（succeeded/success/ready/partial）。partial（部分产物缺失）
 *      也跳转，因为核心结果已可看，缺失项会在页面上单独提示。
 */
export function shouldAutoRedirect({ runId, state, isNewRun }) {
  return Boolean(String(runId || '').trim())
    && Boolean(isNewRun)
    && ['succeeded', 'success', 'ready', 'partial'].includes(String(state || '').toLowerCase());
}

/**
 * RunPollController —— 通用单目标轮询控制器。
 *
 * 解决的问题：页面需要持续刷新"当前关注的 Run"状态，但要同时满足：
 *   - 同一时刻只跟踪一个 runId（watch 新目标会先停掉旧的）；
 *   - 失败时指数退避，避免后端抖动时打爆接口；
 *   - 页面切到后台（document.hidden）时降频，减少无意义请求；
 *   - 旧轮询的迟到响应不能污染新目标（generation 代际校验 + AbortController 取消在途请求）。
 *
 * 设计要点：
 *   - `generation` 是单调递增的代际号：每次 stop()/watch() 都会使其 +1，
 *     所有异步回调拿到响应后先比较自己启动时捕获的代际号，不等则直接丢弃。
 *     这比只取消请求更稳——即使请求已完成、回调已在微任务队列里，也能被拦截。
 *   - 不递归 setInterval，而是"上一次结束后再 schedule"，天然避免请求重叠。
 */
export class RunPollController {
  /**
   * @param {(runId: string, signal: AbortSignal) => Promise<any>} fetcher
   *   实际取数函数；必须尊重 signal（传给 request()），否则 abort 不生效。
   * @param {object} [options]
   * @param {number} [options.baseDelay=1200] 正常轮询间隔（毫秒）。
   * @param {number} [options.maxDelay=10000] 失败退避的上限（毫秒）。
   */
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

  /**
   * 停止轮询：代际 +1 使所有在途回调失效，清除定时器并 abort 在途请求。
   * 幂等，可反复调用。
   */
  stop() {
    this.generation += 1;
    if (this.timer != null) clearTimeout(this.timer);
    this.timer = null;
    this.controller?.abort();
    this.controller = null;
    this.runId = null;
  }

  /**
   * 开始跟踪一个 runId。
   *
   * @param {string} runId 目标 Run ID。
   * @param {object} [handlers]
   * @param {(payload, runId) => void|Promise} [handlers.onData] 每次成功取数的回调。
   * @param {(error, runId, failures) => void|Promise} [handlers.onError] 每次失败回调，
   *   failures 是连续失败次数（从 1 开始），调用方据此区分"首次失败"与"重试中"。
   * @param {(payload) => boolean} [handlers.isTerminal] 返回 false 表示继续轮询；
   *   默认（未提供或返回非 false）视为到达终态并停止。注意语义是"!== false 即停"。
   * @param {(error) => boolean} [handlers.isTerminalError] 返回 true 的错误不再重试
   *   （如 401/403/404 这类重试无意义的错误）。
   */
  watch(runId, handlers = {}) {
    this.stop();
    this.runId = String(runId || '');
    this.failures = 0;
    const generation = this.generation;
    const schedule = (delay) => {
      // 调度前先校验代际与目标：若在等待期间用户切换了 Run 或调用了 stop()，
      // 直接放弃本次调度，不再发起请求。
      if (generation !== this.generation || !this.runId) return;
      // 页面隐藏（切后台/最小化）时把间隔抬到至少 5 秒：轮询结果用户看不见，
      // 降频既省请求也省后端压力；回到前台后下一次 tick 会恢复 baseDelay。
      const visibilityDelay = typeof document !== 'undefined' && document.hidden ? Math.max(delay, 5000) : delay;
      this.timer = setTimeout(tick, visibilityDelay);
    };
    const tick = async () => {
      if (generation !== this.generation || !this.runId) return;
      // 每轮 tick 新建 AbortController：stop() 只 abort 当前在途请求，
      // 不会影响后续轮次。
      this.controller = new AbortController();
      try {
        const payload = await this.fetcher(this.runId, this.controller.signal);
        // 响应到达时目标可能已切换——丢弃迟到数据，不触发回调。
        if (generation !== this.generation) return;
        this.failures = 0;
        await handlers.onData?.(payload, this.runId);
        // 注意反向语义：isTerminal 未提供或返回 true 都停止；
        // 只有显式返回 false 才继续轮询（调用方传的是"是否仍活跃"的判断）。
        if (handlers.isTerminal?.(payload) !== false) {
          this.stop();
          return;
        }
        schedule(this.baseDelay);
      } catch (error) {
        // AbortError 是 stop()/watch() 主动取消导致的，不算失败，静默忽略。
        if (generation !== this.generation || error?.name === 'AbortError') return;
        this.failures += 1;
        await handlers.onError?.(error, this.runId, this.failures);
        if (handlers.isTerminalError?.(error)) {
          this.stop();
          return;
        }
        // 指数退避：base * 2^failures，指数封顶 4（即最多 16 倍），再被 maxDelay 截断。
        // 封顶 4 是为了让退避增长快但不过度——1200ms 起步最多到 ~19s 前就被 10s 上限截住。
        schedule(Math.min(this.baseDelay * (2 ** Math.min(this.failures, 4)), this.maxDelay));
      } finally {
        if (generation === this.generation) this.controller = null;
      }
    };
    schedule(0);
  }
}

/** document.getElementById 的短别名，全文大量使用。 */
function byId(id) {
  return document.getElementById(id);
}

/**
 * 从后端各种可能的字段位置提取统一的 Run 状态字符串（小写）。
 *
 * 为什么需要这么多 fallback：新旧接口混用时，状态可能在 run.result_state
 * （run-result-v1）、payload.result_state、run.state、payload.state、
 * run.status、payload.status 中任意一处出现。全部缺失时按 'pending' 处理，
 * 让轮询继续而不是误判为终态。
 */
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

/** 判断 payload 代表的 Run 是否仍在活跃（排队/训练中），用于决定是否继续轮询。 */
function isActivePayload(payload) {
  return ACTIVE_STATES.has(canonicalState(payload));
}

/**
 * 把旧版 Run 详情接口（GET /api/training/runs/{id}）的扁平字段投影成
 * 与 `run-result-v1` 结构对齐的对象，使后续渲染代码可以只写一套逻辑。
 *
 * 设计意图：这不是"伪造"新契约——schema_version 明确标为 'legacy-run-projection'，
 * 并在 warnings 中向用户说明部分信息可能无法恢复。各字段尽量做合理映射：
 *   - 旧 status 'success' → run.state 'succeeded'，result_state 'ready'；
 *   - 评估口径：leave_one_sample_id_cv → pooled_oof，其余 → direct；
 *   - 指标：旧结构没有分 split 的层次，test 直接取 metrics.test 或整个 metrics；
 *   - confusion_matrix / classification_report 提升到 analysis 层；
 *   - history 只保留"是否有数据 + 行"，供训练曲线卡片判断。
 *
 * @param {object} payload 旧接口返回的扁平 Run 对象。
 * @returns {object} 形似 run-result-v1 的投影对象。
 */
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

/**
 * 归一化结果 payload：是 run-result-v1 原样返回，否则走 legacy 投影。
 * 这是结果渲染层的统一入口，渲染代码永远面对同一种结构。
 */
export function normalizeResult(payload) {
  if (payload?.schema_version === 'run-result-v1' && payload?.run) return payload;
  return normalizedLegacyResult(payload || {});
}

/**
 * 后端 health 是否声明支持 run-result-v1 契约。
 * 用于决定 404 时能否回退旧接口：新后端支持契约却 404，说明 Run 真不存在，
 * 不应回退；只有旧后端（未声明契约）的 404 才可能是"没有 result 端点"。
 */
export function backendSupportsResultContract(health) {
  return health?.contracts?.run_result === 'run-result-v1';
}

/**
 * 是否应对 404 使用旧版接口回退：错误是 404 且已拿到 health 且后端不支持 v1 契约。
 */
export function shouldUseLegacyResultFallback(error, health) {
  return error?.status === 404
    && health != null
    && !backendSupportsResultContract(health);
}

/**
 * 获取后端 health 能力（带 window 级缓存）。
 * @param {AbortSignal} signal 取消信号。
 * @param {boolean} [forceRefresh=false] 为 true 时绕过缓存重新请求
 *   （fetchResult 在 404 后需要最新的契约声明，强制刷新）。
 * 请求失败返回 null 而不是抛错——health 只是能力探测，拿不到不应阻断主流程。
 */
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

/**
 * 读取 Run 结果：优先新契约 `GET .../result`；404 时探测 health，
 * 确认后端不支持 v1 契约后回退旧版 Run 详情接口，再经 normalizeResult 对齐结构。
 * 其他错误（401/403/500…）原样上抛，由轮询层按 isTerminalError 处理。
 */
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

// ---- 模块级共享状态 ----

// 训练进度轮询器（建模页"正在训练"用）与结果轮询器（结果页用），
// 各自独立，互不影响。
const trainingPoller = new RunPollController(
  (runId, signal) => request(`/api/training/runs/${encodeURIComponent(runId)}`, { signal }),
);
const resultPoller = new RunPollController(fetchResult);
// 本会话中"用户刚发起"的 Run ID 集合：只有这些 Run 完成后才允许自动跳转
// （见 shouldAutoRedirect / markNewRun）。
const newRunIds = new Set();
let currentResultRunId = null;
// 渲染代际号：结果页存在多个异步渲染（可解释性文件下载后再绘制），
// 每次重新加载/渲染 +1，迟到的异步结果比对代际号后丢弃，避免旧 Run 的内容
// 覆盖新 Run 的页面。
let resultRenderGeneration = 0;
// 最近一次完整渲染的归一化结果，供模型目录/health 就绪后局部重绘概览。
let lastRenderedResult = null;
// 自动跳转相关：倒计时 setTimeout、秒数刷新 setInterval、当前跳转目标。
let redirectTimer = null;
let redirectTicker = null;
let redirectRunId = null;
// 对话框焦点管理：打开前记录焦点元素，关闭后归还（无障碍要求）。
let authReturnFocus = null;
let trainingResultReturnFocus = null;
let trainingDialogFocusTimer = null;
// canvas 图表在窗口 resize 时需要重绘；保存 handler 引用以便替换内容前解绑，
// 防止监听器泄漏和闭包引用旧 DOM。
let sampleExplanationResizeHandler = null;
let historyResizeHandler = null;
// 单样品可解释性 JSON 的下载缓存（download_url → Promise），上限 4 条 LRU 式淘汰；
// 缓存 Promise 而非结果，可同时去重并发请求。
const explainabilityPayloadCache = new Map();

/**
 * 状态 → [中文标签, CSS 状态类] 映射。第二个元素决定状态徽章配色
 * （pending/running/success/warning/failed/cancelled）。未知状态返回"状态未知"。
 */
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
    cancelled: ['STOP', 'cancelled'],
    paused: ['STOP', 'cancelled'],
  };
  return states[state] || ['状态未知', ''];
}

/** 从归一化结果对象提取当前状态（优先 result_state，兼容旧字段）。 */
function resultState(result) {
  return String(result?.run?.result_state || result?.run?.state || result?.run?.status || 'pending').toLowerCase();
}

/**
 * 评估策略 → 中文说明。三种口径对应后端契约：
 * stratified_holdout（分层留出）、leave_one_sample_id_cv（按 Sample_ID 留一 CV）、
 * external_test_holdout（独立测试集）。未知值原样显示，避免吞掉新策略名。
 */
function evaluationLabel(strategy) {
  const labels = {
    stratified_holdout: '分层留出（Train / Valid / Test）',
    leave_one_sample_id_cv: '按 Sample_ID 留一交叉验证',
    external_test_holdout: '独立测试集最终评估',
  };
  return labels[strategy] || strategy || '-';
}

/**
 * 指标聚合口径 → 中文说明。
 * CV 场景必须明确区分：pooled_oof（所有折 OOF 预测合并计算，是主指标）、
 * fold_mean / fold_std（各折指标的均值/标准差，仅供参考）；direct 是非 CV 直接计算。
 * 项目契约禁止把 fold_mean 当作主指标展示，本函数是标注层。
 */
function aggregationLabel(value) {
  const labels = {
    pooled_oof: '合并交叉验证预测',
    fold_mean: '各折均值',
    fold_std: '各折标准差',
    direct: '直接计算',
  };
  return labels[value] || value || '-';
}

/** 空值（null/undefined/空串）显示为 '-'，其余转字符串；表格单元格通用。 */
function valueOrDash(value) {
  return value == null || value === '' ? '-' : String(value);
}

/**
 * 训练耗时的人性化文本。
 * 优先用后端给的 duration_seconds；缺失时用 started_at/finished_at 推算。
 * < 60 秒显示"X.X 秒"，否则显示"X.X 分钟"；两端时间缺失或非法返回 '-'。
 */
function durationText(run) {
  const supplied = Number(run?.duration_seconds);
  if (Number.isFinite(supplied)) return supplied < 60 ? `${supplied.toFixed(1)} 秒` : `${(supplied / 60).toFixed(1)} 分钟`;
  const start = new Date(run?.started_at).getTime();
  const end = new Date(run?.finished_at).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return '-';
  const seconds = Math.max(0, (end - start) / 1000);
  return seconds < 60 ? `${seconds.toFixed(1)} 秒` : `${(seconds / 60).toFixed(1)} 分钟`;
}

/**
 * 训练时间的展示文本：有 started_at 显示开始时间；尚未开始则显示创建时间并
 * 加"（任务创建）"后缀说明它还不是真正的训练开始时刻；两者都没有返回 '—'。
 */
export function trainingTimeText(run) {
  if (run?.started_at) return formatTime(run.started_at);
  if (run?.created_at) return `${formatTime(run.created_at)}（任务创建）`;
  return '—';
}

/**
 * 构建一条通知条（notice）DOM。
 * @param {'error'|'warning'|'empty'|'loading'|'success'|'status'} kind 决定配色与
 *   ARIA role：错误用 role="alert"（屏幕阅读器立即播报），其余 role="status"。
 * @param {string} title 加粗标题。
 * @param {string} message 正文。
 * @param {HTMLElement[]} [actions] 可选操作按钮/链接。
 */
function notice(kind, title, message, actions = []) {
  const content = element('div', {}, element('strong', { text: title }), element('span', { text: message }));
  const actionWrap = actions.length ? element('div', { className: 'notice-actions' }, ...actions) : null;
  return element('div', { className: `notice notice-${kind}`, role: kind === 'error' ? 'alert' : 'status' }, content, actionWrap);
}

/**
 * 清空结果页所有区块并解绑 resize 监听器。
 * 每次加载新 Run / 重新渲染前调用，防止旧 Run 的 DOM 与监听器残留
 * （resize handler 闭包里引用着旧 canvas，不解绑会泄漏并可能重绘旧图）。
 */
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

/**
 * 复制到剪贴板的小按钮：成功后短暂显示"已复制"（1.5 秒）再还原；
 * clipboard API 失败（如非安全上下文）显示"复制失败"，不抛错打断页面。
 */
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

/**
 * 概览网格中的一项（dt/dd 结构）：值过长时 CSS 截断，完整值放 title 悬停可见；
 * extra 用于追加附加元素（如 Run ID 旁的复制按钮）。
 */
function overviewItem(label, value, extra = null) {
  return element('div', { className: 'overview-item' },
    element('dt', { text: label }),
    element('dd', {}, element('span', { className: 'truncate-text', text: valueOrDash(value), title: valueOrDash(value) }), extra),
  );
}

/**
 * 渲染结果页顶部"概览"区：模型名 + 状态徽章 + 关键元数据网格 + 后端 warnings。
 *
 * 要点：
 *   - 模型显示名优先用结果里的 display_name，其次查全局模型目录
 *     `window.SpecAutoAIModelMeta`（目录就绪晚于渲染时，会靠
 *     'specautoai:model-catalog-ready' 事件触发本函数局部重绘）；
 *   - 数据集指纹只显示 SHA-256 前 16 位，足够人工比对又不刷屏；
 *   - warnings 逐条渲染为 warning 通知（如 legacy 投影的降级提示）。
 */
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
    overviewItem('训练时间', trainingTimeText(run)),
    overviewItem('结束时间', formatTime(run.finished_at)),
    overviewItem('训练耗时', durationText(run)),
    overviewItem('数据量', dataset.curve_count),
    overviewItem('样本数', dataset.sample_id_count),
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

/**
 * 核心指标卡片定义：[字段名, 显示名, 解释文案]。
 * 分类任务的五个主指标；balanced_accuracy 在类别不均衡时比 accuracy 更有参考意义，
 * macro 系列是"先按类算再等权平均"，与小类表现更相关。
 */
const METRIC_DEFINITIONS = [
  ['accuracy', 'Accuracy', '总体预测正确比例。'],
  ['balanced_accuracy', 'Balanced Accuracy', '各类别召回率等权平均，类别不均衡时更有参考意义。'],
  ['macro_precision', 'Macro Precision', '先分别计算每类 Precision，再对类别等权平均。'],
  ['macro_recall', 'Macro Recall', '先分别计算每类 Recall，再对类别等权平均。'],
  ['macro_f1', 'Macro F1', '先分别计算每类 F1，再对类别等权平均。'],
];

/**
 * 取指标值，兼容旧后端的简写：macro_f1 缺失时尝试 f1，macro_recall → recall ……
 * 因为旧接口可能只存不带 macro_ 前缀的键。取不到返回 undefined（调用方跳过该卡片）。
 */
function metricValue(metrics, key) {
  if (!metrics || typeof metrics !== 'object') return undefined;
  if (metrics[key] != null) return metrics[key];
  if (key.startsWith('macro_')) return metrics[key.replace('macro_', '')];
  return undefined;
}

/**
 * 渲染"核心指标"卡片区。
 * 只渲染实际存在且为有限数字的指标（避免训练未完成/部分产物缺失时出现 NaN 卡片）；
 * 一个都没有时显示空态提示，而不是留白。标题注明主指标口径（pooled_oof / direct…），
 * 这是 CV 场景的契约要求：主指标必须标注聚合方式。
 */
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

/**
 * 渲染"训练、验证与测试集"分指标表格（train/valid/test 三行）。
 *
 * 聚合口径的推断优先级：split.aggregation 显式值 → test 行用 evaluation.primary_aggregation
 * → CV 策略默认 fold_mean → 兜底 direct。契约要求"合并 OOF 与折均值分开标注"，
 * 所以每行都带聚合口径列，防止读者把 fold_mean 误当 pooled OOF 主指标。
 */
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
      element('p', { text: '全部交叉验证折合并的测试指标与折均值分开标注，避免混用。' }),
    )),
    element('div', { className: 'table-scroll' }, table),
  );
}

/**
 * 提取各 split 的分析数据（混淆矩阵、分类报告等）为统一数组。
 * 新契约：analysis.splits.{train,valid,test} 各自成对象；
 * 旧契约 fallback：analysis 顶层直接放 test 的字段，此时包成单个 test 条目。
 * @returns {{name: string, label: string, analysis: object}[]}
 */
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

/**
 * 生成某个 split 分析卡片的聚合口径标注。
 * 推断链：splitAnalysis.aggregation → test 行取 evaluation.primary_aggregation →
 * CV 策略兜底 fold_mean，否则 direct。
 * 特殊值 pooled_cross_fold 单独措辞：跨折合并预测里同一样本可能出现在多折，
 * 计数会重复，必须提醒读者不能与单折数字直接比较。
 */
function splitAggregationLabel(splitAnalysis, result, splitName) {
  const aggregation = splitAnalysis?.aggregation
    || (splitName === 'test' ? result?.evaluation?.primary_aggregation : null)
    || (result?.evaluation?.strategy === 'leave_one_sample_id_cv' ? 'fold_mean' : 'direct');
  if (aggregation === 'pooled_cross_fold') return '跨折合并预测（样本可能重复）';
  return aggregationLabel(aggregation);
}

/**
 * 解析混淆矩阵数据，兼容两种后端形状：
 *   - 直接是二维数组；
 *   - { matrix, labels } 对象。
 * 类别名优先级：source.labels → result.label_names → 从 classification_report 的键
 * 推导（剔除 accuracy / macro avg 等汇总行）。矩阵非法时返回空数组，调用方跳过渲染。
 */
function confusionData(splitAnalysis, result) {
  const source = splitAnalysis?.confusion_matrix;
  const matrix = Array.isArray(source) ? source : source?.matrix;
  const report = splitAnalysis?.classification_report || {};
  const labels = source?.labels || result.label_names || Object.keys(report).filter((key) => (
    !['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(key)
  ));
  return { matrix: Array.isArray(matrix) ? matrix : [], labels };
}

/**
 * 渲染单个 split 的混淆矩阵卡片（行=真实类别，列=预测类别）。
 * 标签数量与矩阵维度不一致时按行索引补齐 `class_N`，保证表格始终对齐；
 * 矩阵为空返回 null（调用方 filter 掉）。
 */
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

/**
 * 从 classification_report 中提取逐类别行：值为对象且键不是
 * 'macro avg'/'weighted avg' 等汇总行。注意 'accuracy' 是数字不是对象，
 * 被 typeof 检查自然过滤，无需额外排除。
 */
function classRows(report) {
  if (!report || typeof report !== 'object') return [];
  return Object.entries(report).filter(([label, values]) => (
    values && typeof values === 'object' && !['macro avg', 'weighted avg', 'micro avg', 'samples avg'].includes(label)
  ));
}

/**
 * 渲染单个 split 的逐类别 Precision/Recall/F1/support 表格。
 * F1 键兼容 sklearn 风格 'f1-score' 与下划线风格 'f1_score'。
 */
function renderSplitClassMetrics(splitName, splitAnalysis, result) {
  const rows = classRows(splitAnalysis?.classification_report);
  if (!rows.length) return null;
  const splitLabel = { train: 'Train', valid: 'Valid', test: 'Test' }[splitName] || splitName;
  const table = element('table', {},
    element('thead', {}, element('tr', {}, ...['类别', 'Precision', 'Recall', 'F1', '数据量'].map((label) => element('th', { text: label })))),
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

/**
 * 提取"预测结果分布"的行数据（每类真实数量 vs 预测数量），兼容三种来源：
 *   1. 后端直接给数组；
 *   2. 后端给对象（{labels, true_counts, predicted_counts} 或 {label: counts}）；
 *   3. 都没有时**由混淆矩阵推导**：行和=真实数，列和=预测数。
 * 第 3 条是"只展示真实生成或可直接推导的分析"原则的体现——分布图数据来自
 * 已展示的混淆矩阵，不引入新口径。
 */
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

/**
 * 计算整数计数的"漂亮"坐标轴刻度（1-2-5 nice number 算法）。
 *
 * 原理：rawStep = max/(ticks-1)，取其数量级 magnitude，把余数归到 1/2/5/10
 * 中最接近的"好看"步长；再向上取整出轴顶 top，从 top 递减到 0 生成刻度。
 * 计数是整数，所以步长至少为 1，刻度都是整数，避免出现 3.7 这类轴标签。
 *
 * @param {number} maxValue 数据最大值。
 * @param {number} [desiredTicks=5] 期望刻度个数（实际可能 ±1）。
 * @returns {number[]} 从大到小的刻度数组；maxValue 为 0 时返回 [0]。
 */
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

/**
 * 渲染单个 split 的"真实 vs 预测数量"竖向分组柱状图（纯 CSS 高度实现，不用 canvas）。
 *
 * 无障碍设计：图形容器 role="img" + aria-label 描述内容，同时附一张
 * visually-hidden 的数据表（summaryTable），屏幕阅读器可直接读到精确数字。
 * 柱高 = 值 / scaleTop * 100%，scaleTop 来自 distributionScaleTicks 的轴顶，
 * 与左侧刻度轴严格一致。
 */
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
    element('thead', {}, element('tr', {}, element('th', { text: '类别' }), element('th', { text: '真实数据量' }), element('th', { text: '预测数据量' }))),
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
      'aria-label': `${splitLabel} 各类别真实数据量与预测数据量竖向柱状图`,
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

/**
 * 参数审计表格：把 best_params 对象逐行渲染；嵌套对象 JSON.stringify 展示。
 * 无参数时显示空态而非空白区域。
 */
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

/**
 * 渲染"训练与参数审计"折叠卡片（<details>，默认收起，避免占据主视觉）。
 *
 * 内容分两块（按模型类型二选一或同时存在）：
 *   - traditional：传统模型的超参搜索结果——最终 best_params + 选择指标得分，
 *     CV 时还有 best_params_by_fold 逐折表（每折独立搜索，参数可能不同）；
 *   - deep：深度模型训练摘要——最佳验证 loss、实际 epoch 数（可能早停小于上限）、
 *     最低学习率（学习率衰减的下限）。
 * 两者都没有（老 Run 未生成审计数据）返回 null。
 */
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

/**
 * 提取逐 Epoch 训练历史行，兼容两种形状：直接是数组，或 { rows, reason, truncated }。
 */
function historyRows(result) {
  const history = result.analysis?.history;
  if (Array.isArray(history)) return history;
  return Array.isArray(history?.rows) ? history.rows : [];
}

/**
 * 计算图表 Y 轴（或值域）[min, max]，带边距与退化处理。
 *
 * 逻辑：
 *   1. options 显式给了 min 和 max（如验证指标固定 [0,1]）直接使用；
 *   2. 无有效数据回退 [0, 1]；
 *   3. includeZero 时把 0 纳入范围（柱状/计数图常用）；
 *   4. min === max（曲线是平的）时向两侧各扩 max(|v|*10%, 0.1)，避免除零和
 *      无法读图的单点区间；
 *   5. 正常情况上下各加 8% padding，曲线不顶到边框。
 *
 * @param {number[]} values 数据值。
 * @param {{min?: number, max?: number, includeZero?: boolean}} [options]
 * @returns {[number, number]}
 */
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

/** 在 [min, max] 上均匀取 count 个刻度（含两端），至少 2 个。 */
function linearTicks(min, max, count = 5) {
  const safeCount = Math.max(2, Math.floor(count));
  return Array.from({ length: safeCount }, (_, index) => min + ((max - min) * index) / (safeCount - 1));
}

/**
 * 在 canvas 上手绘训练过程折线图（不引入图表库，保持前端零构建依赖）。
 *
 * 实现要点：
 *   - 按 devicePixelRatio 放大画布再 setTransform 缩回，保证高分屏下线条清晰；
 *   - X 轴取 row.epoch（缺省用行号 1 基）；只有 1 个 epoch 时人为扩 ±0.5 防止除零；
 *   - Y 域由 chartDomain 计算（可被 options.domain 固定，如指标图固定 [0,1]）；
 *   - 缺失值（NaN/undefined）直接跳过——只 moveTo 不 lineTo 的点不连线，
 *     因此某 epoch 缺 valid 数据时折线会自然断开而不是画错；
 *   - 图例手绘在画布顶部，每条序列固定 150px 宽排列。
 *
 * @param {HTMLCanvasElement} canvas 目标画布。
 * @param {object[]} rows 训练历史行。
 * @param {{key: string, label: string, color: string}[]} series 要画的序列。
 * @param {object} [options] xLabel/yLabel/decimals/domain 等。
 */
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

/**
 * 渲染"训练过程曲线"卡片（Loss + 验证指标两张 canvas 图）。
 *
 * 分支逻辑：
 *   - 传统 ML 模型（按 family 或模型 ID 判断）不产生逐 Epoch 历史，直接返回 null，
 *     不渲染任何卡片；
 *   - 深度模型但没有 history 行时显示空态卡片并给出后端返回的 reason；
 *   - CV 场景 history 行带 fold_index，出现多折时提供下拉切换；
 *     折数 ≤ 1 时不显示选择器。
 *
 * history.truncated 为真时提示"页面只加载了摘要"——完整记录在下载区，
 * 防止用户误以为曲线被截断是 bug。窗口 resize 通过重绑 handler 触发重绘
 * （canvas 位图不会随 CSS 自动重排）。
 */
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

/**
 * 渲染"图表与分析"整区：混淆矩阵组 + 逐类指标组 + 预测分布组 + 训练曲线 + 参数审计。
 * 每组按 train/valid/test 并列成卡片；某类卡片全空时整组不渲染（group 返回 null）。
 * 设计约束：只展示后端真实生成或可由现有指标直接推导的内容，不伪造 ROC 等不存在
 * 的产物（项目契约明确：当前没有正式 ROC-AUC/ROC/PR 产物）。
 */
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
    group('各类别指标', '分别查看每个数据分区中各类别的 Precision、Recall、F1 与数据量。', classCards),
    group('预测结果分布', '每个类别使用竖向分组柱比较真实数据量和预测数据量。', distributionCards),
    historyCard,
    auditCard,
  );
}

/**
 * 重要区间的展示标签：优先用真实 X 坐标范围（如拉曼位移/HPLC 分钟），
 * 没有 X 坐标时退化为特征索引范围，最后兜底区间自带 label/name。
 * X 坐标来自 wide-feature 宽表表头的真实坐标，比索引更有业务意义。
 */
function segmentLabel(segment) {
  if (segment?.start_x != null || segment?.end_x != null) return `${valueOrDash(segment.start_x)} – ${valueOrDash(segment.end_x)}`;
  if (segment?.start_index != null || segment?.end_index != null) return `特征 ${valueOrDash(segment.start_index)} – ${valueOrDash(segment.end_index)}`;
  return valueOrDash(segment?.label || segment?.name);
}

/**
 * 渲染"重要区间明细"横向条形列表（最多前 8 条）。
 * 条宽按 |importance| / 最大值 归一化为百分比，最小 2% 保证小值也可见；
 * 数值列优先显示未归一化的 importance（更贴近原始量级），条宽则用归一化值
 * （normalized_importance 优先）保证可比性。无数据时显示空态。
 */
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

/**
 * 在 artifacts 清单里定位"单样品可解释性 JSON"那个产物。
 *
 * 先按 explainability.samples.artifact/name 指定的文件名精确匹配；
 * 没有指定时按特征猜：category === 'explainability' 且 format json 且文件名含 'sample'。
 * 匹配前统一过滤掉模型二进制与 joblib——这些文件按契约不在结果页公开，
 * 即使后端清单里出现也不能被选中（如 DSCARNet 的 AggMap/PCA joblib）。
 */
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

/**
 * 获取单样品可解释性 payload：优先用结果内嵌的 samples；否则下载 artifact JSON。
 *
 * 防护：
 *   - artifact 必须 downloadable 且 download_url 通过 safeDownloadUrl 校验
 *     （同源 + /api/training/runs/ 前缀），否则返回 null；
 *   - 下载 Promise 缓存去重（同一 URL 并发只发一次），失败时从缓存删除以便重试；
 *   - 缓存上限 4 条，按插入序淘汰最旧条目（简单 LRU），避免大 JSON 撑爆内存；
 *   - generation 校验：下载完成后若页面已切到别的 Run（代际号变了），返回 null
 *     丢弃旧数据，不污染新页面。
 */
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

/**
 * 取样品曲线的 X 轴：优先样品自带的 sample_x_axis，其次 payload 级 x_axis，
 * 两者都必须与曲线等长才采用；否则退化为 0..N-1 索引轴。
 * X 轴对应宽表表头的真实坐标（拉曼 cm⁻¹ / HPLC 分钟），等长校验防止错位绘图。
 */
function sampleAxis(sample, payload) {
  const curve = Array.isArray(sample?.curve) ? sample.curve.map(Number) : [];
  const supplied = Array.isArray(sample?.sample_x_axis) && sample.sample_x_axis.length === curve.length
    ? sample.sample_x_axis
    : Array.isArray(payload?.x_axis) && payload.x_axis.length === curve.length
      ? payload.x_axis
      : curve.map((_, index) => index);
  return supplied.map(Number);
}

/**
 * 过滤出合法的重要性窗口：start/end 都是整数、0 ≤ start ≤ end < 曲线长度。
 * 后端数据异常（越界、非整数）时直接丢弃该窗口，不让坏数据破坏热力条渲染。
 */
function sampleWindows(sample, curveLength) {
  return (Array.isArray(sample?.windows) ? sample.windows : []).filter((windowItem) => {
    const start = Number(windowItem?.start_index);
    const end = Number(windowItem?.end_index);
    return Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end >= start && end < curveLength;
  });
}

/**
 * 校验单样品可解释性 payload 是否可绘制。
 *
 * 规则：
 *   - payload.status 存在且非 'ready'：解释产物还没生成完，返回 invalid + reason；
 *   - 逐样品过滤：曲线长度 > 1、X 轴与曲线等长、曲线与轴全部为有限数值；
 *   - 过滤后至少剩 1 个样品才 valid，否则返回 invalid 并给出可读原因。
 *
 * 设计意图：解释文件可能正在异步发布（后端 409 语义）或部分样品数据异常，
 * 前端宁可降级显示原因，也不画错图。
 *
 * @returns {{valid: boolean, reason: string|null, samples: object[]}}
 */
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

/**
 * 重要性值 [0,1] → 热力颜色：浅橙 rgb(255,247,237) 线性插值到深红 rgb(220,38,38)。
 * 输入先 clamp 到 [0,1]，非数字按 0 处理。色带选橙色→红色是为了与
 * primary_segment 高亮（红色系）在视觉上保持一致语义：越红越重要。
 */
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
      ? [progress.training_stage_label, progress.message, progress.phase, progress.fold_progress_text, progress.current_fold != null ? `第 ${progress.current_fold} 折` : null, progress.current_epoch != null ? `Epoch ${progress.current_epoch}` : progress.epoch != null ? `Epoch ${progress.epoch}` : null].filter(Boolean)
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
    replaceChildren(target, notice('warning', 'STOP', '本次训练已停止，模型、指标和中间产物均未保留。'));
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

/**
 * 渲染结果加载失败的错误态，按 HTTP 状态码定制文案：
 * 404 任务不存在/已删除（也可能因 server 模式下 owner 为空的历史 Run 不可见）；
 * 403 无权查看（Principal scope 隔离）；401 需要先在认证对话框输入令牌。
 */
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

/**
 * 渲染"结果落地页"（无 run_id 时）：Run ID 输入表单 + 最近 10 条任务列表。
 * 列表优先用 projection=summary 轻量投影（含可空 test_macro_f1），旧后端不支持
 * 该参数时回退完整列表接口。列表项兼容新旧两种 Run 形状（扁平字段或嵌套 run/model）。
 */
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
      const timeLabel = trainingTimeText(run);
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

/**
 * 加载并跟踪某个 Run 的结果（结果页主入口）。
 *
 * 空 runId → 落地页；否则清区、显示加载态，并用 resultPoller 轮询：
 *   - onData 直接走 renderResultPayload（活跃状态也会渲染，页面显示进度）；
 *   - onError：首次失败或 403/404 立即显示错误页；后续失败显示"第 N 次重试"，
 *     避免网络抖动时页面在错误/内容间闪烁；
 *   - isTerminal：归一化后不再是活跃状态即停轮询；
 *   - isTerminalError：401/403/404 重试无意义，直接停。
 */
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

/**
 * 供建模页使用的训练进度跟踪：用 trainingPoller 轮询 Run 详情接口，
 * 活跃状态持续回调 onData，到达终态自动停止。与结果页轮询相互独立。
 */
export function watchTrainingRun(runId, onData, onError) {
  trainingPoller.watch(runId, {
    onData,
    onError,
    isTerminal: (payload) => !isActivePayload(payload),
    isTerminalError: (error) => [401, 403, 404].includes(error?.status),
  });
}

/** 停止训练进度轮询（离开建模页或手动取消跟踪时调用）。 */
export function stopTrainingWatch() {
  trainingPoller.stop();
}

/**
 * 把一个 Run 标记为"本会话新发起"。这是自动跳转资格的唯一来源：
 * 只有 newRunIds 里的 Run 训练完成后才弹倒计时并自动跳结果页
 * （见 shouldAutoRedirect），防止浏览历史 Run 时被打断。
 * 同时取消上一个 Run 未完成的跳转（新训练取代旧引导）。
 */
export function markNewRun(runId) {
  const normalized = String(runId || '').trim();
  if (!normalized) return;
  newRunIds.add(normalized);
  cancelAutoRedirect();
}

/**
 * 取消自动跳转：清倒计时定时器、秒表刷新器、目标 runId，并关闭结果对话框。
 * @param {{restoreFocus?: boolean}} [options] restoreFocus 默认 true；
 *   跳转发生时传 false——焦点应落在新页面而不是还给旧按钮。
 */
export function cancelAutoRedirect(options = {}) {
  const restoreFocus = options?.restoreFocus !== false;
  if (redirectTimer != null) clearTimeout(redirectTimer);
  if (redirectTicker != null) clearInterval(redirectTicker);
  redirectTimer = null;
  redirectTicker = null;
  redirectRunId = null;
  hideTrainingResultDialog(restoreFocus);
}

/**
 * 关闭"训练完成"对话框并按需归还焦点到打开前的元素（无障碍焦点管理）。
 * 已隐藏时直接返回，幂等。
 */
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

/**
 * 弹出"训练完成"对话框（立即查看 / 留在本页 + 3 秒倒计时文案）。
 * 记录当前焦点元素以便关闭时归还；打开后异步把焦点移到"立即查看"按钮，
 * 配合 initialize 中的 Tab 焦点环与 Escape 关闭，满足模态对话框无障碍要求。
 * @returns {HTMLElement|null} 倒计时文案元素；DOM 缺失（旧页面结构）返回 null。
 */
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

/**
 * 训练完成后的引导入口（建模页轮询到终态时调用）。
 *
 * 守卫条件：shouldAutoRedirect（新 Run + 成功类状态）且当前停留在建模页，
 * 且同一 runId 未在引导中（重复调用幂等返回 true）。
 * 动作：清空建模页旧的结果区块、显示成功提示与"立即查看结果"链接、
 * 弹出倒计时对话框；3 秒后自动把 hash 切到专属结果页。
 * 倒计时用 250ms interval 刷新文案，实际跳转以 Date.now() 差值计算剩余秒数，
 * 不受 interval 漂移影响。
 *
 * @returns {boolean} 是否接管了完成引导（false 表示由调用方自行处理）。
 */
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

/**
 * 跳转到某 Run 的结果页。若当前 hash 已是目标（重复点击同一链接），
 * hashchange 不会触发，需手动 loadResult；否则改 hash 走正常路由流程。
 */
export function navigateToResult(runId) {
  const destination = buildResultHash(runId);
  if (window.location.hash === destination) loadResult(runId);
  else window.location.hash = destination;
}

/** 跳转到某个视图；hash 未变化时手动重放路由（同 navigateToResult 的幂等处理）。 */
export function navigateToView(view) {
  const destination = buildViewHash(view);
  if (window.location.hash === destination) applyCurrentRoute();
  else window.location.hash = destination;
}

/**
 * 应用当前 hash 路由（hashchange 与初始化时调用）。
 *
 * 流程：解析 hash → 通知全局视图切换器 → 结果视图则 loadResult（含轮询），
 * 其他视图停掉结果轮询；进入结果页时把焦点移到状态区（tabindex=-1 程序化聚焦，
 * 不进入 Tab 序），让键盘/读屏用户感知页面切换；离开建模页时取消未完成的
 * 自动跳转引导（用户已主动离开，不应再被倒计时拽走）。
 */
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

/**
 * 弹出服务器认证对话框（响应 api-client 抛出的 'specautoai:auth-required' 事件）。
 * 记录打开前焦点；清空上次输入并异步聚焦输入框。
 * 令牌本身由 window.SpecAutoAIAuth 管理，只存 sessionStorage，本函数不接触存储。
 */
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

/** 关闭认证对话框：清输入与错误文案，归还焦点到打开前的元素。 */
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

/**
 * 用当前已保存的令牌请求 /api/auth/session 验证有效性。
 * 成功后更新 UI 为"已认证"并把按钮切换为"退出服务器认证"；
 * 失败时错误上抛，由调用方清令牌并重新弹框。
 * authRetry: false 防止 401 时再次触发认证事件造成递归弹框。
 */
async function verifyAuthentication() {
  await request('/api/auth/session', { authRetry: false });
  byId('authStatus')?.classList.remove('hidden');
  byId('authStatus').textContent = '服务器已认证';
  if (byId('authButton')) byId('authButton').textContent = '退出服务器认证';
  hideAuthDialog();
}

/**
 * 初始化 server 模式认证 UI 与交互。
 *
 * 绑定：'specautoai:auth-required' 事件（api-client 收到 401 时广播）→ 弹框；
 * 表单提交 → setServerToken 存 sessionStorage → 验证，失败清令牌并重新弹框；
 * 取消/Escape → cancelAuthentication；Tab 在对话框内做焦点环（focus trap）。
 * 退出登录时：停掉两个轮询器、清解释缓存与已渲染内容、重置会话 UI——
 * 防止旧身份的数据残留在页面上（Principal 隔离的前端侧配合）。
 *
 * 启动时探测 /api/auth/config：仅 server 模式显示认证按钮/状态；
 * 已有令牌则立即验证，失效则清除并弹框。旧后端无此端点时静默按本地模式继续。
 */
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

/**
 * 模块初始化：暴露全局 API、注册路由与跨模块事件、启动认证流程、应用当前路由。
 *
 * 事件订阅：
 *   - hashchange → applyCurrentRoute；
 *   - 'specautoai:model-catalog-ready'：模型目录晚到时局部重绘概览
 *     （模型显示名依赖目录元数据）；
 *   - 'specautoai:health-ready'：health 晚到时刷新活跃 Run 的状态条
 *     （worker 可用性提示依赖 health）；
 *   - keydown：结果对话框的 Escape 关闭与 Tab 焦点环。
 * 无 hash 时 replaceState 到默认视图（不产生多余历史记录）。
 * 最后广播 'specautoai:results-ready' 供其他脚本等待本模块。
 */
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

// 浏览器环境下自启动：DOM 未就绪则等 DOMContentLoaded（once 防重复），
// 已就绪则立即初始化。typeof 守卫让本文件也能在 Node 下被测试 import 而不报错。
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
}
