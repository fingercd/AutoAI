/**
 * 模块说明：SpecAutoAI 前端统一 API 客户端（api-client.js）。
 *
 * 职责：
 * - 封装对后端 FastAPI 同源 `/api/` 接口的 fetch 请求，统一处理结构化错误与可选的 Bearer 认证。
 * - 被 `static/index.html` 与经典前端模块复用；
 *   结果数据契约以后端 `run-result-v1`（见 `docs/run_result_contract.md`）为准，本模块只负责传输，不解释业务字段。
 *
 * 关键设计约束（对应 AGENTS.md / docs/frontend_backend_handoff.md）：
 * - 服务器模式（server）下，浏览器端 token 只能放在当前标签页的 sessionStorage，
 *   严禁写入 URL、localStorage、日志或仓库；本模块是唯一读写该 token 的地方。
 * - Authorization 头只对“同源且路径以 /api/ 开头”的请求注入，防止 token 泄漏到第三方域。
 * - 401 时通过自定义事件 `specautoai:auth-required` 通知 UI 弹出令牌输入，不在这里直接操作 DOM，
 *   保持本模块与具体界面解耦。
 *
 * 协作模块：ui-utils.js（DOM/格式化工具）、各页面内联脚本（通过 window.SpecAutoAIRequest 等全局桥接使用）。
 */

/** 同源 API 的请求、结构化错误和可选服务器 Bearer 认证。 */

/**
 * sessionStorage 中保存服务器访问令牌的键名。
 * 导出常量是为了让 UI 层在提示文案 / 清理逻辑中引用同一个键，避免魔法字符串漂移。
 */
export const SERVER_TOKEN_STORAGE_KEY = 'specautoai.serverToken';

/**
 * 结构化 API 错误类型。
 * 后端错误响应统一为 `{ error: { code, message, details, request_id } }` 结构
 * （或 FastAPI 默认的 `{ detail: ... }`），本类把它们归一化成一个 Error 对象，
 * 让 UI 层可以按 `code` 分支处理（如令牌失效、权限不足），而不是解析裸 HTTP 状态码。
 */
export class ApiError extends Error {
  /**
   * @param {string} message 面向用户的错误信息（取自后端 error.message / detail）
   * @param {object} [options] 附加元数据
   * @param {number|null} [options.status] HTTP 状态码；非 HTTP 错误（如用户取消认证）为 null 或手动指定
   * @param {string|null} [options.code] 后端结构化错误码，用于程序化分支
   * @param {*} [options.details] 后端返回的补充细节（如校验失败字段列表）
   * @param {string|null} [options.requestId] 服务端 request_id，便于在日志/工单中定位
   */
  constructor(message, options = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.details = options.details ?? null;
    this.requestId = options.requestId ?? null;
  }
}

/**
 * 等待用户提供服务器令牌的单例“等待者”。
 * 形状为 { promise, resolve, reject }；同一时刻只允许一个认证请求在途，
 * 多个并发请求遇到 401 时共享同一个 promise，避免弹出一堆令牌输入框。
 */
let tokenWaiter = null;

/**
 * 安全地获取 sessionStorage。
 * 非浏览器环境（如 Node 下跑纯函数测试）返回 null；
 * 某些隐私模式/禁用存储的浏览器访问 sessionStorage 会直接抛异常，因此包 try/catch。
 */
function sessionStore() {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage;
  } catch (_) {
    return null;
  }
}

/**
 * 读取当前标签页保存的服务器令牌；无令牌或存储不可用时返回空字符串。
 * 空字符串（而非 null）让调用方可以做真值判断且不必区分两种“无令牌”情形。
 */
export function getServerToken() {
  return sessionStore()?.getItem(SERVER_TOKEN_STORAGE_KEY) || '';
}

/**
 * 写入（或清空）服务器令牌。
 * - 传入值先 trim 归一化；空串表示清除令牌。
 * - 若此时有请求正在等待令牌（tokenWaiter 存在），立即用新令牌 resolve 它，
 *   让挂起的请求自动重试——这是“弹出令牌框 → 用户输入 → 原请求继续”流程的汇合点。
 * @param {string} token 用户输入的令牌原文
 * @returns {string} 归一化后的令牌（可能为空串）
 */
export function setServerToken(token) {
  const normalized = String(token || '').trim();
  const store = sessionStore();
  if (store) {
    if (normalized) store.setItem(SERVER_TOKEN_STORAGE_KEY, normalized);
    else store.removeItem(SERVER_TOKEN_STORAGE_KEY);
  }
  if (normalized && tokenWaiter) {
    tokenWaiter.resolve(normalized);
    tokenWaiter = null;
  }
  return normalized;
}

/** 清除令牌；复用 setServerToken('') 以保证“归一化 + 清存储”逻辑只有一份。 */
export function clearServerToken() {
  setServerToken('');
}

/**
 * 用户主动取消认证（如关掉令牌输入框）：
 * 以 401 + code 'authentication_cancelled' 拒绝等待中的 promise，
 * 让请求方得到一个可识别的 ApiError，而不是永远挂起。
 * 没有等待者时是安全的空操作。
 */
export function cancelAuthentication() {
  if (!tokenWaiter) return;
  tokenWaiter.reject(new ApiError('未提供服务器访问令牌', { status: 401, code: 'authentication_cancelled' }));
  tokenWaiter = null;
}

/**
 * 判断 URL 是否指向“同源的 /api/ 接口”。
 * 只有这类请求才允许附带 Bearer 令牌、才参与 401 自动重认证——
 * 防止令牌被注入到跨域请求或静态资源请求中泄漏。
 * 非浏览器环境（无 window）退化为前缀判断，供测试使用。
 */
function isSameOriginApi(url) {
  if (typeof window === 'undefined') return String(url).startsWith('/api/');
  const target = new URL(url, window.location.href);
  return target.origin === window.location.origin && target.pathname.startsWith('/api/');
}

/**
 * 发起一次“等待用户提供令牌”的流程：
 * - 若已有等待者，直接复用其 promise（单例，见 tokenWaiter 注释）。
 * - 派发 `specautoai:auth-required` 自定义事件，由 UI 层监听并展示令牌输入界面；
 *   本模块不操作 DOM，保持解耦。
 * @returns {Promise<string>} 用户提交令牌后 resolve 为令牌；用户取消则 reject 为 ApiError。
 */
function requestAuthentication() {
  if (tokenWaiter) return tokenWaiter.promise;
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  tokenWaiter = { promise, resolve, reject };
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('specautoai:auth-required'));
  }
  return promise;
}

/**
 * 把后端错误响应体归一化成 ApiError。
 * 兼容两种后端错误格式：
 * 1. 项目统一的 `{ error: { code, message, details, request_id } }` 结构；
 * 2. FastAPI 默认的 `{ detail: '...' }` 或 `{ detail: { message, code, details } }`。
 * requestId 优先取结构化字段，其次取响应头 x-request-id，方便排障时与后端日志对齐。
 */
function normalizedError(body, response) {
  const structured = body?.error && typeof body.error === 'object' ? body.error : null;
  const detail = body?.detail;
  const message = structured?.message
    || (typeof detail === 'string' ? detail : detail?.message)
    || `HTTP ${response.status}`;
  return new ApiError(message, {
    status: response.status,
    code: structured?.code || detail?.code || null,
    details: structured?.details || detail?.details || null,
    requestId: structured?.request_id || response.headers.get('x-request-id'),
  });
}

/**
 * 实际执行 fetch 的内部函数，带 401 自动重认证能力。
 * @param {string} url 请求地址
 * @param {object} options fetch 选项；额外支持自定义键 `authRetry`（默认 true）
 * @param {boolean} allowAuthRetry 本轮是否允许“等用户输令牌后重试一次”；
 *        重试时传 false，保证最多重试一轮，避免令牌始终无效时死循环。
 *
 * 流程要点：
 * - 克隆调用方 headers 到新的 Headers 对象，不污染调用方传入的对象；
 * - 仅对同源 /api/ 请求注入 Authorization（见 isSameOriginApi）；
 * - 401 且同源：先清掉失效的旧令牌，再等待用户输入新令牌并重试；
 *   非重试轮次也派发 auth-required 事件，让 UI 有机会更新提示。
 */
async function performFetch(url, options, allowAuthRetry) {
  const { authRetry: _authRetry, ...fetchOptions } = options;
  const headers = new Headers(fetchOptions.headers || {});
  const token = getServerToken();
  if (token && isSameOriginApi(url) && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const response = await fetch(url, { ...fetchOptions, headers });
  if (response.ok) return response;

  const body = await response.json().catch(() => ({}));
  const error = normalizedError(body, response);
  if (response.status === 401 && isSameOriginApi(url)) {
    if (token) clearServerToken();
    if (allowAuthRetry) {
      await requestAuthentication();
      return performFetch(url, options, false);
    }
    if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent('specautoai:auth-required'));
  }
  throw error;
}

/**
 * 发起 JSON API 请求的公共入口。
 * - 成功时解析并返回 JSON 响应体；204 No Content 统一返回 null（避免 response.json() 抛错）。
 * - 失败时抛 ApiError（见 performFetch / normalizedError）。
 * @param {string} url 请求地址
 * @param {object} [options] fetch 选项；`authRetry: false` 可关闭 401 自动重认证（如下载接口的显式重试场景）
 * @returns {Promise<*>} 解析后的 JSON，或 204 时的 null
 */
export async function request(url, options = {}) {
  const response = await performFetch(url, options, options.authRetry !== false);
  if (response.status === 204) return null;
  return response.json();
}

/**
 * 从 Content-Disposition 响应头解析下载文件名。
 * 优先解析 RFC 5987 的 `filename*=UTF-8''...` 编码形式（后端对中文文件名使用该形式），
 * 其次回退到普通 `filename="..."`；解析失败时返回原始编码串或空串，绝不抛错，
 * 因为文件名只是锦上添花，不应让下载失败。
 */
function contentDispositionFilename(value) {
  if (!value) return '';
  const encoded = value.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  if (encoded) {
    try { return decodeURIComponent(encoded); } catch (_) { return encoded; }
  }
  return value.match(/filename="?([^";]+)"?/i)?.[1] || '';
}

/**
 * 下载文件类接口（如预处理宽表 CSV、artifact 下载）的公共入口。
 * 与 request() 的区别：返回 Blob 而非 JSON，并从响应头提取服务端给出的文件名。
 * @returns {Promise<{blob: Blob, filename: string}>} filename 可能为空串，由调用方决定兜底文件名。
 */
export async function downloadFile(url, options = {}) {
  const response = await performFetch(url, options, options.authRetry !== false);
  return {
    blob: await response.blob(),
    filename: contentDispositionFilename(response.headers.get('content-disposition')),
  };
}

/**
 * 浏览器环境下的全局桥接：
 * 旧版 `static/index.html` 的内联脚本不是 ES module，无法 import 本文件，
 * 因此把公共 API 挂到 window 上供其使用；ES 模块可直接 import。
 */
if (typeof window !== 'undefined') {
  window.SpecAutoAIRequest = request;
  window.SpecAutoAIDownload = downloadFile;
  window.SpecAutoAIAuth = {
    cancelAuthentication,
    clearServerToken,
    getServerToken,
    setServerToken,
    storageKey: SERVER_TOKEN_STORAGE_KEY,
  };
}
