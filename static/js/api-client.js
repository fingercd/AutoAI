/** 同源 API 的请求、结构化错误和可选服务器 Bearer 认证。 */
export const SERVER_TOKEN_STORAGE_KEY = 'specautoai.serverToken';

export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.details = options.details ?? null;
    this.requestId = options.requestId ?? null;
  }
}

let tokenWaiter = null;

function sessionStore() {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage;
  } catch (_) {
    return null;
  }
}

export function getServerToken() {
  return sessionStore()?.getItem(SERVER_TOKEN_STORAGE_KEY) || '';
}

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

export function clearServerToken() {
  setServerToken('');
}

export function cancelAuthentication() {
  if (!tokenWaiter) return;
  tokenWaiter.reject(new ApiError('未提供服务器访问令牌', { status: 401, code: 'authentication_cancelled' }));
  tokenWaiter = null;
}

function isSameOriginApi(url) {
  if (typeof window === 'undefined') return String(url).startsWith('/api/');
  const target = new URL(url, window.location.href);
  return target.origin === window.location.origin && target.pathname.startsWith('/api/');
}

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

export async function request(url, options = {}) {
  const response = await performFetch(url, options, options.authRetry !== false);
  if (response.status === 204) return null;
  return response.json();
}

function contentDispositionFilename(value) {
  if (!value) return '';
  const encoded = value.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  if (encoded) {
    try { return decodeURIComponent(encoded); } catch (_) { return encoded; }
  }
  return value.match(/filename="?([^";]+)"?/i)?.[1] || '';
}

export async function downloadFile(url, options = {}) {
  const response = await performFetch(url, options, options.authRetry !== false);
  return {
    blob: await response.blob(),
    filename: contentDispositionFilename(response.headers.get('content-disposition')),
  };
}

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
