/**
 * 同源 JSON 请求的最小封装。
 *
 * 业务层只接收解析后的 JSON；失败时优先透传 FastAPI 的 detail，保证上传校验、
 * 状态冲突和 capability 错误都能显示具体原因。
 */
export async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

window.SpecAutoAIRequest = request;
