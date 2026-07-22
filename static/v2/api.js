/** v2 API 层：复用 static/js/api-client.js 的 request/downloadFile 与 Bearer 认证边界。 */
import { request, downloadFile } from '../js/api-client.js';

export { ApiError } from '../js/api-client.js';

export function getHealth(options = {}) {
  return request('/health', options);
}

export function getAuthConfig(options = {}) {
  return request('/api/auth/config', options);
}

export function getAuthSession(options = {}) {
  return request('/api/auth/session', options);
}

export function getModels(options = {}) {
  return request('/api/models', options);
}

/** 上传建模 CSV：multipart 字段名固定为 file。 */
export function uploadDataset(file, options = {}) {
  const form = new FormData();
  form.append('file', file);
  return request('/api/datasets/upload', { method: 'POST', body: form, ...options });
}

/**
 * 预处理：多文件使用重复的 files 字段；kind ∈ raman | hplc | chromatography。
 * params: { start_row, end_row, range_mode, x_min, x_max, baseline_method,
 *           hplc_interpolate }。
 */
export function preprocess(kind, files, params = {}, options = {}) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  const append = (key, value) => {
    if (value === null || value === undefined || value === '') return;
    form.append(key, String(value));
  };
  append('start_row', params.start_row ?? 1);
  append('end_row', params.end_row);
  append('range_mode', params.range_mode ?? 'row');
  append('x_min', params.x_min);
  append('x_max', params.x_max);
  if (kind === 'raman') append('baseline_method', params.baseline_method ?? 'arPLS');
  if (kind === 'hplc') form.append('hplc_interpolate', String(params.hplc_interpolate !== false));
  return request(`/api/preprocess/${encodeURIComponent(kind)}`, { method: 'POST', body: form, ...options });
}

/** HPLC 文件选择后的只读批次检测：返回逐文件点数和可处理状态。 */
export function inspectHplc(files, options = {}) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return request('/api/preprocess/hplc/inspect', { method: 'POST', body: form, ...options });
}

/** 创建训练 Run：只入队 queued，由独立 worker 执行。 */
export function createRun(payload, options = {}) {
  return request('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

/** 轻量分页列表：projection=summary + cursor。 */
export function listRunsSummary({ limit = 20, cursor = null, signal } = {}) {
  const params = new URLSearchParams({ projection: 'summary', limit: String(limit) });
  if (cursor) params.set('cursor', cursor);
  return request(`/api/training/runs?${params.toString()}`, { signal });
}

export function getRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}`, options);
}

export function getRunResult(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/result`, options);
}

export function cancelRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST', ...options });
}

export function deleteRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}`, { method: 'DELETE', ...options });
}

export function download(url, options = {}) {
  return downloadFile(url, options);
}

/** 下载 artifact 并解析 JSON（仅用于用户显式点击后的懒加载）。 */
export async function downloadArtifactJson(url, options = {}) {
  const { blob } = await downloadFile(url, options);
  const text = await blob.text();
  return JSON.parse(text);
}
