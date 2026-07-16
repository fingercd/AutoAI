import { request } from './api-client.js';

// createPromise 是页面级 single-flight 锁：用户双击或多个监听器同时触发时只发送
// 一个创建请求。它不会缓存结果，settled 后立即释放，后续 Run 可以正常创建。
let createPromise = null;

export async function createRun(payload) {
  if (createPromise) return createPromise;
  createPromise = request('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).finally(() => { createPromise = null; });
  return createPromise;
}

export const pollRun = (runId) => request(`/api/training/runs/${encodeURIComponent(runId)}`);
export async function cancelRun(runId) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' });
}

window.SpecAutoAITrainingStore = { createRun, pollRun, cancelRun };
window.dispatchEvent(new CustomEvent('specautoai:modules-ready'));
