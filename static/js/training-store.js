import { request } from './api-client.js';

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

window.AutoAITrainingStore = { createRun, pollRun, cancelRun };
window.dispatchEvent(new CustomEvent('autoai:modules-ready'));
