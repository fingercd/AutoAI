/**
 * training-store.js —— 训练任务（Run）创建与轮询的前端数据层模块。
 *
 * 系统位置：
 *   这是 `static/index.html`（旧版主前端）训练流程的网络层封装，负责与后端的
 *   `/api/training/runs` 系列接口通信。上游是建模表单（点击"开始训练"），下游是
 *   `run-results.js` 中的训练进度/结果展示逻辑。
 *
 * 协作模块：
 *   - `api-client.js` 提供统一的 `request()`（封装 fetch、鉴权头、错误规范化），
 *     本模块不直接操作 fetch，保证 token 注入、401 处理等行为全局一致。
 *   - 后端语义约束：训练请求只创建 queued 状态的 Run，不直接启动训练
 *     （训练由独立 worker 执行），因此这里只提供"创建 / 查询 / 取消"三个动词。
 *
 * 关键设计约束：
 *   - `createRun` 使用 single-flight 锁，防止双击或重复触发产生多个重复 Run。
 *   - 模块加载完成后把 API 挂到 `window.SpecAutoAITrainingStore`，并广播
 *     `specautoai:modules-ready` 事件，供非模块化的内联脚本以全局方式使用。
 */
import { request } from './api-client.js';

// createPromise 是页面级 single-flight 锁：用户双击或多个监听器同时触发时只发送
// 一个创建请求。它不会缓存结果，settled 后立即释放，后续 Run 可以正常创建。
let createPromise = null;

/**
 * 创建训练 Run（POST /api/training/runs）。
 *
 * 并发保护（single-flight）：
 *   如果已有一个创建请求在途，直接返回同一个 Promise，而不是再发一次请求。
 *   这解决两个实际场景：用户快速双击"开始训练"按钮、以及多个事件监听器同时触发。
 *   注意它**不缓存结果**——请求 settled（成功或失败）后通过 `finally` 立即释放锁，
 *   所以下一次训练（不同 payload）可以正常创建，不会被旧锁挡住。
 *
 * @param {object} payload 训练请求体（model_type、dataset_id、评估口径等，由后端校验；
 *   server 模式下只接受 dataset_id，不接受浏览器传入的 data_path）。
 * @returns {Promise<object>} 后端返回的 Run 对象（初始为 queued 状态）。
 */
export async function createRun(payload) {
  if (createPromise) return createPromise;
  createPromise = request('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).finally(() => { createPromise = null; });
  return createPromise;
}

/**
 * 轮询单个 Run 的状态（GET /api/training/runs/{runId}）。
 * runId 经 encodeURIComponent 编码，防止特殊字符破坏 URL 路径。
 * 返回一个一次性请求；循环调度由调用方（如 RunPollController）负责。
 */
export const pollRun = (runId) => request(`/api/training/runs/${encodeURIComponent(runId)}`);

/**
 * 请求取消一个 Run（POST /api/training/runs/{runId}/cancel）。
 * 后端按当前 Principal 做 scope 校验，前端只负责发起请求并透传结果/错误。
 */
export async function cancelRun(runId) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' });
}

/**
 * 停止一个排队中/运行中的 Run。后端只保留 STOP 记录并丢弃本次产物。
 */
export async function stopRun(runId) {
  const encoded = encodeURIComponent(runId);
  try {
    return await request(`/api/training/runs/${encoded}/stop`, { method: 'POST' });
  } catch (error) {
    // 页面静态文件可能先于后端进程更新；旧服务没有 /stop 时退回兼容接口。
    if (error?.status !== 404 && error?.status !== 405) throw error;
    return request(`/api/training/runs/${encoded}/cancel`, { method: 'POST' });
  }
}

// 把模块 API 暴露为全局对象，供 `static/index.html` 中的非模块化内联脚本调用；
// 这是 ES module 与旧式脚本之间的桥接方式。
window.SpecAutoAITrainingStore = { createRun, pollRun, cancelRun, stopRun };
// 广播模块就绪事件：内联脚本可能在本模块加载前就已执行，通过监听该事件
// 等待模块可用，避免加载顺序导致的竞态。
window.dispatchEvent(new CustomEvent('specautoai:modules-ready'));
