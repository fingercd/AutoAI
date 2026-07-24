/**
 * 模块：v2 API 层（对后端 REST 接口的薄封装）。
 *
 * 职责与定位：
 * - 本文件只做一件事：把后端 FastAPI 的 REST 端点包装成一组语义化 JS 函数，
 *   供 static/v2 下的各视图（views/*.js）与组件调用。
 * - 自身不实现 HTTP 细节：底层 request/downloadFile 与 Bearer 认证边界
 *   全部复用旧前端的 static/js/api-client.js（v2 与 v1 共用同一份契约实现），
 *   认证 token 的存放（仅当前标签页 sessionStorage）、401 拦截与重试都由它负责。
 * - 与之协作：app.js（getHealth 轮询）、auth-gate 组件（getAuthConfig/getAuthSession）、
 *   建模/预处理/结果视图（其余函数）。
 *
 * 关键设计约束：
 * - 接口路径、字段名、查询参数必须与 docs/frontend_backend_handoff.md 的契约一致，
 *   这里只封装、不发明新协议。
 * - 所有 Run 相关读/写/取消/删除在后端按 Principal（服务端注入的身份）做 scope 隔离，
 *   前端不得、也无法通过请求体传 owner_id/tenant_id 之类的字段。
 */
/** v2 API 层：复用 static/js/api-client.js 的 request/downloadFile 与 Bearer 认证边界。 */
import { request, downloadFile } from '../js/api-client.js';

export { ApiError } from '../js/api-client.js';

/**
 * 健康检查：GET /health。
 * 返回部署模式（deployment_mode）、worker 可用性与契约版本（contracts），
 * app.js 每 30 秒轮询一次并渲染到顶栏徽标。
 * @param {object} [options] 透传给 request 的选项（如 AbortSignal）。
 */
export function getHealth(options = {}) {
  return request('/health', options);
}

/**
 * 获取认证配置：GET /api/auth/config。
 * 用于认证门判断当前部署是否需要 token（server 模式强制，local 模式通常放行）。
 */
export function getAuthConfig(options = {}) {
  return request('/api/auth/config', options);
}

/**
 * 探测当前会话认证状态：GET /api/auth/session。
 * 启动时由 probeAuth 调用，401 时弹出认证门。
 */
export function getAuthSession(options = {}) {
  return request('/api/auth/session', options);
}

/**
 * 获取模型能力目录：GET /api/models。
 * 返回 15 个目标模型的目录（含 available 标记；如 cnn_mamba1d 因 mamba-ssm 依赖
 * 缺失会返回 available=false，前端据此禁用而不是静默替换）。
 */
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
 *
 * 业务约束说明（与后端预处理规则一一对应，详见 AGENTS.md 预处理规则）：
 * - 拉曼（raman）：支持行号或 X 轴范围截取，且后端固定"先选择数据范围、再做基线校正"，
 *   因此 baseline_method 只对 raman 有意义。
 * - HPLC：hplc_interpolate 开关决定是否把所选区间映射到固定目标轴
 *  （第 n 点真实时间 = start + (n-1)*(stop-start)/(point_count-1)），
 *   关闭时导出所选原始 X/Y，但多文件轴不一致会被后端拒绝。
 * - 行号为 1 基、首尾包含；end_row 留空表示按检测出的完整点数处理，前端不得静默截断。
 * - 空值（null/undefined/''）不写入表单，交给后端使用各自的默认值，避免前后端默认值漂移。
 *
 * @param {string} kind 预处理类型：'raman' | 'hplc' | 'chromatography'（后者仅为兼容保留）。
 * @param {File[]} files 待上传的原始数据文件列表，逐一以重复字段名 files 追加。
 * @param {object} [params] 预处理参数，见上。
 * @param {object} [options] 透传选项（如 AbortSignal）。
 * @returns {Promise<object>} 后端预处理结果（含 wide-feature-v2 宽表 CSV 下载信息）。
 */
export function preprocess(kind, files, params = {}, options = {}) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  // 仅当值非空时追加字段：保持"未设置=后端默认"的语义，避免把空字符串传给后端解析。
  const append = (key, value) => {
    if (value === null || value === undefined || value === '') return;
    form.append(key, String(value));
  };
  append('start_row', params.start_row ?? 1);
  append('end_row', params.end_row);
  append('range_mode', params.range_mode ?? 'row');
  append('x_min', params.x_min);
  append('x_max', params.x_max);
  // 基线校正仅拉曼支持；默认 arPLS。
  if (kind === 'raman') append('baseline_method', params.baseline_method ?? 'arPLS');
  // HPLC 插值是布尔开关：后端要求显式字符串，未设置时默认开启（!== false）。
  if (kind === 'hplc') form.append('hplc_interpolate', String(params.hplc_interpolate !== false));
  return request(`/api/preprocess/${encodeURIComponent(kind)}`, { method: 'POST', body: form, ...options });
}

/**
 * HPLC 文件选择后的只读批次检测：返回逐文件点数和可处理状态。
 *
 * 为什么需要单独的 inspect：HPLC 规则要求批次内点数一致（或以唯一众数为期望点数，
 * 不一致时列出全部异常文件并拒绝预处理）。选择文件后先调用本接口做只读解析，
 * 把每个文件的原文件名、点数、时间范围与状态展示给用户确认，再决定是否正式预处理，
 * 避免直接提交后被整体拒绝、反复试错。
 */
export function inspectHplc(files, options = {}) {
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return request('/api/preprocess/hplc/inspect', { method: 'POST', body: form, ...options });
}

/**
 * 创建训练 Run：POST /api/training/runs。
 *
 * 关键契约：本接口只入队一个 queued 状态的 Run，不直接启动训练；
 * 真正的训练由独立 worker 异步执行（HTTP 请求的 BackgroundTasks 不承担训练），
 * 前端提交后轮询 Run 状态或跳转训练记录页等待。
 * server 模式下 payload 只能含 dataset_id / test_dataset_id，不得传浏览器侧 data_path。
 *
 * @param {object} payload 训练配置（数据集、评估口径、模型、超参数等）。
 * @returns {Promise<object>} 新建的 Run 对象（status 为 queued）。
 */
export function createRun(payload, options = {}) {
  return request('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

/**
 * 轻量分页列表：GET /api/training/runs?projection=summary + cursor。
 *
 * 为什么用 projection=summary：训练记录页只需要概要字段（含可空 test_macro_f1），
 * 完整 Run 对象太大；summary 投影中仅"成功且 Manifest 完整"的 Run 才带测试主指标，
 * 且交叉验证必须取 pooled test 而非 fold mean——这些口径由后端保证。
 * cursor 分页避免 offset 分页在并发写入时的错位。
 *
 * @param {object} [args]
 * @param {number} [args.limit=20] 每页条数。
 * @param {string|null} [args.cursor=null] 上一页返回的游标；首页传 null。
 * @param {AbortSignal} [args.signal] 用于页面切换时中止在途请求。
 */
export function listRunsSummary({ limit = 20, cursor = null, signal } = {}) {
  const params = new URLSearchParams({ projection: 'summary', limit: String(limit) });
  if (cursor) params.set('cursor', cursor);
  return request(`/api/training/runs?${params.toString()}`, { signal });
}

/**
 * 获取单个 Run 的完整状态：GET /api/training/runs/{run_id}。
 * 后端按 Principal scope 校验：无权访问的 Run 返回 404/403，前端不额外处理越权。
 */
export function getRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}`, options);
}

/**
 * 获取 Run 的建模结果：GET /api/training/runs/{run_id}/result。
 * 返回 run-result-v1 契约对象（指标、曲线、混淆矩阵、可解释性摘要、artifact 清单），
 * 是结果页的唯一数据来源；CV 结果在其中区分合并预测 / fold mean / fold std。
 */
export function getRunResult(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/result`, options);
}

/**
 * 取消 Run：POST /api/training/runs/{run_id}/cancel。
 * 仅对非终态 Run 有意义；取消语义（是否回收 worker 资源）由后端/worker 决定。
 */
export function cancelRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST', ...options });
}

/** 停止 Run，只保留 STOP 记录并丢弃本次训练产物。 */
export async function stopRun(runId, options = {}) {
  const encoded = encodeURIComponent(runId);
  try {
    return await request(`/api/training/runs/${encoded}/stop`, { method: 'POST', ...options });
  } catch (error) {
    if (error?.status !== 404 && error?.status !== 405) throw error;
    return request(`/api/training/runs/${encoded}/cancel`, { method: 'POST', ...options });
  }
}

/**
 * 删除 Run：DELETE /api/training/runs/{run_id}。
 * 删除会移除记录与其产物目录，属于不可恢复操作，调用方（视图）需自行确认。
 */
export function deleteRun(runId, options = {}) {
  return request(`/api/training/runs/${encodeURIComponent(runId)}`, { method: 'DELETE', ...options });
}

/**
 * 通用文件下载：透传 api-client 的 downloadFile（自动带 Bearer token 与错误处理）。
 * 用于 artifact 白名单内的文件下载（内部模型对象、拟合对象和状态投影等不在白名单内）。
 */
export function download(url, options = {}) {
  return downloadFile(url, options);
}

/**
 * 下载 artifact 并解析 JSON（仅用于用户显式点击后的懒加载）。
 *
 * 设计意图：结果页的部分 JSON 产物（如 sample_feature_importance.json）可能较大，
 * 不随 run-result-v1 内联返回；只有在用户点击"查看"时才下载并解析，
 * 减少结果页首屏传输量。
 *
 * @param {string} url artifact 下载地址（相对路径，由后端 artifact 清单给出）。
 * @returns {Promise<any>} 解析后的 JSON 对象；JSON 非法时抛 SyntaxError。
 */
export async function downloadArtifactJson(url, options = {}) {
  const { blob } = await downloadFile(url, options);
  const text = await blob.text();
  return JSON.parse(text);
}
