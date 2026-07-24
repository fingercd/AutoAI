/**
 * 模块：v2 工作台会话内状态仓库（store）。
 *
 * 职责与定位：
 * - 这是 static/v2 新前端的"单一事实来源"（single source of truth）之一，
 *   只保存生命周期为"当前标签页一次会话"的轻量状态：当前路由、健康检查结果、
 *   本标签页新创建的 Run 集合、以及从训练记录复制过来的建模草稿。
 * - 与之协作的模块：app.js（路由变化时调用 setRoute、轮询后端后调用 setHealth）、
 *   各视图 views/*.js（通过 getState/subscribe 读取或监听状态）。
 *
 * 关键设计约束（安全相关，不能随意改动）：
 * - 只存活于当前标签页内存，刻意不落 localStorage / sessionStorage / Cookie。
 *   项目安全约定要求浏览器侧 token 等敏感信息只能进当前标签页 sessionStorage，
 *   不得写入 URL、localStorage、日志或仓库；本模块更进一步，任何状态都不持久化，
 *   刷新页面即清空，避免跨标签页/跨会话泄露。
 * - 只读巡检不得读取或输出敏感文件；本模块也不保存任何认证信息，认证 token 由
 *   static/js/api-client.js 与 auth-gate 组件单独管理。
 */

/** 订阅者集合：每项形如 { event, handler }，用 Set 便于 O(1) 注销。 */
const listeners = new Set();

/**
 * 全局可变状态对象（模块级单例）。
 * 设计意图：v2 前端状态极少，因此不引入 Redux/Pinia 等框架，
 * 用一个普通对象 + 手写发布订阅即可，符合不引入第三方前端框架与构建链的项目约束。
 */
const state = {
  /** 当前路由快照：{ view: 视图名, runId: 结果页专属 run_id 或 null }。 */
  route: { view: 'workbench', runId: null },
  /** 本标签页新创建的 Run：只有它们允许 3 秒自动跳转。 */
  createdRunIds: new Set(),
  /** 从训练记录复制的建模草稿。 */
  modelingDraft: null,
  /** 最近一次 /health 健康检查结果（含 deployment_mode、worker、contracts）。 */
  health: null,
};

/**
 * 读取整个状态对象。
 * @returns {object} state 单例本身（注意：返回的是引用，调用方不应直接改写）。
 */
export function getState() {
  return state;
}

/**
 * 更新当前路由并广播 'route' 事件。
 * 由 app.js 的 render() 在每次 hash 解析后调用，使视图能感知 runId 等参数。
 * @param {{view: string, runId: string|null}} route parseHash 的解析结果。
 */
export function setRoute(route) {
  state.route = route;
  emit('route', route);
}

/**
 * 更新健康检查结果并广播 'health' 事件。
 * @param {object} health 后端 /health 返回的对象，可能为 null（请求失败场景由调用方决定）。
 */
export function setHealth(health) {
  state.health = health;
  emit('health', health);
}

/**
 * 登记"本标签页新创建的训练 Run"。
 * 用途：建模视图提交训练后，只有本标签页创建的 Run 才允许自动跳转到结果页，
 * 防止用户在训练记录页点开别人的/历史的 Run 时被意外跳转打扰。
 * @param {string} runId 后端返回的 Run 标识；空值直接忽略。
 */
export function markRunCreated(runId) {
  if (runId) state.createdRunIds.add(runId);
}

/**
 * 判断某个 Run 是否由本标签页创建（配合 markRunCreated 使用）。
 * @param {string} runId
 * @returns {boolean} true 表示是本标签页创建，允许自动跳转。
 */
export function isTabCreatedRun(runId) {
  return Boolean(runId) && state.createdRunIds.has(runId);
}

/**
 * 暂存建模草稿（"复制配置"功能）：训练记录页把某个 Run 的参数写入这里，
 * 建模向导进入时通过 consumeModelingDraft 取走并回填表单。
 * @param {object|null} draft 建模配置草稿对象。
 */
export function setModelingDraft(draft) {
  state.modelingDraft = draft;
}

/**
 * 取出并清空建模草稿（"消费"语义：一次性读取，避免旧草稿被重复使用）。
 * @returns {object|null} 草稿对象；没有草稿时返回 null。
 */
export function consumeModelingDraft() {
  const draft = state.modelingDraft;
  state.modelingDraft = null;
  return draft;
}

/**
 * 订阅状态事件。
 * @param {string} event 事件名，目前只有 'route' 和 'health'。
 * @param {(payload: any) => void} handler 回调，收到该事件对应的 payload。
 * @returns {() => void} 注销函数；视图卸载时务必调用，防止闭包泄漏和幽灵更新。
 */
export function subscribe(event, handler) {
  const entry = { event, handler };
  listeners.add(entry);
  return () => listeners.delete(entry);
}

/**
 * 内部发布函数：同步遍历订阅者并调用匹配的 handler。
 * 注意是同步分发——事件简单、订阅者少，无需异步队列；
 * handler 抛错不会在这里捕获，由调用方自行保证健壮性。
 * @param {string} event 事件名。
 * @param {any} payload 事件负载。
 */
function emit(event, payload) {
  for (const entry of listeners) {
    if (entry.event === event) entry.handler(payload);
  }
}
