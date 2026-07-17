/** 会话内状态：只存活于当前标签页内存，不落 localStorage。 */

const listeners = new Set();

const state = {
  route: { view: 'workbench', runId: null },
  /** 本标签页新创建的 Run：只有它们允许 3 秒自动跳转。 */
  createdRunIds: new Set(),
  /** 从训练记录复制的建模草稿。 */
  modelingDraft: null,
  health: null,
};

export function getState() {
  return state;
}

export function setRoute(route) {
  state.route = route;
  emit('route', route);
}

export function setHealth(health) {
  state.health = health;
  emit('health', health);
}

export function markRunCreated(runId) {
  if (runId) state.createdRunIds.add(runId);
}

export function isTabCreatedRun(runId) {
  return Boolean(runId) && state.createdRunIds.has(runId);
}

export function setModelingDraft(draft) {
  state.modelingDraft = draft;
}

export function consumeModelingDraft() {
  const draft = state.modelingDraft;
  state.modelingDraft = null;
  return draft;
}

export function subscribe(event, handler) {
  const entry = { event, handler };
  listeners.add(entry);
  return () => listeners.delete(entry);
}

function emit(event, payload) {
  for (const entry of listeners) {
    if (entry.event === event) entry.handler(payload);
  }
}
