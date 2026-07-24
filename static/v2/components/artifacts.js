/**
 * 模块说明：Artifact 产物列表组件（static/v2/components/artifacts.js）
 * =================================================================
 * 职责：在 v2 结果页渲染"结果产物"下载表格——只消费 run-result-v1 契约中
 * `GET /api/training/runs/{run_id}/result` 返回的 `artifacts[]` 描述数组，
 * 逐项展示文件名、说明、大小、完整性状态，并按 catalog 状态决定是否允许下载。
 *
 * 在系统中的位置：
 *   - 属于并行新前端 `static/v2/index.html` 的结果页组件，由结果页主控脚本调用
 *     `renderArtifacts(container, { artifacts, onDownload })`；
 *     `onDownload(item)` 由主控注入，真正执行带鉴权的下载。
 *   - 同时导出 `artifactActionState` / `isForbiddenArtifactName` 等纯函数，
 *     被 explainability-panel.js 等兄弟组件复用，保证"能不能下载、为什么不行"
 *     的判断口径全站唯一。
 *
 * 协作模块：
 *   - `../lib/dom.js`：`el`/`clear` DOM 工具。
 *   - `../lib/format.js`：`formatBytes` 统一文件大小显示。
 *
 * 关键设计约束（对应后端 Manifest/下载白名单规则）：
 *   - 黑名单优先：`model.pkl`、`model.pt`、`status.json`、`manifest.json` 以及所有
 *     `*.joblib`（如 DSCARNet 的 AggMap/PCA 映射文件）属于内部文件，即使后端
 *     描述里出现也不渲染、不可下载——这是前端兜底，后端 artifact 白名单才是主防线。
 *   - 下载 URL 必须是同源相对 API 地址（`/api/...`）：拒绝任何 scheme
 *     （http:/javascript: 等）、协议相对 `//`、反斜杠与 `..` 路径穿越，防止把
 *     后端返回的 URL 当成任意外链跳转（开放重定向 / XSS 防护）。
 *   - 是否可下载以后端 `downloadable=true` 为必要条件，前端再加 URL 安全检查；
 *     不可下载时给出可读原因，优先透传后端 `reason` 字段，前端只在没有 reason
 *     时按完整性状态（ok/volatile/missing/corrupt/not_generated）拼装兜底文案。
 */
import { el, clear } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';

/** 新结果页不展示、不下载的内部文件；即使描述里出现也直接过滤。 */
export const FORBIDDEN_ARTIFACT_NAMES = new Set(['model.pkl', 'model.pt', 'status.json', 'manifest.json']);

/**
 * 判断文件名是否属于内部黑名单。
 * @param {string} name - artifact 文件名。
 * @returns {boolean} 命中固定名单或以 `.joblib` 结尾（DSCARNet 映射等序列化对象，
 *   下载白名单不开放）时返回 true。
 */
export function isForbiddenArtifactName(name) {
  const normalized = String(name || '');
  return FORBIDDEN_ARTIFACT_NAMES.has(normalized) || normalized.endsWith('.joblib');
}

/**
 * 前端展示白名单过滤：黑名单项不渲染。
 * 同时剔除 null/非对象项，防御后端返回脏数据导致后续渲染崩溃。
 * @param {Array} artifacts - run-result-v1 的 artifacts[]。
 * @returns {Array} 可展示的 artifact 描述数组（非数组输入返回 []）。
 */
export function visibleArtifacts(artifacts) {
  if (!Array.isArray(artifacts)) return [];
  return artifacts.filter((item) => item && typeof item === 'object' && !isForbiddenArtifactName(item.name));
}

/**
 * 下载 URL 安全性校验：必须是同源相对 API 地址，拒绝 scheme、反斜杠与路径穿越。
 * 规则：
 *   - 非字符串或空串 → false；
 *   - 含任何 URI scheme（http:、javascript:、data: 等）→ false；
 *   - 协议相对 "//host/..." → false（可能跳外站）；
 *   - 必须以 "/api/" 开头（只允许打回本后端 API）；
 *   - 含反斜杠（Windows 路径/绕过尝试）→ false；
 *   - 路径段含 ".."（查询串与 hash 剥离后判断）→ false，防路径穿越。
 * @param {string} url - 后端返回的 download_url。
 * @returns {boolean}
 */
export function isSafeDownloadUrl(url) {
  if (typeof url !== 'string' || !url) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return false;
  if (url.startsWith('//')) return false;
  if (!url.startsWith('/api/')) return false;
  if (url.includes('\\')) return false;
  return !url.split(/[?#]/)[0].split('/').includes('..');
}

/**
 * 后端 artifact 完整性状态 → 中文展示名。
 * volatile 表示文件在训练后仍可能被改写（大小/哈希不稳定），
 * missing/corrupt/not_generated 分别对应缺失、SHA-256 校验失败、未生成。
 */
const INTEGRITY_LABELS = {
  ok: '完整',
  volatile: '易变',
  missing: '缺失',
  corrupt: '损坏',
  not_generated: '未生成',
};

/**
 * 单个 artifact 的展示状态：只有 downloadable=true 且 URL 安全才允许下载；
 * 其余情况给出可读原因（优先使用后端 reason）。
 *
 * 判定顺序：
 *   1. 描述不是对象 → 禁用（"描述不可用"）。
 *   2. 命中内部文件黑名单 → 禁用（前端兜底，与 visibleArtifacts 一致）。
 *   3. downloadable===true 且 isSafeDownloadUrl(url) → 启用。
 *   4. 否则按优先级拼装原因：后端 reason > URL 不安全 > 文件缺失 > 完整性异常 > 兜底文案。
 * @param {Object} artifact - run-result-v1 artifacts[] 中的一项
 *   （name/label/category/size_bytes/integrity/downloadable/download_url/reason/exists）。
 * @returns {{enabled: boolean, reason: string}} 下载按钮状态与禁用原因。
 */
export function artifactActionState(artifact) {
  if (!artifact || typeof artifact !== 'object') {
    return { enabled: false, reason: '描述不可用' };
  }
  if (isForbiddenArtifactName(artifact.name)) {
    return { enabled: false, reason: '内部文件，不开放下载' };
  }
  const url = artifact.download_url;
  if (artifact.downloadable === true && isSafeDownloadUrl(url)) {
    return { enabled: true, reason: '' };
  }
  const reason = artifact.reason
    || (artifact.downloadable === true && !isSafeDownloadUrl(url) ? '下载地址不安全，已禁用' : '')
    || (artifact.exists === false ? '文件未生成或已缺失' : '')
    || (artifact.integrity && artifact.integrity !== 'ok' ? `完整性状态：${INTEGRITY_LABELS[artifact.integrity] || artifact.integrity}` : '')
    || '当前不可下载';
  return { enabled: false, reason };
}

/**
 * 渲染 Artifact 下载表格。
 * 逻辑：先清空容器 → visibleArtifacts 过滤黑名单 → 空列表渲染空态提示；
 * 否则逐行渲染文件信息，下载按钮状态完全由 artifactActionState 决定：
 * 可下载时用主色按钮并绑定 onDownload(item)；不可下载时禁用按钮并在
 * 按钮下方附原因 hint（不禁用 title，原因直接可见）。
 * @param {HTMLElement} container - 挂载点，内部内容会被整体替换。
 * @param {Object} options
 * @param {Array} options.artifacts - run-result-v1 的 artifacts[]。
 * @param {Function} [options.onDownload] - 下载回调，由主控注入实际下载逻辑。
 */
export function renderArtifacts(container, { artifacts, onDownload }) {
  clear(container);
  const items = visibleArtifacts(artifacts);
  if (!items.length) {
    container.append(el('div', { className: 'empty' }, [
      el('p', { text: '没有可展示的产物。' }),
      el('p', { className: 'hint', text: '训练成功并生成结果文件后，这里会列出可下载项。' }),
    ]));
    return;
  }
  const wrap = el('div', { className: 'table-wrap' });
  const table = el('table', { className: 'data-table artifacts-table' }, [
    el('caption', { text: '结果产物逐项下载；禁用项附带原因。' }),
    el('thead', {}, el('tr', {}, [
      el('th', { text: '文件', attrs: { scope: 'col' } }),
      el('th', { text: '说明', attrs: { scope: 'col' } }),
      el('th', { text: '大小', attrs: { scope: 'col' } }),
      el('th', { text: '完整性', attrs: { scope: 'col' } }),
      el('th', { text: '操作', attrs: { scope: 'col' } }),
    ])),
    el('tbody', {}, items.map((item) => {
      const state = artifactActionState(item);
      const button = el('button', {
        className: state.enabled ? 'btn btn-primary btn-sm' : 'btn btn-ghost btn-sm',
        text: '下载',
        attrs: { type: 'button', disabled: state.enabled ? null : true },
        on: state.enabled ? { click: () => onDownload?.(item) } : {},
      });
      const actionCell = el('td', {}, [button]);
      if (!state.enabled && state.reason) {
        actionCell.append(el('div', { className: 'hint artifact-reason', text: state.reason }));
      }
      return el('tr', {}, [
        el('td', {}, el('code', { text: item.name || '—' })),
        el('td', { text: item.label || item.category || '—' }),
        el('td', { text: formatBytes(item.size_bytes) }),
        el('td', { text: INTEGRITY_LABELS[item.integrity] || item.integrity || '—' }),
        actionCell,
      ]);
    })),
  ]);
  wrap.append(table);
  container.append(wrap);
}
