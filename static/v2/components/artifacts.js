/** Artifact 列表：只消费 run-result-v1 的 artifacts[]，按 catalog 状态逐项下载。 */
import { el, clear } from '../lib/dom.js';
import { formatBytes } from '../lib/format.js';

/** 新结果页不展示、不下载的内部文件；即使描述里出现也直接过滤。 */
export const FORBIDDEN_ARTIFACT_NAMES = new Set(['model.pkl', 'model.pt', 'status.json', 'manifest.json']);

export function isForbiddenArtifactName(name) {
  const normalized = String(name || '');
  return FORBIDDEN_ARTIFACT_NAMES.has(normalized) || normalized.endsWith('.joblib');
}

/** 前端展示白名单过滤：黑名单项不渲染。 */
export function visibleArtifacts(artifacts) {
  if (!Array.isArray(artifacts)) return [];
  return artifacts.filter((item) => item && typeof item === 'object' && !isForbiddenArtifactName(item.name));
}

/** 下载 URL 必须是同源相对 API 地址，拒绝 scheme、反斜杠与路径穿越。 */
export function isSafeDownloadUrl(url) {
  if (typeof url !== 'string' || !url) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return false;
  if (url.startsWith('//')) return false;
  if (!url.startsWith('/api/')) return false;
  if (url.includes('\\')) return false;
  return !url.split(/[?#]/)[0].split('/').includes('..');
}

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
