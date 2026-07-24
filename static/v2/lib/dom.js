/**
 * dom.js —— v2 前端的 DOM 构建辅助库。
 *
 * 模块职责：为所有 v2 视图组件提供统一的元素创建接口（el / svgEl）、
 * 子节点追加（appendChildren）、清空（clear）与下载触发（saveBlob）。
 *
 * 关键安全约束（为什么全模块禁用 innerHTML）：
 * - 所有文本一律走 textContent / createTextNode，浏览器不会做 HTML 解析，
 *   从根上杜绝 XSS——展示内容里包含用户文件名、后端错误信息等不可信字符串；
 * - escapeHtml 作为唯一兜底导出，仅用于极少数必须拼 HTML 字符串的场景
 *   （如富文本提示），集中在一处便于审计。
 *
 * 在系统中的位置：v2 最底层工具模块，charts.js 及全部视图组件都建立在它之上；
 * 不 import 任何其他 v2 模块，无循环依赖风险。
 */

/**
 * HTML 特殊字符转义（& < > " ' 五件套）。
 * 注意转义顺序：必须先转 &，否则后续生成的 &lt; 等会被二次转义。
 * @param {*} value 任意值；null/undefined 归一为空串。
 * @returns {string} 可安全嵌入 HTML 文本/属性上下文的字符串。
 */
export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

/**
 * 创建 HTML 元素（本模块核心工厂函数）。
 *
 * @param {string} tag 标签名，如 'div'、'button'。
 * @param {Object} [options]
 * @param {string} [options.className] 整串 class（直接覆盖 node.className）。
 * @param {string|number} [options.text] 文本内容，走 textContent——不解析 HTML，
 *        这是全模块防 XSS 的第一道防线；null/undefined 表示不设置文本。
 * @param {Object<string, *>} [options.attrs] 其余属性键值对；值为
 *        null/undefined/false 时跳过（方便条件属性），true 写为空字符串
 *        属性（如 disabled），其他统一 String() 化。
 * @param {Object<string, *>} [options.dataset] data-* 属性；null/undefined 跳过。
 * @param {Object<string, EventListener>} [options.on] 事件监听，键为事件名。
 * @param {...(Node|string|Array|boolean|null|undefined)} children
 *        子节点；字符串一律按文本节点处理（不解析 HTML），数组可任意嵌套，
 *        null/undefined/false 被跳过，便于 `{cond && el(...)}` 这类条件渲染写法。
 * @returns {HTMLElement}
 */
export function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined && options.text !== null) {
    node.textContent = String(options.text);
  }
  if (options.attrs) {
    for (const [key, value] of Object.entries(options.attrs)) {
      if (value === null || value === undefined || value === false) continue;
      node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  if (options.dataset) {
    for (const [key, value] of Object.entries(options.dataset)) {
      if (value === null || value === undefined) continue;
      node.dataset[key] = String(value);
    }
  }
  if (options.on) {
    for (const [event, handler] of Object.entries(options.on)) {
      node.addEventListener(event, handler);
    }
  }
  appendChildren(node, children);
  return node;
}

/**
 * 追加子节点：拍平任意深度嵌套数组；false/null/undefined 跳过；
 * Node 直接 append，其余值（字符串、数字）转成文本节点。
 * flat(Infinity) 让调用方可以随意写 [items.map(...), cond && el(...)] 结构。
 * @param {Node} node 父节点。
 * @param {Array} children 子节点数组（可嵌套）。
 */
export function appendChildren(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** 清空节点的全部子节点（逐首节点移除，比 innerHTML='' 更符合本模块约定）。 */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** SVG 命名空间：createElementNS 必须显式指定，否则创建的是不可见的 HTML 元素。 */
export const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * 创建 SVG 元素。与 el 的差异：
 * - 用 createElementNS（SVG 必须带命名空间）；
 * - SVG 没有 className/dataset 的 HTML 语义，统一走 setAttribute；
 * - 子节点规则与 appendChildren 一致（字符串按文本节点处理）。
 * @param {string} tag SVG 标签名，如 'svg'、'polyline'、'text'。
 * @param {Object<string, *>} [attrs] 属性；null/undefined 跳过。
 * @param {...(Node|string|Array)} children 子节点。
 * @returns {SVGElement}
 */
export function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * 触发浏览器把 Blob 保存为本地文件（用于结果/预处理 CSV 下载）。
 * 原理：创建临时 Object URL，挂一个隐藏 <a download> 并程序化点击，
 * 随后移除节点；Object URL 延迟 1 秒回收——立即 revoke 可能导致
 * 部分浏览器尚未开始读取 URL 而下载失败。
 * 文件名优先用服务端 content-disposition 解析出的值，调用方给兜底名。
 * @param {Blob} blob 要保存的内容。
 * @param {string} [filename] 下载文件名；缺省 'download'。
 */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = el('a', { attrs: { href: url, download: filename || 'download' } });
  anchor.style.display = 'none';
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
