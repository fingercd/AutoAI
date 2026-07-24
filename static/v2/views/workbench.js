/**
 * 【模块说明】预处理工作台视图（static/v2 工作台）
 *
 * 职责：渲染"预处理工作台"页——三张卡片走完全程：
 * ① 上传原始两列曲线文件（X, 强度）→ ② 一键预处理（高级参数默认折叠）→
 * ③ 预览对照曲线、导出建模文件或跳转建模页。支持拉曼（raman）与 HPLC 色谱两种类型。
 *
 * 系统位置与协作：
 * - 这是 v2 前端（static/v2/index.html）的一个视图组件，由 Hash 路由挂载；
 *   下载结果后到 #/modeling（modeling.js）继续建模流程。
 * - 通过 ../api.js 调用后端：inspectHplc（HPLC 批次点数/时间范围预检）、
 *   preprocess（POST /api/preprocess/raman 或 /api/preprocess/hplc）、download（下载结果 CSV）。
 * - 曲线图用 ../lib/charts.js 的 lineChart（SVG）；validateHplcRowRange 来自
 *   ../../js/ui-utils.js，与旧前端共用同一份 HPLC 行号范围校验逻辑（前后端行为一致的关键）。
 *
 * 关键业务约束（与后端契约一致）：
 * - 预处理统一输出 wide-feature-v2 宽表：Index, Label, Sample_ID, Name 四个元数据列 +
 *   第 5 列起真实、严格递增的坐标表头；Label/Sample_ID 由用户下载后人工填写，Name 已保留原文件名。
 * - 拉曼：处理顺序固定为先截取范围、再在范围内做基线校正（arPLS 等 5 种方法）。
 * - HPLC：0–50 分钟时间范围；选择文件后必须先做批次点数检测（inspectHplc），
 *   点数一致（或满足后端众数规则）才允许开始预处理；hplc_interpolate 开关决定
 *   输出固定目标时间轴插值结果还是原始所选轴（关闭时多文件所选轴必须完全一致）。
 *   两种模式都不做面积归一化或消负。
 */
import { el, clear, saveBlob, svgEl } from '../lib/dom.js';
import { lineChart } from '../lib/charts.js';
import { preprocess, inspectHplc, download } from '../api.js';
import { validateHplcRowRange } from '../../js/ui-utils.js';

/** 拉曼可选的基线校正方法，arPLS 为默认（见 paramForm 中选项文案）。 */
const RAMAN_BASELINE_METHODS = ['arPLS', 'airPLS', 'als', 'drPLS', 'poly'];
/** 原始曲线颜色（灰蓝，虚线展示）。 */
const COLOR_RAW = '#94a3b8';
/** 处理后曲线颜色（深青绿，实线展示）。 */
const COLOR_PROCESSED = '#0f766e';

/**
 * 为当前文件选择生成一个指纹串（文件名 + 大小 + 最后修改时间）。
 * 用于检测异步的 HPLC 批次检查返回时，用户是否已经换了一批文件：
 * 指纹不一致则丢弃过期结果。 与换行符分隔避免不同字段拼接出相同串。
 */
const fileSelectionKey = (files) => files
  .map((file) => `${file.name}\u0000${file.size}\u0000${file.lastModified}`)
  .join('\u0001');

/**
 * 取曲线的"处理后强度"序列：新契约字段为 processed_y，旧响应兼容 corrected_y。
 * 两者都没有时返回 null（调用方回退展示原始 raw_y）。
 */
function processedSeries(curve) {
  return curve?.processed_y ?? curve?.corrected_y ?? null;
}

/**
 * 渲染图例列表：每条序列一小段 SVG 线段（颜色/虚线与图一致）+ 名称。
 * @param {Array<{name?: string, color?: string, dashed?: boolean}>} series
 */
function legendList(series) {
  return el('ul', { className: 'legend' },
    (series || []).map((item, index) => el('li', { className: 'legend-item' }, [
      svgEl('svg', { width: 22, height: 10, viewBox: '0 0 22 10', 'aria-hidden': 'true' }, [
        svgEl('line', {
          x1: 0, x2: 22, y1: 5, y2: 5,
          stroke: item.color || COLOR_PROCESSED,
          'stroke-width': 2,
          'stroke-dasharray': item.dashed ? '5 4' : null,
        }),
      ]),
      el('span', { text: item.name || `序列 ${index + 1}` }),
    ])));
}

/**
 * 为一条曲线构造预览图（SVG）与配套图例。
 *
 * HPLC 只画一条处理后的色谱强度线（无处理后数据时回退 raw_y）；
 * 拉曼画"原始 raw_y（灰虚线）+ 基线校正后（绿实线）"双序列对照。
 *
 * @param {object} curve 后端 curves 数组中的一项：{ name, x, raw_y, processed_y?/corrected_y? }。
 * @param {'raman'|'hplc'} kind 预处理类型，决定序列构成与坐标轴文案。
 * @returns {{ svg: SVGElement, legend: HTMLElement }}
 */
function curveChart(curve, kind) {
  const processed = processedSeries(curve);
  const series = kind === 'hplc'
    ? [{ name: '色谱强度', xs: curve.x, ys: processed ?? curve.raw_y, color: COLOR_PROCESSED }]
    : [{ name: '原始 raw_y', xs: curve.x, ys: curve.raw_y, color: COLOR_RAW, dashed: true }];
  if (kind !== 'hplc' && processed) series.push({ name: '基线校正后 corrected_y', xs: curve.x, ys: processed, color: COLOR_PROCESSED });
  return {
    svg: lineChart({
      series,
      title: kind === 'hplc' ? `${curve.name} 色谱曲线` : `${curve.name} 预处理对照`,
      description: kind === 'hplc'
        ? `曲线 ${curve.name}：横轴为保留时间，纵轴为强度。`
        : `曲线 ${curve.name}：灰虚线为原始强度，绿线为基线校正后强度。横轴为 X，纵轴为强度。`,
      xLabel: kind === 'hplc' ? '保留时间（分钟）' : '拉曼位移 / X',
      yLabel: '强度',
    }),
    legend: legendList(series),
  };
}

/**
 * 构造"高级参数"表单：范围截取（全部/按行号/按 X 轴数值）+ 类型专属控件
 * （拉曼：基线校正方法；HPLC：共同时间轴线性插值开关），全部收进 <details> 折叠，
 * 顶部附实时"将执行……"摘要。
 *
 * @param {'raman'|'hplc'} kind 预处理类型。
 * @returns {{ node: HTMLElement, collect: () => object, setPointCount: (n: any) => void }}
 *   node 表单节点；collect 收集并校验为后端 preprocess 接口的 params；
 *   setPointCount 由 HPLC 批次检测结果回填检测出的公共点数（驱动行号范围校验与占位提示）。
 * 边界：HPLC 行号范围经 validateHplcRowRange 严格校验（1 基、首尾包含、不越界），
 * 非法时 collect 抛异常由调用方展示；x_value 模式空边界传 null 表示不限。
 */
function paramForm(kind) {
  const isHplc = kind === 'hplc';
  // 由 setPointCount 回填的批次公共点数；null 表示尚未检测/不可用。
  let detectedPointCount = null;
  const rangeMode = el('select', { className: 'select', attrs: { id: 'v2-pp-range-mode' } }, [
    el('option', { text: '全部数据（不截取）', attrs: { value: 'row' } }),
    el('option', { text: '按行号截取', attrs: { value: 'row_range' } }),
    el('option', { text: isHplc ? '按保留时间截取' : '按 X 轴数值截取', attrs: { value: 'x_value' } }),
  ]);
  const startRow = el('input', { className: 'input', attrs: { id: 'v2-pp-start-row', type: 'number', min: '1', step: '1', value: '1' } });
  const endRow = el('input', { className: 'input', attrs: { id: 'v2-pp-end-row', type: 'number', min: '1', step: '1', placeholder: isHplc ? '选择文件后自动确定' : '留空到末尾' } });
  const xMin = el('input', { className: 'input', attrs: { id: 'v2-pp-x-min', type: 'number', step: 'any', placeholder: '下限，可留空' } });
  const xMax = el('input', { className: 'input', attrs: { id: 'v2-pp-x-max', type: 'number', step: 'any', placeholder: '上限，可留空' } });
  const rowBox = el('div', { className: 'grid grid-2' }, [
    el('div', { className: 'field' }, [el('label', { text: '起始行', attrs: { for: 'v2-pp-start-row' } }), startRow]),
    el('div', { className: 'field' }, [el('label', { text: '终止行', attrs: { for: 'v2-pp-end-row' } }), endRow]),
  ]);
  const xBox = el('div', { className: 'grid grid-2' }, [
    el('div', { className: 'field' }, [el('label', { text: isHplc ? '保留时间下限（分钟）' : 'X 下限', attrs: { for: 'v2-pp-x-min' } }), xMin]),
    el('div', { className: 'field' }, [el('label', { text: isHplc ? '保留时间上限（分钟）' : 'X 上限', attrs: { for: 'v2-pp-x-max' } }), xMax]),
  ]);
  rowBox.hidden = true;
  xBox.hidden = true;

  const extras = el('div', { className: 'stack' });
  if (kind === 'raman') {
    const baseline = el('select', { className: 'select', attrs: { id: 'v2-pp-baseline' } },
      RAMAN_BASELINE_METHODS.map((method) => el('option', { text: method === 'arPLS' ? 'arPLS（默认）' : method, attrs: { value: method } })));
    extras.append(
      el('div', { className: 'field' }, [
        el('label', { text: '基线校正方法', attrs: { for: 'v2-pp-baseline' } }),
        baseline,
        el('p', { className: 'hint', text: '顺序固定：先截取范围，再在范围内做基线校正，不能颠倒。' }),
      ]),
    );
    extras.baselineSelect = baseline;
  } else {
    const interpolate = el('input', {
      attrs: { type: 'checkbox', checked: true, id: 'v2-pp-hplc_interpolate' },
    });
    extras.append(
      el('label', { className: 'row', attrs: { for: 'v2-pp-hplc_interpolate' } }, [interpolate, ' 启用共同时间轴线性插值']),
      el('p', { className: 'hint', text: '开启时在 0–50 分钟固定目标时间轴上选择并插值；关闭时保留原始时间轴，但多文件所选轴必须完全一致，否则拒绝导出。两种模式都不做面积归一化或消负。' }),
    );
    extras.hplcInterpolate = interpolate;
  }

  const summary = el('p', { className: 'hint', attrs: { role: 'status' } });

  /**
   * 生成"范围"摘要文本。HPLC 的行号模式会实时跑 validateHplcRowRange：
   * 合法时展示"第 a–b 行，共 n 点"，非法时把校验异常信息直接展示出来（用户边输边看到错误）。
   */
  const rangeText = () => {
    if (isHplc && rangeMode.value !== 'x_value') {
      try {
        const selected = validateHplcRowRange(
          rangeMode.value === 'row_range' ? startRow.value : 1,
          rangeMode.value === 'row_range' ? endRow.value : null,
          detectedPointCount,
        );
        return `范围：第 ${selected.startRow}–${selected.endRow} 行，共 ${selected.pointCount} 点；最大终止行 ${detectedPointCount}`;
      } catch (error) {
        return `范围错误：${error?.message || '请检查 HPLC 行号'}`;
      }
    }
    if (rangeMode.value === 'row') return '范围：全部数据';
    if (rangeMode.value === 'row_range') return `范围：第 ${startRow.value || 1} 行 ～ ${endRow.value || '末尾'}`;
    return isHplc
      ? `范围：保留时间 ${xMin.value || '—'} ～ ${xMax.value || '—'} 分钟`
      : `范围：X ${xMin.value || '—'} ～ ${xMax.value || '—'}`;
  };

  /** 根据范围模式切换输入区显隐并刷新顶部摘要。 */
  const update = () => {
    rowBox.hidden = rangeMode.value !== 'row_range';
    xBox.hidden = rangeMode.value !== 'x_value';
    summary.textContent = kind === 'raman'
      ? `将执行：${rangeText()} · 基线校正 ${extras.baselineSelect.value}`
      : `将执行：${rangeText()} · 线性插值${extras.hplcInterpolate.checked ? '开' : '关'}`;
  };
  const typeControls = kind === 'raman' ? [extras.baselineSelect] : [extras.hplcInterpolate];
  for (const control of [rangeMode, startRow, endRow, xMin, xMax, ...typeControls]) {
    control.addEventListener('change', update);
    control.addEventListener('input', update);
  }
  update();

  /**
   * 收集为后端 params：range_mode 只有 'row'（行号语义）与 'x_value' 两种，
   * UI 上的"全部数据"与"按行号截取"都归到 'row'（全量即 1..末尾）。
   * HPLC 一律经 validateHplcRowRange 归一化（终止行留空按检测点数补齐，越界抛异常）。
   */
  const collect = () => {
    const params = { range_mode: rangeMode.value === 'x_value' ? 'x_value' : 'row' };
    if (rangeMode.value === 'row_range') {
      if (isHplc) {
        const selected = validateHplcRowRange(startRow.value, endRow.value, detectedPointCount);
        params.start_row = selected.startRow;
        params.end_row = selected.endRow;
      } else {
        params.start_row = Number(startRow.value) || 1;
        params.end_row = endRow.value ? Number(endRow.value) : null;
      }
    } else if (rangeMode.value === 'x_value') {
      params.x_min = xMin.value === '' ? null : Number(xMin.value);
      params.x_max = xMax.value === '' ? null : Number(xMax.value);
    } else {
      if (isHplc) {
        const selected = validateHplcRowRange(1, null, detectedPointCount);
        params.start_row = selected.startRow;
        params.end_row = selected.endRow;
      } else {
        params.start_row = 1;
        params.end_row = null;
      }
    }
    if (kind === 'raman') params.baseline_method = extras.baselineSelect.value;
    if (kind === 'hplc') params.hplc_interpolate = extras.hplcInterpolate.checked;
    return params;
  };

  const node = el('div', { className: 'stack' }, [
    summary,
    el('details', { className: 'advanced' }, [
      el('summary', { text: '高级参数（通常不用改）' }),
      el('div', { className: 'field' }, [
        el('label', { text: '数据范围', attrs: { for: 'v2-pp-range-mode' } }),
        rangeMode,
      ]),
      rowBox,
      xBox,
      extras,
    ]),
  ]);
  /**
   * 回填批次检测出的公共点数（HPLC inspect 结果）。
   * 有效（>=2 的整数）时：限制行号输入 max、默认选中 1..pointCount、终止行占位提示；
   * 无效时：清除限制与占位。之后刷新摘要。
   */
  const setPointCount = (pointCount) => {
    const parsed = Number(pointCount);
    detectedPointCount = Number.isInteger(parsed) && parsed >= 2 ? parsed : null;
    if (detectedPointCount) {
      startRow.max = String(detectedPointCount);
      endRow.max = String(detectedPointCount);
      startRow.value = '1';
      endRow.value = String(detectedPointCount);
      endRow.placeholder = `留空按 ${detectedPointCount}`;
    } else {
      startRow.removeAttribute('max');
      endRow.removeAttribute('max');
      endRow.value = '';
      endRow.placeholder = '选择文件后自动确定';
    }
    update();
  };
  return { node, collect, setPointCount };
}

/**
 * 渲染预处理结果的宽表预览（后端 preview 数组：每行一个对象）。
 * Label / Sample_ID 列此时为空，提示用户下载后人工填写。
 * @returns {HTMLElement|null} 无数据时返回 null（调用方直接跳过 append）。
 */
function previewTable(preview) {
  const rows = Array.isArray(preview) ? preview : [];
  if (!rows.length) return null;
  const columns = Object.keys(rows[0]);
  return el('div', { className: 'table-wrap' }, [
    el('table', { className: 'data-table' }, [
      el('caption', { text: '预处理样品与待填字段预览（Label、Sample_ID 待人工填写）' }),
      el('thead', {}, el('tr', {}, columns.map((column) => el('th', { text: column, attrs: { scope: 'col' } })))),
      el('tbody', {}, rows.map((row) => el('tr', {}, columns.map((column) => el('td', { text: String(row[column] ?? '') }))))),
    ]),
  ]);
}

/**
 * 渲染 HPLC 批次点数检测表：每个文件的原文件名、有效点数、时间范围与状态。
 * 时间范围用 toPrecision(8) 避免浮点长尾巴；status 非 'ready' 时展示后端 message。
 * @returns {HTMLElement|null} 无文件信息时返回 null。
 */
function hplcInspectionTable(inspection) {
  const files = Array.isArray(inspection?.files) ? inspection.files : [];
  if (!files.length) return null;
  return el('div', { className: 'table-wrap hplc-inspection-scroll' }, [
    el('table', { className: 'data-table' }, [
      el('caption', { text: 'HPLC 文件点数检测' }),
      el('thead', {}, el('tr', {}, ['原文件名', '有效点数', '时间范围（分钟）', '状态']
        .map((text) => el('th', { text, attrs: { scope: 'col' } })))),
      el('tbody', {}, files.map((item) => {
        const hasRange = Number.isFinite(Number(item.x_start)) && Number.isFinite(Number(item.x_stop));
        const range = hasRange ? `${Number(item.x_start).toPrecision(8)}–${Number(item.x_stop).toPrecision(8)}` : '—';
        const status = item.status === 'ready' ? '通过' : item.message || '异常';
        return el('tr', {}, [item.name || '—', item.point_count ?? '—', range, status]
          .map((text) => el('td', { text: String(text) })));
      })),
    ]),
  ]);
}

/**
 * 挂载"预处理工作台"视图。
 *
 * 局部状态：kind（raman/hplc）、files（当前选择的文件）、lastResult（最近一次
 * 预处理响应）、busy（请求进行中）、hplcInspection 及其 message/sequence
 * （HPLC 批次检测结果与防竞态序号）。
 *
 * @param {HTMLElement} container 视图挂载点。
 * @param {object} deps announce(msg) 读屏播报；toast(msg, opts) 全局提示。
 * @returns {{ unmount: () => void }} 卸载句柄（本视图无需要清理的计时器/全局节点，空实现）。
 */
export function mountWorkbench(container, { announce, toast }) {
  let kind = 'raman';
  let files = [];
  let lastResult = null;
  let busy = false;
  let hplcInspection = null;
  let hplcInspectionMessage = '';
  let hplcInspectionSequence = 0;

  const root = el('section', { className: 'stack view-workbench', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);

  /* 进度指示：① 上传文件 → ② 一键处理 → ③ 下载/去建模 */
  const stepItems = ['① 上传文件', '② 一键处理', '③ 下载 / 去建模'].map((label) => el('li', { className: 'step', text: label }));
  const stepsBar = el('ol', { className: 'steps' }, stepItems);
  /** 根据当前进度（无文件 / 已选文件 / 已有结果）同步三步指示器的高亮与完成态。 */
  function syncSteps() {
    const stage = lastResult ? 2 : files.length ? 1 : 0;
    stepItems.forEach((item, index) => {
      item.classList.toggle('step-done', index < stage);
      item.classList.toggle('step-active', index === stage || (lastResult && index === 2));
      if (index === stage) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
    });
  }

  const tabs = el('div', { className: 'tabs', attrs: { role: 'tablist', 'aria-label': '预处理类型' } }, [
    el('button', { className: 'tab tab-active', text: '拉曼 Raman', attrs: { type: 'button', role: 'tab', 'aria-selected': 'true' }, on: { click: () => switchKind('raman') } }),
    el('button', { className: 'tab', text: 'HPLC 色谱', attrs: { type: 'button', role: 'tab', 'aria-selected': 'false' }, on: { click: () => switchKind('hplc') } }),
  ]);

  /* 卡片 ①：上传文件 */
  const fileInput = el('input', {
    className: 'input',
    attrs: { type: 'file', multiple: true, accept: '.csv,.txt,.tsv,.dat', id: 'v2-pp-files' },
    on: {
      change: () => {
        files = Array.from(fileInput.files || []);
        // 换文件后旧的批次检测结果全部作废；递增序号使在途的 inspect 响应失效。
        hplcInspection = null;
        hplcInspectionMessage = '';
        hplcInspectionSequence += 1;
        form?.setPointCount?.(null);
        renderFileList();
        syncSteps();
        updateSubmitAvailability();
        if (kind === 'hplc' && files.length) void inspectSelectedFiles();
      },
    },
  });
  const fileList = el('div', { className: 'stack' });
  const uploadCard = el('div', { className: 'card' }, [
    el('div', { className: 'row spread' }, [
      el('h2', { className: 'card-title', text: '上传原始文件' }),
      el('span', { className: 'badge', text: '第 1 步' }),
    ]),
    el('div', { className: 'field' }, [
      el('label', { text: '选择文件（可多选）', attrs: { for: 'v2-pp-files' } }),
      fileInput,
      el('p', { className: 'hint', text: '每个文件是两列数据（X, 强度）的 CSV/文本；一次可上传多条曲线。' }),
    ]),
    fileList,
  ]);

  /** 重绘文件清单；HPLC 模式下附带批次检测状态文案与点数检测表。 */
  function renderFileList() {
    clear(fileList);
    if (!files.length) {
      fileList.append(el('div', { className: 'empty', text: '还没有选择文件。选好文件后点下方“开始预处理”即可，其余都用默认值。' }));
      return;
    }
    fileList.append(el('ul', { className: 'stack' },
      files.map((file) => el('li', { className: 'row spread' }, [
        el('span', { text: file.name }),
        el('span', { className: 'badge', text: `${(file.size / 1024).toFixed(1)} KB` }),
      ]))));
    if (kind === 'hplc' && hplcInspectionMessage) {
      fileList.append(el('p', {
        className: hplcInspection?.processable ? 'hint' : 'error-text',
        text: hplcInspectionMessage,
        attrs: { role: 'status' },
      }));
    }
    const inspectionTable = kind === 'hplc' ? hplcInspectionTable(hplcInspection) : null;
    if (inspectionTable) fileList.append(inspectionTable);
  }

  /* 卡片 ②：一键预处理（高级参数折叠） */
  const paramsHost = el('div');
  let form = paramForm(kind);
  paramsHost.append(form.node);

  const submitButton = el('button', { className: 'btn btn-primary', text: '开始预处理', attrs: { type: 'button', disabled: true }, on: { click: runPreprocess } });
  const errorBox = el('p', { className: 'error-text', attrs: { role: 'alert' } });
  const processCard = el('div', { className: 'card' }, [
    el('div', { className: 'row spread' }, [
      el('h2', { className: 'card-title', text: '一键预处理' }),
      el('span', { className: 'badge', text: '第 2 步' }),
    ]),
    el('p', { className: 'hint', text: '参数已按标准流程填好，直接点开始即可；需要截取范围或改方法时再展开高级参数。' }),
    paramsHost,
    el('div', { className: 'card-actions' }, [submitButton]),
    errorBox,
  ]);

  /* 卡片 ③：结果（处理成功后出现） */
  const resultsHost = el('div', { className: 'stack' });

  /**
   * 提交按钮门禁：请求进行中、未选文件时禁用；
   * HPLC 还必须批次检测通过（processable）才允许开始——点数不一致的批次直接挡住。
   */
  function updateSubmitAvailability() {
    submitButton.disabled = busy || !files.length || (kind === 'hplc' && !hplcInspection?.processable);
  }

  /**
   * 异步执行 HPLC 批次点数检测（inspectHplc）。
   * 防竞态双保险：sequence 序号 + 文件指纹 fileSelectionKey——
   * 请求返回时若用户已换文件或切到拉曼，则丢弃过期结果。
   * 检测通过时把公共点数回填到参数表单（驱动行号范围校验）。
   */
  async function inspectSelectedFiles() {
    const sequence = ++hplcInspectionSequence;
    const key = fileSelectionKey(files);
    hplcInspection = null;
    hplcInspectionMessage = '正在检测每个文件的有效点数…';
    form.setPointCount?.(null);
    updateSubmitAvailability();
    renderFileList();
    try {
      const result = await inspectHplc(files);
      if (sequence !== hplcInspectionSequence || key !== fileSelectionKey(files) || kind !== 'hplc') return;
      hplcInspection = result;
      hplcInspectionMessage = result.message || (result.processable ? '批次点数一致，可以开始预处理。' : '批次检查未通过。');
      form.setPointCount?.(result.processable ? result.common_point_count : null);
    } catch (error) {
      if (sequence !== hplcInspectionSequence) return;
      hplcInspection = null;
      hplcInspectionMessage = `检测失败：${error?.message || '未知错误'}`;
    }
    updateSubmitAvailability();
    renderFileList();
  }

  /**
   * 切换拉曼 / HPLC：重建参数表单（两种类型控件不同）、清空旧结果与错误、
   * 作废在途批次检测；切到 HPLC 且已有文件时立即重新检测。
   * busy 中禁止切换，避免请求中途状态错乱。
   */
  function switchKind(next) {
    if (next === kind || busy) return;
    kind = next;
    for (const tab of tabs.children) {
      const active = tab.textContent.includes(next === 'raman' ? 'Raman' : 'HPLC');
      tab.classList.toggle('tab-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    }
    clear(paramsHost);
    form = paramForm(kind);
    paramsHost.append(form.node);
    clear(resultsHost);
    lastResult = null;
    errorBox.textContent = '';
    hplcInspection = null;
    hplcInspectionMessage = '';
    hplcInspectionSequence += 1;
    renderFileList();
    updateSubmitAvailability();
    if (kind === 'hplc' && files.length) void inspectSelectedFiles();
    syncSteps();
    announce(`已切换到 ${next === 'raman' ? '拉曼' : 'HPLC'} 预处理`);
  }

  /**
   * 执行预处理：前置门禁（文件、HPLC 批次检测）→ collect 参数（校验失败就地提示）→
   * 调 preprocess 接口。期间 busy 禁用按钮并显示骨架屏；成功渲染结果卡片，
   * 失败在错误框给出带"下一步"指引的文案。
   */
  async function runPreprocess() {
    errorBox.textContent = '';
    if (!files.length) {
      errorBox.textContent = '请先在第 1 步选择至少一个原始数据文件，然后再点开始预处理。';
      fileInput.focus();
      return;
    }
    if (kind === 'hplc' && !hplcInspection?.processable) {
      errorBox.textContent = hplcInspection?.message || '请等待 HPLC 文件点数检测通过后再开始预处理。';
      return;
    }
    let params;
    try {
      params = form.collect();
    } catch (error) {
      errorBox.textContent = error?.message || 'HPLC 行号范围无效，请检查后重试。';
      announce('预处理参数校验未通过');
      return;
    }
    busy = true;
    updateSubmitAvailability();
    submitButton.textContent = '处理中…';
    clear(resultsHost);
    resultsHost.append(el('div', { className: 'card' }, [
      el('div', { className: 'skeleton' }),
      el('div', { className: 'skeleton' }),
      el('p', { className: 'hint', text: '正在处理并生成对照曲线…', attrs: { role: 'status' } }),
    ]));
    announce('预处理请求已提交');
    try {
      const result = await preprocess(kind, files, params);
      lastResult = result;
      renderResults();
      syncSteps();
      announce(`预处理完成，共 ${result.rows ?? files.length} 行数据`);
      toast('预处理完成。下一步：下载统一 CSV，填写 Label / Sample_ID 后去建模。', {
        type: 'success',
        action: { label: '下载 CSV', onClick: () => downloadResult() },
      });
    } catch (error) {
      clear(resultsHost);
      errorBox.textContent = `预处理失败：${error?.message || '未知错误'}。下一步：确认每个文件都是两列数值数据（X, 强度）；多文件轴不一致时请先对齐，HPLC 也可启用共同时间轴插值。`;
      announce('预处理失败');
    } finally {
      busy = false;
      updateSubmitAvailability();
      submitButton.textContent = '开始预处理';
    }
  }

  /** 按 download_url 拉取结果 Blob 并触发浏览器保存；文件名缺失时用兜底名。 */
  async function downloadResult() {
    if (!lastResult?.download_url) return;
    try {
      const { blob, filename } = await download(lastResult.download_url);
      saveBlob(blob, filename || `${kind}_preprocessed.csv`);
    } catch (error) {
      toast(`下载失败：${error?.message || '未知错误'}。请重试，或刷新页面后重新处理。`, { type: 'error' });
    }
  }

  /**
   * 渲染第 3 步结果卡片：警告面板（如有）+ 指标卡 + 曲线对照预览 + 宽表预览 + 下载/去建模按钮。
   *
   * 展示的契约信息：
   * - output_precision：wide-feature-v2 格式名、特征数、总列数。
   * - HPLC 专属：插值开关、X 轴一致性、实际点数与完整网格点数、真实保留时间轴摘要
   *   （真实坐标逐列写在 CSV 特征表头，这是建模读取坐标轴的依据）。
   * - intensity_summary：每条曲线处理后的点数/最小/最大/均值；全零时提示检查范围或方法。
   */
  function renderResults() {
    clear(resultsHost);
    if (!lastResult) return;
    const curves = Array.isArray(lastResult.curves) ? lastResult.curves : [];

    const chartHost = el('div', { className: 'chart-card' });
    const infoLine = el('p', { className: 'hint' });
    /** 渲染第 index 条曲线的对照图与强度摘要。 */
    const renderCurve = (index) => {
      const curve = curves[index];
      clear(chartHost);
      if (!curve) {
        infoLine.textContent = '';
        return;
      }
      const { svg, legend } = curveChart(curve, kind);
      chartHost.append(el('div', { className: 'scroll-x' }, [svg]), legend);
      const info = (lastResult.intensity_summary || []).find((item) => item?.name === curve.name);
      infoLine.textContent = info
        ? `点数 ${info.point_count}，最小 ${info.min?.toFixed?.(4) ?? '—'}，最大 ${info.max?.toFixed?.(4) ?? '—'}，均值 ${info.mean?.toFixed?.(4) ?? '—'}${info.all_zero ? '；处理后强度全为 0，请在高级参数中检查范围或方法' : ''}`
        : '';
    };

    const curveSelect = el('select', {
      className: 'select',
      attrs: { id: 'v2-pp-curve', 'aria-label': '选择要查看的曲线' },
      on: { change: () => renderCurve(Number(curveSelect.value) || 0) },
    }, curves.map((curve, index) => el('option', { text: curve.name || `曲线 ${index + 1}`, attrs: { value: String(index) } })));

    const metaCards = [
      ['类型', kind === 'raman' ? '拉曼' : 'HPLC'],
      ['输出行数', String(lastResult.rows ?? '—')],
      ['数据量', String(curves.length)],
    ];
    if (lastResult.output_precision) {
      metaCards.push(['输出格式', lastResult.output_precision.format || 'wide-feature-v2']);
      metaCards.push(['特征数', String(lastResult.output_precision.feature_count ?? '—')]);
      metaCards.push(['总列数', String(lastResult.output_precision.total_column_count ?? '—')]);
    }
    if (kind === 'raman') metaCards.push(['基线方法', lastResult.baseline_method || '—']);
    if (kind === 'hplc') {
      metaCards.push(['插值', lastResult.hplc_interpolate ? '开' : '关']);
      metaCards.push(['X 轴', lastResult.x_axis_consistent === false ? '不一致' : '共同轴已验证']);
      if (lastResult.hplc_axis) {
        metaCards.push(['实际点数', String(lastResult.hplc_axis.point_count ?? '—')]);
        metaCards.push(['完整网格点数', String(lastResult.hplc_axis.grid_point_count ?? lastResult.hplc_axis.input_point_count_required ?? '—')]);
      }
    }

    const axis = kind === 'hplc' ? lastResult.hplc_axis : null;
    /** 轴端点数值格式化：6 位小数后去尾零；非有限值显示 '—'。 */
    const axisNumber = (value) => {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(6).replace(/\.?0+$/, '') : '—';
    };
    const axisSummary = axis
      ? `XXX：真实保留时间 ${axisNumber(axis.start)}–${axisNumber(axis.stop)} 分钟，第 ${axis.selected_start_row ?? '—'}–${axis.selected_end_row ?? '—'} 点，共 ${axis.point_count ?? '—'} 点；每个真实时间坐标逐列保存在 CSV 特征表头中。`
      : null;

    const warnings = Array.isArray(lastResult.warnings) ? lastResult.warnings : [];
    const warningPanel = warnings.length
      ? el('div', { className: 'card warning-panel', attrs: { role: 'status' } }, [
        el('h2', { className: 'card-title', text: 'X 轴提示' }),
        el('ul', { className: 'warning-list' }, warnings.map((warning) => el('li', { text: String(warning) }))),
      ])
      : null;

    const downloadButton = el('button', {
      className: 'btn btn-primary',
      text: '下载统一建模 CSV',
      attrs: { type: 'button', disabled: lastResult.download_url ? null : true },
      on: { click: downloadResult },
    });
    if (warningPanel) resultsHost.append(warningPanel);
    resultsHost.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: '下载结果，去建模' }),
        el('span', { className: 'badge status-succeeded', text: '第 3 步 · 已完成处理' }),
      ]),
      el('div', { className: 'metric-grid' }, metaCards.map(([label, value]) => el('div', { className: 'metric-card' }, [
        el('span', { className: 'metric-label', text: label }),
        el('span', { className: 'metric-value', text: value }),
      ]))),
      axisSummary ? el('p', { className: 'hint', text: axisSummary }) : null,
      curves.length
        ? el('div', { className: 'field' }, [el('label', { text: kind === 'hplc' ? '查看色谱曲线' : '查看曲线对照（灰虚线原始 / 绿线校正后）', attrs: { for: 'v2-pp-curve' } }), curveSelect])
        : null,
      curves.length ? chartHost : el('div', { className: 'empty', text: '响应中没有曲线预览，可直接下载 CSV 查看数据。' }),
      infoLine,
      previewTable(lastResult.preview),
      el('p', { className: 'hint', text: '下一步：下载 CSV 后只需填写 Label 和 Sample_ID 两列；Name 已保留每行原文件名，请不要改动。第 5 列起是真实坐标表头，然后到建模页上传训练。' }),
      el('div', { className: 'card-actions' }, [
        downloadButton,
        el('a', { className: 'btn btn-ghost', text: '填写后去建模页 →', attrs: { href: '#/modeling' } }),
      ]),
    ]));
    if (curves.length) renderCurve(0);
  }

  root.append(
    stepsBar,
    tabs,
    uploadCard,
    processCard,
    resultsHost,
  );
  renderFileList();
  updateSubmitAvailability();
  syncSteps();

  return {
    unmount() {},
  };
}
