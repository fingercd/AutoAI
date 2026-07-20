/** 预处理工作台：三张卡走完全程（上传文件 → 一键处理 → 下载/去建模），高级参数默认折叠。 */
import { el, clear, saveBlob, svgEl } from '../lib/dom.js';
import { lineChart } from '../lib/charts.js';
import { preprocess, download } from '../api.js';

const RAMAN_BASELINE_METHODS = ['arPLS', 'airPLS', 'als', 'drPLS', 'poly'];
const COLOR_RAW = '#94a3b8';
const COLOR_PROCESSED = '#0f766e';

function processedSeries(curve) {
  return curve?.processed_y ?? curve?.corrected_y ?? null;
}

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
      xLabel: kind === 'hplc' ? '保留时间' : '拉曼位移 / X',
      yLabel: '强度',
    }),
    legend: legendList(series),
  };
}

/** 范围截取 + 各类型的方法/开关，收进高级参数并附带实时摘要。 */
function paramForm(kind) {
  const rangeMode = el('select', { className: 'select', attrs: { id: 'v2-pp-range-mode' } }, [
    el('option', { text: '全部数据（不截取）', attrs: { value: 'row' } }),
    el('option', { text: '按行号截取', attrs: { value: 'row_range' } }),
    el('option', { text: '按 X 轴数值截取', attrs: { value: 'x_value' } }),
  ]);
  const startRow = el('input', { className: 'input', attrs: { id: 'v2-pp-start-row', type: 'number', min: '1', step: '1', value: '1' } });
  const endRow = el('input', { className: 'input', attrs: { id: 'v2-pp-end-row', type: 'number', min: '1', step: '1', placeholder: '留空到末尾' } });
  const xMin = el('input', { className: 'input', attrs: { id: 'v2-pp-x-min', type: 'number', step: 'any', placeholder: '下限，可留空' } });
  const xMax = el('input', { className: 'input', attrs: { id: 'v2-pp-x-max', type: 'number', step: 'any', placeholder: '上限，可留空' } });
  const rowBox = el('div', { className: 'grid grid-2' }, [
    el('div', { className: 'field' }, [el('label', { text: '起始行', attrs: { for: 'v2-pp-start-row' } }), startRow]),
    el('div', { className: 'field' }, [el('label', { text: '结束行', attrs: { for: 'v2-pp-end-row' } }), endRow]),
  ]);
  const xBox = el('div', { className: 'grid grid-2' }, [
    el('div', { className: 'field' }, [el('label', { text: 'X 下限', attrs: { for: 'v2-pp-x-min' } }), xMin]),
    el('div', { className: 'field' }, [el('label', { text: 'X 上限', attrs: { for: 'v2-pp-x-max' } }), xMax]),
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
      el('p', { className: 'hint', text: '开启时映射到服务端固定时间轴；关闭时保留所选原始 X/Y，轴不一致仍会生成 CSV 并提示；不做面积归一化或消负。' }),
    );
    extras.hplcInterpolate = interpolate;
  }

  const summary = el('p', { className: 'hint', attrs: { role: 'status' } });

  const rangeText = () => {
    if (rangeMode.value === 'row') return '范围：全部数据';
    if (rangeMode.value === 'row_range') return `范围：第 ${startRow.value || 1} 行 ～ ${endRow.value || '末尾'}`;
    return `范围：X ${xMin.value || '—'} ～ ${xMax.value || '—'}`;
  };

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

  const collect = () => {
    const params = { range_mode: rangeMode.value === 'x_value' ? 'x_value' : 'row' };
    if (rangeMode.value === 'row_range') {
      params.start_row = Number(startRow.value) || 1;
      params.end_row = endRow.value ? Number(endRow.value) : null;
    } else if (rangeMode.value === 'x_value') {
      params.x_min = xMin.value === '' ? null : Number(xMin.value);
      params.x_max = xMax.value === '' ? null : Number(xMax.value);
    } else {
      params.start_row = 1;
      params.end_row = null;
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
  return { node, collect };
}

function previewTable(preview) {
  const rows = Array.isArray(preview) ? preview : [];
  if (!rows.length) return null;
  const columns = Object.keys(rows[0]);
  return el('div', { className: 'table-wrap' }, [
    el('table', { className: 'data-table' }, [
      el('caption', { text: '统一 CSV 前 5 行预览（Label 与 Sample_ID 待人工补齐）' }),
      el('thead', {}, el('tr', {}, columns.map((column) => el('th', { text: column, attrs: { scope: 'col' } })))),
      el('tbody', {}, rows.map((row) => el('tr', {}, columns.map((column) => el('td', { text: String(row[column] ?? '') }))))),
    ]),
  ]);
}

export function mountWorkbench(container, { announce, toast }) {
  let kind = 'raman';
  let files = [];
  let lastResult = null;
  let busy = false;

  const root = el('section', { className: 'stack view-workbench', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);

  /* 进度指示：① 上传文件 → ② 一键处理 → ③ 下载/去建模 */
  const stepItems = ['① 上传文件', '② 一键处理', '③ 下载 / 去建模'].map((label) => el('li', { className: 'step', text: label }));
  const stepsBar = el('ol', { className: 'steps' }, stepItems);
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
        renderFileList();
        syncSteps();
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
  }

  /* 卡片 ②：一键预处理（高级参数折叠） */
  const paramsHost = el('div');
  let form = paramForm(kind);
  paramsHost.append(form.node);

  const submitButton = el('button', { className: 'btn btn-primary', text: '开始预处理', attrs: { type: 'button' }, on: { click: runPreprocess } });
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
    syncSteps();
    announce(`已切换到 ${next === 'raman' ? '拉曼' : 'HPLC'} 预处理`);
  }

  async function runPreprocess() {
    errorBox.textContent = '';
    if (!files.length) {
      errorBox.textContent = '请先在第 1 步选择至少一个原始数据文件，然后再点开始预处理。';
      fileInput.focus();
      return;
    }
    busy = true;
    submitButton.disabled = true;
    submitButton.textContent = '处理中…';
    clear(resultsHost);
    resultsHost.append(el('div', { className: 'card' }, [
      el('div', { className: 'skeleton' }),
      el('div', { className: 'skeleton' }),
      el('p', { className: 'hint', text: '正在处理并生成对照曲线…', attrs: { role: 'status' } }),
    ]));
    announce('预处理请求已提交');
    try {
      const result = await preprocess(kind, files, form.collect());
      lastResult = result;
      renderResults();
      syncSteps();
      announce(`预处理完成，共 ${result.rows ?? files.length} 行数据`);
      toast('预处理完成。下一步：下载统一 CSV，补齐 Label / Sample_ID 后去建模。', {
        type: 'success',
        action: { label: '下载 CSV', onClick: () => downloadResult() },
      });
    } catch (error) {
      clear(resultsHost);
      errorBox.textContent = `预处理失败：${error?.message || '未知错误'}。下一步：确认每个文件都是两列数值数据（X, 强度）后重试；仍失败请在高级参数中改用“全部数据”。`;
      announce('预处理失败');
    } finally {
      busy = false;
      submitButton.disabled = false;
      submitButton.textContent = '开始预处理';
    }
  }

  async function downloadResult() {
    if (!lastResult?.download_url) return;
    try {
      const { blob, filename } = await download(lastResult.download_url);
      saveBlob(blob, filename || `${kind}_preprocessed.csv`);
    } catch (error) {
      toast(`下载失败：${error?.message || '未知错误'}。请重试，或刷新页面后重新处理。`, { type: 'error' });
    }
  }

  async function downloadVisibleAxis() {
    if (!lastResult?.xxx_download_url) return;
    try {
      const { blob, filename } = await download(lastResult.xxx_download_url);
      saveBlob(blob, filename || 'hplc_xxx.csv');
    } catch (error) {
      toast(`XXX 时间轴下载失败：${error?.message || '未知错误'}。请重试。`, { type: 'error' });
    }
  }

  function renderResults() {
    clear(resultsHost);
    if (!lastResult) return;
    const curves = Array.isArray(lastResult.curves) ? lastResult.curves : [];

    const chartHost = el('div', { className: 'chart-card' });
    const infoLine = el('p', { className: 'hint' });
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
      ['曲线数', String(curves.length)],
    ];
    if (kind === 'raman') metaCards.push(['基线方法', lastResult.baseline_method || '—']);
    if (kind === 'hplc') {
      metaCards.push(['插值', lastResult.hplc_interpolate ? '开' : '关']);
      metaCards.push(['X 轴', lastResult.x_axis_consistent ? '一致' : '不一致']);
      if (lastResult.hplc_axis) {
        metaCards.push(['固定点数', String(lastResult.hplc_axis.point_count ?? '—')]);
      }
    }

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
    const axisDownloadButton = kind === 'hplc' && lastResult.xxx_download_url
      ? el('button', {
        className: 'btn btn-secondary',
        text: '下载可见 XXX 时间轴',
        attrs: { type: 'button' },
        on: { click: downloadVisibleAxis },
      })
      : null;

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
      curves.length
        ? el('div', { className: 'field' }, [el('label', { text: kind === 'hplc' ? '查看色谱曲线' : '查看曲线对照（灰虚线原始 / 绿线校正后）', attrs: { for: 'v2-pp-curve' } }), curveSelect])
        : null,
      curves.length ? chartHost : el('div', { className: 'empty', text: '响应中没有曲线预览，可直接下载 CSV 查看数据。' }),
      infoLine,
      previewTable(lastResult.preview),
      el('p', { className: 'hint', text: '下一步：下载 CSV 后用表格软件补齐 Label 和 Sample_ID 两列（同一 Sample_ID 的重复测量会整组划分，不会跨 train/valid/test），然后到建模页上传训练。' }),
      el('div', { className: 'card-actions' }, [
        downloadButton,
        axisDownloadButton,
        el('a', { className: 'btn btn-ghost', text: '补齐后去建模页 →', attrs: { href: '#/modeling' } }),
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
  syncSteps();

  return {
    unmount() {},
  };
}
