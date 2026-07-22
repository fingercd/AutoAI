/** AI 建模向导：四步门禁（数据 → 评估口径 → 模型 → 参数确认提交），每步只保留必要选择。 */
import { el, clear } from '../lib/dom.js';
import { EVALUATION_STRATEGIES } from '../lib/format.js';
import { uploadDataset, getModels, createRun } from '../api.js';
import { renderModelCatalog, findModel } from '../components/model-catalog.js';
import { markRunCreated, consumeModelingDraft } from '../store.js';
import { naturalCompare, renderSampleIdList } from '../../js/ui-utils.js';

const STEPS = ['上传数据', '评估口径', '选择模型', '参数与提交'];
const REDIRECT_SECONDS = 3;
let summaryListSerial = 0;

function datasetSummaryCard({ name, summary, reuse }) {
  const sampleSummary = summary?.sample_id || {};
  const metrics = [
    ['数据量', summary?.samples ?? '—'],
    ['类别数', summary?.classes ?? '—'],
    ['样本数', sampleSummary.group_count ?? sampleSummary.groups?.length ?? '—'],
    ['每样本测量数', sampleSummary.expected_repeats_per_group ?? '—'],
    ['特征数', summary?.curve_length ?? '—'],
  ];
  const labelRows = Object.entries(summary?.label_counts || {})
    .sort(([first], [second]) => naturalCompare(first, second));
  const groupsHost = el('div');
  renderSampleIdList(groupsHost, sampleSummary.groups || [], {
    listId: `v2-sample-groups-${summaryListSerial += 1}`,
  });
  const card = el('div', { className: 'card' }, [
    el('div', { className: 'row spread' }, [
      el('h3', { className: 'card-title', text: name || '—' }),
      reuse ? el('span', { className: 'badge status-queued', text: '沿用原 Run 数据集' }) : el('span', { className: 'badge status-succeeded', text: '校验通过' }),
    ]),
    el('div', { className: 'metric-grid' }, metrics.map(([label, value]) => el('div', { className: 'metric-card' }, [
      el('span', { className: 'metric-label', text: label }),
      el('span', { className: 'metric-value', text: String(value) }),
    ]))),
    el('h4', { text: '按样本分组' }),
    groupsHost,
    el('h4', { text: '类别分布' }),
    labelRows.length
      ? el('div', { className: 'table-wrap' }, [
        el('table', { className: 'data-table' }, [
          el('thead', {}, el('tr', {}, [
            el('th', { text: '类别', attrs: { scope: 'col' } }),
            el('th', { text: '数据量', attrs: { scope: 'col' } }),
          ])),
          el('tbody', {}, labelRows.map(([label, count]) => el('tr', {}, [
            el('td', { text: label }),
            el('td', { text: String(count) }),
          ]))),
        ]),
      ])
      : el('p', { className: 'hint', text: '没有可展示的类别统计。' }),
    el('p', { className: 'hint', text: 'Label 始终按分类处理；同一 Sample_ID 整组划分，不会跨 train/valid/test。' }),
  ]);
  return card;
}

/** 单个 CSV 上传区：选文件 → 点“上传并校验”，结果卡片就地展示。 */
function uploadBox({ inputId, buttonClass, onUploaded }) {
  const input = el('input', { className: 'input', attrs: { type: 'file', accept: '.csv', id: inputId } });
  const button = el('button', { className: buttonClass, text: '上传并校验', attrs: { type: 'button' } });
  const status = el('p', { className: 'hint', attrs: { role: 'status' } });
  const errorBox = el('p', { className: 'error-text', attrs: { role: 'alert' } });
  const host = el('div', { className: 'stack' });
  button.addEventListener('click', async () => {
    errorBox.textContent = '';
    const file = input.files?.[0];
    if (!file) {
      errorBox.textContent = '请先选择 CSV 文件，再点“上传并校验”。';
      input.focus();
      return;
    }
    button.disabled = true;
    status.textContent = '上传并校验中…';
    try {
      const result = await uploadDataset(file);
      status.textContent = '';
      clear(host);
      host.append(datasetSummaryCard({ name: result.dataset_name, summary: result.summary }));
      onUploaded(result);
    } catch (error) {
      status.textContent = '';
      errorBox.textContent = `上传失败：${error?.message || '未知错误'}。下一步：新文件应为 Index、Label、Sample_ID、Name 四个元数据列，第 5 列起为严格递增的真实数值坐标；旧无 Name 宽表仍兼容。`;
    } finally {
      button.disabled = false;
    }
  });
  return { node: el('div', { className: 'stack' }, [
    el('div', { className: 'field' }, [
      el('label', { text: '选择 CSV 文件', attrs: { for: inputId } }),
      input,
    ]),
    el('div', { className: 'card-actions' }, [button]),
    status,
    errorBox,
    host,
  ]), host };
}

export function mountModeling(container, { announce, toast, navigate }) {
  const draft = consumeModelingDraft();
  const wizard = {
    step: 1,
    datasetId: draft?.datasetId || null,
    datasetName: draft?.datasetName || null,
    datasetSummary: null,
    datasetReuse: Boolean(draft?.datasetId),
    testDatasetId: draft?.testDatasetId || null,
    testDatasetName: draft?.testDatasetName || null,
    testSummary: null,
    testReuse: Boolean(draft?.testDatasetId),
    strategy: draft?.config?.split_mode && EVALUATION_STRATEGIES[draft.config.split_mode]
      ? draft.config.split_mode
      : 'stratified_holdout',
    models: null,
    modelsError: null,
    modelId: draft?.config?.model_type || null,
    params: {
      epochs: Number(draft?.config?.epochs) || 200,
      batch_size: Number(draft?.config?.batch_size) || 8,
      learning_rate: Number(draft?.config?.learning_rate) || 0.001,
      normalization: draft?.config?.normalization || 'zscore',
      seed: draft?.config?.seed ?? '',
    },
    countdownTimer: null,
    successDialog: null,
  };

  const root = el('section', { className: 'stack view-modeling', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);
  const stepper = el('ol', { className: 'steps' });
  const body = el('div', { className: 'stack' });
  if (draft) {
    root.append(el('div', { className: 'card' }, [
      el('p', { className: 'hint', text: `已从 Run ${draft.sourceRunId || ''} 复制训练配置，请核对数据与参数后再提交。` }),
    ]));
  }
  root.append(stepper, body);

  function cleanupCountdown() {
    if (wizard.countdownTimer) {
      clearInterval(wizard.countdownTimer);
      wizard.countdownTimer = null;
    }
    if (wizard.successDialog) {
      wizard.successDialog.remove();
      wizard.successDialog = null;
    }
  }

  function renderStepper() {
    clear(stepper);
    STEPS.forEach((label, index) => {
      const number = index + 1;
      stepper.append(el('li', {
        className: `step${number === wizard.step ? ' step-active' : ''}${number < wizard.step ? ' step-done' : ''}`,
        attrs: { 'aria-current': number === wizard.step ? 'step' : null },
        text: `${number}. ${label}`,
      }));
    });
  }

  function navButtons({ prev = true, nextLabel = '下一步', onNext, nextDisabled = false, nextHint = '' }) {
    return el('div', { className: 'card-actions' }, [
      prev ? el('button', { className: 'btn btn-ghost', text: '上一步', attrs: { type: 'button' }, on: { click: () => goto(wizard.step - 1) } }) : null,
      onNext ? el('button', { className: 'btn btn-primary', text: nextLabel, attrs: { type: 'button', disabled: nextDisabled || null }, on: { click: onNext } }) : null,
      nextDisabled && nextHint ? el('span', { className: 'hint', text: nextHint }) : null,
    ]);
  }

  function goto(step) {
    wizard.step = Math.min(4, Math.max(1, step));
    render();
  }

  function render() {
    renderStepper();
    clear(body);
    if (wizard.step === 1) renderStep1();
    if (wizard.step === 2) renderStep2();
    if (wizard.step === 3) renderStep3();
    if (wizard.step === 4) renderStep4();
  }

  function renderStep1() {
    const mainBox = uploadBox({
      inputId: 'v2-up-main',
      buttonClass: 'btn btn-primary',
      onUploaded: (result) => {
        wizard.datasetId = result.dataset_id;
        wizard.datasetName = result.dataset_name;
        wizard.datasetSummary = result.summary;
        wizard.datasetReuse = false;
        render();
      },
    });
    const testBox = uploadBox({
      inputId: 'v2-up-test',
      buttonClass: 'btn btn-ghost',
      onUploaded: (result) => {
        wizard.testDatasetId = result.dataset_id;
        wizard.testDatasetName = result.dataset_name;
        wizard.testSummary = result.summary;
        wizard.testReuse = false;
        render();
      },
    });
    body.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: '上传建模 CSV' }),
        el('span', { className: 'badge', text: '第 1 步，共 4 步' }),
      ]),
      el('p', { className: 'hint', text: '使用预处理工作台下载并填写 Label / Sample_ID 后的 wide-feature-v2 CSV：前三列固定为 Index、Label、Sample_ID，第 4 列 Name 保留原文件名，第 5 列起是真实 XXX 坐标；旧 wide-feature-v1 仍兼容。' }),
      wizard.datasetId
        ? el('div', { className: 'stack' }, [
          wizard.datasetSummary
            ? datasetSummaryCard({ name: wizard.datasetName, summary: wizard.datasetSummary, reuse: wizard.datasetReuse })
            : el('p', { className: 'hint', text: `当前数据集：${wizard.datasetName || wizard.datasetId}（沿用，可重新上传替换）` }),
        ])
        : null,
      wizard.datasetSummary ? el('p', { className: 'hint', text: '需要更换数据集？重新选择文件并上传校验即可替换。' }) : null,
      mainBox.node,
      el('details', { className: 'advanced' }, [
        el('summary', { text: '可选：独立测试 CSV（必须与主数据使用完全相同的真实坐标表头）' }),
        wizard.testDatasetId
          ? el('p', { className: 'hint', text: `当前独立测试集：${wizard.testDatasetName || wizard.testDatasetId}` })
          : null,
        testBox.node,
      ]),
      navButtons({
        prev: false,
        onNext: () => goto(2),
        nextDisabled: !wizard.datasetId,
        nextHint: '上传并校验主 CSV 后才能进入下一步。',
      }),
    ]));
  }

  function renderStep2() {
    if (wizard.strategy === 'external_test_holdout' && !wizard.testDatasetId) {
      wizard.strategy = 'stratified_holdout';
    }
    const group = el('div', { className: 'stack', attrs: { role: 'radiogroup', 'aria-label': '评估口径' } });
    for (const [key, meta] of Object.entries(EVALUATION_STRATEGIES)) {
      const needsTest = key === 'external_test_holdout';
      const disabled = needsTest && !wizard.testDatasetId;
      const radio = el('input', {
        attrs: {
          type: 'radio', name: 'v2-strategy', value: key,
          checked: wizard.strategy === key ? true : null,
          disabled: disabled ? true : null,
          id: `v2-strategy-${key}`,
        },
        on: { change: () => { wizard.strategy = key; } },
      });
      group.append(el('label', { className: 'card', attrs: { for: `v2-strategy-${key}` } }, [
        el('span', { className: 'row' }, [radio, el('strong', { text: meta.label })]),
        el('span', { className: 'hint', text: meta.description }),
        disabled ? el('span', { className: 'error-text', text: '需要先在第 1 步的“可选”区域上传独立测试 CSV。' }) : null,
      ]));
    }
    body.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: '选择评估口径' }),
        el('span', { className: 'badge', text: '第 2 步，共 4 步' }),
      ]),
      el('p', { className: 'hint', text: '默认“分层留出”适合大多数情况，直接下一步即可。三种口径互斥；交叉验证的 Test 主指标由全部折的测试预测合并计算，不会与逐折均值混用。' }),
      group,
      navButtons({ onNext: () => goto(3) }),
    ]));
  }

  function renderStep3() {
    const host = el('div', { className: 'stack' });
    const nextButton = el('button', {
      className: 'btn btn-primary',
      text: '下一步',
      attrs: { type: 'button', disabled: wizard.modelId ? null : true },
      on: { click: () => { if (wizard.modelId) goto(4); } },
    });
    body.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: '选择模型' }),
        el('span', { className: 'badge', text: '第 3 步，共 4 步' }),
      ]),
      el('p', { className: 'hint', text: '标有“可训练”的模型都可以直接用；不确定就选第一个可训练模型。不可用模型已折叠，不会静默替代。' }),
      host,
      el('div', { className: 'card-actions' }, [
        el('button', { className: 'btn btn-ghost', text: '上一步', attrs: { type: 'button' }, on: { click: () => goto(2) } }),
        nextButton,
        wizard.modelId ? null : el('span', { className: 'hint', text: '请先选择一个可训练模型。' }),
      ]),
    ]));
    if (wizard.modelsError) {
      host.append(
        el('p', { className: 'error-text', attrs: { role: 'alert' }, text: `模型目录加载失败：${wizard.modelsError}。请检查网络后重试。` }),
        el('button', { className: 'btn btn-ghost', text: '重试', attrs: { type: 'button' }, on: { click: () => { wizard.models = null; wizard.modelsError = null; render(); } } }),
      );
      return;
    }
    if (wizard.models) {
      renderModelCatalog(host, {
        models: wizard.models,
        selectedId: wizard.modelId,
        onSelect: (id) => { wizard.modelId = id; nextButton.disabled = false; },
      });
      return;
    }
    host.append(
      el('div', { className: 'skeleton' }),
      el('div', { className: 'skeleton' }),
      el('p', { className: 'hint', attrs: { role: 'status' }, text: '正在加载模型目录…' }),
    );
    getModels()
      .then((result) => {
        wizard.models = Array.isArray(result?.models) ? result.models : [];
        if (wizard.modelId && !findModel(wizard.models, wizard.modelId)?.available) {
          wizard.modelId = null;
        }
      })
      .catch((error) => {
        wizard.modelsError = error?.message || '模型目录加载失败';
      })
      .finally(() => {
        if (wizard.step === 3 && !wizard.models && !wizard.modelsError) return;
        if (wizard.step === 3) render();
      });
  }

  function renderStep4() {
    const model = findModel(wizard.models, wizard.modelId);
    const inputs = {
      epochs: el('input', { className: 'input', attrs: { type: 'number', min: '1', step: '1', value: String(wizard.params.epochs), id: 'v2-p-epochs' } }),
      batch_size: el('input', { className: 'input', attrs: { type: 'number', min: '1', step: '1', value: String(wizard.params.batch_size), id: 'v2-p-batch' } }),
      learning_rate: el('input', { className: 'input', attrs: { type: 'number', min: '0', step: 'any', value: String(wizard.params.learning_rate), id: 'v2-p-lr' } }),
      normalization: el('select', { className: 'select', attrs: { id: 'v2-p-norm' } },
        ['zscore', 'minmax', 'area', 'none'].map((value) => el('option', { text: value, attrs: { value, selected: wizard.params.normalization === value ? true : null } }))),
      seed: el('input', { className: 'input', attrs: { type: 'number', step: '1', value: wizard.params.seed === '' ? '' : String(wizard.params.seed), placeholder: '留空使用默认', id: 'v2-p-seed' } }),
    };
    const splitInfo = {
      stratified_holdout: 'train/valid/test 目标 8:1:1（按 Sample_ID 整组；Valid/Test 每类至少 1 个）',
      leave_one_sample_id_cv: '每折留 1 个 Sample_ID 作 test，其余 8:2；Test 主指标为合并交叉验证预测',
      external_test_holdout: '主数据 8:2 划分 train/valid，独立测试集作 test',
    }[wizard.strategy];
    const summaryRows = [
      ['主数据集', wizard.datasetName || wizard.datasetId || '—'],
      ['独立测试集', wizard.strategy === 'external_test_holdout' ? (wizard.testDatasetName || wizard.testDatasetId || '—') : '不使用'],
      ['评估口径', EVALUATION_STRATEGIES[wizard.strategy].shortLabel],
      ['划分', splitInfo],
      ['模型', model ? `${model.display_name}（${model.id}）` : wizard.modelId || '—'],
    ];
    const errorBox = el('p', { className: 'error-text', attrs: { role: 'alert' } });
    const submit = el('button', { className: 'btn btn-primary', text: '提交训练', attrs: { type: 'button' } });
    const syncParams = () => {
      wizard.params = {
        epochs: Number(inputs.epochs.value) || wizard.params.epochs,
        batch_size: Number(inputs.batch_size.value) || wizard.params.batch_size,
        learning_rate: Number(inputs.learning_rate.value) || wizard.params.learning_rate,
        normalization: inputs.normalization.value || wizard.params.normalization,
        seed: inputs.seed.value === '' ? '' : Number(inputs.seed.value),
      };
    };
    const paramsHint = el('p', { className: 'hint' });
    const updateParamsHint = () => {
      paramsHint.textContent = `训练参数：epochs ${inputs.epochs.value || 200} · batch_size ${inputs.batch_size.value || 8} · learning_rate ${inputs.learning_rate.value || 0.001} · normalization ${inputs.normalization.value}${inputs.seed.value === '' ? '' : ` · seed ${inputs.seed.value}`}（默认值已适合大多数情况）`;
    };
    for (const input of Object.values(inputs)) {
      input.addEventListener('change', updateParamsHint);
      input.addEventListener('input', updateParamsHint);
    }
    updateParamsHint();
    submit.addEventListener('click', async () => {
      errorBox.textContent = '';
      syncParams();
      const config = {
        model_type: wizard.modelId,
        epochs: Number(inputs.epochs.value),
        batch_size: Number(inputs.batch_size.value),
        learning_rate: Number(inputs.learning_rate.value),
        normalization: inputs.normalization.value,
        split_mode: wizard.strategy,
      };
      if (inputs.seed.value !== '') config.seed = Number(inputs.seed.value);
      if (!config.model_type) {
        errorBox.textContent = '还没有选择模型。请返回第 3 步选择一个可训练模型。';
        return;
      }
      submit.disabled = true;
      submit.textContent = '提交中…';
      try {
        const result = await createRun({
          dataset_id: wizard.datasetId,
          test_dataset_id: wizard.strategy === 'external_test_holdout' ? wizard.testDatasetId : null,
          config,
        });
        markRunCreated(result.run_id);
        toast(`Run ${result.run_id} 已入队（queued），等待 worker 执行。`, {
          type: 'success',
          action: { label: '查看进度', onClick: () => goResult(result.run_id) },
        });
        announce(`训练已入队：${result.run_id}`);
        renderSuccess(result);
      } catch (error) {
        errorBox.textContent = `创建失败：${error?.message || '未知错误'}。下一步：检查网络与登录状态后重新点击提交；参数本身已保留。`;
      } finally {
        submit.disabled = false;
        submit.textContent = '提交训练';
      }
    });
    body.append(el('div', { className: 'card' }, [
      el('div', { className: 'row spread' }, [
        el('h2', { className: 'card-title', text: '确认并提交' }),
        el('span', { className: 'badge', text: '第 4 步，共 4 步' }),
      ]),
      el('div', { className: 'table-wrap' }, [
        el('table', { className: 'data-table' }, [
          el('caption', { text: '提交前确认' }),
          el('tbody', {}, summaryRows.map(([key, value]) => el('tr', {}, [
            el('th', { text: key, attrs: { scope: 'row' } }),
            el('td', { text: String(value) }),
          ]))),
        ]),
      ]),
      paramsHint,
      el('details', { className: 'advanced' }, [
        el('summary', { text: '高级训练参数（默认即可，通常不用改）' }),
        el('div', { className: 'grid grid-3' }, [
          el('div', { className: 'field' }, [el('label', { text: 'epochs（深度模型上限 200）', attrs: { for: 'v2-p-epochs' } }), inputs.epochs]),
          el('div', { className: 'field' }, [el('label', { text: 'batch_size', attrs: { for: 'v2-p-batch' } }), inputs.batch_size]),
          el('div', { className: 'field' }, [el('label', { text: 'learning_rate', attrs: { for: 'v2-p-lr' } }), inputs.learning_rate]),
          el('div', { className: 'field' }, [el('label', { text: 'normalization', attrs: { for: 'v2-p-norm' } }), inputs.normalization]),
          el('div', { className: 'field' }, [el('label', { text: 'seed（可选）', attrs: { for: 'v2-p-seed' } }), inputs.seed]),
        ]),
      ]),
      el('p', { className: 'hint', text: '提交只会创建 queued Run，由独立 worker 异步执行；成功响应不代表训练已开始或完成。' }),
      el('div', { className: 'card-actions' }, [
        el('button', { className: 'btn btn-ghost', text: '上一步', attrs: { type: 'button' }, on: { click: () => { syncParams(); goto(3); } } }),
        submit,
      ]),
      errorBox,
    ]));
  }

  function goResult(runId) {
    cleanupCountdown();
    navigate(`#/results?run_id=${encodeURIComponent(runId)}`);
  }

  function renderSuccess(result) {
    cleanupCountdown();
    clear(body);
    clear(stepper);
    const runId = result.run_id;
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    const countdownText = el('p', { className: 'hint', attrs: { 'aria-live': 'polite' } });
    const dialog = el('div', { className: 'dialog-backdrop', attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'v2-created-title' } }, [
      el('div', { className: 'auth-card' }, [
        el('h2', { text: 'Run 已入队', attrs: { id: 'v2-created-title' } }),
        el('p', {}, ['Run ID：', el('code', { text: runId }), '，当前状态 ', el('span', { className: 'badge status-queued', text: 'queued（排队中）' }), '。']),
        el('p', { className: 'hint', text: '训练由独立 worker 执行；该响应不代表训练已开始或完成。' }),
        countdownText,
        el('div', { className: 'card-actions' }, [
          el('button', { className: 'btn btn-primary', text: '立即查看进度', attrs: { type: 'button' }, on: { click: () => goResult(runId) } }),
          el('button', { className: 'btn btn-ghost', text: '留在本页', attrs: { type: 'button' }, on: { click: stay } }),
        ]),
      ]),
    ]);
    function stay() {
      cleanupCountdown();
    }
    dialog.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') stay();
    });
    body.append(
      el('div', { className: 'card' }, [
        el('div', { className: 'row spread' }, [
          el('h2', { className: 'card-title', text: '训练任务已创建' }),
          el('span', { className: 'badge status-queued', text: 'queued' }),
        ]),
        el('p', {}, ['Run ID：', el('code', { text: runId })]),
        el('p', { className: 'hint', text: '状态：queued。可在训练记录或建模结果页查看进度。' }),
        warnings.length ? el('div', { className: 'stack' }, warnings.map((warning) => el('p', { className: 'error-text', text: `注意：${warning}` }))) : null,
        el('div', { className: 'card-actions' }, [
          el('button', { className: 'btn btn-primary', text: '查看进度', attrs: { type: 'button' }, on: { click: () => goResult(runId) } }),
          el('button', { className: 'btn btn-ghost', text: '再建一个 Run', attrs: { type: 'button' }, on: { click: () => { cleanupCountdown(); wizard.step = 1; render(); } } }),
        ]),
      ]),
    );
    document.body.append(dialog);
    wizard.successDialog = dialog;
    dialog.querySelector('button.btn-primary')?.focus();
    let remaining = REDIRECT_SECONDS;
    countdownText.textContent = `${remaining} 秒后自动跳转到建模结果（仅本次新建 Run 自动跳转）`;
    wizard.countdownTimer = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        goResult(runId);
        return;
      }
      countdownText.textContent = `${remaining} 秒后自动跳转到建模结果（仅本次新建 Run 自动跳转）`;
    }, 1000);
  }

  render();

  return {
    unmount() {
      cleanupCountdown();
    },
  };
}
