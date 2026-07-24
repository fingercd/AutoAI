/**
 * 【模块说明】AI 建模向导视图（static/v2 工作台）
 *
 * 职责：渲染"AI 建模"四步向导页——上传数据 → 评估口径 → 选择模型 → 参数确认提交。
 * 这是 v2 前端（static/v2/index.html）的一个视图组件，由 Hash 路由 #/modeling 挂载。
 *
 * 系统位置与协作：
 * - 通过 ../api.js 调用后端接口：uploadDataset（POST /api/datasets 上传并校验 CSV）、
 *   getModels（GET /api/training/models 模型能力目录）、createRun（POST /api/training/runs 创建 queued Run）。
 * - 通过 ../store.js 的 consumeModelingDraft 读取"从历史 Run 复制配置"的草稿（训练记录页跳转过来时携带），
 *   markRunCreated 记录本次会话新建的 run_id（用于训练记录页高亮/识别）。
 * - 通过 ../components/model-catalog.js 渲染模型目录卡片（含 available=false 的不可用模型折叠展示）。
 * - EVALUATION_STRATEGIES 来自 ../lib/format.js，定义三种评估口径的展示文案：
 *   stratified_holdout（分层留出 8:1:1）、leave_one_sample_id_cv（留一样本交叉验证）、
 *   external_test_holdout（独立测试集，需要第 1 步上传独立测试 CSV）。
 *
 * 关键设计约束（与后端契约一致）：
 * - 训练 HTTP 请求只创建 queued Run，不直接启动训练；成功响应不代表训练已开始或完成（见 renderSuccess）。
 * - 上传的 CSV 必须满足 wide-feature-v2 宽表契约：Index, Label, Sample_ID, Name 四个元数据列 +
 *   第 5 列起严格递增的真实数值坐标表头；旧 wide-feature-v1（无 Name 列）仍兼容。
 * - external_test_holdout 必须携带 test_dataset_id；未上传独立测试集时该口径禁用并自动回退。
 * - Label 始终按分类处理；同一 Sample_ID 整组划分，不会跨 train/valid/test。
 */
import { el, clear } from '../lib/dom.js';
import { EVALUATION_STRATEGIES } from '../lib/format.js';
import { uploadDataset, getModels, createRun } from '../api.js';
import { renderModelCatalog, findModel } from '../components/model-catalog.js';
import { markRunCreated, consumeModelingDraft } from '../store.js';
import { naturalCompare, renderSampleIdList } from '../../js/ui-utils.js';

/** 向导四步的展示标题，顺序即路由内步骤顺序。 */
const STEPS = ['上传数据', '评估口径', '选择模型', '参数与提交'];
/** 提交成功后自动跳转到结果页前的倒计时秒数。 */
const REDIRECT_SECONDS = 3;
/**
 * 单调递增序号，为每张数据集摘要卡片的 Sample_ID 列表生成页面内唯一的 listId，
 * 避免同一页多张卡片的 aria/id 冲突。
 */
let summaryListSerial = 0;

/**
 * 渲染数据集校验摘要卡片。
 *
 * 用途：上传并校验成功（或沿用了历史 Run 数据集草稿）后，就地展示后端返回的数据集
 * 统计信息，让用户在提交前确认数据规模、样本分组与类别分布。
 *
 * @param {object} options
 * @param {string} options.name 数据集显示名（通常是上传的文件名）。
 * @param {object|null} options.summary 后端 upload 接口返回的 summary：
 *   samples（数据量/行数）、classes（类别数）、curve_length（特征数）、
 *   label_counts（各类别数据量字典）、sample_id（Sample_ID 分组信息：
 *   group_count / groups / expected_repeats_per_group）。
 * @param {boolean} [options.reuse] 为 true 时表示沿用历史 Run 的数据集（展示"沿用原 Run 数据集"
 *   徽章），否则展示"校验通过"徽章。
 * @returns {HTMLElement} 组装好的卡片节点。
 * 边界：summary 中任一字段缺失时显示 '—'；label_counts 为空时显示提示文案而不是空表。
 */
function datasetSummaryCard({ name, summary, reuse }) {
  const sampleSummary = summary?.sample_id || {};
  // 五项核心指标；任一缺失用 '—' 占位，保证卡片结构稳定。
  const metrics = [
    ['数据量', summary?.samples ?? '—'],
    ['类别数', summary?.classes ?? '—'],
    ['样本数', sampleSummary.group_count ?? sampleSummary.groups?.length ?? '—'],
    ['每样本测量数', sampleSummary.expected_repeats_per_group ?? '—'],
    ['特征数', summary?.curve_length ?? '—'],
  ];
  // 类别分布按类别名自然序排序（naturalCompare 保证 "class2" 排在 "class10" 前）。
  const labelRows = Object.entries(summary?.label_counts || {})
    .sort(([first], [second]) => naturalCompare(first, second));
  const groupsHost = el('div');
  // Sample_ID 分组列表单独渲染到宿主节点；listId 用递增序号保证全页唯一。
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

/**
 * 构造单个 CSV 上传区（选文件 → 点"上传并校验"，结果卡片就地展示）。
 *
 * 主数据集与独立测试集各用一个实例。上传成功后会清空宿主区域并替换为
 * datasetSummaryCard，同时通过 onUploaded 回调把后端结果回写给向导状态。
 *
 * @param {object} options
 * @param {string} options.inputId 文件 input 的 id（label 的 for 属性需要对应，保证无障碍）。
 * @param {string} options.buttonClass 按钮样式类（主数据集用主按钮，测试集用次要按钮）。
 * @param {(result: object) => void} options.onUploaded 上传校验成功回调，
 *   result 为后端响应：{ dataset_id, dataset_name, summary }。
 * @returns {{ node: HTMLElement, host: HTMLElement }} node 为完整上传区节点，
 *   host 为结果卡片的宿主容器（外部一般只用 node）。
 * 边界：未选文件直接点按钮时给出错误提示并聚焦 input；上传失败时错误文案中
 * 附带 wide-feature-v2 格式要求作为"下一步"指引。
 */
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
    // 上传期间禁用按钮防重复提交；finally 中恢复。
    button.disabled = true;
    status.textContent = '上传并校验中…';
    try {
      const result = await uploadDataset(file);
      status.textContent = '';
      // 成功后替换掉旧的结果卡片（重新上传即替换数据集）。
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

/**
 * 挂载"AI 建模"视图（四步向导），v2 路由进入 #/modeling 时调用。
 *
 * 状态集中在 wizard 对象：当前步骤、主/测试数据集 id 与摘要、评估口径、
 * 模型目录与所选模型、训练参数、提交成功后的倒计时与对话框引用。
 * 每次状态变化通过 render() 全量重绘 body（简单可靠，向导数据量小）。
 *
 * @param {HTMLElement} container 视图挂载点。
 * @param {object} deps 由 v2 外壳注入的依赖：
 *   announce(msg) 向屏幕阅读器播报；toast(msg, opts) 弹全局提示；
 *   navigate(hash) 进行 Hash 路由跳转。
 * @returns {{ unmount: () => void }} 卸载句柄；路由离开时调用，清理倒计时与成功对话框。
 */
export function mountModeling(container, { announce, toast, navigate }) {
  // 从训练记录页"复制配置"跳转过来时，草稿里带有原 Run 的 datasetId/config 等。
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
    // 草稿里的 split_mode 若不在当前能力目录的三种口径内则忽略，回退默认分层留出。
    strategy: draft?.config?.split_mode && EVALUATION_STRATEGIES[draft.config.split_mode]
      ? draft.config.split_mode
      : 'stratified_holdout',
    models: null,
    modelsError: null,
    modelId: draft?.config?.model_type || null,
    // 参数默认值与后端约定一致：epochs 200 / batch_size 8 / lr 0.001 / zscore；seed 留空表示用默认。
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

  /**
   * 清理提交成功后的自动跳转倒计时与模态对话框。
   * 在跳转、留页、卸载、再次创建 Run 等任何脱离成功态的路径上都必须调用，
   * 避免计时器在组件销毁后仍触发导航。
   */
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

  /** 重绘顶部步骤条：当前步高亮（aria-current="step"），已过步标记完成态。 */
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

  /**
   * 构造每步底部的"上一步 / 下一步"按钮行。
   * @param {object} options prev=false 隐藏上一步（第 1 步）；nextDisabled 时可用
   *   nextHint 在按钮旁解释为何不能前进（门禁式向导：不满足条件不给走）。
   */
  function navButtons({ prev = true, nextLabel = '下一步', onNext, nextDisabled = false, nextHint = '' }) {
    return el('div', { className: 'card-actions' }, [
      prev ? el('button', { className: 'btn btn-ghost', text: '上一步', attrs: { type: 'button' }, on: { click: () => goto(wizard.step - 1) } }) : null,
      onNext ? el('button', { className: 'btn btn-primary', text: nextLabel, attrs: { type: 'button', disabled: nextDisabled || null }, on: { click: onNext } }) : null,
      nextDisabled && nextHint ? el('span', { className: 'hint', text: nextHint }) : null,
    ]);
  }

  /** 跳转到指定步骤并钳制在 1..4，随后全量重绘。 */
  function goto(step) {
    wizard.step = Math.min(4, Math.max(1, step));
    render();
  }

  /** 全量重绘：步骤条 + 当前步骤内容。 */
  function render() {
    renderStepper();
    clear(body);
    if (wizard.step === 1) renderStep1();
    if (wizard.step === 2) renderStep2();
    if (wizard.step === 3) renderStep3();
    if (wizard.step === 4) renderStep4();
  }

  /**
   * 第 1 步：上传建模 CSV（必传）+ 可选独立测试 CSV（折叠在 details 里）。
   * 门禁：未上传并校验主 CSV 前禁用"下一步"。
   * 草稿沿用场景：已有 datasetId 但没有新 summary 时只显示沿用提示，可重新上传替换。
   */
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

  /**
   * 第 2 步：选择评估口径（三种互斥单选）。
   * external_test_holdout 依赖第 1 步上传的独立测试集：未上传时该选项禁用；
   * 若当前恰好选中它（例如草稿带入），自动回退为 stratified_holdout，避免提交无效配置。
   */
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

  /**
   * 第 3 步：选择模型。模型目录异步拉取（getModels），有三种 UI 状态：
   * 加载中（骨架屏）、失败（错误 + 重试按钮）、成功（renderModelCatalog 渲染卡片）。
   * 门禁：未选模型时"下一步"禁用；目录加载后若草稿带入的 modelId 已不可用
   * （available=false，例如 cnn_mamba1d 缺依赖），自动清空选择而不是静默替代。
   */
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
    // 状态一：目录加载失败 —— 显示错误与重试（重试清空缓存状态后重新 render 触发再次拉取）。
    if (wizard.modelsError) {
      host.append(
        el('p', { className: 'error-text', attrs: { role: 'alert' }, text: `模型目录加载失败：${wizard.modelsError}。请检查网络后重试。` }),
        el('button', { className: 'btn btn-ghost', text: '重试', attrs: { type: 'button' }, on: { click: () => { wizard.models = null; wizard.modelsError = null; render(); } } }),
      );
      return;
    }
    // 状态二：目录已就绪 —— 渲染模型卡片；选中后即刻解锁"下一步"。
    if (wizard.models) {
      renderModelCatalog(host, {
        models: wizard.models,
        selectedId: wizard.modelId,
        onSelect: (id) => { wizard.modelId = id; nextButton.disabled = false; },
      });
      return;
    }
    // 状态三：尚未拉取 —— 骨架屏 + 发起异步请求。
    host.append(
      el('div', { className: 'skeleton' }),
      el('div', { className: 'skeleton' }),
      el('p', { className: 'hint', attrs: { role: 'status' }, text: '正在加载模型目录…' }),
    );
    getModels()
      .then((result) => {
        wizard.models = Array.isArray(result?.models) ? result.models : [];
        // 草稿带入的模型当前不可用时清空选择，强制用户重新选可训练模型。
        if (wizard.modelId && !findModel(wizard.models, wizard.modelId)?.available) {
          wizard.modelId = null;
        }
      })
      .catch((error) => {
        wizard.modelsError = error?.message || '模型目录加载失败';
      })
      .finally(() => {
        // 只在用户仍停留在第 3 步时重绘；若期间已切走则丢弃本次结果渲染。
        if (wizard.step === 3 && !wizard.models && !wizard.modelsError) return;
        if (wizard.step === 3) render();
      });
  }

  /**
   * 第 4 步：参数确认与提交。
   * 上半部分是只读确认表（数据集/口径/划分方式/模型），高级训练参数折叠在 details 里；
   * 底部"提交训练"调用 createRun 创建 queued Run，成功后进入 renderSuccess。
   *
   * 提交契约要点：只传 dataset_id / test_dataset_id（external_test_holdout 时才传）与 config；
   * 不在浏览器侧传 data_path（server 模式安全约束）。seed 留空时不出现在 config 中。
   */
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
    // 各评估口径的划分方式说明，展示在确认表中帮助用户最终核对。
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
    // 把输入框当前值同步回 wizard.params；无效数字回退到原值，保证往返第 3 步不丢参数。
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
    // 实时摘要行：参数修改即时反映，未填项回退默认值展示。
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
      // seed 可选：留空则不传，由后端使用默认随机种子。
      if (inputs.seed.value !== '') config.seed = Number(inputs.seed.value);
      if (!config.model_type) {
        errorBox.textContent = '还没有选择模型。请返回第 3 步选择一个可训练模型。';
        return;
      }
      submit.disabled = true;
      submit.textContent = '提交中…';
      try {
        // 只创建 queued Run；test_dataset_id 仅在独立测试口径下传递，其余口径传 null。
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

  /** 清理倒计时/对话框后跳转到专属结果页（Hash 路由契约：#/results?run_id=...）。 */
  function goResult(runId) {
    cleanupCountdown();
    navigate(`#/results?run_id=${encodeURIComponent(runId)}`);
  }

  /**
   * 提交成功后的成功态：body 内展示成功卡片，同时弹出模态对话框，
   * 并在 REDIRECT_SECONDS 倒计时后自动跳转到结果页。
   * "立即查看进度"立即跳转；"留在本页"/Esc 仅取消倒计时与对话框。
   * 后端 warnings（如有）逐条展示在成功卡片里。
   */
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
    // 焦点移入对话框主按钮，保证键盘/读屏用户立即可操作。
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
