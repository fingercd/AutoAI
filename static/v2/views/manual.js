/** 使用手册：短 FAQ，覆盖数据格式、评估口径、队列语义、令牌与下载边界。 */
/*
 * 模块说明
 * ========
 * 本文件是 v2 工作台的“使用手册”视图（纯静态内容，无网络请求、无交互状态）。
 *
 * 在系统中的位置：
 * - 属于 static/v2 原生 JS 前端的 views 层，由路由层调用 `mountManual(container)` 挂载。
 * - 只依赖 `lib/dom.js` 的 `el` 助手构建 DOM，是三个视图中最简单的一个。
 *
 * 内容定位：
 * - 以 FAQ 形式向用户解释平台的关键业务契约，文案必须与后端真实行为一致：
 *   wide-feature-v2 宽表格式、Sample_ID 整组划分、三种评估口径、queued 队列语义、
 *   Bearer 令牌边界、artifact 下载白名单、ROC/PR 暂不可用等。
 * - 修改后端规则（如宽表契约、评估口径、下载白名单）时应同步更新这里的说明，
 *   否则手册会误导用户。
 */
import { el } from '../lib/dom.js';

/**
 * 构造一条 FAQ 卡片。
 *
 * @param {string} question 问题标题，渲染为卡片标题（h2）。
 * @param {...HTMLElement} children 正文内容（段落、列表、表格等），按序追加。
 * @returns {HTMLElement} `<section class="card">`，每条 FAQ 独立成卡，便于扫读。
 */
function faq(question, ...children) {
  return el('section', { className: 'card' }, [
    el('h2', { className: 'card-title', text: question }),
    ...children,
  ]);
}

/**
 * 挂载“使用手册”视图。
 *
 * 全部内容一次性静态渲染，没有任何异步逻辑或事件监听；
 * 因此返回的 unmount 是空操作，DOM 清理由路由层统一处理。
 *
 * @param {HTMLElement} container 路由层提供的挂载容器。
 * @returns {{ unmount(): void }} 与其他视图保持一致的卸载接口（此处为空实现）。
 */
export function mountManual(container) {
  const root = el('div', { className: 'stack', attrs: { 'aria-labelledby': 'v2-view-title' } });
  container.append(root);

  root.append(
    el('p', { className: 'page-sub', text: '常见问题速查；每条只讲一件事，看完即可回到工作台操作。' }),

    // 以下每条 faq 对应平台的一条业务契约，文案按后端当前实际行为撰写：
    // wide-feature-v2 宽表列结构、Sample_ID 整组划分、三种评估口径、
    // 异步队列语义、CV 主指标口径、Run 操作边界、令牌存储边界、下载白名单、
    // 以及“没有真实产物就不展示”的 ROC/PR 原则。
    faq('建模 CSV 需要什么格式？',
      el('p', { text: '预处理新导出使用 wide-feature-v2 宽表。前四列名称与顺序固定，第 5 列起每一列代表一个真实坐标：' }),
      el('div', { className: 'table-wrap' }, [
        el('table', { className: 'data-table' }, [
          el('thead', {}, el('tr', {}, ['列名', '含义', '要求'].map((head) => el('th', { text: head, attrs: { scope: 'col' } })))),
          el('tbody', {}, [
            ['Index', '曲线标识', '第 1 列；不可为空'],
            ['Label', '类别标签', '第 2 列；必填，即使为数字也始终按分类类别处理'],
            ['Sample_ID', '样品组标识', '第 3 列；同一组重复测量必须使用相同 Sample_ID'],
            ['Name', '原始文件名', '第 4 列；预处理自动写入，保留扩展名'],
            ['第 5 列起', '真实 XXX 坐标 / 标量 Intensity', '列名必须是有限、唯一、严格递增的数值；每个单元格必须是有限强度值'],
          ].map((row) => el('tr', {}, row.map((cell) => el('td', { text: cell }))))),
        ]),
      ]),
      el('p', { className: 'hint', text: '预处理导出的 CSV 中 Label 与 Sample_ID 为空，需要人工填写后再上传；Name 已写入各自原始文件名。已有 wide-feature-v1（三列元数据）仍可训练，但新的预处理结果统一输出 v2。' }),
      el('p', { className: 'hint', text: '所有行必须共享同一真实坐标轴；HPLC 关闭插值时若多文件轴不一致会拒绝导出。旧的六列数组/JSON CSV 不再可训练。Excel 最多 16,384 列，v2 扣除四列元数据后最多支持 16,380 个特征。' }),
    ),

    faq('为什么 Sample_ID 必须整组划分？',
      el('p', { text: '同一 Sample_ID 只能对应一个 Label；划分 train/valid/test 时按 Sample_ID 整组进行，同一样品的重复测量不会泄漏到不同集合。这是避免“同一样品既在训练又在测试”导致指标虚高的关键约束。' }),
    ),

    faq('三种评估口径怎么选？',
      el('ul', {}, [
        el('li', { text: 'stratified_holdout：按 Sample_ID 整组、以 8:1:1 为目标划分；Valid/Test 至少各包含每类 1 个样品组，因此每类至少需要 3 个不同 Sample_ID。' }),
        el('li', { text: 'leave_one_sample_id_cv：每折留一个 Sample_ID 作 test，适合小样本；Test 主指标由全部交叉验证折的测试预测合并计算，逐折均值只作审计。' }),
        el('li', { text: 'external_test_holdout：主数据 8:2 划分 train/valid，独立测试集作为最终 test；与交叉验证互斥。' }),
      ]),
    ),

    faq('提交训练后为什么还要等？',
      el('p', { text: '提交只创建 queued Run，由独立 worker 进程异步执行；HTTP 202 不代表训练已开始或完成。状态以记录为准：queued → running → succeeded / failed / cancelled。' }),
      el('p', { text: '结果页每 3 秒自动刷新一次，进入终态自动停止。若 worker 不可用或版本不兼容，创建 Run 会被拒绝，需要同时重启 Web 与 Worker。' }),
    ),

    faq('交叉验证的 Test 指标看哪个？',
      el('p', { text: '只看主指标卡：它由全部交叉验证折的测试预测合并计算。逐折均值（fold mean ± std）放在“评估与审计明细”里仅供审计，两者口径不同，不能混用比较。' }),
    ),

    faq('如何复用、取消或删除 Run？',
      el('ul', {}, [
        el('li', { text: '复制配置：把该 Run 的训练配置带回建模向导，可改参数后再次提交。' }),
        el('li', { text: '取消：仅排队中 / 运行中的 Run 可取消。' }),
        el('li', { text: '删除：仅终态 Run 可删除，记录与产物一并移除，不可恢复。' }),
      ]),
    ),

    faq('为什么有时要求输入服务器令牌？',
      el('p', { text: '服务器模式下，除 /static、/health 与 /api/auth/config 外的 API 都需要 Bearer 令牌。令牌只保存在当前标签页的 sessionStorage，关闭标签页即失效；不会写入 URL、localStorage、日志或下载文件。令牌失效时会弹出认证框，输入后被中断的请求自动重试。' }),
    ),

    faq('为什么有些文件不能下载？',
      el('p', { text: '结果下载由后端 catalog 逐项决定，每个文件都经过大小与 SHA-256 校验；禁用项会显示原因，不提供打包下载。' }),
      el('p', { text: '以下内容不开放：model.pkl、model.pt 等模型对象/权重、*.joblib 内部拟合对象、status.json 易变投影、manifest.json 内部索引；config.json 仅在不含服务器路径时开放。' }),
      el('p', { className: 'hint', text: '预处理只提供一个统一建模 CSV，通过响应中的 download_url 下载；响应里的 output_path 等服务器路径不会也不应作为浏览器链接使用。' }),
    ),

    faq('为什么没有 ROC / PR 曲线？',
      el('p', { text: '只有真实计算出的分析产物才会展示；当前训练没有 ROC / Precision-Recall 产物时，结果页会在“暂不可用的分析”中说明原因，不绘制空图、不伪造数值。' }),
    ),
  );

  return { unmount() {} };
}
