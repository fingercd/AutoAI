# 混淆矩阵 PNG 下载与模型特征可视化实施计划

> 日期：2026-08-20
> 状态：待评审、尚未实施
> 目标项目：SpecAutoAI / `D:\PythonProject\AutoAI`

## 1. 目标与成功标准

本轮交付包含两个可独立验收的功能：

1. Train、Valid、Test 三个混淆矩阵各自提供“下载 PNG”按钮，浏览器直接生成并下载清晰、白底、含标题/类别轴/计数/口径说明的 PNG，不依赖后端截图或第三方图表库。
2. 在建模结果页“重要性分析”之后增加“模型特征可视化”，只为具有模型原生、可审计低维表征的模型生成对应图；不为所有模型统一套 PCA/t-SNE，也不把描述性图误写成泛化证据。

完成后的可观察结果：

- 经典正式结果页和并行 v2 结果页均能下载三个分区的混淆矩阵 PNG。
- `pls_da` 的 holdout 结果展示 PLS 潜变量得分图、R²Y/Q² 标签置换检验、VIP 图。
- `pca_lda` 展示其 Pipeline 内真实 PCA 步骤的得分图；`pca_mlp` 展示其真实 PCA 输入表征。
- `random_forest` 展示基于森林叶节点 proximity 的 MDS 二维图和层次聚类树。
- 其他模型与 `leave_one_sample_id_cv` 首版显示明确“不支持/暂不可用”原因，不显示空图或伪造图。
- 新图的数据能追溯到新 artifact，Manifest 包含 SHA-256/大小校验，`run-result-v1` 不暴露本机路径。

## 2. 图片辨认与方法结论

用户提供的第一张组合图可辨认为化学计量学常见输出：

- 左上：PCA 得分图，坐标常写为 `t[1]`、`t[2]`，属于无监督降维描述。
- 右上：PLS-DA/OPLS-DA 得分图，类别标签参与潜变量构造，属于监督降维描述。
- 左下：标签置换检验图，比较原模型与随机置换标签模型的 R²/Q²。
- 右下：VIP（Variable Importance in Projection）变量重要性柱图；示例中带误差条通常还依赖 jackknife/bootstrap，首版不能在没有重采样估计时伪造误差条。

第二张图是样品层次聚类树（dendrogram），不是随机森林中的一棵决策树。为了让它与当前分类模型存在真实联系，本计划不对原始谱线直接做通用欧氏聚类，而是对 `random_forest` 使用 Breiman 定义的森林 proximity：两样品落入同一叶节点的树占比作为相似度，再进行 MDS 和平均连接层次聚类。

关键证据：

- scikit-learn 的 [`PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) 明确提供 `x_scores_`、`x_weights_` 与 `transform()`，可直接导出拟合模型的潜变量坐标。
- scikit-learn 的 [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 提供 `transform()` 与 `explained_variance_ratio_`，可从现有 `pca_lda` Pipeline 原样导出。
- Westerhuis 等关于 [PLS-DA 交叉验证与置换检验](https://link.springer.com/article/10.1007/s11306-007-0099-6) 的研究指出，校准得分图容易显得过度乐观，类别差异不能只靠得分图下结论；因此页面必须把得分图与 Test 指标、置换检验并列解释。
- [Can We Trust Score Plots?](https://www.mdpi.com/2218-1989/10/7/278) 进一步说明，当 R² 与 Q² 差异很大时，校准得分图可能误导；不同交叉验证折的坐标还存在符号/旋转对齐问题。
- Breiman 的 [Random Forests 使用说明](https://www.stat.berkeley.edu/~breiman/Using_random_forests_V3.1.pdf) 定义了 leaf proximity，并明确将其用于 metric scaling 与聚类可视化。
- PCA/PLS 得分空间生成聚类树有先例，但原始研究也强调距离选择和统计验证的重要性：[PCAtoTree](https://pmc.ncbi.nlm.nih.gov/articles/PMC3534867/)。

## 3. 模型—可视化适配矩阵

| 当前模型 | 首版图形 | 数据来源 | 支持口径 | 不支持/限制说明 |
|---|---|---|---|---|
| `pls_da` | PLS 得分图、R²Y/Q² 置换检验、VIP | 已拟合 `PLSDAClassifier.model` 的 `transform/x_weights_/x_scores_/y_loadings_` | `stratified_holdout`、`external_test_holdout` | 当前项目没有 OPLS-DA，标题必须写 PLS-DA；CV 不拼接跨折坐标 |
| `pca_lda` | PCA 得分图 | Pipeline 的 `named_steps['pca']` | 两种 holdout | 只描述 PCA 空间，不把点间分离写成 LDA 泛化性能 |
| `pca_mlp` | PCA 输入得分图 | 模型真实 `pca_model` | 两种 holdout | 不是 MLP 隐层嵌入；标题需明确“PCA 输入” |
| `random_forest` | proximity-MDS、proximity dendrogram | `RandomForestClassifier.apply(X)` 的叶节点索引 | 两种 holdout | 树是森林相似性聚类，不是单棵决策树 |
| `logistic_regression`、`svm`、`xgboost` | 不生成 | — | — | 没有与示例等价的原生二维潜变量；首版不使用通用 PCA/t-SNE 冒充模型解释 |
| CNN/ResNet/Inception/TCN/Transformer/DSCARNet 等 | 不生成 | — | — | 需要单独定义并验证 penultimate embedding 与跨折对齐，超出本轮范围；现有单样品重要性继续保留 |
| `cnn_mamba1d` | 不生成 | — | — | 模型本身 `available=false` |

### 交叉验证门禁

`leave_one_sample_id_cv` 首版返回 `status="unsupported"`，原因写入 artifact 和页面：不同折分别拟合的 PCA/PLS/MDS 方向存在符号、旋转和尺度不唯一；直接把各折坐标合并在同一二维平面会产生伪结构。后续若要支持，应另做“跨折 Procrustes 对齐 + OOF 坐标稳定性”方案并单独验收，不能复用本轮 holdout 实现。

## 4. 总体实现方案

### 4.1 混淆矩阵 PNG

新增一个经典与 v2 共用的无依赖前端模块。下载时直接根据 `run-result-v1.analysis.splits.{train,valid,test}.confusion_matrix` 在离屏 `<canvas>` 重绘，而不是截取 DOM：

- 2× 像素密度，白色背景；下载后在微信、Word、PPT 中仍可直接使用。
- 标题包含 Run ID 与 Train/Valid/Test；副标题包含 `direct`、`pooled_oof` 或 `pooled_cross_fold` 的中文口径。
- 行为真实类别、列为预测类别；单元格写计数，对角线绿色、误测红色，颜色强度与当前页面一致。
- 布局按最长类别名和矩阵阶数动态计算；对超长类别名截断显示但保留互不覆盖的轴标签。
- 文件名固定为 `run_<安全RunID>__confusion_matrix_<train|valid|test>.png`；只允许字母、数字、`-`、`_`，避免路径字符或控制字符进入下载名。
- Canvas 不可用或 `toBlob()` 失败时，在矩阵卡片下的 `aria-live` 状态区给出失败原因，不影响结果页其他区域。

该功能是纯前端派生下载，不进入 Manifest；下载的数据与页面已经展示的混淆矩阵完全同源。

### 4.2 模型特征可视化 artifact

新增 `model_feature_visualization.json`，建议 schema：

```json
{
  "schema_version": "model-feature-visualization-v1",
  "status": "ready|unsupported|unavailable",
  "model_type": "pls_da",
  "evaluation_strategy": "stratified_holdout",
  "scope": "holdout_fitted_model",
  "reason": null,
  "warning": "得分图只描述模型表征，不能单独证明泛化能力",
  "plots": [
    {
      "id": "pls_scores",
      "type": "scatter",
      "title": "PLS-DA 潜变量得分图",
      "x_label": "PLS latent variable 1",
      "y_label": "PLS latent variable 2",
      "points": [
        {"index": 0, "name": "S1", "sample_id": "1", "label": "A", "split": "train", "x": 0.1, "y": -0.2}
      ]
    }
  ]
}
```

原则：

- 所有数值落盘前检查为有限数，不允许 NaN/Infinity 污染 JSON。
- 点记录只含业务元数据和坐标，不含 `data_path`、`run_dir` 等服务器路径。
- `unsupported` 也生成该 JSON，使历史审计与前端空态原因一致。
- artifact 在 `ARTIFACT_CATALOG` 中登记为可选、可下载、`category="analysis"`；结果页同时将通过 `analysis.model_feature_visualization` 内联读取这个小型载荷，不要求用户再点击下载才能看图。
- `run-result-v1` 只在 Manifest 条目存在且完整性状态为 `ok` 时读取该文件；损坏时返回 unavailable 原因并追加 warning。

### 4.3 各算法的精确定义

#### PLS 得分图

- 使用最终 train+valid 拟合模型的 `PLSRegression.transform()` 投影 Train/Valid/Test。
- 只使用前两个潜变量；不足两个有效成分时该图返回 unavailable，不用随机抖动制造第二轴。
- 颜色编码真实类别，点形编码 split；外部 Test 只做 transform，不参与成分拟合。
- 置信边界只使用最终拟合池（train+valid）的二维得分计算整体 95% 协方差椭圆，并在图注中明确这是描述性边界，不是显著性检验；样本不足或协方差奇异时省略椭圆。

#### PLS 标签置换检验

- 使用选参阶段的 train-only 标准化矩阵和固定 Train/Valid 划分，绝不使用 Test。
- 将 Train+Valid 池标签整体随机置换后，按原索引拆回 Train/Valid；每次用固定的已选 `n_components` 重拟合 one-hot `PLSRegression`。
- R²Y 在 Train 上计算；Q² 在未参与拟合的 Valid 上计算，分母基准使用 Train one-hot 标签均值。
- 默认 50 次、随机种子沿用 Run `seed`；artifact 记录实际次数、原模型 R²Y/Q²、所有置换点和经验 `p=(1+#Q²_perm>=Q²_original)/(B+1)`。
- 置换次数不进入前端训练表单，避免扩大本轮配置面；后端用具名常量，测试可注入较小次数。

#### VIP

- 根据已拟合 PLS 的 `x_weights_`、`x_scores_`、`y_loadings_` 按标准 VIP 汇总公式计算每个真实 X 坐标的 VIP。
- 页面显示前 24 个并画 `VIP=1` 参考线；JSON 保留全量有限结果，便于下载审计。
- 首版不显示误差条；图注明确 `VIP>1` 只是常用启发式，不等于统计显著性或生物学重要性。

#### PCA 得分图

- `pca_lda` 直接取最终 Pipeline 中已拟合 PCA；`pca_mlp` 取模型自身 PCA。
- 坐标轴标注 `explained_variance_ratio_`；Test 只 transform。
- 若配置只保留一个 PCA 成分，则返回 unavailable，不制造第二维。

#### 随机森林 proximity-MDS 与聚类树

- 对选中样品调用 `model.apply(X)`，proximity 定义为每对样品落在相同叶节点的树数/总树数。
- 经典 MDS 使用 `1-proximity` 作为平方距离矩阵；坐标轴标注正特征值解释比例。
- 聚类距离使用 `sqrt(1-proximity)`，平均连接法生成 dendrogram。
- 为避免 O(n²) 内存与不可读的叶标签，超过 120 条曲线时按 `真实类别 × split` 分层确定性抽样；JSON 记录 `source_sample_count`、实际绘图数和 `sampled=true`，页面显式提示。
- 叶标签优先 `Name`，其次 `Sample_ID`，再退化为 `S<行号>`；叶文字按真实类别着色。

## 5. 影响文件与职责

### 新建

- `backend/app/training_visualization.py`：schema 常量、有限数清洗、PLS/PCA/RF 可视化计算、unsupported 状态生成；不负责 HTTP 或 DOM。
- `static/js/confusion-matrix-download.js`：共享 Canvas 布局、文件名清洗、PNG Blob 下载。
- `static/js/model-feature-charts.js`：共享原生 SVG 渲染器，支持 scatter、permutation、bar、dendrogram；禁止 `innerHTML`。
- `backend/tests/test_training_visualization.py`：算法级、确定性、有限数和口径门禁测试。

### 修改

- `backend/app/training.py`：保留最终传统/深度模型的最小可视化上下文；训练评估和解释性完成后生成 `model_feature_visualization.json` 摘要；不改变现有指标口径。
- `backend/app/runs/artifacts.py`：把新 JSON 加入显式 catalog，保持模型对象与 joblib 私有。
- `backend/app/runs/result_projection.py`：仅从完整 artifact 投影 `analysis.model_feature_visualization`，损坏/缺失安全降级。
- `static/index.html`：在 `resultExplainability` 后、`resultArtifacts` 前增加 `resultFeatureVisualization` 容器和必要响应式样式。
- `static/js/run-results.js`：三个混淆矩阵卡片各接 PNG 下载按钮；新增模型可视化结果区渲染；重置/加载流程纳入新容器。
- `static/v2/components/confusion-matrix.js`：接收 `runId/split/aggregation` 下载参数并渲染 PNG 按钮。
- `static/v2/views/result.js`：把分区参数传给混淆矩阵组件，在重要性区后渲染模型特征可视化。
- `static/v2/styles.css`：图表网格、图例、下载按钮与窄屏滚动样式。
- `docs/run_result_contract.md`：记录新 analysis 字段、artifact schema、unsupported 语义和 CV 门禁。
- `docs/frontend_backend_handoff.md`：更新结果页顺序、下载行为和模型适配矩阵。
- `backend/tests/test_run_artifact_manifest.py`：catalog、下载白名单、完整性与 descriptor 测试。
- `backend/tests/test_run_result_contract_v1.py`：ready/unsupported/corrupt artifact 的投影测试。
- `backend/tests/test_result_frontend_contract.py`：经典结果页容器、纯函数、文件名与禁止 `innerHTML` 回归。
- `backend/tests/test_frontend_v2_contract.py`、`static/v2/tests/run-tests.mjs`：新共享模块被引用、矩阵下载布局/文件名、图形数据归一化与空态测试。
- `backend/tests/test_smoke.py`：至少一个 PLS-DA holdout 和一个 RF 小数据训练断言新 artifact 为 ready；既有其他模型断言 unsupported 不阻塞 Run 成功。

## 6. 按依赖顺序的实施任务

### 任务 1：冻结 schema、科学性门禁与纯算法测试

**涉及文件或模块：**

- 创建：`backend/app/training_visualization.py`
- 创建：`backend/tests/test_training_visualization.py`

**接口与依赖：**

- 消费：已拟合 sklearn 模型、训练已标准化矩阵、`splits`、标签和样品元数据。
- 产出：`model-feature-visualization-v1` 字典；只含 JSON 可序列化有限值。

**实施内容：**

1. 先实现统一的 ready/unsupported/unavailable 构造器和有限数清洗。
2. 实现 PLS score/VIP/permutation；测试置换只使用 Train/Valid，改变 Test 标签不得改变置换结果。
3. 实现 PCA score；测试坐标等于已拟合 PCA 的 `transform()`。
4. 实现 RF proximity/MDS/dendrogram；用手工小叶矩阵验证 proximity 对角线为 1、对称、相同叶越多距离越小。
5. 对 CV、单成分、样品不足、非适配模型返回有原因的状态。

**验收标准：**

- 固定 seed 重复调用得到相同 JSON。
- 输出可经 `json.dumps(..., allow_nan=False)` 序列化。
- Test 数据不参与 PLS 置换模型拟合；外部 Test 仅 transform。
- 不适配模型不抛异常、不生成伪坐标。

**验证：**

```powershell
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_training_visualization.py' -q
```

预期：全部通过，且测试总耗时通过降低注入的 permutation 次数保持可控。

### 任务 2：接入训练流水线与 Manifest

**涉及文件或模块：**

- 修改：`backend/app/training.py`
- 修改：`backend/app/runs/artifacts.py`
- 修改：`backend/tests/test_run_artifact_manifest.py`
- 修改：`backend/tests/test_smoke.py`

**接口与依赖：**

- 消费：任务 1 的 builder。
- 产出：Run 目录中的 `model_feature_visualization.json` 和 Manifest descriptor。

**实施内容：**

1. 传统模型分支分别保留选参阶段标准化矩阵和最终模型标准化矩阵，避免 PLS 置换错误使用 train+valid scaler。
2. holdout 在最终模型完成后生成可视化；CV 生成 unsupported JSON；可视化失败转 unavailable 并写公开原因，不让附加绘图失败覆盖已成功的训练指标。
3. artifact catalog 设为 `required=false`、`downloadable=true`、`category=analysis`。
4. Manifest 继续由现有 finalize 扫描生成 SHA-256 和大小；未知二进制策略不变。

**验收标准：**

- PLS/RF holdout Run 的 artifact 存在、可下载、哈希校验通过。
- 逻辑回归/深度模型和 CV Run 仍能成功结束，artifact 明确 unsupported。
- 训练指标、混淆矩阵、单样品重要性与模型文件行为不变。

**验证：**

```powershell
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_run_artifact_manifest.py' 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
```

### 任务 3：扩展 `run-result-v1` 投影

**涉及文件或模块：**

- 修改：`backend/app/runs/result_projection.py`
- 修改：`backend/tests/test_run_result_contract_v1.py`
- 修改：`docs/run_result_contract.md`

**接口与依赖：**

- 消费：Manifest descriptor 与 `model_feature_visualization.json`。
- 产出：`analysis.model_feature_visualization`。

**实施内容：**

1. 只在 succeeded Run 且 descriptor `integrity=ok` 时读取 JSON。
2. 校验 schema、status、plots 类型和有限数；格式损坏时不把异常传播为结果接口 500。
3. 缺失（历史 Run）返回 `null`；存在但 unsupported 原样返回原因；损坏返回 unavailable 并写 warning。
4. 保留 `run-result-v1` 版本字符串不变：这是向后兼容的可选新增字段。

**验收标准：**

- 新 Run 页面可直接取得绘图载荷。
- 历史 Run 仍可正常打开。
- 载荷和 artifact descriptor 均不含绝对路径。

**验证：**

```powershell
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_run_result_contract_v1.py' -q
```

### 任务 4：实现共享混淆矩阵 PNG 下载

**涉及文件或模块：**

- 创建：`static/js/confusion-matrix-download.js`
- 修改：`static/js/run-results.js`
- 修改：`static/v2/components/confusion-matrix.js`
- 修改：`static/v2/views/result.js`
- 修改：`static/index.html`
- 修改：`static/v2/styles.css`

**接口与依赖：**

- 消费：现有矩阵二维数组、类别名、split、aggregation、runId。
- 产出：用户浏览器下载的 PNG；不调用后端。

**实施内容：**

1. 将矩阵清洗、布局计算、文件名生成做成纯函数；Canvas 绘制和 Blob 下载作为薄适配层。
2. 经典页在每张矩阵表格下放按钮；v2 虽使用 split tabs，也在每个 tab 的矩阵下放对应按钮。
3. 处理 1 类、长标签、多类矩阵、0 计数、非法项、Canvas 失败和连续点击。
4. 下载按钮不依赖 artifact 状态：只要当前矩阵已成功渲染即可下载。

**验收标准：**

- 三个按钮的文件名和内容分别对应 Train/Valid/Test，不会全部下载 Test。
- PNG 打开后无透明黑底、无文字截断重叠，行列方向明确。
- server token 不写入文件名、URL、Canvas 文本或日志。

**验证：**

```powershell
node --check 'D:\PythonProject\AutoAI\static\js\confusion-matrix-download.js'
node --check 'D:\PythonProject\AutoAI\static\js\run-results.js'
node --check 'D:\PythonProject\AutoAI\static\v2\components\confusion-matrix.js'
node --check 'D:\PythonProject\AutoAI\static\v2\views\result.js'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_result_frontend_contract.py' 'D:\PythonProject\AutoAI\backend\tests\test_frontend_v2_contract.py' -q
```

人工检查：在经典页和 v2 页分别下载三张 PNG，打开核对标题、类别顺序、计数与页面表格逐格一致。

### 任务 5：渲染模型特征可视化

**涉及文件或模块：**

- 创建：`static/js/model-feature-charts.js`
- 修改：`static/index.html`
- 修改：`static/js/run-results.js`
- 修改：`static/v2/views/result.js`
- 修改：`static/v2/styles.css`
- 修改：`static/v2/tests/run-tests.mjs`
- 修改：`backend/tests/test_result_frontend_contract.py`
- 修改：`backend/tests/test_frontend_v2_contract.py`

**接口与依赖：**

- 消费：`analysis.model_feature_visualization`。
- 产出：重要性分析之后的 SVG 图表或可解释空态。

**实施内容：**

1. 共享渲染模块只允许四种已知 plot type；未知类型跳过并提示，不执行 artifact 中的任意代码/HTML。
2. scatter 颜色表示真实类别、点形表示 split，支持训练池椭圆；悬停标题显示 Name/Sample_ID/类别/split/坐标。
3. permutation 同时画 R²Y 与 Q²、原模型点、经验 p 值和“不能仅凭得分图判断”的提示。
4. VIP 画前 24 项与阈值线；不画误差条。
5. dendrogram 画可横向滚动 SVG、类别图例和抽样提示。
6. 经典结果页容器顺序固定为：单样品重要性 → 模型特征可视化 → 结果产物；v2 顺序一致。

**验收标准：**

- PLS/PCA/RF fixture 分别只出现其适配图。
- unsupported/CV 显示原因，不出现空坐标轴。
- 所有不可信文本用 `textContent`/SVG text node；共享模块和 `run-results.js` 不使用 `innerHTML`。
- 多类别和长样品名在窄屏可滚动，不挤破页面。

**验证：**

```powershell
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_result_frontend_contract.py' 'D:\PythonProject\AutoAI\backend\tests\test_frontend_v2_contract.py' -q
```

人工检查：使用 PLS-DA、PCA-LDA、Random Forest 三个 fixture/真实 Run 打开结果页，核对图形类型、图例、点数、抽样提示和 section 顺序。

### 任务 6：文档、全量回归与真实数据闭环

**涉及文件或模块：**

- 修改：`docs/frontend_backend_handoff.md`
- 修改：`docs/run_result_contract.md`
- 必要时修改：与实现直接相关的上述测试文件

**实施内容：**

1. 文档写清模型适配矩阵、CV 限制、score/VIP/permutation 的解释边界和 artifact 下载行为。
2. 运行 smoke、全套 `backend/tests`、compileall 和 v2 Node 测试。
3. 检查 `git diff --check`、`git status --short`，确认未 stage/修改用户已有的两份未跟踪计划文件。
4. 项目根目录当前没有 `data.csv`（2026-08-20 已检查），因此本轮计划不能预先保证真实数据闭环；实施结束时再次检查。若出现 `data.csv`，用 `pls_da + stratified_holdout` 跑一次轻量真实训练并核对 artifact/结果页；若仍缺失，在交付中明确记录“未运行：本地验证数据不存在”。

**验证：**

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
git diff --check
git status --short
```

预期：全部自动化测试通过；若 `data.csv` 缺失，只有真实数据闭环被明确跳过，其余验证不受影响。

## 7. 风险、性能与回滚

### 风险与缓解

- **PLS 得分图过度乐观：** 页面固定显示科学性警告；置换检验只用 Train/Valid；Test 指标仍为主验收口径。
- **置换训练增加耗时：** 仅 `pls_da` holdout 执行 50 次；阶段进度显示“生成模型可视化”；附加图失败不覆盖成功训练结果。
- **RF proximity O(n²)：** 120 条确定性分层抽样上限；页面和 JSON记录抽样事实。
- **CV 坐标误拼接：** 首版明确 unsupported，后续必须单独做跨折对齐。
- **旧 Run 兼容：** 新 analysis 字段可空，历史 Manifest 不强制缺失新 artifact，`run-result-v1` 版本不升级。
- **前端 PNG 字体差异：** 使用系统中文字体栈并以白底/文本计数保证即使颜色或字体回退仍可读。
- **大类别数图表拥挤：** SVG/矩阵容器横向滚动；RF 树限制叶数；不删除真实类别。

### 回滚边界

两个功能可独立回滚：

1. 混淆矩阵下载只涉及共享前端模块与按钮调用，移除后不影响后端契约。
2. 模型可视化是可选 artifact 和可选 analysis 字段；停止生成并移除前端 section 后，现有指标、单样品解释和 artifact 下载继续工作。已生成的历史 JSON 可继续保留，不需要删除 Run 数据。

## 8. 评审时需要确认的非阻塞决策

默认按以下选择实施；若需要调整，应在编码前确认：

1. **随机森林图采用模型原生 proximity，而不是原始谱线欧氏 HCA。** 这样图确实反映当前森林，但与用户示例的通用 HCA 数值不会完全一致。
2. **PLS 置换默认 50 次。** 若论文发表要求更高 p 值分辨率，可改为 200/1000 次，但会显著增加每次 PLS 训练耗时。
3. **首版 VIP 不带误差条。** 要复现示例误差条，需要新增 jackknife/bootstrap 定义、次数和稳定性验收，建议另列需求。
4. **首版 CV 不画跨折得分图。** 这是科学性限制，不是技术遗漏。
5. **同步支持经典正式页与 v2。** 经典页是当前正式入口，v2 复用同一契约；共享模块避免两套算法漂移。

## 9. 推荐执行与审查顺序

建议按“任务 1–3 后端可视化产物 → 任务 4 混淆矩阵 PNG → 任务 5 两套前端展示 → 任务 6 全量验证”执行。任务 4 可在任务 1–3 之后独立审查，但不建议先合并任务 5，因为前端模型图依赖稳定的 artifact schema。

可选提交检查点（仅便于审查，不是实施前提）：

1. `feat: add model-native visualization artifacts`
2. `feat: add confusion matrix png downloads`
3. `feat: render model feature visualizations in result pages`
4. `docs(test): document and verify result visualizations`
