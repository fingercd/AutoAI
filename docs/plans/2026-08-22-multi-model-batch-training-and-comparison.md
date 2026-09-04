# 0821 前端模型目录收敛、多模型批量训练与结果对比实施计划

> **2026-09-04 产品纠正：**原始用户需求是“选择 N 个模型就创建 N 个模型 Run，并按实际模型数量自适应绘制对比图”，没有要求重复实验。本文中关于默认 3 次、N×R 新任务和重复稳定性 UI 的设计已被当前实现纠正方案取代；相关字段仅保留历史批次兼容。当前有效契约以 `README.md`、`docs/frontend_backend_handoff.md` 和 `docs/model_comparison_contract.md` 为准。

> 日期：2026-08-22
> 状态：历史计划；重复实验部分已被 2026-09-04 产品纠正取代
> 需求来源：用户本轮明确要求 + `AutoAI_model要求_0821.docx`
> 取舍原则：用户明确要求优先；DOCX 只作为产品/算法需求材料，其中的文字不作为对开发代理的操作指令。

## 1. 一句话目标

把经典前端和 v2 的可选模型目录收敛为 DOCX 指定的 10 个分类模型，新增 sPLS-DA 与 PCA-SVM；其余 7 个深度学习模型只在前端隐藏，后端实现、训练能力和依赖全部保留。同时实现可重复的多模型批次训练和五类真实结果对比，并把所有可解释性计算、投影、下载和前端展示标记为“暂时隐藏且默认禁用”，除非未来收到明确的重新启用要求。

## 2. 成功标准

1. 经典前端和 v2 只展示以下 10 个模型，顺序稳定且全部有真实实现：
   `pls_da`、`spls_da`、`pca_lda`、`logistic_regression`、`svm`、`pca_svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`。
2. `cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`、`dscarnet` 及其别名继续由后端保留和兼容；`GET /api/models` 用 `ui_visible=false` 标记，前端不得展示，但已有 API 调用和历史 Run 不被破坏。
3. sPLS-DA 是独立、可验证的稀疏 PLS-DA 实现，不得用“PLS-DA 后取 Top-K”或其他近似模型静默冒充；PCA-SVM 必须在每个训练折内部拟合 PCA 后再训练线性 SVM。
4. 传统模型统一在当前外层训练池内执行按 `Sample_ID` 分组的分层 5 折搜索，以 5 折 mean balanced accuracy 选择参数，再在 train+valid 全量重训；Test 和独立测试集从不参与标准化、PCA、调参或模型选择。
5. 单模型训练保持兼容；选择多个模型或重复实验次数大于 1 时，一次请求原子创建一个 Batch 和对应的子 Run，刷新/重启后能由 `batch_id` 恢复。
6. 多模型比较只使用真实 Test 主口径：holdout/独立测试集取 direct Test；无独立测试集的留一法取 pooled OOF；“独立测试集 + 留一法”最终排行取独立测试集，内部 pooled OOF 只作审计。
7. 比较页按真实产物提供：混淆矩阵入口、四项总体指标图、Sample_ID×模型正确率热图、重复实验一致性图、类别 Recall/Sensitivity 热图；模型数量从 2 到 10 时布局自适应。
8. 可解释性默认关闭：新 Run 不执行特征遮挡、Grad-CAM 或模型特征可视化，不生成对应 artifact；结果 API 不公开其内容或下载地址；经典前端和 v2 均不渲染、不懒加载、不显示占位卡片。
9. 代码中存在唯一明确标记：`TEMPORARILY_HIDDEN: do not re-enable without an explicit product requirement and synchronized backend/frontend tests`。不能通过普通请求参数或环境变量意外重新开启。
10. 自动化测试覆盖 10 个前端可见模型、两个新增模型、七个后端保留但前端隐藏的深度模型、5 折选参、无泄漏、批次事务、重复实验、比较图数据、可解释性禁用和 server Principal 权限；存在本地 `data.csv` 时跑一个轻量多模型真实闭环。

## 3. DOCX 要求解析与采用范围

### 3.1 直接采用

- 适用数据画像：分类任务，N 约 50–1000，L 约 500–10000，输入为 `1 × L`。
- 所有标准化参数只从当前训练折拟合，并应用于对应验证/测试数据。
- 支持无独立测试集 8:1:1、独立测试集下主数据 8:2，以及基于 `Sample_ID` 的留一交叉训练验证；前端允许用户自定义合法比例。
- 同一 `Sample_ID` 的重复测量不得跨 train/valid/test。
- 两套前端的可选模型目录严格使用 DOCX 的 10 项；后端不删除现有额外深度模型。
- 传统模型使用分层 5 折 mean balanced accuracy 选参，展示最优参数；留一法展示每一外层轮次的最优参数。
- 深度模型使用 AdamW、ReduceLROnPlateau、early stopping、最佳 validation loss checkpoint。
- PCA-MLP 和 1D-CNN 按 N/L 档位选择结构。
- 结果包含总体指标比较、Sample_ID 预测热图、重复实验一致性和类别 Recall 热图。

### 3.2 明确不采用或需要澄清的内容

- DOCX 同时写了回归输出头，但文档标题、任务类型和当前平台都明确是分类；本计划继续只支持分类，不新增 PLSR、SVR 或 MSELoss 回归入口。
- DOCX 未给 PCA-SVM 的独立搜索网格。本计划默认使用 PCA-LDA 的 PCA 维数网格与线性 SVM 的 C 网格做笛卡尔积；这项默认必须记录在模型契约中。
- DOCX 未指定“多次重复实验”的次数。本计划默认 Batch 重复次数为 3，可选 1–5；该默认会使总任务数变为“模型数 × 重复次数”，实施前可由产品方调整，但不能静默取消重复能力。
- DOCX 的“正确且高/低置信度”没有阈值定义。首版选择无任意阈值的“正确率热图”：每格为该 Sample_ID 在各重复实验中预测正确的比例；不人为划分高/低置信度。
- N/L 是适用范围而非数学硬边界。超出范围返回醒目 warning，并在结果审计中保留；除非模型自身维数或 5 折分组条件不满足，不仅因超范围直接拒绝训练。

## 4. 前端可见目录与后端保留矩阵

### 4.1 前端最终可见目录（10 项）

| 顺序 | 模型 ID | 展示名 | 类型 | 处理 |
|---:|---|---|---|---|
| 1 | `pls_da` | PLS-DA | 传统化学计量学 | 保留并对齐 5 折搜索 |
| 2 | `spls_da` | sPLS-DA | 传统化学计量学 | 新增真实稀疏实现 |
| 3 | `pca_lda` | PCA-LDA | 传统化学计量学 | 保留并对齐 5 折搜索 |
| 4 | `logistic_regression` | Logistic Regression | 线性模型 | 保留，改为 elastic-net/saga 网格 |
| 5 | `svm` | SVM | 支持向量机 | 保留，固定 linear kernel |
| 6 | `pca_svm` | PCA-SVM | 支持向量机 | 新增 fold 内 PCA+linear SVM |
| 7 | `random_forest` | Random Forest | 树集成 | 保留，移除 OOB 选参，改 5 折网格 |
| 8 | `xgboost` | XGBoost | 树集成 | 保留并对齐 DOCX 网格 |
| 9 | `pca_mlp` | PCA-MLP | 深度学习 | 保留并对齐 BN/输出头结构 |
| 10 | `cnn1d` | 1D-CNN | 深度学习 | 保留并对齐 N/L profile |

### 4.2 后端保留、前端隐藏目录（7 项）

以下模型不出现在经典前端和 v2 的模型选择器中，但后端代码、注册、训练入口、依赖和历史兼容全部保留：

- `cnn1d_se`
- `resnet1d`
- `inception1d`
- `tcn1d`
- `cnn_transformer1d` 及 `transformer1d` 等别名
- `cnn_mamba1d`
- `dscarnet`

隐藏语义：

1. 保留能力目录、别名表、可训练集合、构造器、模型源码、DSCARNet 映射代码和专属依赖。
2. `GET /api/models` 继续返回这些模型以保持后端/API 兼容，但增加 `ui_visible=false` 和 `visibility_reason="temporarily_hidden_from_ui"`；前端只能渲染 `ui_visible=true`。
3. 直接 API 请求仍可按原有能力训练这些模型；`cnn_mamba1d` 仍按依赖事实返回 `available=false`，不得自动替换。
4. 历史 Run 的字符串、指标、预测、混淆矩阵继续由 `run-result-v1` 读取；用户本地 `storage/runs`、权重和历史产物不移动、不删除。
5. 后端隐藏模型不进入普通前端的多选 Batch，但由受控 API 创建的历史/兼容 Batch 仍可比较，比较页按返回结果渲染，不因模型隐藏而丢数据。

## 5. 训练与评估目标契约

### 5.1 随机性拆分

新增两个内部种子，解决重复实验与公平比较：

- `split_seed`：只控制 train/valid/test、外层留一和内层 5 折划分；同一 Batch 全模型、全重复固定一致，默认 42。
- `model_seed`：控制模型初始化、Dropout、DataLoader、RF/XGBoost 随机性；第 r 次重复使用 `base_seed + r`。
- 旧请求的 `seed` 同时映射到二者，保持兼容；新 Batch 显式固化两个值。

这样重复实验改变模型随机性但不改变测试 Sample_ID，Sample_ID×模型热图和跨模型比较使用完全相同的测试对象。

### 5.2 四种评估策略

1. `stratified_holdout`：无独立测试集，按 Sample_ID 分组分层，默认 8:1:1。
2. `external_test_holdout`：主数据默认 8:2，独立测试集只作最终 Test。
3. `leave_one_sample_id_cv`：无独立测试集，每个 Sample_ID 轮流作外层 Test；主指标是 pooled OOF。
4. `leave_one_sample_id_cv_with_external_test`：主数据执行留一 OOF 审计，独立测试集完全不进入外层/内层 CV；选择最终模型后在全部主数据上重训，只对独立测试集评估一次，独立 Test 为主指标，pooled OOF 为次级审计。

自定义比例仍使用正整数权重并校验总和为 10。前端开放 7:2:1、6:2:2 等合法配置，但必须满足每个非空分区类别完整和 Sample_ID 分组约束。

### 5.3 传统模型内层 5 折选参

对 8 个传统模型统一执行：

1. 固定外层 Test，构造当前外层 train+valid 候选池。
2. 使用 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)`；groups=`Sample_ID`。
3. 每个内层折单独拟合 normalizer、PCA/PLS/sPLS 和分类器，禁止先对完整候选池标准化。
4. 对每组候选参数计算 5 折 balanced accuracy 均值与标准差。
5. 按 mean balanced accuracy 降序选择；平分时使用参数复杂度更低者，再按候选固定顺序稳定选择。
6. 用选中参数在外层 train+valid 全量重新拟合 normalizer 和模型，再评估外层 Test。
7. `hyperparameter_search.csv` 保存所有候选的五折得分、mean/std、是否选中；结果页训练审计显示最终参数。留一外层 CV 时，每轮独立保存最优参数。

如果任一类别在候选池中少于 5 个不同 Sample_ID，训练前拒绝并列出类别/分组数；不得静默退化为 3 折、普通 StratifiedKFold 或按曲线行拆分。

### 5.4 文档指定参数网格

- PLS-DA：`n_components=[1,2,3,4,5,6,8,10,12,15]`，上限为当前内折可用样本秩/特征数。
- sPLS-DA：`n_components=[1,2,3,5,8,10]`，`keepX=[25,50,100,300]`，`keepX<=L`。
- PCA-LDA：`n_components=[2,3,5,8,10,15,20,30,40,50]`，每内折动态限制维数。
- Logistic Regression：`penalty="elasticnet"`、`solver="saga"`、`C=[0.01,0.1,1,10,100]`、`l1_ratio=[0.1,0.5,0.9]`。
- SVM：`kernel="linear"`、`C=[0.01,0.1,1,10,100]`。
- PCA-SVM：PCA 维数使用 PCA-LDA 网格，C 使用 SVM 网格；PCA 和 SVM 必须位于同一 fold-local Pipeline。
- Random Forest：`n_estimators=500`、`max_depth=[None,5,10]`、`min_samples_leaf=[1,5]`、`max_features=["sqrt",0.1]`。
- XGBoost：`n_estimators=[100,300]`、`learning_rate=0.1`、`max_depth=[2,3,5]`、`min_child_weight=[3,5]`、`colsample_bytree=0.3`、`subsample=0.8`、`reg_lambda=10`。

### 5.5 sPLS-DA 真值要求

创建独立 `SPLSDAClassifier`，采用 mixOmics 风格的监督稀疏 PLS-DA：one-hot Y、逐成分稀疏权重求解、每成分 `keepX` 非零载荷约束、成分 deflation 和类别响应预测。具体要求：

- 不调用普通 PLS-DA 后再截取 VIP/系数来伪装 sPLS-DA。
- 训练折以外的数据不得参与成分或变量选择。
- 输出 `selected_feature_indices_by_component`、实际成分数和实际 keepX 到模型元数据/参数审计；这属于模型参数审计，不恢复“可解释性”面板。
- 使用固定小数据集生成独立参考 fixture，与 R `mixOmics::splsda` 或其他经确认的权威实现比较选中特征、类别预测和 balanced accuracy；参考工具只用于测试 fixture，不成为生产运行依赖。
- 如果无法达到已定义的参考容差，则模型目录返回 `available=false` 并说明原因，不得发布近似实现。

### 5.6 PCA-SVM 真值要求

使用 sklearn Pipeline：fold-local normalization → PCA → `SVC(kernel="linear", probability=True)`。PCA 只能在当前内折训练子集拟合；`probability=True` 仅为预测置信度/热图提供概率，超参选择仍使用类别预测的 balanced accuracy。

### 5.7 深度模型对齐

统一参数：AdamW、batch size 8、learning rate 1e-3、weight decay 1e-4、ReduceLROnPlateau factor 0.5/patience 10/min lr 1e-6、最多 200 epochs、early stopping patience 20、默认 seed 42。

PCA-MLP：

- N≤100：D=32，H=[32,16]，p=0.5。
- 100<N≤300：D=64，H=[64,32]，p=0.4。
- N>300：D=128，H=[128,64]，p=0.3。
- 每个隐藏层固定为 `Linear → BatchNorm1d → ReLU → Dropout`。
- 分类头统一输出 `num_classes` logits，并使用 CrossEntropyLoss；不采用 DOCX 中的回归头。

1D-CNN：

- N 档位通道与 Dropout 完全按 DOCX。
- L≤1000：K=[7,5,3]，P=[2,2,2]。
- 1000<L≤3000：K=[9,5,3]，P=[4,2,2]。
- L>3000：K=[9,7,5]，P=[4,2,2]；修正当前代码长序列第三档池化不一致。
- 三段 `Conv1D → BN → ReLU → MaxPool`，接 AdaptiveAvgPool1d(1)、Flatten、Dropout 和 `num_classes` logits。
- 架构/输出头变化后把新 Run 的 `architecture_version` 提升为 `docx-classification-v3-0821`；旧 v2 checkpoint 只读，不静默加载。

## 6. 可解释性“暂时隐藏”设计

### 6.1 唯一开关

创建内部模块 `backend/app/feature_flags.py`：

```python
# TEMPORARILY_HIDDEN: do not re-enable without an explicit product requirement
# and synchronized backend/frontend contract tests.
EXPLAINABILITY_ENABLED = False
```

该常量不是环境变量、请求字段或管理员 UI 设置。未来只能通过明确代码变更、契约更新和测试重新开启。

### 6.2 后端行为

- 训练主链不调用 `training_explainability.py`、`feature_selection.py` 中的遮挡/Grad-CAM，也不调用 `training_visualization.py` 的模型特征可视化。
- 新 Run 不生成 `sample_feature_importance.*`、`feature_importance.*`、`model_feature_visualization.json`、`dscarnet_mapping.json`。
- `config.json`/`model_metadata.json` 写入 `explainability_status="temporarily_hidden"`，而非某个算法名。
- `run-result-v1` 保留兼容顶层键但只返回：`explainability: {"status":"temporarily_hidden"}`；`analysis` 不再投影 `model_feature_visualization`。
- artifact catalog 对所有 explainability/feature-visualization 名称设为不可公开；历史 Manifest 中即使登记，也不在 descriptors 列出，直接下载端点返回 404 `artifact unavailable`。
- `/api/models` 移除 `explainability_method` 字段，避免目录层触发解释模块导入。
- 现有解释代码、七个前端隐藏深度模型的解释分支和 DSCARNet 专属依赖全部保留，但 active training path 不得调用；统一加暂时隐藏标记，未来恢复仍需明确需求和测试。

### 6.3 前端行为

- 经典结果页不创建“单样品可解释性”“模型特征可视化”区块，不请求相关 JSON，不显示相关下载分类。
- v2 `result.js` 不导入/调用 `explainability-panel.js` 和模型特征图组件；DOM 中不保留空白卡片或“暂不可用”提示。
- 模型选择卡不显示“解释方法”。
- 前端未引用的可解释性组件文件保留，并在顶部增加同一 `TEMPORARILY_HIDDEN` 注释；当前页面不导入、不调用，恢复功能必须另立计划。
- 测试用请求 spy 断言打开结果页不会访问任何 explainability artifact URL。

## 7. 多模型 Batch 与重复实验架构

### 7.1 执行单元

- 一个子 Run 只训练“一个模型 × 一个 repeat”。
- 一个 Batch 包含 N 个模型、R 次重复，共 N×R 个有序子 Run。
- 选择一个模型且 R=1：继续调用现有单 Run API。
- 模型数>1 或 R>1：调用 Batch API。
- 所有子 Run一次事务性入队；一个失败不覆盖其他结果。
- 当前 worker 仍是单进程单并发；部署多个相同契约 worker 时可并发 claim。页面显示真实 queued/running 数，不承诺 GPU 一定同时运行。

### 7.2 请求草案

```http
POST /api/training/batches
```

```json
{
  "dataset_id": "ds_abc123",
  "test_dataset_id": null,
  "model_types": ["pls_da", "spls_da", "pca_svm"],
  "repeat_count": 3,
  "base_seed": 42,
  "config": {
    "split_mode": "stratified_holdout",
    "split_train": 8,
    "split_valid": 1,
    "split_test": 1,
    "normalization": "zscore"
  }
}
```

约束：前端只提交 10 个 `ui_visible=true` 模型；后端为兼容受控 API，接受全部已注册且 `available=true` 的模型，规范化后无重复；`repeat_count` 1–5；N×R≤50；`config` 不允许 `model_type`、`seed` 或可解释性开关。所有模型、数据快照和配置先验证，任何失败时 Batch/Run 均不落库。

成功返回：`batch_id`、模型数、重复次数、总 Run 数、有序的 `{model_type, repeat_index, run_id, state}`。

### 7.3 数据库

新增 `training_batches`，并为 `runs` 增加可空字段：

- `batch_id`
- `batch_model_order`
- `batch_repeat_index`
- `split_seed`
- `model_seed`

Batch 表保存主/测试数据快照、公共配置、规范模型列表、repeat_count、base_seed、Principal scope 和时间。Batch 状态由子 Run实时聚合，不双写冗余状态。

### 7.4 生命周期

- `queued`：全部 queued。
- `running`：任一子 Run 已开始且仍有 active 子 Run。
- `succeeded`：全部 succeeded 且必需指标完整。
- `partial`：全部终态，成功与 failed/cancelled/partial 混合。
- `failed`：无成功结果且至少一个 failed。
- `cancelled`：全部 cancelled。

Batch STOP 在仓库事务中取消所有 queued/running 子 Run，终态不回滚。Batch 子 Run允许单独 STOP，但禁止单独 DELETE；批次删除只允许所有子 Run终态，并统一删除 DB 记录和精确解析后的子 Run目录。历史数据/大文件不自动清理。

## 8. 多模型比较契约与图表

### 8.1 `model-comparison-v1`

```http
GET /api/training/batches/{batch_id}/comparison
```

后端只从 Manifest 完整性通过的子 Run读取 `metrics.primary`、`predictions.csv`、`label_map.json`、`split.json` 和训练审计。比较前校验：

- 主/独立测试数据 SHA-256 一致。
- evaluation strategy 与主聚合口径一致。
- 同一 repeat 内各模型 `split.json` digest 一致。
- Test/OOF Sample_ID 集合和标签一致。
- 指标有限且位于 `[0,1]`。

不一致时 `comparable=false`，保留单 Run链接和错误原因，但不生成排行或热图。

### 8.2 总体指标比较（DOCX 9.2）

每个模型聚合 R 次重复的：

- Accuracy
- Balanced Accuracy
- Macro-F1
- Weighted-F1

返回 mean/std/min/max 和每次重复原值。R=1 时只显示值，不伪造误差条。主排行默认按 mean Accuracy，平分时依次 mean Balanced Accuracy、mean Macro-F1、目录顺序稳定排序。

布局：

- 2–5 个模型：Accuracy 横向排名 + 四指标分组柱状图。
- 6 个及以上模型：Accuracy 横向排名 + 四指标热力表，避免拥挤柱状图；普通前端最多 10 个，兼容 API Batch 可按实际返回数量继续滚动展示。
- 明细表始终展示 mean±std、成功重复数/总重复数、耗时和单 Run入口。

### 8.3 Sample_ID×模型正确率热图（DOCX 9.3）

每个子 Run先按 Sample_ID 聚合重复测量：对同一 Sample_ID 的 `prob_<label>` 取算术平均，再 argmax 得到 Sample_ID 级预测；同组真实 Label 必须唯一。

Batch 级每格为：该模型对该 Sample_ID 在成功 repeats 中预测正确的比例 `correct_count / available_repeat_count`。颜色 0–1，格内同时显示百分比；缺失预测为 `—`，不能当错误或 0。

### 8.4 重复实验预测一致性图（DOCX 9.4）

对每个模型、每个 Sample_ID 计算预测一致性：`各重复预测中出现次数最多的类别数 / available_repeat_count`。模型级图展示所有 Sample_ID 一致性的 mean、min、max；R<2 时显示“需要至少 2 次重复实验”，不画虚假一致性图。

### 8.5 类别 Recall/Sensitivity 热图（DOCX 9.5）

从每个 repeat 的 Test/pooled OOF 混淆矩阵或分类报告读取每类 recall，按模型×类别返回 mean/std。只有所有模型 label_map 一致时绘制；类别缺失不补 0，返回不可比较原因。

### 8.6 混淆矩阵（DOCX 9.1）

Batch 比较页不把多个矩阵叠在一张图中。每个模型提供“查看混淆矩阵”入口：R=1 打开对应单 Run结果；R>1 默认展示基于所有成功 repeats 的合并预测矩阵，并允许切换具体 repeat。继续沿用当前真实 Train/Valid/Test/pooled OOF 口径。

## 9. 影响文件清单

### 9.1 模型目录与训练器

- 修改 `backend/app/models/registry.py`：保留现有模型集合并加入 sPLS-DA/PCA-SVM；为前端可见模型建立独立集合；按模型记录架构版本。
- 修改 `backend/app/models/__init__.py`：导出 sPLS-DA/PCA-SVM，保留所有现有深度模型导出。
- 创建 `backend/app/models/spls_da.py`：真实稀疏 PLS-DA estimator。
- 创建 `backend/app/models/pca_svm.py`：fold-local PCA+linear SVM Pipeline。
- 修改 `backend/app/models/logistic_regression.py`：elastic-net/saga/l1_ratio。
- 修改 `backend/app/models/svm.py`、`pca_lda.py`、`random_forest.py`、`xgboost.py`：文档参数与概率输出。
- 修改 `backend/app/models/pca_mlp.py`、`cnn1d.py`、`profiles.py`：BN、pool profile、num_classes logits。
- 保留 `backend/app/models/cnn_se1d.py`、`resnet1d.py`、`inception1d.py`、`tcn1d.py`、`cnn_transformer1d.py`、`cnn_mamba1d.py`、`dscarnet.py`，不得因前端隐藏而删除或失去后端训练能力。
- 保留 `backend/app/dscarnet_mapping.py`、`backend/requirements-dscarnet.txt` 和 AggMap/mamba capability 探测。
- 修改 `backend/app/training.py`：内层 5 折、split/model seed、两个新增模型、重复实验所需预测字段、跳过可解释性。
- 修改 `backend/app/contracts.py`、`classification_policy.py`：模型别名、四种评估策略、自定义比例、seed 和隐藏功能字段。

### 9.2 可解释性隐藏

- 创建 `backend/app/feature_flags.py`。
- 修改 `backend/app/training_explainability.py`、`feature_selection.py`：保留全部现有模型可复用代码但无 active call，并加暂时隐藏标记；不删除深度模型/DSCARNet 分支。
- 修改 `backend/app/training_visualization.py`：默认不调用并标记暂时隐藏；不得覆盖当前工作区已有未提交实现。
- 修改 `backend/app/runs/artifacts.py`、`result_projection.py`、`status_projection.py`：不生成/投影/下载解释性内容。
- 修改 `backend/app/routers/catalog.py`：保留现有模型并加入两个新模型；增加 `ui_visible`/`visibility_reason`，10 个为 true、7 个额外深度模型为 false；不返回解释方法。
- 修改 `static/js/run-results.js`、`static/index.html`、`static/v2/views/result.js`、`static/v2/components/artifacts.js`：不展示、不请求。
- 保留并标记 `static/v2/components/explainability-panel.js`、`static/js/model-feature-charts.js`，但从 active 前端 import/render 链路断开。

### 9.3 Batch 与比较

- 修改 `backend/app/runs/contracts.py`、`repository.py`：BatchRecord、表迁移、原子创建、scope、STOP/DELETE。
- 创建 `backend/app/runs/batch_projection.py`：状态聚合、repeat 聚合、Sample_ID/Recall/一致性计算。
- 创建 `backend/app/routers/batches.py` 并在 `backend/app/main.py` 注册。
- 修改 `backend/app/version.py`：登记 `training-batch-v1`、`model-comparison-v1`。
- 修改 `static/js/training-store.js`，创建 `static/js/model-comparison.js`。
- 修改 `static/v2/api.js`、`store.js`、`app.js`、`views/modeling.js`、`styles.css`。
- 修改 `static/v2/components/model-catalog.js` 为 checkbox。
- 创建 `static/v2/views/comparison.js`、`static/v2/components/model-comparison.js`。
- 修改 `static/v2/lib/charts.js`：0–1 横向排名、四指标图和热力图。

### 9.4 文档

- 修改 `AGENTS.md`、`CONTEXT.md`、`README.md`：10 个前端可见模型、7 个后端保留/前端隐藏模型、解释性暂时隐藏和训练策略。
- 修改 `docs/frontend_backend_handoff.md`、`docs/run_result_contract.md`。
- 创建 `docs/model_comparison_contract.md` 和模型验收说明。
- 更新只描述 15 模型或解释性正式启用的当前说明；DSCARNet/其余深度模型文档改为“后端保留、普通前端隐藏”。历史计划文件可保留并标注已被 0821 前端目录要求取代，不重写历史事实。

## 10. 按依赖排序的实施任务

### 任务 1：冻结 0821 分类模型和隐藏功能契约

**实施：**先更新模型 ID、别名、架构版本、评估策略、Batch/Comparison schema 和 explainability hidden 状态。把本计划第 3–8 节同步到正式契约文档，避免代码先行造成前后端口径分叉。

**验收：**OpenAPI 继续接受后端已注册模型；能力目录用 `ui_visible` 精确区分 10 个可见和 7 个隐藏模型；两套前端只展示 10 个；`/health.contracts` 返回新版本；没有公开 explainability 开关。

### 任务 2：隐藏七个额外深度模型并保持后端能力

**实施：**在 catalog 增加独立 UI 可见性字段，两套前端只渲染 true；保留 registry、别名、构造器、源码、DSCARNet mapping、AggMap/mamba 依赖与直接 API 训练路径。为隐藏模型补后端回归测试，防止 UI 收敛误伤训练能力。

**验收：**前端恰好显示 10 项；API catalog 仍含后端模型及可见性元数据；当前可用隐藏模型仍能通过后端构造/训练，`cnn_mamba1d` 保持真实 unavailable；历史 Run仍能显示指标。

### 任务 3：新增 sPLS-DA 与 PCA-SVM

**实施：**按第 5.5/5.6 节实现 estimator、预测概率、序列化和元数据；为 sPLS-DA 建独立权威参考 fixture；PCA-SVM 用 sklearn Pipeline 防泄漏。

**验收：**两模型通过二/多分类、维数上限、概率有限且归一、pickle round-trip、固定 seed、fold-local fit 和 reference fixture；若 sPLS-DA 未通过参考验收则 catalog 必须 unavailable，不能标可训练。

### 任务 4：统一传统模型 5 折搜索和文档网格

**实施：**重构 `_traditional_candidate_configs`/fit 流程为分组分层内层 5 折；移除 RF OOB 选参；更新 Logistic elastic-net 和全部参数网格；每候选记录五折分数。

**验收：**测试故意在 valid/test 放极端值，证明 scaler/PCA/变量选择未见外部数据；选中参数等于最高 mean balanced accuracy；不足 5 组/类别明确拒绝；留一法每轮参数独立可见。

### 任务 5：对齐 PCA-MLP、1D-CNN 和四种评估策略

**实施：**修正 BN、长序列 pool、num_classes+CrossEntropy；拆分 split/model seed；实现独立测试集+留一模式和最终外部 Test；开放合法自定义比例。

**验收：**N/L 三档结构精确匹配 DOCX；二分类为 2 logits；外部 Test 从未进入 CV；LOSO pooled OOF 与 external direct 明确分开；v2 checkpoint 不被 v3 静默加载。

### 任务 6：四层关闭可解释性

**实施：**添加硬编码 hidden flag；训练跳过计算，Manifest 不登记，结果投影只给 hidden 状态，下载拒绝，前端不渲染/不请求。所有现有后端解释分支保留但不进入 active call graph。

**验收：**用 monkeypatch 让解释函数一调用就失败，完整训练仍成功；Run目录没有相关文件；结果 API/HTML/网络请求不出现解释性内容；历史直接 artifact URL 不可下载。

### 任务 7：实现 Batch、repeat 和资源安全队列

**实施：**新增表/可空列、原子创建、N×R 上限、scope、状态聚合、STOP/DELETE；子 Run注入一致 split_seed 和逐 repeat model_seed。继续复用现有 claim/lease。

**验收：**3 模型×3 repeats 原子创建 9 Run；中途数据库失败零残留；多个 worker 不重复 claim；单 worker 真实显示排队；其他 Principal 对 Batch 全部 404。

### 任务 8：实现 `model-comparison-v1`

**实施：**聚合主指标和 repeats，按 Sample_ID 聚合概率，生成总体四指标、正确率矩阵、一致性、类别 recall 和可选合并混淆矩阵；校验 digest/标签/样品集合。

**验收：**direct、pooled OOF、external+LOSO 三种 fixture 指标取值正确；缺失/失败 repeat 不补 0；不一致 split/label 返回不可比较；响应无服务器路径或 traceback。

### 任务 9：改造 v2 多选和比较页

**实施：**radio 改 checkbox；显示模型数×重复次数=任务数和计算成本提示；单 Run/Batch 分流；新增 comparison 深链、串行轮询、AbortController、3/5/6/10 模型响应式布局。

**验收：**一个模型 R=1 仍走旧入口；三个模型 R=3 只发一个 Batch 请求；刷新恢复；R<2 不画一致性；Explainability DOM/请求均不存在；键盘和读屏可完成多选。

### 任务 10：改造经典前端并保持兼容

**实施：**动态 checkbox 目录、重复次数、公共参数、Batch Hash 路由和比较模块；移除解释性区块及下载分类；保持现有导航顺序与单 Run结果 URL。

**验收：**经典脚本抽取后 Node 语法通过；现有字符串契约更新为 10 模型/无 explainability；单/多模型、防重复提交、错误恢复和 STOP 均可用。

### 任务 11：全量测试、真实数据和文档交付

**实施：**先跑模型/5折/隐藏功能定向测试，再跑 Batch/Comparison 和前端纯函数，最后跑整个后端。若根目录有 `data.csv`，用 `pls_da + logistic_regression`、R=2 做真实闭环；sPLS-DA 另跑固定参考 fixture。

**验收：**全部自动化通过；真实比较值与各子 Run `metrics.primary` 一致；没有数据、模型、Run、渲染图、缓存、DOCX 副本或二进制进入 Git。

## 11. 测试文件计划

新增：

- `backend/tests/test_model_catalog_0821.py`
- `backend/tests/test_spls_da.py`
- `backend/tests/test_pca_svm.py`
- `backend/tests/test_grouped_inner_cv_search.py`
- `backend/tests/test_explainability_temporarily_hidden.py`
- `backend/tests/test_training_batch_repository.py`
- `backend/tests/test_training_batch_router.py`
- `backend/tests/test_model_comparison_contract_v1.py`

重点修改：

- `backend/tests/test_model_spec_v2.py`（改为 v3/10 模型）
- `backend/tests/test_model_catalog.py`
- `backend/tests/test_training_config_validation.py`
- `backend/tests/test_classification_policy.py`
- `backend/tests/test_run_result_contract_v1.py`
- `backend/tests/test_run_artifact_manifest.py`
- `backend/tests/test_result_frontend_contract.py`
- `backend/tests/test_frontend_v2_contract.py`
- `backend/tests/test_smoke.py`
- `static/v2/tests/run-tests.mjs`

七个前端隐藏模型的结构/训练测试继续保留；另增 UI 可见性测试，证明“前端不显示”和“后端仍可用”同时成立。仅更新与前端可见目录数量相矛盾的断言。

## 12. 验证命令

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'

& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest `
  'D:\PythonProject\AutoAI\backend\tests\test_model_catalog_0821.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_spls_da.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_pca_svm.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_grouped_inner_cv_search.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_explainability_temporarily_hidden.py' -q

& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest `
  'D:\PythonProject\AutoAI\backend\tests\test_training_batch_repository.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_training_batch_router.py' `
  'D:\PythonProject\AutoAI\backend\tests\test_model_comparison_contract_v1.py' -q

node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'

& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest `
  'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q

& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest `
  'D:\PythonProject\AutoAI\backend\tests' -q

& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall `
  'D:\PythonProject\AutoAI\backend\app' -q
```

前端语法：

```powershell
node --check 'D:\PythonProject\AutoAI\static\js\training-store.js'
node --check 'D:\PythonProject\AutoAI\static\js\model-comparison.js'
node --check 'D:\PythonProject\AutoAI\static\v2\api.js'
node --check 'D:\PythonProject\AutoAI\static\v2\app.js'
node --check 'D:\PythonProject\AutoAI\static\v2\views\modeling.js'
node --check 'D:\PythonProject\AutoAI\static\v2\views\comparison.js'
```

经典 `static/index.html` 的内联脚本按现有测试辅助方式抽取到 `work/`，再执行 `node --check --input-type=commonjs`。不新增 Playwright/JSDOM 依赖。

## 13. 人工验收矩阵

| 场景 | 预期 |
|---|---|
| 打开模型目录 | 前端只看到 10 个指定模型，无解释方法字段 |
| 直接请求隐藏深度模型 | 后端按原能力训练；前端仍不显示，不能静默替代 |
| 选择 3 模型、R=3 | 原子创建 9 Run，显示真实 queued/running/succeeded 数 |
| 选择 5 模型 | 四指标分组柱图可读，五行模型汇总 |
| 选择 6–10 模型 | 自动切换四指标热力表和 Accuracy 横向排行 |
| 一个 repeat 失败 | 其他任务继续；成功次数清晰，失败值不补 0 |
| Holdout 重复实验 | 各 repeat 测试 Sample_ID 相同，model seed 不同 |
| LOSO | 主指标为 pooled OOF；每一轮最优参数可查看 |
| 外部 Test+LOSO | 外部 Test 不参加 CV，外部指标用于最终排行 |
| R=1 | 不显示一致性图，只给“至少 2 次”说明 |
| 结果页 | 无可解释性卡片、请求、artifact 下载 |
| 历史隐藏模型 Run | 指标和混淆矩阵可读；普通前端复制配置时提示该模型当前不在可见目录，但不删除后端能力 |
| server 跨 Principal | Batch、子 Run、比较、STOP、DELETE 全部 404 |

## 14. 风险与控制

1. **sPLS-DA 算法真值风险**：必须有独立参考 fixture；未通过时显示 unavailable，不发布近似实现。
2. **计算量放大**：10 模型×3 repeats 最多 30 Run，且传统模型有内层 5 折；前端在提交前展示总任务数，后端限制 N×R≤50，worker 不自动超卖 GPU。
3. **5 折分组样本不足**：训练前按类别列出 Sample_ID 数量并拒绝，不回退为按曲线拆分。
4. **架构兼容**：PCA-MLP/1D-CNN 输出头变化必须提升 architecture version，旧权重只读。
5. **解释性只隐藏 UI 但后台仍耗时**：通过训练调用 spy、目录产物断言、artifact 路由测试三重防止。
6. **工作区已有未提交修改**：实施前重新运行 `git status --short`；`training.py`、`result_projection.py`、`artifacts.py`、两套结果页及新增 `training_visualization.py` 均有重叠，必须逐块合并，不得覆盖用户改动。
7. **前端过滤误伤后端能力**：以 registry 构造/轻量训练回归测试锁定七个隐藏模型；只改 `ui_visible` 和前端过滤，不删除模型、映射、依赖或 storage 产物。
8. **DOCX 的独立 Test+LOSO 语义复杂**：契约明确 external Test 是最终主指标、pooled OOF 仅审计，测试用不同数值证明没有混用。

## 15. 回滚与发布

- 发布顺序：后端 10 模型/隐藏解释性 → 新模型与 5 折训练 → Batch/Comparison API → v2 → 经典前端。
- Web 与 worker 必须同时从同一版本重启；通过 `/health` 确认 worker contract compatible。
- Batch 前端可临时关闭而不影响单 Run；子 Run本身仍是普通 Run，可通过旧结果页查看。
- 新数据库列均可空，历史 Run无需回填；不从相邻历史 Run猜 Batch。
- 回滚代码时保留新增表/可空列，不做破坏性数据库降级。
- 解释性重新启用必须另立计划，重新评估目标 10 模型的方法、artifact 白名单、结果契约和前端；不得只把常量改成 True 就发布。

## 16. 本轮明确不做

- 不新增回归任务、PLSR、SVR 或回归损失。
- 不实现模型集成投票、stacking 或自动部署最佳模型。
- 不恢复任何可解释性、Grad-CAM、遮挡重要性、VIP/特征图前端展示；sPLS 选中特征只进入参数审计。
- 不伪造 ROC/AUC、Precision-Recall、综合评分、置信区间或没有真实重复数据的误差条。
- 不自动移动/删除历史 Run、用户数据、模型文件或压缩包。
- 不在一个 Python 进程内为多个深度模型擅自开线程并发，也不自动抢占多张 GPU。

## 17. 可选审查检查点

1. 任务 1–2：确认“前端 10 项、后端全部保留”的可见性边界与历史兼容。
2. 任务 3–5：独立审查 sPLS-DA 真值、PCA-SVM 泄漏和 5 折选参。
3. 任务 6：确认可解释性在计算、投影、下载、UI 四层全部关闭。
4. 任务 7–8：审查 Batch 事务、repeat seed 和比较统计口径。
5. 任务 9–11：完成两套前端和全量/真实数据门禁后再发布。
