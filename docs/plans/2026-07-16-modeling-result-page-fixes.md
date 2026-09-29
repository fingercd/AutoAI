# 建模页与建模结果页问题修复实施计划

> **历史计划归档（原计划日期：2026-07-16；归档标识：2026-09-29）：** 本文保留当时的目标、方案和验收记录，不能据此判断功能已交付或按文中的分支、推送、部署命令操作。当前事实与工作流程以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 和[文档导航](../README.md)为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`。不得创建新分支、worktree、fork 或可开发复制；开发用 Git 命令仅在该服务器 Pan 根目录执行。本次整理不提交、推送或部署，GitHub 其他分支保留。

> 日期：2026-07-16
> 状态：已实施
> 范围：FastAPI 后端、原生静态前端、Run/Artifact 契约、测试与文档
> 约束：不引入 React/Vue/Vite，不让训练 HTTP 请求直接执行训练，不伪造当前产物不存在的指标或图表。

## 1. 目标与成功标准

本轮目标是在不重写现有静态前端的前提下，解决用户截图中建模页布局、训练完成跳转、结果字段缺失、三分区分析、训练曲线、解释性、下载和最近任务展示问题，并消除 Web、Worker 与静态前端版本不一致时的静默错误。

完成后应满足：

1. 建模页数据集区域略宽于训练配置区域，两列间距增加；“每组测量数”在正常桌面宽度下一行显示。
2. 新训练完成后出现浏览器中央对话框，默认 3 秒进入 `#/results?run_id=...`，可立即查看，也可选择留在当前页。
3. 新 Run 不再被错误标为“历史 Run 的兼容投影”；前后端版本不一致时明确提示需要重启服务。
4. 结果概览能展示真实可恢复的时间、数据集、样本、类别和特征信息；无法恢复的旧字段明确显示 `-`，不得猜测。
5. Train、Valid、Test 均展示混淆矩阵、各类别指标和预测结果分布；三个混淆矩阵在宽屏同一行。
6. 预测结果分布改为竖向柱状图，显示真实数、预测数和数值刻度。
7. 传统机器学习模型完全不渲染“训练过程曲线”区域；深度模型曲线具有横纵轴、刻度、网格和明确单位。
8. 新 Run 不再计算、写出、投影或展示全局重要性；结果页只保留单样品解释。
9. 单样品解释能够从真实 artifact 懒加载，显示样品曲线、第一重要区间、窗口热力条和中文图例。
10. 结果下载区按 Manifest 白名单展示真实文件；每个文件单独下载，失败和禁用原因明确。
11. “最近任务”显示 CSV 原始文件名、训练时间和训练耗时。
12. 页面刷新、服务重启后仍可由 Run ID 恢复结果；历史 Run 的只读访问不被破坏。

## 2. 本次已确认的代码与运行事实

### 2.1 当前运行中的 Web 与磁盘静态前端版本不一致

现场检查 `http://127.0.0.1:8000` 得到：

- `/health` 只返回 `{"status":"ok"}`，没有当前代码应返回的 `deployment_mode` 和 `worker` 摘要。
- 当前运行服务的 OpenAPI 中没有 `GET /api/training/runs/{run_id}/result`。
- 当前运行服务也没有 `projection=summary` 列表参数。
- 同一个服务动态托管的 `static/js/run-results.js` 已包含：
  - `run-result-v1`
  - `/api/training/runs/{run_id}/result`
  - 3 秒跳转
  - 新结果页逻辑

因此浏览器实际处于“新静态前端 + 旧 Web/Worker 进程”的混合版本。

`static/js/run-results.js:192` 在 `/result` 返回 404 时无条件退回旧 `/api/training/runs/{run_id}`，随后 `static/js/run-results.js:129-184` 将响应归一化为 `legacy-run-projection`，并固定加入：

> 这是历史 Run 的兼容投影；部分时间、数据快照或下载清单可能未记录。

这正是截图中“刚训练却被说成历史 Run”的直接原因，不是该 Run 真正来自历史数据。

### 2.2 截图中的两个 Run 已生成真实指标和解释性文件

已检查：

- `62ad9f778e0d416dae9066f8c005c628`（PLS-DA）
- `6dd0f7edcfff4b4e92747c2d2dab7ae9`（DSCARNet）

两者目录都真实包含：

- `metrics.json`
- `cv_metrics.json`
- `predictions.csv`
- `fold_metrics.csv`
- `sample_feature_importance.json/csv`
- `manifest.json`
- 模型元数据和标签映射

其中 `metrics.json` 已包含 Train、Valid、Test 三个分区的：

- Accuracy / Balanced Accuracy
- Macro Precision / Recall / F1
- 混淆矩阵
- classification report

单样品解释文件也不是空文件：

- PLS-DA：9 个测试样品、160 点曲线、80 个解释窗口。
- DSCARNet：9 个测试样品、160 点曲线、160 个回投窗口。

因此“指标、单样品解释、下载全部不显示”的主要问题发生在结果接口和前端投影层，不是本次训练没有生成产物。

### 2.3 这两个 Run 由旧 Worker 代码生成

两个 Run 的 `manifest.json` 均缺少当前 `run-artifact-manifest-v2` 所需的 `schema_version`；SQLite 中的：

- `started_at`
- `finished_at`
- `dataset_snapshot_json`

也没有被旧 Worker 正确更新。

而当前磁盘代码已经具备：

- `GET /api/training/runs/{run_id}/result`
- `run-result-v1`
- v2 Manifest
- Run 创建时数据集名称和 SHA-256 快照
- Worker claim 时记录 `started_at`
- Worker 完成时记录 `finished_at`

所以实施前必须先解决进程版本一致性，否则只改静态文件仍会重复出现相同问题。

### 2.4 当前结果页只把 Test 投影到分析区域

`backend/app/runs/result_projection.py:188` 已构造 `metrics.splits.train/valid/test`，但 `analysis` 中只有从主指标（Test）提取的：

- `confusion_matrix`
- `classification_report`
- `prediction_distribution`

对应代码位于 `backend/app/runs/result_projection.py:330-415`。

前端：

- `renderConfusion()`：`static/js/run-results.js:426`
- `renderClassMetrics()`：`static/js/run-results.js:451`
- `renderDistribution()`：`static/js/run-results.js:494`

也只读取 `result.analysis` 的单份 Test 数据。因此 Train/Valid 数据虽然已在 artifact 中存在，但结果页没有消费。

### 2.5 当前训练曲线与模型类型处理不完整

- `static/js/run-results.js:625` 在没有曲线时仍返回一个“训练过程”空卡片，所以传统模型仍会显示这一部分。
- `backend/app/training.py` 会为传统模型人为写入一行 `epoch=1` 的 `history.csv`，但这实际是参数选择摘要，不是真实 epoch 曲线。
- `static/js/run-results.js:581` 只画外框和折线，没有 X/Y 数值刻度、网格线或轴标题。

### 2.6 当前全局重要性仍在正式结果模块中

虽然 `backend/tests/test_task13_ui_contract.py:133` 的测试名是“只渲染单样品重要性”，但该测试只检查了 `static/index.html` 中没有旧函数，没有检查正式模块 `static/js/run-results.js`。

当前仍存在：

- `renderGlobalImportance()`：`static/js/run-results.js:728`
- 全局 artifact 自动加载：`static/js/run-results.js:819-825`
- 全局聚合：`backend/app/feature_selection.py:848`
- 全局文件写出：`backend/app/feature_selection.py:963`
- 训练主链路生成全局文件：`backend/app/training.py:718-732`
- 全局 artifact 白名单：`backend/app/runs/artifacts.py:33-34`
- 结果接口投影 `explainability.global`：`backend/app/runs/result_projection.py:421`

### 2.7 当前单样品解释前端只显示文字条目

`static/js/run-results.js:743` 当前加载完成后只展示：

- 真实标签
- 预测标签
- 预测概率
- Top 区间横条

没有实现项目文档中已约定的：

- 样品曲线
- 第一重要红色区间
- 窗口热力条
- 中文颜色图例
- 数值坐标轴

旧结果回退时又没有 `artifacts[]` 下载描述，导致懒加载按钮也不会出现。

### 2.8 当前下载 UI 依赖 `run-result-v1.artifacts[]`

`static/js/run-results.js:886` 只根据 `result.artifacts` 渲染下载卡片。旧接口回退投影通常没有安全 artifact 描述，所以页面显示“没有可展示的下载清单”。

当前后端磁盘代码的 `RunArtifactWriter.descriptors()` 已能提供：

- label
- category
- format
- size
- integrity
- downloadable
- reason
- download_url
- suggested_filename

因此恢复 `/result` 路由和 Manifest 兼容描述后，不需要建立第二套下载机制。

## 3. 问题优先级清单

### P0

本次未发现会直接写错训练标签、越权下载或破坏 Run 状态机的新 P0 问题。现有下载仍受 Principal scope、Manifest 和路径边界控制；修复时必须保持这些约束。

### P1-1：Web、Worker 和静态前端版本不一致

- **现象：** 新页面请求 `/result` 得到 404，静默回退为旧接口。
- **证据：** 运行中 OpenAPI 无 `/result`，但服务返回的静态 JS 已包含 `/result`。
- **影响：** 新 Run 被错误称为历史 Run；概览、三分区指标、解释性、下载全部降级。
- **推荐方案：** 增加接口/Worker 契约版本握手；部署时协调重启 Web 与 Worker；前端不再把“路由缺失”当“历史 Run”。
- **修改范围：** 健康接口、Worker 心跳、启动器、前端结果请求、部署文档和测试。
- **风险与兼容性：** Worker 心跳表只做加列迁移；旧 Worker 没有版本时标为不兼容，但不删除其数据。
- **验证方法：** OpenAPI 存在 `/result`；`/health` 返回兼容版本；新 Run Manifest 为 v2。

### P1-2：新 Run 的时间和数据快照可能为空

- **现象：** 创建时间、开始时间、耗时、Sample_ID 数、类别数、特征数显示 `-`。
- **证据：** 示例 Run 的 SQLite `started_at/finished_at` 为空，`dataset_snapshot_json={}`；旧 status 只保存了部分字段。
- **影响：** 结果不可追溯，用户无法判断训练时长和数据规模。
- **推荐方案：** 新进程严格以 SQLite 时间和创建时快照为准；对旧 Run 从现有 artifact 做只读恢复。
- **修改范围：** Run 创建、Worker、结果投影、摘要投影和测试。
- **风险与兼容性：** 不回写或猜测无法恢复的旧时间；只补充有证据的字段。
- **验证方法：** 新 Run 的三个时间字段和耗时非空；旧 Run 至少恢复文件名、曲线数、类别数和特征数。

### P1-3：单样品解释与下载入口被兼容投影截断

- **现象：** artifact 文件存在，但页面没有加载按钮和下载卡片。
- **证据：** 示例 Run 的解释文件约 0.46–0.57 MB；旧回退结果没有 `artifacts[]`。
- **影响：** 核心结果不可使用。
- **推荐方案：** 由 `/result` 统一返回安全描述；旧 Manifest 使用兼容描述，不再退回无清单的旧页面模型。
- **修改范围：** 结果请求、Manifest 描述、解释性懒加载、下载 UI 和测试。
- **风险与兼容性：** 不放开模型文件和 joblib；下载前继续校验 SHA-256/大小。
- **验证方法：** 示例旧 Run 和新 Run 都能显示允许下载的真实文件；禁用项说明原因。

### P1-4：三分区分析数据未进入页面契约

- **现象：** 混淆矩阵、各类别指标、预测分布只有 Test。
- **证据：** `metrics.json` 已有三份数据，`analysis` 只取主 Test。
- **影响：** 用户无法判断拟合、验证和泛化差异。
- **推荐方案：** 在 `run-result-v1` 中增加 `analysis.splits.train/valid/test`。
- **修改范围：** 结果投影、前端分析组件、契约文档和测试。
- **风险与兼容性：** 仅新增字段，保留现有 `analysis.confusion_matrix` 等 Test 兼容字段一个发布周期。
- **验证方法：** Holdout 和 CV 都有三组分析；CV 口径标签正确。

### P2-1：传统模型显示虚假的训练曲线区域

- **现象：** PLS-DA 等传统模型显示空曲线或单点历史。
- **证据：** 前端无数据仍返回卡片；训练端写入合成的 epoch 1 行。
- **影响：** 容易让用户误以为传统模型也进行了 epoch 优化。
- **推荐方案：** 传统模型不生成 `history.csv`，不渲染训练曲线；参数搜索保留在训练审计和 CSV。
- **修改范围：** 训练写出、artifact 适用性、前端和测试。
- **风险与兼容性：** 旧 Run 的 `history.csv` 仍可只读，但在结果页标为不适用。
- **验证方法：** 六个传统模型无曲线区；深度模型仍正常显示。

### P2-2：深度学习曲线缺少数值坐标

- **现象：** 只有折线和图例，没有 X/Y 刻度。
- **影响：** 无法判断 epoch、loss 和指标的绝对大小。
- **推荐方案：** 补齐通用坐标、刻度、网格、域计算和 resize 重绘。
- **修改范围：** `static/js/run-results.js` 和前端纯函数测试。
- **风险与兼容性：** Canvas 尺寸和高 DPI 处理需避免文字模糊。
- **验证方法：** 固定历史数据截图检查；常数序列、单 epoch 和多折均可读。

### P2-3：全局重要性与当前产品方向不一致

- **现象：** 页面和新 Run 仍生成、展示全局重要性。
- **影响：** 页面冗余，且用户已明确只需要单样品解释。
- **推荐方案：** 新主链路停止生成和展示；历史两个文件名仅保留只读兼容。
- **修改范围：** 训练、解释性模块、状态/结果投影、artifact catalog、前端、测试和文档。
- **风险与兼容性：** “彻底删除全部兼容代码”会破坏历史下载入口；本计划不采取该破坏性方案。
- **验证方法：** 新 Run 目录和结果接口均无全局重要性；历史直接链接仍按原权限策略工作。

### P2-4：单样品解释展示不完整

- **现象：** 当前即使加载成功，也只有 Top 区间文字条。
- **影响：** 无法将解释区间与原始光谱/色谱位置对应。
- **推荐方案：** 实现曲线、红色主区间、热力条、坐标和图例。
- **修改范围：** 前端结果模块与测试。
- **风险与兼容性：** 不改变解释算法，只消费现有 JSON。
- **验证方法：** PLS-DA、CNN1D、DSCARNet 各验证一个真实样品。

### P2-5：最近任务信息不足

- **现象：** 只显示 Run ID、模型和状态。
- **影响：** 多个任务难以识别。
- **推荐方案：** 增加 CSV 名、训练开始时间和耗时。
- **修改范围：** 摘要接口、结果页 landing 表格和测试。
- **风险与兼容性：** 响应只新增字段。
- **验证方法：** 最近任务和训练记录的文件名与时间一致。

### P3-1：建模页局部布局拥挤

- **现象：** 两列距离偏近，“每组测量数”换行。
- **影响：** 桌面端观感和信息扫描效率下降。
- **推荐方案：** 只增加建模页专用网格类，不修改全站通用 `.grid`。
- **修改范围：** `static/index.html` 的 HTML class 与内联 CSS。
- **风险与兼容性：** 需在 900px、600px 断点回归。
- **验证方法：** 1440px 宽度下标签单行；窄屏仍能自然换列。

## 4. 目标结果页信息架构

结果页保持当前分区结构，但调整为：

1. **任务概览**
   - Run ID、模型、数据集文件名、评估方式、主指标口径。
   - 创建、开始、结束时间和耗时。
   - 曲线数、Sample_ID 数、类别数、特征数和数据指纹。
   - 只有真正的兼容 Manifest 才显示兼容说明，且文案描述“产物格式版本”，不把刚生成的任务称为历史任务。

2. **核心指标**
   - Test 或 pooled OOF 主指标卡片。
   - 下方保留 Train/Valid/Test 标量指标对照表。

3. **混淆矩阵**
   - Train、Valid、Test 三卡同一行。
   - 小屏堆叠；类别多时每卡内部横向滚动。
   - 每卡显示聚合口径。

4. **各类别指标**
   - Train、Valid、Test 三卡。
   - 表头统一为 Precision、Recall、F1、Support。

5. **预测结果分布**
   - Train、Valid、Test 三个竖向柱状图。
   - 每个类别两根柱：真实、预测。
   - Y 轴从 0 开始，显示整数刻度和精确数值。

6. **训练过程**
   - 仅深度学习模型渲染。
   - Loss 与验证指标两个图。
   - 多折任务保留折选择器。

7. **训练与参数审计**
   - 传统模型显示参数搜索及选择指标。
   - 深度模型显示最佳验证 Loss、实际 epoch 和最低学习率。
   - 默认折叠。

8. **单样品解释**
   - 单列宽卡片。
   - 懒加载后显示样品选择器、分类结果、曲线、红色主区间、热力条和 Top 区间。
   - 不再出现全局重要性卡片。

9. **结果下载**
   - 继续按指标、预测、训练、解释、元数据分组。
   - 每张卡同时显示业务名称、实际文件名、格式、大小和状态。

## 5. 结果接口与数据结构调整

### 5.1 健康与版本能力

扩展 `/health`，建议返回：

```json
{
  "status": "ok",
  "deployment_mode": "local",
  "contracts": {
    "run_result": "run-result-v1",
    "artifact_manifest": "run-artifact-manifest-v2",
    "run_summary": "v1"
  },
  "worker": {
    "available": true,
    "compatible": true,
    "contract_version": "run-artifact-manifest-v2"
  }
}
```

Worker 心跳表增加可空的 `contract_version` 字段。旧 Worker 心跳没有版本时：

- 不删除心跳；
- `available` 可继续表示进程存活；
- `compatible=false`；
- 前端禁止误导用户开始一个必然生成旧格式的新任务，并提示重启 Worker。

### 5.2 三分区分析

在 `analysis` 中新增：

```json
{
  "analysis": {
    "splits": {
      "train": {
        "aggregation": "direct",
        "confusion_matrix": [],
        "classification_report": {},
        "prediction_distribution": {}
      },
      "valid": {},
      "test": {}
    }
  }
}
```

口径规则：

- 普通 holdout：三组均为 `direct`。
- 外部测试集：Train/Valid 为主数据直接划分，Test 为外部测试直接结果。
- 留一 Sample_ID CV：
  - Test 使用 pooled OOF，单个样本只出现于其 test 折。
  - Train/Valid 图表使用跨折预测合并结果；同一样本可能在不同折重复出现，必须标注“跨折合并审计，不等同于独立样本集”。
  - Train/Valid 标量主展示仍使用 fold mean，不能拿 pooled 图表的标量替换折均值。

为兼容现有调用方，旧的：

- `analysis.confusion_matrix`
- `analysis.classification_report`
- `analysis.prediction_distribution`

暂时继续映射 Test，一个发布周期后再评估是否移除。

### 5.3 最近任务摘要

`projection=summary` 增加：

```json
{
  "dataset_name": "data.csv",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "duration_seconds": 12.5
}
```

前端“训练时间”显示规则：

1. 优先 `started_at`；
2. 尚未开始时显示 `created_at` 并标记“创建”；
3. 耗时单独显示，不把创建到完成的排队时间误当训练耗时。

### 5.4 解释性契约

新 Run 的结果接口只提供：

```json
{
  "explainability": {
    "samples": {
      "status": "ready",
      "artifact": "sample_feature_importance.json",
      "csv_artifact": "sample_feature_importance.csv",
      "sample_count": 9,
      "method": "sample_occlusion_log_loss"
    }
  }
}
```

新训练不再生成：

- `feature_importance.json`
- `feature_importance.csv`
- `status.feature_importance`
- `explainability.global`

兼容原则：

- 不修改或删除历史 Run 目录中的上述文件。
- 结果页和最近任务不展示它们。
- 历史已知下载 URL 在原 Principal、Manifest 和完整性约束下保持只读兼容。
- 不允许新 Manifest 再登记这两个文件。

### 5.5 Artifact 下载

继续以 `ARTIFACT_CATALOG` 和 Manifest 为权威，不建立前端硬编码文件列表。

新主链路公开：

- `metrics.json`
- `cv_metrics.json`
- `fold_metrics.csv`
- `predictions.csv`
- `cv_predictions.csv`（适用时）
- `history.csv`（仅深度模型）
- `hyperparameter_search.csv`（仅传统模型）
- `sample_feature_importance.json/csv`
- `dscarnet_mapping.json`（仅 DSCARNet）
- `model_metadata.json`
- `label_map.json`
- `split.json`
- 安全的 `config.json`

继续不公开：

- `model.pkl`
- `model.pt`
- `*.joblib`
- `status.json`
- 含服务器路径的 `config.json`

## 6. 分阶段实施任务

### 阶段一：修复运行版本一致性和兼容判定

#### 任务 1：增加 Web/Worker 契约版本握手

**目标：** 让浏览器、Web 和 Worker 能明确判断彼此是否属于同一结果契约版本，阻止混合版本继续产生不可解释的降级结果。

**前置依赖：** 复用现有 `/health`、Worker heartbeat 和数据库初始化机制；部署方能够在升级时协调重启 Web 与 Worker。

**涉及文件或模块：**

- 创建：`backend/app/version.py`（集中声明接口和 Manifest 契约版本）
- 修改：`backend/app/routers/catalog.py`
- 修改：`backend/app/runs/repository.py`
- 修改：`backend/app/runs/worker.py`
- 修改：`static/js/run-results.js`
- 修改：`static/index.html`
- 修改：`run.py`
- 测试：`backend/tests/test_run_worker.py`
- 测试：`backend/tests/test_model_catalog.py` 或新增健康接口专项测试

**接口与依赖：**

- 消费：现有 Worker heartbeat。
- 产出：`/health.contracts`、`worker.contract_version`、`worker.compatible`。

**实施内容：**

1. 为 `worker_heartbeats` 添加可空 `contract_version` 列，使用现有初始化流程幂等 `ALTER TABLE`。
2. Worker 每次心跳写入当前 Manifest 契约版本。
3. `/health` 返回 Web 支持的结果契约和当前活跃 Worker 的契约版本。
4. 前端启动后保存能力信息。
5. `fetchResult()` 只有在后端明确不支持结果契约时才允许旧接口兼容；如果后端宣称支持而某个 Run 返回 404，应直接显示“任务不存在”。
6. Web 支持新接口但 Worker 不兼容时，建模页显示明确错误，开始训练按钮禁用或二次确认，不再悄悄生成旧格式 Run。
7. `run.py` 启动后检查 Web/Worker 版本一致性；部署说明要求升级时同时停止并重启两者。

**验收标准：**

- `/health` 可判断 Web 和 Worker 是否兼容。
- 新前端连接旧 Web 时显示“后端版本过旧”，不显示“历史 Run”。
- 新 Web 连接旧 Worker 时明确提示 Worker 版本不一致。
- 正常版本下训练入口不受影响。

**验证方式：**

- 单元测试旧 heartbeat、无版本 heartbeat、新版本 heartbeat。
- 启动旧/新组合的模拟 API，验证前端兼容分支。
- 人工检查 OpenAPI 必须存在 `/api/training/runs/{run_id}/result`。

**风险与回滚：**

- 风险：加列迁移失败会影响健康摘要。
- 回滚：版本列保持可空；读取失败时返回 `compatible=false`，不影响 Run 主表。

#### 任务 2：修正新 Run 与旧 Manifest 的提示语义

**目标：** 准确区分“后端接口过旧”“旧 Manifest 兼容读取”和“真正历史 Run”，避免把刚完成的训练误报为历史任务。

**前置依赖：** 任务 1 提供的后端能力信息和 Worker 契约版本；现有 legacy Run 归一化逻辑保持可测试。

**涉及文件或模块：**

- 修改：`static/js/run-results.js`
- 修改：`backend/app/runs/result_projection.py`
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_run_result_contract_v1.py`

**实施内容：**

1. 将“旧接口投影”和“旧 Manifest 格式”区分为两个状态。
2. 只有真正通过显式 legacy capability 进入旧接口时才使用 `legacy-run-projection`。
3. Manifest 缺少 v2 schema 时，提示：
   - “该 Run 的产物清单由旧版 Worker 生成，已按兼容策略读取。”
4. 不再使用“历史 Run”描述刚刚完成但格式较旧的任务。

**验收标准：**

- 示例两个 Run 在新 Web 下不再出现截图中的固定“历史 Run”文案。
- 真实历史 Run 仍能看到准确的兼容说明。

**验证方式：**

- 构造新 DB Run + v1 Manifest。
- 构造无 DB 的真实 legacy Run。
- 分别断言提示文本和下载能力。

**风险与回滚：**

- 仅调整判定和文案；可回滚到旧兼容归一化，但不建议恢复无条件 404 fallback。

### 阶段二：完善结果数据契约和可恢复元数据

#### 任务 3：补齐时间、数据快照和最近任务摘要

**目标：** 为新 Run 建立稳定、可刷新恢复的概览元数据，并在不猜测的前提下尽可能恢复旧 Run 的可追溯信息。

**前置依赖：** 任务 1 确保执行新训练的 Web/Worker 版本一致；现有 Run 数据库、status 和 artifact 文件仍可只读访问。

**涉及文件或模块：**

- 修改：`backend/app/routers/runs.py`
- 修改：`backend/app/runs/result_projection.py`
- 修改：`backend/app/runs/status_projection.py`
- 修改：`backend/app/runs/execution.py`
- 修改：`backend/app/runs/repository.py`
- 测试：`backend/tests/test_run_result_page_backend.py`
- 测试：`backend/tests/test_run_result_contract_v1.py`
- 测试：`backend/tests/test_run_status_projection.py`

**实施内容：**

1. 保证新 Run 创建时快照至少包含：
   - 数据集原始文件名
   - SHA-256
2. Worker 解析训练数据后补齐：
   - curve_count
   - sample_id_count
   - class_count
   - feature_count
   - test_curve_count
3. 保证 claim 和 finish 后的最新 `RunRecord` 被写入最终 status 投影，避免 final status 继续携带训练前的空时间。
4. 对旧 Run 只读恢复：
   - 曲线数：`status.sample_count`
   - Sample_ID 数：`split.json` 中出现的唯一 Sample_ID
   - 类别数：`label_map.json`
   - 特征数：`model_metadata.json`
   - 文件名：DatasetRepository 的 `original_name`
   - 结束时间：数据库优先，其次 status；Manifest 时间只能标注为“产物发布时间”，不能伪装成训练结束时间
5. `_summary_projection()` 增加 `duration_seconds`。

**验收标准：**

- 新 Run 概览字段完整。
- 示例旧 Run 至少能恢复 `data.csv`、90 条曲线、30 个 Sample_ID、2 类、160 特征。
- 无法恢复的开始时间仍为 `-`，不生成假时间。

**验证方式：**

- 新 Run 全字段测试。
- 空快照旧 Run 恢复测试。
- 服务重启后重复请求结果一致。

**风险与回滚：**

- 读取 `split.json` 必须限制为本 Run 目录，解析失败只返回空值。
- 回滚时可只撤销旧 Run 恢复，不影响新 Run 权威字段。

#### 任务 4：新增三分区分析投影

**目标：** 把已有 Train、Valid、Test 真实评估产物统一投影成前端可稳定消费的三分区分析契约。

**前置依赖：** 现有 `metrics.splits` 和预测审计数据可用；任务 3 明确 Run 元数据与评估口径的权威来源。

**涉及文件或模块：**

- 修改：`backend/app/runs/result_projection.py`
- 修改：`docs/run_result_contract.md`
- 修改：`docs/frontend_backend_handoff.md`
- 测试：`backend/tests/test_run_result_contract_v1.py`

**实施内容：**

1. 创建统一 helper，把每个 split 的图表来源解析为：
   - 标量值来源
   - 图表/分类报告来源
   - aggregation 标签
2. 普通 holdout 从 `metrics.splits.<name>.values` 读取。
3. CV：
   - Test 从 pooled OOF 读取。
   - Train/Valid 的图表从 pooled 审计数据读取。
   - 标量仍保留 fold mean/fold std。
4. 生成 `analysis.splits`。
5. 保留 Test 旧字段作为兼容别名。

**验收标准：**

- 三个 split 均有矩阵、分类报告和分布。
- CV Test 矩阵来自 pooled OOF。
- 不从 fold mean 标量反推不存在的混淆矩阵。

**验证方式：**

- Holdout 二分类。
- CV 二分类，故意构造 pooled OOF F1 与 fold mean F1 不同。
- 多分类矩阵。

**风险与回滚：**

- CV Train/Valid 含重复样本，必须在契约和 UI 明示。
- 新字段为增量扩展，可独立回滚。

### 阶段三：简化解释性主链路并恢复单样品展示

#### 任务 5：停止新 Run 的全局重要性生成与展示

**目标：** 从新训练主链路和正式结果页删除全局重要性，只保留用户实际需要的单样品解释，同时不破坏历史文件。

**前置依赖：** 任务 2 已明确旧 Manifest 的兼容边界；现有单样品解释产物能够独立生成和下载。

**涉及文件或模块：**

- 修改：`backend/app/training.py`
- 修改：`backend/app/feature_selection.py`
- 修改：`backend/app/runs/artifacts.py`
- 修改：`backend/app/runs/status_projection.py`
- 修改：`backend/app/runs/result_projection.py`
- 修改：`static/js/run-results.js`
- 测试：`backend/tests/test_smoke.py`
- 测试：`backend/tests/test_run_artifact_manifest.py`
- 测试：`backend/tests/test_task13_status_projection.py`
- 测试：`backend/tests/test_task13_ui_contract.py`

**实施内容：**

1. `_write_deep_explainability_artifacts()` 改为只写 `sample_feature_importance.json/csv`。
2. 删除新训练路径对：
   - `aggregate_sample_feature_importance`
   - `write_feature_importance_artifacts`
   - `feature_summary`
   的依赖。
3. 删除不再有调用者的全局聚合和序列化函数。
4. 新 status 不再写 `feature_importance`。
5. 新结果响应不再写 `explainability.global`。
6. 新 Manifest catalog 不再主动列出全局文件。
7. 前端删除全局卡片、加载函数和文案。
8. 对历史文件名只在下载解析层保留窄范围只读兼容，不加入新页面，不让新 Run 生成。

**验收标准：**

- 新 Run 目录没有 `feature_importance.json/csv`。
- 新 `/result` 没有全局解释摘要。
- 结果页模型解释区域只有单样品解释。
- 历史文件不被删除。

**验证方式：**

- 搜索生产代码中无全局渲染和聚合调用。
- 传统、CNN、DSCARNet 各跑解释性测试。
- 历史 URL 兼容测试。

**风险与回滚：**

- 风险：直接删除 catalog 会影响历史 v2 Manifest 下载。
- 回滚/兼容：保留独立的 legacy-only 文件名策略；不得让它重新进入新 Manifest。

#### 任务 6：实现完整单样品解释组件

**目标：** 使用项目已经生成的真实单样品解释 artifact，恢复可选择、可读、可重试且适配窄屏的完整解释视图。

**前置依赖：** 任务 5 固化只保留单样品解释的新契约；任务 1、2 保证 `/result` 和 artifact descriptor 能被可靠获取。

**涉及文件或模块：**

- 修改：`static/js/run-results.js`
- 修改：`static/index.html`
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_task13_ui_contract.py`
- 测试：`backend/tests/test_explainability_v2.py`

**实施内容：**

1. 保留“用户点击后才下载大 JSON”的懒加载策略。
2. 加载后校验：
   - `status=ready`
   - `samples` 非空
   - curve 与 sample_x_axis 长度一致
   - windows 的索引在曲线范围内
3. 样品选择器显示：
   - Sample_ID / Name
   - 真实类别
   - 预测类别
   - 正确与否
4. 主图：
   - 画当前样品曲线。
   - X/Y 轴有数值刻度。
   - `primary_segment` 使用半透明红色区域。
5. 主图下方画窗口热力条：
   - 使用 `normalized_importance`。
   - 低值浅色，高值红色。
   - 提供“重要性低—高”中文图例。
6. 下方保留 Top 区间明细。
7. 使用 payload 缓存，切换样品不重复请求。
8. Canvas 随容器 resize 重绘；窄屏不横向溢出。
9. 403、404、409、网络失败分别给出可重试提示。

**验收标准：**

- 示例两个 Run 均可加载并切换 9 个样品。
- PLS-DA 遮挡解释和 DSCARNet 回投解释均能画图。
- 第一重要区间位置与 JSON 索引一致。
- 页面首屏未点击时不下载单样品大文件。

**验证方式：**

- 纯函数测试坐标域、窗口裁剪和颜色映射。
- 用示例 artifact 做浏览器人工检查。
- 模拟损坏和长度不一致的 payload。

**风险与回滚：**

- 不修改训练算法，前端绘图可独立回滚为 Top 列表。

### 阶段四：重构分析图表和训练曲线展示

#### 任务 7：三分区混淆矩阵与各类别指标

**目标：** 将 Train、Valid、Test 的混淆矩阵和各类别指标分层展示，并在桌面宽屏保持三个矩阵同一行。

**前置依赖：** 任务 4 已提供 `analysis.splits.train/valid/test` 及其 aggregation 说明。

**涉及文件或模块：**

- 修改：`static/js/run-results.js`
- 修改：`static/index.html`
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_task13_ui_contract.py`

**实施内容：**

1. 将单结果函数改为接收 `splitName + splitAnalysis`。
2. 混淆矩阵区域使用专用三列网格：
   - 桌面端三卡同一行。
   - 小于约 1100px 时允许横向滚动或两列。
   - 小于 700px 时单列。
3. 每张卡显示 Train/Valid/Test 和 aggregation 说明。
4. 各类别指标同样生成三份表。
5. 类别数较多时卡片内部滚动，不压缩到不可读。

**验收标准：**

- Holdout 三矩阵同一行。
- CV 标签明确，不混淆 pooled OOF 与 fold mean。
- 多分类表格可滚动。

**验证方式：**

- 二分类、多分类、空 split 和部分缺失数据。
- 1440、1024、768、390px 人工检查。

**风险与回滚：**

- 三列在中等屏幕可能过窄；通过专用断点回退，不修改全局网格。

#### 任务 8：竖向预测分布

**目标：** 把三分区真实/预测类别数量改为带数值刻度的竖向分组柱状图，并保持数据与混淆矩阵一致。

**前置依赖：** 任务 4 已提供三个 split 的分布数据；任务 7 已建立三分区响应式卡片布局。

**涉及文件或模块：**

- 修改：`static/js/run-results.js`
- 修改：`static/index.html`
- 测试：`backend/tests/test_result_frontend_contract.py`

**实施内容：**

1. 移除当前水平 `.distribution-bars`。
2. 每个 split 绘制竖向分组柱状图。
3. 每类显示“真实”和“预测”两根柱。
4. Y 轴从 0 开始，生成 4–6 个整数刻度。
5. 柱顶显示数值；类别名过长时截断并用 title 提供完整文本。
6. Canvas/DOM 添加可访问的文字摘要或隐藏表格。

**验收标准：**

- 三个 split 均为竖向图。
- 柱高和文字值与混淆矩阵行列和一致。
- 全零、单类和不平衡类别仍可读。

**验证方式：**

- 用固定矩阵断言 true/predicted counts。
- 浏览器检查窄屏布局。

**风险与回滚：**

- 若 Canvas 可访问性不足，保留同数据的简短表格作为 fallback。

#### 任务 9：传统模型隐藏曲线，深度模型补齐坐标

**目标：** 消除传统模型的伪训练曲线，并让深度模型训练曲线具有可解释的数值轴、刻度和异常数据保护。

**前置依赖：** 结果契约能够稳定提供模型 family 和真实 history 数据；历史传统 Run 允许保留文件但不展示。

**涉及文件或模块：**

- 修改：`backend/app/training.py`
- 修改：`backend/app/runs/artifacts.py`
- 修改：`static/js/run-results.js`
- 测试：`backend/tests/test_classification_policy.py`
- 测试：`backend/tests/test_run_result_contract_v1.py`
- 测试：`backend/tests/test_task13_ui_contract.py`

**实施内容：**

1. 传统模型不再向 `history_rows` 写合成 epoch。
2. 传统模型不创建 `history.csv`；参数搜索使用 `hyperparameter_search.csv`。
3. `renderHistory()` 在 `model.family=traditional_ml` 时直接返回 `null`。
4. 深度曲线：
   - X 轴使用真实 epoch。
   - Loss 自动计算数值域并增加边距。
   - Accuracy/F1 默认使用 0–1 域。
   - 显示 X/Y 轴标题、刻度和网格。
   - 单点或常数序列使用安全域，避免空图。
   - 只有存在有限值的 series 才画。
5. 多折切换后重新计算域。

**验收标准：**

- PLS-DA、SVM、随机森林等页面没有训练曲线标题或空卡片。
- CNN/DSCARNet 图上可读出 epoch、loss、accuracy/F1 数值。
- 单 epoch 数据不会画成空白。

**验证方式：**

- 传统/深度接口契约测试。
- 纯函数测试 tick 和 domain。
- 真实 CNN1D Run 浏览器检查。

**风险与回滚：**

- 历史传统 Run 的 `history.csv` 继续存在但不展示。
- 绘图升级可独立回滚，不影响 artifact。

### 阶段五：建模页局部布局和完成跳转

#### 任务 10：调整建模页两列宽度与摘要卡片

**目标：** 用局部 CSS 改善建模页数据集区域、训练配置区域和摘要卡片的可读性，不影响其他功能页。

**前置依赖：** 无后端依赖，可与阶段二至四并行；必须沿用现有 DOM 和视觉变量。

**涉及文件或模块：**

- 修改：`static/index.html`
- 测试：`backend/tests/test_task13_ui_contract.py`

**实施内容：**

1. 给 AI 建模首屏网格增加专用 class，例如 `.modeling-primary-grid`。
2. 桌面端建议：
   - 数据集列：约 `1.1–1.15fr`
   - 配置列：约 `0.85–0.9fr`
   - gap：由 16px 增至 20–24px
3. 不修改全站 `.grid`，避免预处理页面受影响。
4. `.summary-metrics` 的卡片最小宽度提高；“每组测量数”标签在桌面端 `white-space: nowrap`。
5. 在 900px 以下回到单列；在窄屏允许标签换行。

**验收标准：**

- 截图同等宽度下“每组测量数”一行显示。
- 两个主面板有更清晰间距。
- 预处理页布局无变化。

**验证方式：**

- 1440/1920px 桌面截图。
- 900/600px 断点检查。

**风险与回滚：**

- 只使用页面专用 class，回滚不影响其他页面。

#### 任务 11：中央 3 秒完成对话框与“留在当前页”

**目标：** 将训练完成提示改为可访问的中央对话框，保留 3 秒自动跳转，同时给予用户明确的立即查看和留页选择。

**前置依赖：** 任务 1、2 保证结果路由可用且错误语义准确；复用现有新 Run 标记和自动跳转清理逻辑。

**涉及文件或模块：**

- 修改：`static/index.html`
- 修改：`static/js/run-results.js`
- 测试：`backend/tests/test_result_frontend_contract.py`
- 测试：`backend/tests/test_task13_ui_contract.py`

**实施内容：**

1. 新增居中 modal/dialog，沿用认证对话框的遮罩和卡片风格。
2. 内容：
   - 训练完成
   - Run ID
   - “3 秒后进入建模结果”
3. 操作：
   - “立即查看结果”：取消定时器并跳转。
   - “留在当前页”：取消定时器、关闭对话框，保留当前训练成功状态。
4. 自动跳转失败时按钮始终可用。
5. 仅 `markNewRun()` 标记的当前标签页新 Run 自动弹出；查看历史 Run 不触发。
6. 支持：
   - Escape 留在当前页
   - 焦点锁定和关闭后焦点恢复
   - `prefers-reduced-motion`
7. 新训练开始时取消旧计时器和旧 modal。

**验收标准：**

- 新训练完成后 modal 位于浏览器中心。
- 3 秒自动跳转。
- 两个按钮行为正确。
- 页面刷新和历史 Run 不重复弹出。

**验证方式：**

- 纯函数测试计时和 Run 绑定。
- 人工测试立即查看、留在当前页、切换导航、连续两次训练。

**风险与回滚：**

- 防止多个 timer 并存；继续复用 `cancelAutoRedirect()` 作为唯一清理入口。

#### 任务 12：最近任务显示文件名与训练时间

**目标：** 让“最近任务”直接显示数据集 CSV 文件名、训练时间和耗时，并与训练记录页保持同一口径。

**前置依赖：** 任务 3 已补齐摘要投影中的数据集名称、时间和 `duration_seconds`。

**涉及文件或模块：**

- 修改：`static/js/run-results.js`
- 修改：`static/index.html`
- 修改：`backend/app/routers/runs.py`
- 测试：`backend/tests/test_run_result_contract_v1.py`
- 测试：`backend/tests/test_result_frontend_contract.py`

**实施内容：**

1. 最近任务表列调整为：
   - Run ID
   - 数据集
   - 模型
   - 状态
   - 训练时间
   - 耗时
   - 操作
2. 文件名优先：
   - `dataset_name`
   - `dataset.name`
   - `config.dataset_name`
3. 训练时间按第 5.3 节规则。
4. 长文件名使用省略号和 title。
5. 窄屏使用表格横向滚动，不删除关键列。

**验收标准：**

- 示例 Run 显示 `data.csv`。
- 新 Run 显示正确开始时间和耗时。
- 训练记录页和最近任务页口径一致。

**验证方式：**

- queued、running、succeeded、failed 四种状态。
- 长中文/英文文件名。

**风险与回滚：**

- 只新增摘要字段和列，可独立回滚。

### 阶段六：测试、真实数据验证与文档同步

#### 任务 13：补齐自动化测试，消除当前假阳性

**目标：** 为版本判定、三分区分析、解释性、下载、曲线和跳转建立能真实约束正式模块的回归门禁。

**前置依赖：** 任务 1–12 的接口和交互方案已经固定；测试不得依赖用户 Run 目录或提交本地数据。

**涉及文件或模块：**

- 修改：`backend/tests/test_result_frontend_contract.py`
- 修改：`backend/tests/test_task13_ui_contract.py`
- 修改：`backend/tests/test_run_result_contract_v1.py`
- 修改：`backend/tests/test_run_artifact_manifest.py`
- 修改：`backend/tests/test_run_status_projection.py`
- 修改：`backend/tests/test_smoke.py`
- 修改：`backend/tests/test_run_worker.py`

**实施内容：**

1. 修正“只渲染单样品解释”测试，使其同时扫描 `static/js/run-results.js`。
2. 增加以下纯函数测试：
   - 版本能力与 legacy fallback 判定
   - split 分析选择
   - 曲线可见性
   - chart domain/ticks
   - 最近任务显示模型
3. 后端契约测试：
   - 三分区分析
   - CV pooled OOF 语义
   - v1 Manifest 兼容
   - 新 Run 无全局 artifact
   - 单样品下载 descriptor
   - Worker 版本心跳
4. 下载测试：
   - 正常下载
   - 缺失
   - hash/size 不一致
   - 403
   - 旧 Manifest
5. modal 测试：
   - 3 秒
   - 立即跳转
   - 留在当前页
   - 非新 Run 不弹出

**验收标准：**

- 测试能在正式结果模块重新加入全局卡片时失败。
- 测试能在 `/result` 路由缺失但前端继续静默 fallback 时失败。
- 下载和单样品解释覆盖成功与失败路径。

**验证命令：**

```powershell
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_result_frontend_contract.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_run_result_contract_v1.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_run_artifact_manifest.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端内联脚本和模块分别执行 Node 语法检查。

**风险与回滚：**

- Windows pytest 临时目录可能受权限影响；验证时显式使用项目 `work/pytest-*` 作为 `--basetemp`，不把环境权限错误当测试失败。

#### 任务 14：真实数据闭环和部署升级验证

**目标：** 用同版本 Web/Worker、真实 `data.csv` 和现场旧 Run 完成最终闭环，并同步所有面向开发和部署的文档。

**前置依赖：** 任务 1–13 完成且专项测试通过；本机存在可用 `data.csv` 和约定的 PyTorch Python 环境。

**涉及文件或模块：**

- 不修改训练数据。
- 按结果同步更新：`README.md`
- 修改：`docs/frontend_backend_handoff.md`
- 修改：`docs/run_result_contract.md`
- 修改：`deploy/server_deploy.md`
- 视启动行为修改：`run.py`

**实施内容：**

1. 停止旧 Web 和旧 Worker，使用同一提交、同一 Python 环境重新启动。
2. 启动后检查：
   - `/health.contracts`
   - Worker compatible
   - OpenAPI `/result`
3. 使用本地 `data.csv` 跑：
   - 一个轻量传统模型（建议 PLS-DA）
   - 一个轻量深度模型（建议 CNN1D；DSCARNet 可作为额外验证）
4. 对每个 Run 验证：
   - 自动跳转
   - 刷新恢复
   - 三分区分析
   - 下载
   - 单样品解释
   - 时间和文件名
5. 再打开两个现场旧 Run，验证兼容读取。
6. 文档删除“新 Run 会生成全局重要性”的陈述，明确只保留历史只读兼容。

**验收标准：**

- Web 和 Worker 契约一致。
- 新 PLS-DA 无训练曲线、无全局文件，有三分区分析和单样品解释。
- 新 CNN1D 有带刻度曲线和单样品解释。
- 旧 Run 可读取真实产物，不被错误描述为历史任务本身。

**验证与证据：**

- 保存 API 响应摘要、Manifest 文件名清单和关键页面截图到 `work/`。
- 不提交 `data.csv`、Run 目录、截图缓存、模型和下载文件。

**风险与回滚：**

- 真实深度训练耗时较长时先用少量 epoch 验证 UI 契约，但正式验收仍需至少一次正常配置闭环。
- 升级前不改写历史 artifact；回滚代码后历史目录仍可用。

## 7. 任务依赖关系

推荐顺序：

1. 任务 1 → 任务 2：先建立版本事实，才能正确区分 legacy。
2. 任务 3 → 任务 4：先保证元数据，再扩展分析契约。
3. 任务 5 → 任务 6：先简化解释性数据源，再实现单样品 UI。
4. 任务 4 → 任务 7/8：三分区前端依赖后端 split 分析。
5. 任务 9 可与任务 7/8 并行，但必须在任务 13 前完成。
6. 任务 10/11/12 依赖任务 1 的版本能力，但不依赖训练算法修改。
7. 任务 13 覆盖所有任务。
8. 任务 14 是最终发布门禁。

可选提交检查点：

1. Web/Worker 版本握手与兼容判定。
2. 结果契约和三分区分析。
3. 解释性简化与单样品组件。
4. 前端布局、图表和完成 modal。
5. 测试与文档。

## 8. 兼容性与迁移策略

1. 不修改训练 HTTP 语义：请求仍只创建 queued Run。
2. 不改变 Hash 路由和导航顺序。
3. `run-result-v1` 以新增字段为主；Test 旧分析字段暂时保留。
4. Worker heartbeat 数据库迁移只增加可空列。
5. 历史 Run 目录不批量重写、不删除。
6. 旧 Manifest 只读兼容，但 UI 文案不再笼统称为“历史 Run”。
7. 新 Run 停止生成全局文件；历史全局文件不进入新 UI。
8. 下载仍经过 Principal scope、Manifest、catalog、路径和完整性校验。
9. 不把模型权重、pickle 或 joblib 放开下载。
10. 旧 Run 无法恢复的开始时间和耗时保持 `-`。

## 9. 明确不建议本轮实施的内容

1. 不引入 React、Vue、Vite、Chart.js 或其他前端框架/图表库。
2. 不重写整个 `static/index.html`。
3. 不改变模型结构、搜索空间、数据划分算法或解释算法。
4. 不新增 ROC-AUC、ROC、PR 等当前训练产物没有的数据。
5. 不把 Train/Valid/Test 缺失数据用 Test 复制填充。
6. 不直接把 `model.pkl/model.pt/joblib` 开放给浏览器。
7. 不为了修复当前两个 Run 批量改写所有历史 Manifest。
8. 不删除用户已有 Run、模型或训练数据。
9. 不让前端根据文件系统路径直接下载 artifact。
10. 不把静态文件更新等同于服务升级；Web 与 Worker 必须协调重启。

## 10. 尚需确认的问题

当前没有阻塞实施的问题，计划默认按以下方式执行：

- 用户所说“删除所有和全局重要性有关的代码”解释为：删除新主链路的计算、生成、契约、页面和新下载入口。
- 为遵守项目现有历史兼容约束，历史 `feature_importance.json/csv` 的窄范围只读下载兼容仍保留，但不会出现在新页面，也不会被新 Run 生成。

如果后续明确要求连历史文件下载也彻底移除，需要单独确认，因为那会破坏已有 Run 的历史下载 URL，属于兼容性变更。

## 11. 实施完成定义

只有同时满足以下条件才视为完成：

1. 当前运行 Web、Worker 与静态前端契约一致。
2. 两个现场旧 Run 可正确打开和下载允许的产物。
3. 新 PLS-DA 和新 CNN1D 各完成一次真实数据闭环。
4. 11 项用户需求逐项通过人工验收。
5. 全量 `backend/tests` 通过。
6. 前端脚本语法检查通过。
7. README、接口契约和部署说明与实现一致。
8. Git 中没有数据、Run artifact、模型、缓存或验证截图。
