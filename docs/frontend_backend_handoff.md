# SpecAutoAI 前后端接口契约

2026-09-05 增量：经典多模型对比页新增批次历史、归档与绘图接口，详见 `docs/model_comparison_contract.md` 的“经典对比页与持久归档”。Run summary 新增可空 batch_id；原 Run/Batch 契约、v2 和单模型结果页保持兼容。本轮没有改动训练请求和算法。

当前建模入口仅支持分类，具体六模型与 0904 特征工程见第六节。

> 最近核对：2026-08-22。本文记录当前 FastAPI + 静态前端的稳定接口、状态和下载边界。实现与自动化测试优先于历史计划；`AutoAI_开发计划.md` 仅作历史资料。

## 1. 当前架构

- 经典前端入口为 `/`（`static/index.html`），v2 独立工作台入口为 `/v2`（重定向到 `static/v2/index.html`）。两者都使用原生 HTML/CSS/JavaScript，不引入 React、Vue 或 Vite。
- 两套前端共享 `static/js/api-client.js`、Principal 鉴权、Dataset/Run API、artifact 规则和 `run-result-v1`；v2 是并行正式入口，不改变经典前端 URL。
- FastAPI 同源托管网页和 API，公共启动器为 `run.py`；`run_classic.py` 与 `run_v2.py` 仅分别选择自动打开 `/` 或 `/v2`，后端和 worker 生命周期完全复用。手动入口仍为 `backend.app.main:app`。
- `POST /api/training/runs` 只创建 SQLite 中的 `queued` Run；训练由独立 `backend.app.runs.worker` 进程执行。
- `POST /api/training/batches` 原子创建统一 Batch 与其所有 queued 子 Run；仍不在 HTTP 请求或 `BackgroundTasks` 中启动训练。
- FastAPI BackgroundTasks 不承担训练执行。
- SQLite `RunRepository` 是任务状态权威；`status.json` 只是历史兼容投影。
- 每次训练通过唯一 Run ID 关联训练记录、结果页和 artifact。
- 结果页使用 `#/results?run_id=<Run ID>`；v2 的可复制完整地址为 `/static/v2/index.html#/results?run_id=<Run ID>`。刷新页面后重新请求后端，不依赖浏览器内存中的旧结果。
- v2 批次比较使用 `#/comparison?batch_id=<Batch ID>`；经典入口在建模页内显示同一比较接口的紧凑视图。

## 2. 本机与服务器认证

### 2.1 本机模式

默认：

```text
AUTOAI_DEPLOYMENT_MODE=local
```

本机模式不要求令牌，所有请求使用空 Principal。建议仅监听 `127.0.0.1`；`run.py` 会拒绝在 local 模式绑定 `0.0.0.0`。

### 2.2 服务器模式

服务器对外监听时必须配置：

```text
AUTOAI_DEPLOYMENT_MODE=server
AUTOAI_API_TOKEN=<至少 32 个字符的随机令牌>
```

可选：

```text
AUTOAI_PRINCIPAL_ID=server-admin
AUTOAI_TENANT_ID=default
AUTOAI_ALLOWED_ORIGINS=https://autoai.example.edu
```

规则：

- API 令牌只通过 `Authorization: Bearer <token>` 发送，不能放在 URL、请求体或 Git 配置中。
- 浏览器只把令牌保存在当前标签页的 `sessionStorage`，键名为 `specautoai.serverToken`；关闭标签页后失效。
- `AUTOAI_ALLOWED_ORIGINS` 使用逗号分隔的明确来源，禁止 `*`。
- `/`、`/static/**`、`/health` 和 `/api/auth/config` 可匿名访问。
- 其他 `/api/**`、`/docs`、`/redoc` 和 `/openapi.json` 需要有效 Bearer 令牌。
- `/api/auth/session` 用于校验当前令牌，并返回由服务端注入的 Principal 摘要。
- 请求体不接受 `owner_id` 或 `tenant_id`；身份只能由服务端认证边界注入。
- server Principal 只能查询、取消、删除和下载自身作用域内的 Dataset 与 Run。旧的 owner 为空 Run 在 server 模式默认不可见，必须通过显式迁移绑定，不能静默继承。

认证发现：

```http
GET /api/auth/config
```

```json
{
  "mode": "server",
  "auth_required": true,
  "token_storage": "session"
}
```

未认证响应：

```json
{
  "error": {
    "code": "authentication_required",
    "message": "需要有效的服务器访问令牌"
  }
}
```

HTTP 状态为 401，并包含 `WWW-Authenticate: Bearer`。

`GET /health` 始终匿名可访问，返回：

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
    "contract_version": "run-artifact-manifest-v2",
    "live_count": 1,
    "last_seen_at": "...",
    "active_run_count": 0
  }
}
```

它不返回 token、owner、tenant、worker_id 或底层数据库路径。旧 Worker 没有 `contract_version`，或同时存在不同版本的活跃 Worker 时，`compatible=false` 且汇总 `contract_version=null`。

## 3. 错误处理

前端应同时兼容两种错误结构：

```json
{"detail": "错误原因"}
```

```json
{
  "error": {
    "code": "artifact_missing",
    "message": "预测结果文件未生成",
    "details": {},
    "request_id": "..."
  }
}
```

训练后台失败不是创建请求失败。创建接口可能已经返回 HTTP 202，随后结果接口返回：

```json
{
  "run": {
    "state": "failed",
    "result_state": "failed",
    "error": {
      "code": "invalid_training_data_or_config",
      "stage": "training_validation",
      "message": "数据集标签无效",
      "retryable": false
    }
  }
}
```

前端不得显示服务器 traceback、绝对路径或令牌。

worker 当前收敛的稳定错误码包括：数据完整性变化用 `dataset_changed`，数据文件不存在或不可读用 `dataset_unavailable`，数据内容或训练配置校验失败用 `invalid_training_data_or_config`，其余训练异常用 `training_failed`。

## 4. 数据上传与预处理

新预处理固定输出 `wide-feature-v2` 宽表，一条曲线占一行：

```csv
Index,Label,Sample_ID,Name,0,0.0066675556740898788,...,50
1,A,S001,GSGC-001.csv,0.12,0.15,...,0.08
2,A,S001,GSGC-002.csv,0.11,0.16,...,0.09
```

- 前四列名称与顺序固定为 `Index, Label, Sample_ID, Name`；`Name` 保存原始文件名，`Label` 必填且始终按分类类别处理。
- 第 5 列起均为特征列。列名必须能解析为有限浮点数，数值唯一且严格递增；它们就是逐点真实 `XXX` 坐标，生成端使用 float64 可往返文本。
- 每个特征单元格是对应坐标处的有限标量 `Intensity`，不是数组或 JSON。预处理输出的强度最多保留 5 位小数，不再按单元格字符数自适应降低精度。
- 同一 `Sample_ID` 只能对应一个 Label；重复测量组不能跨 train/valid/test。所有行共享同一特征表头所表示的公共轴。
- 原始文件名写入 `Name`，预处理响应也通过 `curves[].name` 保留名称用于预览。训练结果优先显示 `Name`；读取没有 `Name` 的 `wide-feature-v1` 时以 `Index` 回退。
- Excel 总列数最多 16,384；v2 扣除四个元数据列后最多 16,380 个特征。超过上限必须由业务侧先做明确的范围选择或降采样，导出器不得静默删点。
- 新上传与 worker 训练接受 `wide-feature-v2`，并兼容既有三列元数据的 `wide-feature-v1`。旧 `Index,Name,XXX,Intensity,Label,Sample_ID` 数组/JSON 文件（包括 `linspace-v1`、`linspace-slice-v1`）返回明确迁移错误，不自动取首行坐标或有损转换。
- 上传使用 `multipart/form-data`，文件字段名为 `file`；预处理多文件字段名为重复的 `files`。

上传建模数据：

```http
POST /api/datasets/upload
```

响应包含稳定的 `dataset_id`、原始 `dataset_name` 和 `summary`。本机模式暂时保留 `dataset_path` 兼容字段；server 模式不返回服务器绝对路径。

预处理：

```text
POST /api/preprocess/raman
POST /api/preprocess/hplc/inspect       # 选择文件后逐文件检测
POST /api/preprocess/hplc
POST /api/preprocess/chromatography   # 简单截取兼容接口
```

拉曼顺序固定为先选择行号/X 轴范围，再执行基线校正。HPLC 同样保留行号/X 轴范围选择和 `hplc_interpolate` 开关。固定轴使用服务端 `HplcGridConfig`，覆盖包含首尾端点的 0–50 分钟，点数由当前批次实际公共点数动态构造；算法函数和前端均不内嵌固定点数。

选择 HPLC 文件后，前端调用检查接口并逐行展示原文件名、点数、时间范围与状态。批次点数一致时，任意不少于 2 的公共点数都可处理；不一致时，后端以唯一众数作为期望点数并列出所有异常文件及实际点数，若众数并列则列出全部点数组，二者都拒绝预处理。每个源 X 必须严格递增。行号为 1 基、首尾包含，起止都必须在动态 `1..point_count` 内；终止行留空时才使用检测出的完整点数。开启插值时，范围参数选择固定目标轴的对应切片：行号 1–4000 产生 4000 个目标点，100–4000 产生 3901 个目标点，而不是把所选源点重新扩展为完整点数。第 n 点的真实保留时间为 `start_minutes + (n-1)*(stop_minutes-start_minutes)/(point_count-1)`；强度始终从完整源曲线中寻找左右邻点。仪器轴与目标轴在边界只有不超过一个采样间隔的相位差时，允许使用首两个或末两个源点线性延伸，超过一个间隔则拒绝。

宽表只有一组特征表头，所以同一预处理批次必须共享公共轴：

- HPLC 开启插值时使用所选固定目标轴，真实分钟坐标逐点写入特征表头。
- HPLC 关闭插值时保留所选原始 X/Y，但只有所有文件所选轴逐点完全一致才允许导出；不一致返回 HTTP 400，并提示开启插值或先对齐数据。
- 拉曼和简单色谱多文件也执行相同公共轴校验；不得把第一条曲线的坐标静默套到其他曲线。
- 独立测试集的特征坐标和顺序必须与主数据集完全一致，仅特征数相同不够。
- 所有预处理模式都不执行消负或面积归一化。

一次成功的预处理只原子写入一个宽表主 CSV。HPLC 文件名仍为 `hplc_<token>.csv`，响应只通过 `download_url` 提供主文件，不生成 `_xxx.csv`，也不返回 `xxx_download_url`/`xxx_rows`。`preview` 展示 `Index/Label/Sample_ID/Name`，其中 `Name` 是对应原始 CSV 文件名；完整真实轴继续在 `curves[].x`、`common_time` 或 `hplc_axis` 中用于前端曲线与摘要显示。

成功响应包含实际输出精度：

```json
{
  "output_precision": {
    "format": "wide-feature-v2",
    "xxx_encoding": "column_headers",
    "xxx_precision": "float64-roundtrip",
    "intensity_decimal_places": 5,
    "adaptive": false,
    "feature_count": 7500,
    "total_column_count": 7504,
    "excel_column_limit": 16384,
    "excel_compatible": true
  }
}
```

`adaptive=false` 表示宽表不再为适应单元格字符限制降精度；`xxx_encoding=column_headers` 表示真实坐标直接位于第 5 列起的表头。开启 HPLC 插值时，`curves[].x` 和 `common_time` 是本次实际选择的固定分钟轴；`hplc_axis.start/stop/point_count` 描述实际输出，`grid_start/grid_stop/grid_point_count` 描述动态完整网格，`selected_start_row/selected_end_row` 是完整网格中的 1 基位置。关闭时 `curves[].x` 对应已验证一致的原始公共轴，`common_time=[]`、`hplc_axis=null`。

预处理主文件统一使用响应中的 `download_url`；两套前端都只展示“下载统一建模 CSV”。`output_path` 等服务器路径不得直接作为浏览器链接；HPLC 不生成第二时间轴文件或 `common_time_path`。

## 5. 创建训练 Run

### 5.1 创建

```http
POST /api/training/runs
Content-Type: application/json
```

```json
{
  "dataset_id": "ds_abc123",
  "test_dataset_id": null,
  "config": {
    "model_type": "cnn1d",
    "epochs": 200,
    "batch_size": 8,
    "learning_rate": 0.001,
    "normalization": "zscore",
    "split_mode": "stratified_holdout",
    "split_train": 8,
    "split_valid": 1,
    "split_test": 1
  }
}
```

`dataset_id` 是正式入口；`data_path` 只保留给受控本机兼容。主数据和独立测试数据各自的 ID/路径不能同时传。

服务端在入队前验证模型、评估方式、比例和数值范围。无效配置返回 422 且不创建 Run：

```json
{
  "detail": {
    "code": "invalid_training_config",
    "message": "不支持的分类模型：..."
  }
}
```

未知训练字段会被忽略并在成功响应的 `warnings` 中说明；兼容别名 `transformer1d` 会规范化为 `cnn_transformer1d`。

成功返回 HTTP 202：

```json
{
  "run_id": "abc123",
  "status": "pending",
  "state": "queued",
  "warnings": []
}
```

该响应不代表训练已开始或完成。

发现活跃但与当前 Web 不兼容的 Worker 时，创建接口在写入 queued Run 前返回 HTTP 503：

```json
{
  "detail": {
    "code": "worker_contract_mismatch",
    "message": "训练 Worker 版本与当前 Web 不兼容，请同时重启 Web 和 Worker"
  }
}
```

### 5.2 列表

兼容完整列表：

```http
GET /api/training/runs
```

结果页入口和训练记录首屏应使用轻量分页：

```http
GET /api/training/runs?projection=summary&limit=20&cursor=<Run ID>
```

```json
{
  "items": [
    {
      "run_id": "abc123",
      "state": "succeeded",
      "status": "success",
      "result_state": "ready",
      "model_type": "pls_da",
      "dataset_name": "teacher-data.csv",
      "created_at": "...",
      "started_at": "...",
      "finished_at": "...",
      "duration_seconds": 12.5,
      "test_macro_f1": 0.9234
    }
  ],
  "next_cursor": null
}
```

summary 不包含完整指标、历史曲线或服务器路径。`test_macro_f1` 可为 `null`：仅成功且结果完整的 Run 从通过 Manifest 大小/SHA-256 校验的必要指标文件中读取；holdout/external test 使用 `metrics.test.macro_f1`，留一交叉验证使用 `cv_summary.pooled_test.macro_f1`，不得使用 fold mean。

### 5.3 单个状态

```http
GET /api/training/runs/{run_id}
```

规范状态与旧兼容状态：

```text
queued     -> pending
running    -> running
succeeded  -> success
failed     -> failed
cancelled  -> paused
```

新页面逻辑应以 `state` 为准；`paused` 只为旧客户端兼容。

### 5.4 专属建模结果

```http
GET /api/training/runs/{run_id}/result
```

响应使用 `schema_version="run-result-v1"`。完整结构见 `run_result_contract.md`。

### 5.5 停止与删除

```text
POST   /api/training/runs/{run_id}/stop
POST   /api/training/runs/{run_id}/cancel   # 旧客户端兼容别名
DELETE /api/training/runs/{run_id}
```

- 停止只适用于 queued/running；内部规范状态仍使用 `cancelled`，界面统一显示 `STOP`。
- STOP 只保留 SQLite 训练记录，删除该 Run 的模型、指标、映射和其他中间产物。
- worker 租约过期或缺失表示训练进程意外中断，同样收敛为 `cancelled/STOP`，不静默重跑。
- 删除只适用于 succeeded/failed/cancelled，并永久移除记录与产物。
- server 模式下，其他 Principal 的 Run 与不存在的 Run 都返回 404，避免泄露标识是否存在。

### 5.6 下载

```http
GET /api/training/runs/{run_id}/artifact/{name}
```

下载必须同时满足：

1. DB 中的 Run 属于当前 Principal。
2. 有 DB 记录的 Run 已进入 `succeeded`。
3. Manifest 中登记该文件。
4. 新 Manifest 的显式 artifact catalog 允许公开。
5. 文件存在且大小、SHA-256 与 Manifest 一致。

不满足时分别返回 403、404 或 409。没有 DB 记录的旧 Run 目录仅在 local Principal 下按旧 Manifest 做只读兼容；server 模式不会开放该兼容旁路。

## 6. 分类模型与评估契约

能力目录保留 17 个后端分类模型，UI 通过 ui_visible=true 展示六类：PLS-DA、Elastic Net、SVM、Random Forest、XGBoost、1D-CNN。Label 即使为数字也按类别处理。

UI 配置新增 experiment_version=word-0904，对应架构 docx-classification-v4-0904。未指定版本的旧请求保留原训练行为。隐藏模型继续通过兼容 API 调用；cnn_mamba1d 依赖不可用时必须真实返回 available=false。

评估方式：

- `stratified_holdout`：按 `Sample_ID` 整组、以 8:1:1 为目标划分。Train/Valid/Test 都必须包含全部类别；Valid/Test 至少为每类 1 个 `Sample_ID`，该下限优先于比例，因此每类少于 3 个不同 `Sample_ID` 时拒绝训练。
- `leave_one_sample_id_cv`：每折留一个 Sample_ID 作 test，其余按 8:2 形成 train/valid。
- `external_test_holdout`：主数据 8:2，独立测试集作为最终 test。
- `leave_one_sample_id_cv_with_external_test`：保留主数据 OOF 审计并单独报告独立测试集最终指标；外部数据绝不进入 CV 拆分。

交叉验证的 Test 主指标来自所有折合并后的 OOF 预测。Train/Valid 标量展示折均值，折标准差作为审计值；图表分析使用跨折预测合并并标记 `pooled_cross_fold`，样本可能重复。不得把 fold mean、pooled cross-fold 和 pooled OOF 混在同一口径中。

传统模型以 `Sample_ID` 分组的内层 5 折 Balanced Accuracy 选优，所有 normalizer/PCA 仅在内层训练折拟合，再使用外层 train+valid 重训。深度模型使用 AdamW、batch size 8、最多 200 epochs，并保存最低 validation loss 权重。

可解释性实现保留，但由内部常量 `TEMPORARILY_HIDDEN` 关闭。新训练不计算或生成解释性 artifact；前端不显示入口或发起解释性请求；结果投影仅返回 `{"status":"temporarily_hidden"}`，历史解释性 artifact 也不得下载。

### 6.1 多模型 Batch 与比较

`POST /api/training/batches` 请求包含 `model_types`、`base_seed` 和通用 `config`。新产品流程遵循“一模型一个 Run”，`repeat_count` 缺省且只允许为 1，普通前端不再展示或提交重复实验次数。仓储与结果投影仍能读取既有历史 R>1 批次，但创建接口不再接受新的重复训练。Batch 为所有模型写入相同 `split_seed=base_seed`，确保在相同划分上比较。单个子 Run 失败或取消不会破坏已成功结果；Batch 聚合状态为 `queued`、`running`、`succeeded`、`partial`、`failed` 或 `cancelled`。

`GET /api/training/batches/{batch_id}/comparison` 返回 `model-comparison-v1`。仅完整成功、划分 digest 和评估口径一致的子 Run 可比较；它返回 Accuracy、Balanced Accuracy、Macro-F1、Weighted-F1、Sample_ID × 模型正确率、类别 Recall 和每模型的单 Run 混淆矩阵入口。历史响应可能仍带 `repeat_stability` 兼容字段，新前端不渲染该区块。缺失值为显式缺失，不补零、不伪造 ROC/PR。

## 7. 结果页数据能力

当前真实支持：

- Accuracy、Balanced Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1。
- Train/Valid/Test 分区指标。
- CV pooled OOF、fold mean、fold std。
- 混淆矩阵、分类报告、由混淆矩阵计算的真实/预测类别分布。
- Train/Valid/Test 三分区混淆矩阵、各类别指标和竖向预测分布。
- 三个混淆矩阵各自提供纯前端 PNG 下载；PNG 直接由同一矩阵数据重绘，标题包含 Run、分区和聚合口径，不依赖后端截图 artifact。
- 深度模型训练历史；传统模型不生成 `history.csv`，结果页不显示空曲线。
- Batch 比较中的 Accuracy 排名、四项总体指标自适应图、Sample_ID × 模型正确率、类别 Recall 与单模型混淆矩阵入口。
- 可解释性和模型特征图当前均为 `temporarily_hidden`，不在 Run 结果页投影或渲染。

当前没有正式计算 ROC-AUC、ROC 曲线和 Precision-Recall 曲线。结果契约会返回 `available=false` 和原因；前端不得绘制空图或伪造数值。

## 8. Artifact catalog

新 Run 的公开下载由后端显式 catalog 决定。主要公开项：

| 文件 | 用途 |
|---|---|
| `metrics.json` | 总体指标 |
| `cv_metrics.json` | CV 汇总与 OOF 口径 |
| `fold_metrics.csv` | 分折审计指标 |
| `predictions.csv` | 测试预测明细 |
| `cv_predictions.csv` | OOF/兼容预测明细 |
| `history.csv` | 深度模型训练过程，适用时 |
| `hyperparameter_search.csv` | 传统模型参数搜索，适用时 |
| `config.json` | 已移除服务器路径的训练配置；只在内容审查通过时公开 |
| `model_metadata.json` | 模型元数据 |
| `label_map.json` | 类别映射 |
| `split.json` | 数据划分信息 |

以下文件不在新结果页下载白名单：

- `status.json`：易变的兼容投影。
- `model.pkl`、`model.pt`：本轮不开放裸模型对象/权重下载。
- `*.joblib`：内部 PCA/AggMap 等拟合对象。
- `manifest.json`：内部索引。
- `sample_feature_importance.*`、`feature_importance.*`、`model_feature_visualization.json`、`dscarnet_mapping.json`：`TEMPORARILY_HIDDEN` 期间不公开；即使历史 Manifest 声明可下载也返回 404。

`config.json` 采用内容审查：新训练生成的无路径配置可下载；历史或异常配置只要包含 `data_path`、`test_data_path`、`*_path` 等服务器路径字段，就会自动标为不可下载并说明原因。

前端只消费 `/result` 返回的 `artifacts[]`，不得维护自己的固定文件数组。

历史 Manifest 中的解释性 artifact 同样受当前隐藏下载限制；新 descriptors 不列出，前端也不展示。

## 9. 前端页面与状态流

Hash 页面：

```text
#/raman
#/chromatography
#/modeling
#/results
#/results?run_id=<encoded Run ID>
#/runs
#/manual
```

导航顺序必须满足：AI 建模 → 建模结果 → 训练记录。

新建 Run 成功后：

1. AI 建模页显示简洁成功状态。
2. 浏览器中央对话框显示 Run ID 和 3 秒倒计时。
3. 提供“立即查看结果”和“留在当前页”；留页、Escape、手动导航或新训练都会清理定时器。
4. 只对当前标签页新创建的 Run 自动跳转；打开历史成功 Run 不自动抢占页面。
5. 跳转定时器在新训练、手动跳转或页面切换时清理。

没有指定 Run ID 的建模结果入口显示最近任务，至少包括 Run ID、原始 CSV 文件名、模型、状态、训练开始时间（未开始时标记创建时间）、耗时和查看操作。

`/result` 返回 404 时，只有 `/health.contracts.run_result` 明确缺失/旧版才允许回退兼容接口；当前 Web 已声明 `run-result-v1` 时必须按任务不存在/不可见处理。

轮询必须串行执行；切换 Run 后取消旧请求，终态停止。queued/running 显示进度，failed/cancelled/404/403/网络失败和部分产物缺失分别处理。

经典入口和 v2 对相同业务概念使用同一可见术语：建模摘要固定按“数据量、类别数、样本数、每样本测量数、特征数”展示；分组区使用“按样本分组”和“样本编号、类别、每样本测量数”；类别统计使用“类别分布”和“类别、数据量”。结果页把 `curve_count` 显示为“数据量”、`sample_id_count` 显示为“样本数”，各类别 `support` 显示为“数据量”。创建/开始时间合并为一个“训练时间”，优先开始时间，未开始时回退创建时间并标注“任务创建”。训练记录显示可空“测试集 Macro F1”，固定四位小数；无值显示 `—`。内部 CSV/API 字段名不随界面术语改名。

两套 HPLC 表单在选中文件后先调用 `/api/preprocess/hplc/inspect`，逐文件展示检测结果，并把起始/终止行边界动态更新为 `1–point_count`。请求前调用共享 `validateHplcRowRange(start, end, pointCount)`；后端仍执行同一动态点数的第二道校验。时间输入显示为“保留时间下限/上限（分钟）”，成功结果展示服务端返回的实际首末分钟、完整网格中的首末点和实际点数。

## 10. 分页面说明边界

- 拉曼页：上传、范围、先截取后基线、下载。
- HPLC 页：输入、三步默认流程、何时调整参数、下载。
- AI 建模页：`wide-feature-v2` 宽表（兼容 v1）、原文件名、真实坐标表头、评估方式、queued/worker、提交。
- 建模结果页：Run ID、交叉验证测试主口径、结果完整性、逐项下载。
- 训练记录页：测试集 Macro F1、训练时间、查看、取消、删除和不可恢复提示。
- 全局帮助：只保留宽表格式、公共轴与 Excel 列数限制、Sample_ID 整组原则、worker 排查和服务器认证等跨页面规则。

容易误操作的字段使用就近提示；页面内说明不复制本技术契约。

## 11. 验证

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端脚本还必须执行 Node 语法检查和纯函数测试。项目根目录存在本地 `data.csv` 时，使用轻量模型完成上传 → queued → worker → succeeded → 结果 URL → 刷新 → 下载的真实闭环；数据、Run 目录和模型产物不得提交进 Git。
