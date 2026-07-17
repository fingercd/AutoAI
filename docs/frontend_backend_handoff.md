# SpecAutoAI 前后端接口契约

> 最近核对：2026-07-17。本文记录当前 FastAPI + 静态前端的稳定接口、状态和下载边界。实现与自动化测试优先于历史计划；`AutoAI_开发计划.md` 仅作历史资料。

## 1. 当前架构

- 经典前端入口为 `/`（`static/index.html`），v2 独立工作台入口为 `/v2`（重定向到 `static/v2/index.html`）。两者都使用原生 HTML/CSS/JavaScript，不引入 React、Vue 或 Vite。
- 两套前端共享 `static/js/api-client.js`、Principal 鉴权、Dataset/Run API、artifact 规则和 `run-result-v1`；v2 是并行正式入口，不改变经典前端 URL。
- FastAPI 同源托管网页和 API，公共启动器为 `run.py`；`run_classic.py` 与 `run_v2.py` 仅分别选择自动打开 `/` 或 `/v2`，后端和 worker 生命周期完全复用。手动入口仍为 `backend.app.main:app`。
- `POST /api/training/runs` 只创建 SQLite 中的 `queued` Run；训练由独立 `backend.app.runs.worker` 进程执行。
- FastAPI BackgroundTasks 不承担训练执行。
- SQLite `RunRepository` 是任务状态权威；`status.json` 只是历史兼容投影。
- 每次训练通过唯一 Run ID 关联训练记录、结果页和 artifact。
- 结果页使用 `#/results?run_id=<Run ID>`；v2 的可复制完整地址为 `/static/v2/index.html#/results?run_id=<Run ID>`。刷新页面后重新请求后端，不依赖浏览器内存中的旧结果。

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

建模 CSV 固定为：

```text
Index, Name, XXX, Intensity, Label, Sample_ID
```

- `XXX` 和 `Intensity` 是等长数值数组。
- `Label` 必填且始终按分类类别处理。
- 同一 `Sample_ID` 只能对应一个 Label；重复测量组不能跨 train/valid/test。
- 上传使用 `multipart/form-data`，文件字段名为 `file`；预处理多文件字段名为重复的 `files`。

上传建模数据：

```http
POST /api/datasets/upload
```

响应包含稳定的 `dataset_id`、原始 `dataset_name` 和 `summary`。本机模式暂时保留 `dataset_path` 兼容字段；server 模式不返回服务器绝对路径。

预处理：

```text
POST /api/preprocess/raman
POST /api/preprocess/hplc
POST /api/preprocess/chromatography   # 简单截取兼容接口
```

拉曼顺序固定为先选择行号/X 轴范围，再执行基线校正。HPLC 默认顺序固定为共同时间轴插值、逐条减最小值、按真实时间轴面积归一化；响应曲线使用 `raw_y` 与 `processed_y`。

预处理下载使用响应中的 `download_url`。`output_path`、`common_time_path` 等服务器路径不得直接作为浏览器链接。

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
      "duration_seconds": 12.5
    }
  ],
  "next_cursor": null
}
```

summary 不包含完整指标、历史曲线或服务器路径。

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

### 5.5 取消与删除

```text
POST   /api/training/runs/{run_id}/cancel
DELETE /api/training/runs/{run_id}
```

- 取消只适用于 queued/running。
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

能力目录固定公开 **15 个目标分类模型**。当前仅支持分类；`Label` 即使为数字也按类别处理。

当前通常可训练 14 项；`cnn_mamba1d` 因依赖不可用返回 `available=false`，不得静默替换。新 Run 使用 `architecture_version="docx-classification-v2"`。

评估方式：

- `stratified_holdout`：按 `Sample_ID` 整组进行 8:1:1。
- `leave_one_sample_id_cv`：每折留一个 Sample_ID 作 test，其余按 8:2 形成 train/valid。
- `external_test_holdout`：主数据 8:2，独立测试集作为最终 test；与 CV 互斥。

交叉验证的 Test 主指标来自所有折合并后的 OOF 预测。Train/Valid 标量展示折均值，折标准差作为审计值；图表分析使用跨折预测合并并标记 `pooled_cross_fold`，样本可能重复。不得把 fold mean、pooled cross-fold 和 pooled OOF 混在同一口径中。

传统模型按 valid balanced accuracy 选优，再使用 train+valid 重训。深度模型使用 AdamW、batch size 8、最多 200 epochs，并保存最低 validation loss 权重。

可解释性：

- 六个传统模型、`pca_mlp`、`cnn_transformer1d` 使用真实类别 Log-loss 窗口遮挡。
- `cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM-like，并保留输入梯度 sanity check。
- `dscarnet` 使用 SAR/CAR 双通路 2D 映射和 2D Grad-CAM 回投。

## 7. 结果页数据能力

当前真实支持：

- Accuracy、Balanced Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1。
- Train/Valid/Test 分区指标。
- CV pooled OOF、fold mean、fold std。
- 混淆矩阵、分类报告、由混淆矩阵计算的真实/预测类别分布。
- Train/Valid/Test 三分区混淆矩阵、各类别指标和竖向预测分布。
- 深度模型训练历史；传统模型不生成 `history.csv`，结果页不显示空曲线。
- 单样品解释摘要及 JSON/CSV artifact；新 Run 不生成或展示全局重要性。

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
| `sample_feature_importance.json/csv` | 单样品解释结果，适用时 |
| `dscarnet_mapping.json` | DSCARNet 映射说明，适用时 |
| `config.json` | 已移除服务器路径的训练配置；只在内容审查通过时公开 |
| `model_metadata.json` | 模型元数据 |
| `label_map.json` | 类别映射 |
| `split.json` | 数据划分信息 |

以下文件不在新结果页下载白名单：

- `status.json`：易变的兼容投影。
- `model.pkl`、`model.pt`：本轮不开放裸模型对象/权重下载。
- `*.joblib`：内部 PCA/AggMap 等拟合对象。
- `manifest.json`：内部索引。

`config.json` 采用内容审查：新训练生成的无路径配置可下载；历史或异常配置只要包含 `data_path`、`test_data_path`、`*_path` 等服务器路径字段，就会自动标为不可下载并说明原因。

前端只消费 `/result` 返回的 `artifacts[]`，不得维护自己的固定文件数组。

历史 Manifest 已登记的 `feature_importance.json/csv` 只保留原 Principal、Manifest 和完整性校验下的直接 URL 兼容；新 descriptors 不列出，前端也不展示。

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

## 10. 分页面说明边界

- 拉曼页：上传、范围、先截取后基线、下载。
- HPLC 页：输入、三步默认流程、何时调整参数、下载。
- AI 建模页：六列 CSV、评估方式、queued/worker、提交。
- 建模结果页：Run ID、OOF、结果完整性、逐项下载。
- 训练记录页：查看、取消、删除和不可恢复提示。
- 全局帮助：只保留六列格式、Sample_ID 整组原则、worker 排查和服务器认证等跨页面规则。

容易误操作的字段使用就近提示；页面内说明不复制本技术契约。

## 11. 验证

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端脚本还必须执行 Node 语法检查和纯函数测试。项目根目录存在本地 `data.csv` 时，使用轻量模型完成上传 → queued → worker → succeeded → 结果 URL → 刷新 → 下载的真实闭环；数据、Run 目录和模型产物不得提交进 Git。
