# AutoAI 项目与前后端对接交接文档

本文档给另一个前端 agent 使用。目标是：在不修改后端接口的前提下，重新实现前端页面，并保证前端可以稳定调用后端，不因为请求格式、字段名、异步训练状态或 CSV 格式问题导致后端报错。

本文只描述项目、接口契约和对接注意事项，不包含 UI 视觉或布局建议。

> 当前稳定接口契约：主前端为 `static/index.html`，不是 React/Vite 主链路；色谱主界面默认调用 `/api/preprocess/hplc` 并开启 HPLC 三步标准流程。当前建模仅支持分类任务，评估口径支持分层 holdout、`Repeat_index` 留一交叉验证和独立测试集 holdout。训练 HTTP 请求只创建 SQLite 中的 `queued` Run，由独立本机 worker 执行；`status.json` 只是兼容投影。当前稳定模型和算法以 master 为准，模型数学改动必须使用独立模型计划。

## 分类模型 v2 接口契约

`GET /api/models` 返回 **15 个目标分类模型**：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`、`dscarnet`。当前仅支持分类；`Repeat_index` 不改名并等同 Sample_ID。新 Run 使用 `architecture_version="docx-classification-v2"`，旧类、旧权重和旧 artifact 名只读兼容。

模型目录始终列出15项，但 capability 决定是否可选。`cnn_mamba1d` 在当前 Windows Conda 环境中因 `mamba-ssm` 依赖不可用而返回 `available=false`。`dscarnet_input_mode` 接受 `sar`、`car`、`dual`（默认）；二分类深度模型使用单 logit + `BCEWithLogitsLoss`，接口仍输出两列类别概率。

划分契约：`stratified_holdout` 默认 8:1:1；`leave_one_repeat_index_cv` 每折留一个 `Repeat_index` 作 test，其余按 8:2 分 train/valid；`external_test_holdout` 主数据 8:2，独立数据作唯一 test，并禁止 CV。传统模型按验证集 balanced accuracy 选优，锁定参数后使用 train+valid 重训。深度模型使用 AdamW、batch size 8、最多 200 epochs，以最低 validation loss 保存最佳权重。

解释性契约：六个传统模型使用窗口置换；`pca_mlp`、`cnn_transformer1d`、`cnn_mamba1d` 使用 `abs(gradient * input)`；`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM；`dscarnet` 使用模式对应的 2D Grad-CAM 回投。继续支持 `feature_importance.json/csv`、`sample_feature_importance.json/csv`、`model.pt/model.pkl` 等旧下载名。

## 1. 项目概况

AutoAI 是一个部署在服务器或本机的谱学数据预处理及分类建模平台。当前后端使用 FastAPI，前端是静态 HTML/CSS/JS，由 FastAPI 同源托管。

核心功能：

1. 拉曼原始 CSV 多文件上传。
2. 色谱原始 CSV 多文件上传。
3. 拉曼数据基线校正和范围截取。
4. 色谱/HPLC 数据范围截取、共同时间轴插值、消负和面积归一化。
5. 统一生成建模 CSV。
6. 上传补全 `Label` 和 `Repeat_index` 的建模 CSV。
7. 按选择的评估口径划分训练/验证/测试集，支持无独立测试集的 8:1:1 分层划分、`Repeat_index` 留一交叉验证，以及有独立测试集时的 8:2 train/valid + external test。
8. 运行分类模型训练。
9. 查看训练状态、指标、混淆矩阵、训练历史，并下载结果文件。

当前后端入口：

```text
backend/app/main.py
```

当前静态文件目录：

```text
static/
```

当前存储目录：

```text
storage/uploads       上传的原始文件和建模 CSV
storage/preprocessed  预处理生成的统一 CSV
storage/runs          每次训练的 Run 产物；状态命令以 SQLite RunRepository 为准
```

## 2. 运行与服务地址

本地开发默认服务：

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

如果前端由同一个 FastAPI 服务托管，接口可以直接使用相对路径：

```text
/health
/api/datasets/upload
/api/training/runs
```

如果前端由其他 dev server 托管，例如 Vite/React 独立端口，则接口 base URL 使用：

```text
http://127.0.0.1:8000
```

后端当前已经开启 CORS：

```text
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

## 3. 统一错误处理规则

所有接口失败时，前端不要只显示 HTTP 状态码，应优先读取响应 JSON 的 `detail` 字段。

FastAPI 主动抛错时通常返回：

```json
{
  "detail": "错误原因"
}
```

训练任务后台失败时，创建任务接口仍可能已经返回 `pending`，真实错误在轮询结果中：

```json
{
  "run_id": "xxxx",
  "status": "failed",
  "error": "错误原因",
  "traceback": "后端完整 traceback"
}
```

前端必须同时处理两类错误：

1. 请求本身失败：读取 `response.detail`。
2. 训练任务失败：轮询 `/api/training/runs/{run_id}` 后读取 `error`。

## 4. 建模 CSV 格式约束

用于 AI 建模的 CSV 必须包含这些列：

```text
Index, Name, XXX, Intensity, Label, Repeat_index
```

字段含义：

| 字段 | 类型 | 要求 |
|---|---|---|
| Index | 数字或字符串 | 样本行编号 |
| Name | 字符串 | 文件名或样本名 |
| XXX | 字符串数组 | X 轴数组，例如 `[100.0, 101.0]` |
| Intensity | 字符串数组 | 强度数组，例如 `[0.12, 0.13]` |
| Label | 字符串 | 分类标签，训练前必须填写 |
| Repeat_index | 字符串或数字 | 同一样品的重复测量分组编号 |

后端校验规则：

1. `XXX` 和 `Intensity` 必须能解析为数组。
2. 每一行的 `XXX` 和 `Intensity` 长度必须相同。
3. 整个训练数据中所有曲线长度必须一致。
4. `Label` 不能为空。
5. `Repeat_index` 不能为空。
6. 同一个 `Repeat_index` 内只能对应一个 `Label`。
7. 每个 `Repeat_index` 的重复测量条数必须一致，例如每种样品都 5 条。
8. 划分训练/验证/测试时按 `Repeat_index` 整组划分，不会把同一样品的重复测量拆到不同集合。

预处理生成的 CSV 中 `Label` 和 `Repeat_index` 默认是空的。用户必须补完这两列后，才能上传到建模接口训练。

## 5. API 总览

### 5.1 健康检查

```http
GET /health
```

返回（稳定 ID 是长期引用；`dataset_path` 仅为受控本地兼容字段）：

```json
{
  "status": "ok"
}
```

用途：页面加载后检查后端是否在线。

### 5.2 项目根目录 data.csv 摘要

```http
GET /api/sample/summary
```

如果项目根目录存在 `data.csv`，返回它的建模摘要。若不存在，返回 404。

返回结构和上传数据集的 `summary` 一致。

### 5.3 上传建模数据集

```http
POST /api/datasets/upload
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | File | 是 | 建模 CSV |

前端必须用 `FormData` 上传，不能用 JSON 上传文件。

返回：

```json
{
  "dataset_id": "ds_abc123",
  "dataset_path": "D:\\PythonProject\\AutoAI\\storage\\uploads\\xxxx.csv",
  "summary": {
    "path": "D:\\PythonProject\\AutoAI\\storage\\uploads\\xxxx.csv",
    "samples": 90,
    "classes": 2,
    "label_counts": {
      "A": 45,
      "B": 45
    },
    "repeat_index": {
      "group_count": 18,
      "expected_repeats_per_group": 5,
      "groups": [
        {
          "repeat_index": "1",
          "count": 5,
          "label": "A"
        }
      ],
      "inconsistent_labels": [],
      "incomplete_groups": []
    },
    "curve_length": 160,
    "curve_lengths": {
      "160": 90
    },
    "columns": ["Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"],
    "preview": [],
    "curves": []
  }
}
```

重要规则：

1. `dataset_id` 是训练接口的首选稳定引用；前端不要试图在浏览器里直接打开服务器路径。
2. `dataset_path` 仅为受控本地兼容字段，可原样作为训练接口的 `data_path` 传回后端。
3. 如果用户上传独立测试集，也调用同一个上传接口；训练时优先把返回的 `dataset_id` 作为顶层 `test_dataset_id` 传给 `/api/training/runs`。

### 5.4 拉曼/色谱/HPLC 预处理

```http
POST /api/preprocess/{kind}
Content-Type: multipart/form-data
```

`kind` 只能是：

```text
raman
chromatography
hplc
```

当前主前端的色谱页面默认使用 `hplc`，不是 `chromatography`。`chromatography` 只做简单范围截取，保留给兼容或手动调用。

表单字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| files | File[] | 是 | 无 | 可多选上传 |
| range_mode | string | 否 | row | `row` 或 `x_value` |
| start_row | int | 否 | 1 | 按行号截取时使用，1-based |
| end_row | int | 否 | 空 | 按行号截取时使用，包含 Python 切片意义上的结束位置 |
| x_min | float | 否 | 空 | 按 X 轴数值截取时的下限 |
| x_max | float | 否 | 空 | 按 X 轴数值截取时的上限 |
| baseline_order | string | 否 | range_then_baseline | 只对 raman 有意义 |
| baseline_method | string | 否 | arPLS | 只对 raman 有意义 |
| hplc_interpolate | bool | 否 | true | 只对 hplc 有意义，线性插值到共同时间轴 |
| hplc_subtract_min | bool | 否 | true | 只对 hplc 有意义，逐条曲线减最小值 |
| hplc_normalize_area | bool | 否 | true | 只对 hplc 有意义，按真实时间轴面积归一化 |

`range_mode` 规则：

```text
row      使用 start_row / end_row 按行号截取。
x_value  使用 x_min / x_max 按第一列 X 轴数值截取。
```

`x_value` 模式下，`x_min` 和 `x_max` 至少填一个。两者都不填会返回 400。

拉曼 `baseline_order` 可选值：

```text
range_then_baseline  先选范围，后基线校正
baseline_then_range  先基线校正，后选范围
```

拉曼 `baseline_method` 默认：

```text
arPLS
```

也可传 rampy.baseline 支持的方法名，例如 `poly`。如果 rampy 不可用，后端会使用简化基线处理兜底；如果 rampy 可用但方法失败，会返回 400。

返回：

```json
{
  "output_path": "D:\\PythonProject\\AutoAI\\storage\\preprocessed\\raman_xxxx.csv",
  "download_url": "/api/files?path=D:\\PythonProject\\AutoAI\\storage\\preprocessed\\raman_xxxx.csv",
  "rows": 6,
  "range_mode": "x_value",
  "x_min": 400.0,
  "x_max": 1800.0,
  "baseline_order": "range_then_baseline",
  "baseline_method": "arPLS",
  "curves": [
    {
      "name": "sample_01",
      "x": [400.0, 401.0],
      "raw_y": [100.0, 101.0],
      "corrected_y": [1.0, 1.2]
    }
  ],
  "preview": [
    {
      "Index": 1,
      "Name": "sample_01",
      "Label": "",
      "Repeat_index": ""
    }
  ]
}
```

色谱返回的 `curves` 没有 `corrected_y`：

```json
{
  "name": "chrom_01",
  "x": [0.0, 0.1],
  "raw_y": [10.0, 11.0]
}
```

HPLC 返回的 `curves` 使用 `processed_y` 表示三步处理后的强度，并额外返回共同时间轴：

```json
{
  "output_path": "D:\\PythonProject\\AutoAI\\storage\\preprocessed\\hplc_xxxx.csv",
  "download_url": "/api/files?path=D:\\PythonProject\\AutoAI\\storage\\preprocessed\\hplc_xxxx.csv",
  "rows": 6,
  "range_mode": "row",
  "hplc_interpolate": true,
  "hplc_subtract_min": true,
  "hplc_normalize_area": true,
  "common_time": [0.0, 0.4],
  "common_time_path": "D:\\PythonProject\\AutoAI\\storage\\preprocessed\\common_time_hplc_xxxx.npy",
  "curves": [
    {
      "name": "hplc_01",
      "x": [0.0, 0.4],
      "raw_y": [-0.1, 1.2],
      "processed_y": [0.0, 2.5]
    }
  ],
  "preview": [
    {
      "Index": 1,
      "Name": "hplc_01",
      "Label": "",
      "Repeat_index": ""
    }
  ]
}
```

重要规则：

1. 预处理接口必须用 `FormData`。
2. 多文件字段名必须重复使用 `files`。
3. 不要把 `files` 做成 JSON 数组。
4. `download_url` 是浏览器下载统一 CSV 的地址。
5. `output_path` 是服务器本地路径，可以作为后续训练 `data_path`，但前提是用户已经补全 `Label` 和 `Repeat_index`。预处理刚生成时这两列为空，直接训练会报错。
6. `common_time_path` 当前只是服务器端复用路径，不在训练 artifact 白名单内，前端不要把它当通用下载链接。

## 5. 创建训练 Run

### 5.5 创建训练任务

```http
POST /api/training/runs
Content-Type: application/json
```

请求体：

```json
{
  "dataset_id": "ds_abc123",
  "data_path": null,
  "test_dataset_id": null,
  "test_data_path": null,
  "config": {
    "model_type": "cnn1d",
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001,
    "normalization": "zscore",
    "split_mode": "stratified_holdout",
    "split_train": 8,
    "split_valid": 1,
    "split_test": 1,
    "early_stopping_patience": 20
  }
}
```

字段规则：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| dataset_id | string | 否 | 首选稳定数据集引用；与 `data_path` 不能同时传 |
| data_path | string | 否 | 受控本地兼容路径；不传则使用项目根目录 `data.csv` |
| test_dataset_id | string | 否 | 独立测试集的稳定数据集引用；与 `test_data_path` 不能同时传 |
| test_data_path | string | 否 | 受控本地兼容路径；传入时后端使用 `external_test_holdout` |
| config | object | 否 | 训练参数 |

创建任务返回：

```json
{
  "run_id": "abc123def456",
  "status": "pending",
  "state": "queued"
}
```

重要：创建接口返回 HTTP 202，只表示 queued Run 已持久化；独立 worker 在请求生命周期之外执行训练。前端不能认为创建任务返回后训练已经完成，必须轮询 `/api/training/runs/{run_id}`。

### 5.6 获取训练记录列表

```http
GET /api/training/runs
```

返回一个数组，每项是一次训练的 `status.json` 内容。最新训练排在前面。

列表和单个 Run 的状态以 SQLite RunRepository 为准；`status.json` 只作为兼容字段投影。

### 5.7 获取单个训练状态

```http
GET /api/training/runs/{run_id}
```

规范状态与兼容状态映射为：

```text
queued     -> pending
running    -> running
succeeded  -> success
failed     -> failed
cancelled  -> paused
```

成功时返回示例：

```json
{
  "run_id": "abc123def456",
  "status": "success",
  "evaluation_strategy": "stratified_holdout",
  "fold_count": 1,
  "metrics": {
    "accuracy": 0.92,
    "macro_f1": 0.91,
    "weighted_f1": 0.92,
    "macro_precision": 0.93,
    "macro_recall": 0.90,
    "confusion_matrix": [[23, 2], [2, 23]],
    "classification_report": {}
  },
  "fold_metrics": [],
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.69,
      "valid_accuracy": 0.8,
      "valid_macro_f1": 0.75,
      "best_valid_macro_f1": 0.75,
      "bad_epochs": 0
    }
  ],
  "model_type": "cnn1d",
  "model_family": "deep_learning",
  "model_artifact": "model.pt",
  "sample_count": 90,
  "test_sample_count": 0,
  "label_names": ["A", "B"],
  "target_epochs": 50,
  "actual_epochs": 26,
  "best_valid_macro_f1": 0.9,
  "not_used_for_reported_cv_metrics": true,
  "config": {},
  "data_path": "D:\\...",
  "test_data_path": null,
  "completed_at": "2026-06-13T12:00:00"
}
```

传统机器学习模型的注意点：

1. `model_family` 是 `traditional_ml`。
2. `model_artifact` 是 `model.pkl`。
3. `target_epochs` 和 `actual_epochs` 都是 1。
4. `history[0].train_loss` 是 `null`。
5. 前端不要假设所有模型都有 loss 曲线。
6. 传统模型会写入 `hyperparameter_search.csv`，记录每折候选参数和验证集表现。

深度学习模型的注意点：

1. `model_family` 是 `deep_learning`。
2. `model_artifact` 是 `model.pt`。
3. `history` 中通常有多轮 epoch。
4. 可能因为早停导致 `actual_epochs < target_epochs`。
5. 留一交叉验证时模型权重只保存最后一折；页面展示的最终性能必须读汇总指标。

### 5.8 下载训练产物

```http
GET /api/training/runs/{run_id}/artifact/{name}
```

允许下载的 `name` 必须同时存在于该 Run 的 `manifest.json` 且标记为 `downloadable=true`：

```text
config.json
label_map.json
split.json
metrics.json
cv_metrics.json
fold_metrics.csv
history.csv
predictions.csv
cv_predictions.csv
hyperparameter_search.csv
feature_importance.json
feature_importance.csv
sample_feature_importance.json
sample_feature_importance.csv
model.pt
model.pkl
status.json
```

如果文件不存在，返回 404；Manifest 中声明为私有的文件（包括 DSCARNet `*.joblib`）返回 403。Run 只有在 Manifest 原子提交后才会进入规范状态 `succeeded`。

### 5.9 取消训练任务

```http
POST /api/training/runs/{run_id}/cancel
```

取消由 SQLite 事务执行；已取消 Run 的兼容状态为 `paused`。worker 使用 claim token 校验，不能用陈旧 claim 覆盖取消结果。

本机启动独立 worker：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe -m backend.app.runs.worker
```

### 5.10 下载预处理文件

```http
GET /api/files?path={absolute_server_path}
```

只允许下载这些目录内的文件：

```text
storage/uploads
storage/preprocessed
```

前端应直接使用预处理接口返回的 `download_url`，不要手动拼接任意本地路径；Run 产物必须使用上一节的 Manifest-backed 路由。

## 6. 训练参数契约

`config` 支持字段如下。前端可以只传用户修改过的字段，没传的后端使用默认值。

```json
{
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001,
  "seed": 42,
  "normalization": "zscore",
  "split_mode": "stratified_holdout",
  "split_train": 8,
  "split_valid": 1,
  "split_test": 1,
  "class_balance": "none",
  "model_type": "cnn1d",
  "early_stopping_patience": 20,
  "dropout": null,
  "hidden_size": 64,
  "transformer_heads": 4,
  "dscarnet_inception_blocks": 1,
  "dscarnet_pca_components": 30,
  "dscarnet_cluster_channels": 9,
  "pls_components": null,
  "random_forest_n_estimators": 100,
  "random_forest_max_depth": 3,
  "random_forest_min_samples_leaf": 2,
  "random_forest_max_features": "sqrt",
  "svm_kernel": "rbf",
  "svm_c": 1.0,
  "svm_gamma": 0.03,
  "xgboost_n_estimators": 50,
  "xgboost_max_depth": 2,
  "xgboost_learning_rate": 0.1,
  "xgboost_subsample": 0.9,
  "xgboost_colsample_bytree": 0.9,
  "xgboost_reg_lambda": 2.0,
  "xgboost_min_child_weight": 1.0,
  "xgboost_gamma": 0.0,
  "feature_selection_enabled": true,
  "feature_window_count": 100,
  "feature_top_k": 5,
  "feature_n_repeats": 5,
  "feature_eval_split": "valid"
}
```

支持的 `model_type`：

```text
pls_da
svm
random_forest
xgboost
cnn1d
transformer1d
resnet1d
inception1d
tcn1d
dscarnet
```

模型别名也可以被后端识别，例如 `PLS-DA`、`1D-Transformer`、`1D-ResNet`、`1D-Inception`、`1D-TCN`、`DSCARNet` 会被规范化；但前端最好传上面的标准小写值。`PLSR` 和 `SVR` 是回归变体，本版分类训练入口不启用；旧 `knn`、`mlp`、`unet1d` 新请求会报错。

输入特征数分为三档：小维度 `1000-3000`、中维度 `3000-6000`、高维度 `6000-10000`。每个模型只保留一种代表性分类架构或小范围搜索策略：

| 模型 | 1000-3000 特征 | 3000-6000 特征 | 6000-10000 特征 |
|---|---|---|---|
| `pls_da` | `n_components=[1,2,3,5]`，上限 `min(5,n_train-2,p)` | `n_components=[1,2,3,5,8]`，上限 `min(8,n_train-2,p)` | `n_components=[1,2,3,5,8,10]`，上限 `min(10,n_train-2,p)` |
| `svm` | 标准化后搜索 `linear` 与小范围 `rbf`，`C=[0.1,1,10]` | 优先 `linear`；`rbf` 只搜索 `gamma≈[0.1/p,1/p,10/p]` | 默认 `linear`，仅保留一个保守 `rbf` 候选 |
| `random_forest` | `n_estimators=300`，`max_depth=[3,5,None]`，`max_features=['sqrt','log2',0.2]` | `n_estimators=500`，`max_depth=[3,5,8]`，`max_features=['sqrt','log2',0.1]` | `n_estimators=600`，`max_depth=[3,5]`，`max_features=['sqrt','log2',0.05]` |
| `xgboost` | 浅树：`max_depth=[2,3]`，`colsample=[0.6,1.0]`，`lambda=[1,5]` | 更强正则：`max_depth=[2,3]`，`colsample=[0.3,0.6]`，`min_child_weight=[3,5]` | 最保守：`max_depth=2`，`colsample=[0.2,0.3]`，`min_child_weight=[5,10]` |
| `cnn1d` | Conv blocks 通道 `[16,32,64]`，kernel `7/5/3`，pool 后 GAP | Stem stride=2，通道 `[24,48,64,96]`，总压缩约 8 倍 | Stem stride=4 或两次 stride=2，通道 `[32,64,96]`，总压缩 8-16 倍 |
| `transformer1d` | Patch `32`/stride `16`，`d_model=48`，2 heads，1-2 layers | Patch `64`/stride `32`，`d_model=64`，4 heads，2 layers | Patch `96`/stride `48`，`d_model=64-96`，4 heads，2 layers，token 控制不超过 256 |
| `resnet1d` | 2 residual stages，通道 `[32,64]`，每 stage 2 conv，GAP | 3 stages `[32,64,96]`，stage 间 stride=2，GAP | 3 stages `[32,64,96]`，首层 stride=4，后续 dilation 扩感受野 |
| `inception1d` | 3 modules，kernels `{10,20,40}`，bottleneck 16，filters 32 | 4 modules，kernels `{16,32,64}`，bottleneck 16，filters 32 | 4 modules，kernels `{20,40,80}`，bottleneck 16，filters 32，早期 stride 压缩 |
| `tcn1d` | 4 residual dilated blocks，kernel 3，dilation `1,2,4,8`，channels 32 | 5 blocks，dilation `1,2,4,8,16`，channels 32-48 | 6 blocks，dilation `1,2,4,8,16,32`，channels 32-48，输入先 stride 压缩 |
| `dscarnet` | 训练折内 AggMap/PCA；SAR 约 `32x32-56x56`，CAR `25-55` PCs | SAR 约 `56x56-80x80`，CAR `40-80` PCs，主体保持 1 个 inception block | SAR 约 `64x64-96x96`，CAR `50-100` PCs，主体不加深，只加强正则和 early stopping |

支持的 `normalization`：

```text
zscore
minmax
area
none
```

支持的 `class_balance`：

```text
none
class_weight
```

划分与评估规则：

1. 无独立测试集且 `split_mode=stratified_holdout` 时，按 `Repeat_index` 整组并尽量按标签比例划分，默认 `split_train=8, split_valid=1, split_test=1`。
2. 无独立测试集且 `split_mode=leave_one_repeat_index_cv` 时，每个 `Repeat_index` 恰好作为一次 test；剩余样品组内部默认按 8:2 生成 train/valid。旧值 `outer_leave_one_repeat_index_cv` 作为兼容别名仍可传入。
3. 有独立测试集且传入顶层 `test_data_path` 时，后端使用 `external_test_holdout`；主数据默认 `split_train=8, split_valid=2, split_test=0`，独立测试集作为最终 test。
4. 每一折或每次 holdout 的 scaler、PLS、AggMap/PCA 等预处理对象只能由当前训练集拟合，并应用到同一口径下的 valid/test。
5. 后端状态中的 `evaluation_strategy` 只会输出 `stratified_holdout`、`leave_one_repeat_index_cv` 或 `external_test_holdout`。

解释性配置规则：

1. `feature_selection_enabled=false` 时仍会在状态中返回 disabled 摘要，并写入兼容的解释性 artifact 摘要。
2. `feature_window_count` 只影响传统 ML 和无卷积模型的窗口重要性；指标为 `baseline_macro_f1 - perturbed_macro_f1`，正值越大表示遮挡/置乱该窗口后 macro-F1 下降越多，窗口越重要。
3. 卷积模型和 `transformer1d` 继续使用原有单样品解释：卷积/ResNet/Inception/TCN 用 Grad-CAM-like，Transformer 用输入梯度，DSCARNet 用 SAR/CAR 双通路 2D Grad-CAM 回投。
4. `feature_top_k` 控制每个样品或聚合结果展示的前若干重要区间。
5. `dscarnet_pca_components` 和 `dscarnet_cluster_channels` 只对 `model_type=dscarnet` 生效。

## 7. 推荐调用流程

### 7.1 只做预处理

1. 用户选择拉曼或色谱原始 CSV。
2. 拉曼调用 `/api/preprocess/raman`；主色谱流程调用 `/api/preprocess/hplc`；只有需要简单截取时才调用 `/api/preprocess/chromatography`。
3. 显示返回的 `curves`。
4. 使用 `download_url` 下载统一 CSV。
5. 用户在 CSV 中补全 `Label` 和 `Repeat_index`。

### 7.2 建模训练

1. 用户上传补全后的建模 CSV。
2. 前端调用 `/api/datasets/upload`。
3. 保存返回的 `dataset_path`。
4. 根据 `summary` 展示数据摘要。
5. 用户选择 10 类分类模型之一，并选择评估口径：分层 8:1:1、留一交叉验证，或上传独立测试集。
6. 如上传独立测试集，前端再次调用 `/api/datasets/upload` 并把返回路径作为顶层 `test_data_path`。
7. 前端调用 `/api/training/runs` 创建任务；无独立测试集时传 `stratified_holdout` 或 `leave_one_repeat_index_cv`，有独立测试集时主数据传 8:2 train/valid。
8. 保存返回的 `run_id`。
9. 每 1 到 2 秒轮询 `/api/training/runs/{run_id}`。
10. 当 `status` 是 `success` 或 `failed` 时停止轮询。
11. 成功后读取 `evaluation_strategy`、`metrics`、`cv_metrics.json`、`fold_metrics.csv`、`cv_predictions.csv`、`label_names`、`model_artifact` 和下载链接。

## 8. 前端最容易导致后端报错的点

### 8.1 文件上传格式错误

错误方式：

```js
fetch("/api/datasets/upload", {
  method: "POST",
  body: JSON.stringify({ file })
})
```

正确方式：

```js
const form = new FormData();
form.append("file", file);
fetch("/api/datasets/upload", { method: "POST", body: form });
```

多文件预处理时：

```js
const form = new FormData();
for (const file of files) {
  form.append("files", file);
}
```

### 8.2 把服务器路径当成本地浏览器路径

`dataset_path`、`output_path` 是服务器路径，只能传回后端，不能直接当作浏览器下载链接。

下载预处理 CSV 使用：

```text
download_url
```

下载训练产物使用：

```text
/api/training/runs/{run_id}/artifact/{name}
```

### 8.3 训练任务没有轮询

`POST /api/training/runs` 只表示任务创建成功，不表示训练完成。必须轮询。

### 8.4 传统模型没有 loss 曲线

`pls_da`、`random_forest`、`svm`、`xgboost` 的 `history[0].train_loss` 是 `null`。前端绘图前必须判断是否为数字。

### 8.5 传入外部测试集

`test_data_path` 必须放在 `/api/training/runs` 请求体顶层，而不是放进 `config`。传入后，前端应隐藏或禁用内部 test 比例，主数据使用 `split_train=8, split_valid=2, split_test=0`，后端状态会显示 `evaluation_strategy=external_test_holdout`。

### 8.6 建模 CSV 没补 Label 或 Repeat_index

预处理刚生成的 CSV 不能直接训练，因为 `Label` 和 `Repeat_index` 为空。直接训练会返回类似：

```text
第 2 行 Label 为空，建模前请补充标签
```

### 8.7 Repeat_index 分组不满足规则

常见错误：

1. 同一个 Repeat_index 中混入多个 Label。
2. 有的样品重复测量 5 次，有的只有 4 次。
3. 独立 `Repeat_index` 组数太少，无法完成分层 holdout、留一 CV 或外部测试集场景下的内部 8:2 train/valid 划分。

这些都属于数据问题，前端应显示后端返回的 `detail` 或训练状态中的 `error`。

## 9. 最小请求示例

### 9.1 健康检查

```js
async function checkHealth() {
  const res = await fetch("/health");
  if (!res.ok) throw new Error("后端不可用");
  return res.json();
}
```

### 9.2 上传数据集

```js
async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/datasets/upload", {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "上传失败");
  return data;
}
```

### 9.3 预处理

```js
async function preprocessRaman(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("range_mode", "x_value");
  form.append("x_min", "400");
  form.append("x_max", "1800");
  form.append("baseline_order", "range_then_baseline");
  form.append("baseline_method", "arPLS");

  const res = await fetch("/api/preprocess/raman", {
    method: "POST",
    body: form,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "预处理失败");
  return data;
}
```

### 9.4 创建训练任务并轮询

```js
async function startTraining(datasetPath) {
  const res = await fetch("/api/training/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data_path: datasetPath,
      config: {
        model_type: "cnn1d",
        epochs: 50,
        batch_size: 16,
        split_mode: "stratified_holdout",
        split_train: 8,
        split_valid: 1,
        split_test: 1,
      },
    }),
  });
  const created = await res.json();
  if (!res.ok) throw new Error(created.detail || "创建训练任务失败");
  return created.run_id;
}

async function pollRun(runId) {
  const res = await fetch(`/api/training/runs/${runId}`);
  const run = await res.json();
  if (!res.ok) throw new Error(run.detail || "读取训练状态失败");
  return run;
}
```

## 10. 结果文件说明

每次训练会在：

```text
storage/runs/{run_id}/
```

生成：

| 文件 | 说明 |
|---|---|
| status.json | 前端轮询读取的完整状态 |
| config.json | 实际训练配置 |
| label_map.json | 类别 id 到 label 的映射 |
| split.json | 当前评估口径下 train/valid/test 的样本行索引和 `Repeat_index`；外部测试集场景会额外记录 external test 索引 |
| metrics.json | 当前评估口径的汇总指标，含 accuracy、macro-F1、weighted-F1、混淆矩阵和分类报告 |
| cv_metrics.json | 兼容指标文件；holdout 场景为 1 个 fold，留一 CV 场景为多 fold 汇总 |
| fold_metrics.csv | 每个 holdout/CV fold 的指标 |
| history.csv | 每轮训练历史，传统模型只有一行 |
| predictions.csv | 兼容预测文件，内容为当前评估口径的 test 预测 |
| cv_predictions.csv | 每个 test 样本的真实标签、预测标签、概率和所属 `Repeat_index` |
| hyperparameter_search.csv | 传统模型每折小范围参数搜索记录；深度模型通常为空或仅含固定配置 |
| model.pt | 深度学习模型；留一 CV 时仅为最后一折模型，holdout 时为本次训练模型 |
| model.pkl | 传统机器学习模型；留一 CV 时仅为最后一折模型，holdout 时为本次训练模型 |
| feature_importance.json/csv | 聚合重要区间；传统模型为 `baseline_macro_f1 - perturbed_macro_f1` |
| sample_feature_importance.json/csv | 深度模型 test 集单样品重要区间，前端优先展示 |
| dscarnet_mapping.json | DSCARNet SAR/CAR 映射元数据，仅 DSCARNet 生成 |
| dscarnet_pca.joblib | DSCARNet PCA 对象，仅写入 run 目录，当前下载接口不开放 |
| dscarnet_sar_aggmap.joblib | DSCARNet SAR AggMap 对象，仅写入 run 目录，当前下载接口不开放 |
| dscarnet_car_aggmap.joblib | DSCARNet CAR AggMap 对象，仅写入 run 目录，当前下载接口不开放 |

当前 `GET /api/training/runs/{run_id}/artifact/{name}` 只允许下载白名单文件：`config.json`、`label_map.json`、`split.json`、`metrics.json`、`cv_metrics.json`、`fold_metrics.csv`、`history.csv`、`predictions.csv`、`cv_predictions.csv`、`hyperparameter_search.csv`、`feature_importance.json`、`feature_importance.csv`、`sample_feature_importance.json`、`sample_feature_importance.csv`、`model.pt`、`model.pkl`、`status.json`。不要在前端写死白名单之外的下载链接。

## 11. 对接完成后的验收清单

另一个前端 agent 完成后，至少跑下面这些检查。

### 11.1 静态页面可访问

```bash
curl http://127.0.0.1:8000/health
```

应返回：

```json
{"status":"ok"}
```

### 11.2 上传建模 CSV

调用：

```text
POST /api/datasets/upload
```

必须返回 200，并且响应中包含：

```text
dataset_path
summary.samples
summary.classes
summary.repeat_index.group_count
```

### 11.3 预处理接口

调用：

```text
POST /api/preprocess/raman
POST /api/preprocess/chromatography
```

必须返回 200，并且响应中包含：

```text
download_url
curves
preview
```

拉曼曲线必须有：

```text
raw_y
corrected_y
```

色谱曲线必须有：

```text
raw_y
```

### 11.4 训练任务

调用：

```text
POST /api/training/runs
GET /api/training/runs/{run_id}
```

必须能从：

```text
pending 或 running
```

最终进入：

```text
success
```

如果进入 `failed`，前端必须显示 `error`。

### 11.5 下载文件

训练成功后，以下链接至少应有一个可下载：

```text
/api/training/runs/{run_id}/artifact/predictions.csv
/api/training/runs/{run_id}/artifact/metrics.json
/api/training/runs/{run_id}/artifact/status.json
```

## 12. 后端自动化测试

在项目根目录运行：

```bash
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe -m pytest backend\tests\test_smoke.py -q
```

当前期望：全部通过。

如果另一个 agent 只改前端，原则上不应改动后端测试。如果测试失败，优先检查是否误改了接口字段名、请求方法、路由路径或静态文件路径。
