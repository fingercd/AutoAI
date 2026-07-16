# 建模结果接口契约（run-result-v1）

> 最近核对：2026-07-16。该接口是专属建模结果页的稳定数据源；原 `GET /api/training/runs/{run_id}` 继续作为兼容状态接口。

## 请求与 URL

```http
GET /api/training/runs/{run_id}/result
```

前端深链接：

```text
/#/results?run_id=<URL-encoded Run ID>
```

Run ID 属于当前 Principal 时返回结果；不存在、已删除或不属于当前 Principal 均返回 404。server 模式还需要 Bearer 令牌。

## 顶层结构

```json
{
  "schema_version": "run-result-v1",
  "run": {},
  "dataset": {},
  "model": {},
  "evaluation": {},
  "metrics": {},
  "analysis": {},
  "explainability": {},
  "artifacts": [],
  "warnings": []
}
```

worker 当前使用的稳定错误码为：数据内容被替换或 hash 不符时 `dataset_changed`，文件不存在或不可读时 `dataset_unavailable`，数据内容或训练配置不合法时 `invalid_training_data_or_config`，其他训练异常时 `training_failed`。

接口不会返回 `data_path`、`test_data_path`、Run 目录或其他服务器绝对路径。

## Run 与结果状态

`run.state` 是执行状态：

```text
queued | running | succeeded | failed | cancelled
```

`run.result_state` 是页面可用性状态：

| 值 | 含义 |
|---|---|
| `pending` | 已排队，尚未执行 |
| `running` | 正在训练 |
| `failed` | 训练失败 |
| `cancelled` | 已取消 |
| `ready` | 成功且必需结果完整 |
| `partial` | 成功，但部分必需或已登记文件缺失/损坏 |
| `missing_manifest` | Run 成功，但 Manifest 缺失 |
| `corrupt_manifest` | Manifest 无法解析 |

时间字段：

```json
{
  "created_at": "2026-07-16T01:00:00+00:00",
  "started_at": "2026-07-16T01:00:05+00:00",
  "finished_at": "2026-07-16T01:02:10+00:00",
  "duration_seconds": 125.0
}
```

旧 Run 缺少数据库时间时返回 `null`；如能从历史状态恢复耗时，会同时写入 warning，不凭空推算。

失败对象只提供结构化诊断，不包含 traceback：

```json
{
  "error": {
    "code": "invalid_training_data_or_config",
    "stage": "training_validation",
    "message": "数据集标签无效",
    "retryable": false,
    "type": "ValueError"
  }
}
```

## 数据集快照

```json
{
  "dataset_id": "ds_abc",
  "name": "teacher-data.csv",
  "sha256": "...",
  "curve_count": 90,
  "sample_id_count": 30,
  "class_count": 3,
  "feature_count": 160,
  "test_curve_count": null
}
```

- `curve_count` 表示曲线/CSV 行数。
- `sample_id_count` 表示唯一 Sample_ID 数量。
- 新 Run 在入队时保存数据 hash，并在 worker 读取前核验；旧 Run 缺失字段时返回 `null`。

## 评估和指标

```json
{
  "evaluation": {
    "strategy": "leave_one_sample_id_cv",
    "fold_count": 30,
    "primary_split": "test",
    "primary_aggregation": "pooled_oof"
  }
}
```

普通 holdout：

- `metrics.primary` 是 Test 直接指标。
- `metrics.splits.<train|valid|test>.aggregation="direct"`。
- `metrics.direct` 保存三个直接分区。

留一 Sample_ID CV：

- `metrics.primary` 与 `metrics.pooled_oof` 是所有 test OOF 预测合并后的主指标。
- Test split 的 `aggregation="pooled_oof"`。
- Train/Valid split 的 `aggregation="fold_mean"`。
- `metrics.fold_mean` 与 `metrics.fold_std` 仅用于逐折审计。
- 不得用折 Test Macro F1 平均值覆盖 pooled OOF Macro F1。

当前可能返回的标量：

```text
accuracy
balanced_accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
```

`metrics.primary` 还可能包含 `confusion_matrix` 和 `classification_report`。

## 分析能力

```json
{
  "analysis": {
    "confusion_matrix": [[3, 1], [1, 3]],
    "classification_report": {},
    "prediction_distribution": {
      "labels": ["A", "B"],
      "true_counts": [4, 4],
      "predicted_counts": [4, 4]
    },
    "history": {
      "available": true,
      "rows": [],
      "columns": [],
      "truncated": false,
      "reason": null
    },
    "roc": {
      "available": false,
      "reason": "当前训练产物未计算 ROC 曲线或 ROC-AUC"
    },
    "precision_recall": {
      "available": false,
      "reason": "当前训练产物未计算 Precision-Recall 曲线"
    }
  }
}
```

传统模型通常没有逐 epoch loss；`history.available=false` 时前端显示原因，不绘制空坐标。当前没有正式 ROC-AUC/ROC/PR 产物，前端不得自行从不完整数据猜测。

## 解释性

```json
{
  "explainability": {
    "global": {},
    "samples": {}
  }
}
```

这里提供用于选择和概览的安全摘要。完整 JSON/CSV 由 `artifacts[]` 下载；单样品文件可能较大，前端应在用户明确点击后再懒加载，加载完成后才在本地切换样品，不能把全部内容重复塞入首屏结果响应。

## Artifact 描述

每个条目：

```json
{
  "name": "metrics.json",
  "label": "总体指标",
  "category": "metrics",
  "format": "json",
  "media_type": "application/json",
  "size_bytes": 1234,
  "sha256": "...",
  "required": true,
  "applicable": true,
  "exists": true,
  "integrity": "ok",
  "downloadable": true,
  "reason": null,
  "download_url": "/api/training/runs/.../artifact/metrics.json",
  "suggested_filename": "run_abc123__metrics.json"
}
```

`integrity` 可能为：

```text
ok | volatile | missing | corrupt | not_generated
```

页面按钮只在 `downloadable=true` 且 URL 安全时启用。禁用时直接显示 `reason`；不提供难以理解的单一“下载全部”按钮。

新 Manifest 使用显式 catalog。`model.pkl`、`model.pt`、`*.joblib`、`status.json` 和 Manifest 本身不作为结果页下载；无服务器路径的 `config.json` 可下载，含路径字段的历史配置自动禁用。

## 兼容性

- 原 Run 状态接口、旧顶层字段和旧 artifact URL 不删除。
- 新 Manifest v2 使用大小和 SHA-256 校验。
- 历史 Manifest 继续按旧 `downloadable` 做 local 只读兼容，但结果页会给出兼容 warning。
- server Principal 不使用“无 DB 的旧目录”旁路。
- 部分文件损坏只影响对应分析和下载，不应导致整个结果页 500。

## 最小验收

1. queued、running、failed、cancelled、ready、partial、missing/corrupt Manifest 均有测试。
2. 同一 URL 连续请求返回同一 Run，不依赖前端内存。
3. CV fixture 能证明 pooled OOF 与 fold mean 分离。
4. 私有、缺失、篡改和未成功 Run 的下载分别被拒绝。
5. server 模式下其他 Principal 和旧 owner 为空 Run 返回 404。
6. 响应和公开配置中不存在服务器绝对路径。
