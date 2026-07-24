# 建模结果接口契约（run-result-v1）

> 最近核对：2026-07-17。该接口是经典前端和 v2 工作台专属建模结果页的共同稳定数据源；原 `GET /api/training/runs/{run_id}` 继续作为兼容状态接口。

前端应先读取匿名 `GET /health` 的 `contracts.run_result`。只有明确发现旧 Web 不支持 `run-result-v1` 时才允许回退旧状态接口；当前 Web 返回 404 表示 Run 不存在或不可见，不能静默解释为“历史 Run”。

## 请求与 URL

```http
GET /api/training/runs/{run_id}/result
```

前端深链接：

```text
/#/results?run_id=<URL-encoded Run ID>
/static/v2/index.html#/results?run_id=<URL-encoded Run ID>
```

v2 公共入口 `/v2` 会重定向到静态工作台；结果页刷新、分享链接和自动跳转均以 URL 中的 Run ID 重新取数。两套前端只能下载 `artifacts[]` 中 `downloadable=true` 且 `download_url` 合法的条目。单样品解释在摘要为 `ready` 且 JSON artifact 可下载时默认加载，使用样品自身的 `curve` 与 `sample_x_axis`；旧产物缺少这些字段时才允许回退到全局轴或基线曲线。

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
| `cancelled` | STOP；训练已停止且本次产物不保留 |
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
- 图表分析的 Test 使用 pooled OOF；Train/Valid 使用跨折预测合并，`aggregation="pooled_cross_fold"`，同一样本可能在不同折重复出现。Train/Valid 标量主展示仍是 fold mean。

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
    "splits": {
      "train": {
        "aggregation": "direct",
        "confusion_matrix": [[9, 1], [0, 10]],
        "classification_report": {},
        "prediction_distribution": {
          "labels": ["A", "B"],
          "true_counts": [10, 10],
          "predicted_counts": [9, 11]
        }
      },
      "valid": {},
      "test": {}
    },
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

新页面以 `analysis.splits` 为准。顶层 `confusion_matrix`、`classification_report`、`prediction_distribution` 暂时继续映射 Test，供旧调用方兼容。传统模型不生成 `history.csv`，`history.available=false` 时前端不渲染训练曲线区域；历史传统 Run 即使含单行 history 也按不适用处理。深度模型曲线使用真实 epoch，并展示数值轴、刻度和网格。当前没有正式 ROC-AUC/ROC/PR 产物，前端不得自行猜测。

## 解释性

```json
{
  "explainability": {
    "samples": {
      "status": "ready",
      "artifact": "sample_feature_importance.json",
      "csv_artifact": "sample_feature_importance.csv",
      "method": "sample_occlusion_log_loss",
      "sample_count": 9
    }
  }
}
```

这里仅提供单样品解释的安全摘要。完整 JSON/CSV 由 `artifacts[]` 下载；单样品文件可能较大，前端应在用户明确点击后再懒加载，加载完成后才在本地切换样品，不能把全部内容重复塞入首屏结果响应。新 Run 不生成或投影全局重要性。

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

训练记录列表的 `GET /api/training/runs?projection=summary` 不属于 `run-result-v1` 完整投影，但可额外返回可空标量 `test_macro_f1`。该值必须与本契约的 Test 主口径一致：holdout 取直接测试指标，CV 取 pooled OOF 主指标，禁止用 fold mean；指标文件缺失、损坏、Run 未成功或结果不完整时返回 `null`。

- 原 Run 状态接口、旧顶层字段和旧 artifact URL 不删除。
- 新 Manifest v2 使用大小和 SHA-256 校验。
- 历史 Manifest 继续按旧 `downloadable` 做 local 只读兼容，但结果页会给出兼容 warning。
- 历史 Manifest 已登记的 `feature_importance.json/csv` 保留直接下载兼容，但 descriptors 和新页面不列出；新 Manifest catalog 不再登记这两个文件。
- server Principal 不使用“无 DB 的旧目录”旁路。
- 部分文件损坏只影响对应分析和下载，不应导致整个结果页 500。

## 最小验收

1. queued、running、failed、cancelled、ready、partial、missing/corrupt Manifest 均有测试。
2. 同一 URL 连续请求返回同一 Run，不依赖前端内存。
3. CV fixture 能证明 pooled OOF 与 fold mean 分离。
4. 私有、缺失、篡改和未成功 Run 的下载分别被拒绝。
5. server 模式下其他 Principal 和旧 owner 为空 Run 返回 404。
6. 响应和公开配置中不存在服务器绝对路径。
7. Holdout/CV fixture 均验证三分区分析口径；新结果不含 `explainability.global`。
8. 传统模型没有 history 文件/曲线，深度模型保留真实 history。
