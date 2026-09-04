# 模型比较接口契约（model-comparison-v1）

`GET /api/training/batches/{batch_id}/comparison` 仅返回当前 Principal 可读取 Batch 的比较投影。它从 Manifest SHA-256/大小校验通过的成功子 Run 读取指标、预测、标签映射和划分审计；响应不包含服务器路径、模型文件或 traceback。

比较前必须一致：主/独立数据快照、评估策略和主聚合口径、`split.json` digest、Test/OOF `Sample_ID` 集合和 `label_map.json`。任何一项不一致时返回 `comparable=false` 及原因，不生成排行或热图。

主结构：

```json
{
  "schema_version": "model-comparison-v1",
  "batch_id": "batch_x",
  "state": "partial",
  "comparable": true,
  "overall_metrics": {"metrics": ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"], "rows": []},
  "sample_correctness": {"status": "ready", "sample_ids": [], "values": []},
  "repeat_stability": {"status": "ready", "rows": []},
  "class_recall": {"status": "ready", "labels": [], "rows": []},
  "confusion_matrices": []
}
```

新 Batch 固定一个模型对应一个 Run，因此每个模型的总体指标直接来自该 Run 的真实 Test 主口径；不生成或展示虚假的误差条。`sample_correctness` 在子 Run 内按 `Sample_ID` 平均 `prob_<label>` 后 argmax，生成模型对各 Sample_ID 的正确性；普通新批次在前端明确显示为“正确/错误”，缺失预测使用 `null`/`—`，不补零。类别 Recall 从真实分类报告读取；缺失类别不会补零。`confusion_matrices` 在比较页按模型直接渲染，行是真实类别、列是预测类别。为兼容已经存在的历史 R>1 批次，响应仍可能聚合 mean/std/min/max 并返回 `repeat_stability`，但普通前端不再创建或展示重复实验；历史正确率和合并矩阵必须标明为历史结果，不能伪装成单次结果。

`leave_one_sample_id_cv` 使用 pooled OOF Test。`leave_one_sample_id_cv_with_external_test` 仅使用 `dataset=external_test` 预测行和 direct external Test 主指标；其主数据 OOF 仅是单 Run 审计。当前没有 ROC、PR 或 AUC 产物，接口和前端都不伪造这些值。
