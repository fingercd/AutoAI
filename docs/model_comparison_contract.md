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

新 Batch 固定一模型一 Run。sample_correctness.values 为 0/1/null；details 返回真实／预测类别、测量条数和概率均值或投票的聚合方法。Recall values 的 support 为该类预测记录数；混淆矩阵顺序严格按 label_map 编码顺序，不按字符串重排矩阵。

历史 R>1 批次默认取每模型第一条成功完整 Run，不合计矩阵、不使用重复均值。run_comparisons 按模型提供各单次投影用于切换；历史候选仍须满足数据快照、划分、标签、测试样品集合一致性。

models[].experiment 来自 Manifest 校验通过的 feature_experiments.json，可为空。包含 version、selected_configuration、fold_configurations、schemes。schemes 返回七种方案的状态、原因、四指标与配置；方案未完成全部外层折时不伪造完整指标。普通 LOSO 没有统一最佳配置，selected_configuration=null；所选方案逐折测试预测合并为主指标。

`leave_one_sample_id_cv` 使用 pooled OOF Test。`leave_one_sample_id_cv_with_external_test` 仅使用 `dataset=external_test` 预测行和 direct external Test 主指标；其主数据 OOF 仅是单 Run 审计。当前没有 ROC、PR 或 AUC 产物，接口和前端都不伪造这些值。

## 经典对比页与持久归档（2026-09-05）

临时功能开关：特征工程前后端已停用，代码保留。`feature_policy.FEATURE_ENGINEERING_ENABLED=False` 时，word-0904 训练配置被拒绝，features 图像请求返回 409，新归档不生成特征方案图像；普通训练不变，历史文件不删除。经典对比页不显示特征工程区块或占位说明。混淆矩阵放大图上限 640px，同时受视口高度约束；此调整不作用于其他图的放大窗口。

经典对比页使用 `static/js/comparison-page.js/css`；v2 和单模型结果页不变。样式全部限定在 `#view-comparison`。总体指标按内容宽度 1100/640px 切为四/二/一列；墨绿矩阵默认真实类别归一化，计数共享范围。格内数字按格子宽度缩小（约 8–11px），轴标签 13px、标题 16px。Sample_ID 每页 50 个，缺失不当作错误。

后端 `comparison_figures.py` 使用同一 Matplotlib Figure 输出 SVG/PNG，不依赖服务器浏览器或 Node。基础依赖包含 Matplotlib；非 Windows 部署若使用中文类别名，应安装 Noto Sans CJK SC 或文泉驿字体。

新增接口（全部按 Principal scope）：

- `GET /api/training/batches?limit=20&cursor=<batch_id>`：批次级游标分页，返回 items、next_cursor；含数据集名、模型、状态、计数和归档状态。游标也必须属于当前 Principal。
- Run summary 增加可空 `batch_id`，经典训练记录据此追加“查看所属对比”。
- `GET /api/training/batches/{id}/archive`：返回 pending/missing/failed/ready/discarded；ready 时附归档 comparison。
- `POST /api/training/batches/{id}/archive?force=false`：幂等补建；force=true 重试生成。运行中/已停止或并发生成返回 409。
- `GET /api/training/batches/{id}/archive/files/{name}`：下载校验通过的 JSON/CSV/SVG/PNG；`all.zip` 包含全部默认图集、数据和 Manifest。
- `POST /api/training/batches/{id}/figure?format=svg|png`：kind 为 overall/matrix/recall/samples/features；接受 metric、model、sort、page（0 基）、search、errors、matrix_mode（percent/count）、width（260–1800）。SVG 返回 svg、width、height 和绘图行列数据；PNG 返回图像。筛选图不覆盖默认归档。

归档位于 Run 数据库同级的 `batches/<batch_id>/`，格式为 `batch-comparison-archive-v1`，包含比较快照、默认 SVG/PNG 与 CSV、带 SHA-256/大小的 manifest.json；绘图版本为 comparison-figures-v1。完整 Sample_ID 图集按 50 个分图保存。

Worker 在终态提交之外尝试归档；归档失败不能改写训练成功状态，另存不可下载的错误标记。跨进程文件锁阻止重复生成，临时目录完成后在 SQLite 发布保护下替换；STOP/DELETE 阻止迟到发布并清理该批次归档。部分失败只保存可比较结果；不可比较时保存原因，不画排行。历史批次首次打开时由前端显式 POST 补建，无需重训。

默认归档不包含旧重复一致性图、私有模型文件或服务器路径；配置详情保留逐折选优信息。历史缺特征工程产物时不补造图像。
