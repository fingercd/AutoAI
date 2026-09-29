# 模型比较接口契约（model-comparison-v1）

当前实现核对：2026-09-29。经典比较页按“总体表现、分类指标、样品预测、特征方案、参数与划分”组织为标签页；特征方案标签由 `/api/models.training_scheme.enabled` 控制。类别 Recall 与 Precision 各显示一张热图，精确数值与 Support 可折叠查看。当前绘图版本为 `comparison-figures-v4`，过期或缺图的归档可从已验证 Run 产物补建。

## 2026-09-07 特征方案接通

经典对比页移除总体性能“排序”下拉框和“选择历史对比”入口。四项总体指标图与数值表统一沿用默认 Balanced Accuracy 降序；下载、归档和图表放大保留。历史结果从训练记录的“查看所属对比”进入，未指定批次的对比页提供训练记录链接，不再请求批次列表。

按《AutoAI_model要求_0904》第 6 节保留网页最优参数及留一法逐轮最优参数展示。对比页删除“不适用与缺失说明”和“配置与审计详情”两个折叠区块；底层缺失值仍保留为 null。特征热图标题、方案名称与指标选项使用中文，PCA/F1 和模型缩写保留；绘图版本升级为 comparison-figures-v2，PNG/SVG/CSV 归档按已有结果更新，不重新训练。

经典页面按 `/api/models.training_scheme.enabled` 恢复特征热图，不保留独立的前端开关。七方案为行，模型为列，默认 Balanced Accuracy，可切换四指标；只显示 ready 的真实指标。not_applicable、partial、failed 和历史缺失保留原因，不能填 0。最佳星号只来自 selected_configuration，普通留一法没有全局星号。最佳参数及 fold_configurations 以可读、按需展开的参数表展示。

归档状态新增 outdated：绘图版本不符或应有图集缺失时，按已有产物幂等补建，不创建训练。归档使用批次目录内不可变 `gen-*` 代次和原子 `current.json` 指针发布；历史平铺归档可读。旧代次保留供并发读取，停止/删除批次时整组清理。下载仍只允许 Manifest 列出的公开文件，指针与服务器路径不开放。默认 ZIP 包含四项 features 指标的 SVG、PNG 和 CSV。

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
  "class_metrics": {"status": "ready", "labels": [], "rows": []},
  "confusion_matrices": []
}
```

`class_metrics` 按类别给出 Precision / Recall / F1 / Support。新 Batch 每模型只有一个 Run；历史重复批次默认采用每模型第一条成功且结果完整的 Run，类别指标不跨 Run 求均值或求和。任何字段缺失即为 `null`，不填 0。**Recall（召回率）= TP/(TP+FN)**，二分类时即 Sensitivity；**Precision（精确率）= TP/(TP+FP)**。经典对比页依据 `class_recall` 与 `class_metrics.precision` 显示两张热图，精确数值与 Support 可折叠查看；图像与归档读取同一投影，缺失值保留为空。2026-09-27 两项指标改为热图并加入 precision 图集；随后绘图版本升级为 `comparison-figures-v4`，旧图集可按现有产物补建，无需重训。

## 预测明细 Excel 导出（2026-09-22）

`GET /api/training/batches/{batch_id}/predictions.xlsx` 返回两个工作表的 `.xlsx`（附件下载），同样按 Principal scope，只使用 Manifest 校验通过的成功子 Run 产物，响应不含服务器路径：

- 工作表 1 `预测类别`：固定 `Index` / `Label` / `Sample_ID` / `划分` 四列，其后每个可比较模型一列，单元格是该模型对该记录的预测类别。
- 工作表 2 `预测概率`：布局相同，每个模型单元格是按 `label_map` 数字键顺序给出的逗号分隔六位小数文本。底层逐类概率归一化；写成六位小数后总和可能有轻微误差，列头注释与工作表名标注类别顺序。

覆盖范围由评估口径和已验证产物决定，不重算、不补造：`stratified_holdout` / `external_test_holdout` 从 `all_predictions.csv` 读取最终模型对全量记录的预测，主数据标为 train/valid/test，独立测试记录标为 `external_test`。普通留一法读取主数据 pooled OOF，每条记录只在其作为测试集的折出现一次并标为 `test`。留一法加独立 Test 时同时导出主数据 pooled OOF 行（`test`）和独立测试行（`external_test`）；该模式的独立 Test 是主指标，pooled OOF 仅用于审计。历史 Run 无 `all_predictions.csv` 时回退到 `predictions.csv`，只导出已有的 test/OOF 行，`Index` 留空。批次不可比较或缺少校验通过的预测/类别映射时返回 409 与原因。

新 Batch 固定一模型一 Run。sample_correctness.values 为 0/1/null；details 返回真实／预测类别、测量条数和概率均值或投票的聚合方法。Recall values 的 support 为该类预测记录数；混淆矩阵顺序严格按 label_map 编码顺序，不按字符串重排矩阵。

历史 R>1 批次默认取每模型第一条成功完整 Run，不合计矩阵、不使用重复均值。run_comparisons 按模型提供各单次投影用于切换；历史候选仍须满足数据快照、划分、标签、测试样品集合一致性。

models[].experiment 来自 Manifest 校验通过的 feature_experiments.json，可为空。包含 version、selected_configuration、fold_configurations、schemes。schemes 返回七种方案的状态、原因、四指标与配置；方案未完成全部外层折时不伪造完整指标。普通 LOSO 没有统一最佳配置，selected_configuration=null；所选方案逐折测试预测合并为主指标。

`leave_one_sample_id_cv` 使用 pooled OOF Test。`leave_one_sample_id_cv_with_external_test` 仅使用 `dataset=external_test` 预测行和 direct external Test 主指标；其主数据 OOF 仅是单 Run 审计。当前没有 ROC、PR 或 AUC 产物，接口和前端都不伪造这些值。

## 经典对比页与持久归档（2026-09-05）

历史基线（2026-09-05，当时特征方案曾暂时停用；已由 2026-09-07 恢复记录覆盖）：当时 `FEATURE_ENGINEERING_ENABLED=False` 会拒绝 word-0904 训练配置、关闭 features 图像请求并阻止新归档生成特征方案图像。此状态不代表当前实现；当前开关为 `True`，前端按 `/api/models.training_scheme.enabled` 显示特征方案结果。

经典对比页使用 `static/js/comparison-page.js/css`；v2 和单模型结果页不变。样式全部限定在 `#view-comparison`。总体指标与矩阵随容器宽度响应式排列；放大弹窗受视口宽高约束。Sample_ID 每页 50 个，缺失不当作错误。

后端 `comparison_figures.py` 使用同一 Matplotlib Figure 输出 SVG/PNG，不依赖服务器浏览器或 Node。基础依赖包含 Matplotlib；非 Windows 部署若使用中文类别名，应安装 Noto Sans CJK SC 或文泉驿字体。

新增接口（全部按 Principal scope）：

- `GET /api/training/batches?limit=20&cursor=<batch_id>`：批次级游标分页，返回 items、next_cursor；含数据集名、模型、状态、计数和归档状态。游标也必须属于当前 Principal。
- Run summary 增加可空 `batch_id`，经典训练记录据此追加“查看所属对比”。
- `GET /api/training/batches/{id}/archive`：返回 pending/missing/failed/ready/discarded；ready 时附归档 comparison。
- `POST /api/training/batches/{id}/archive?force=false`：幂等补建；force=true 重试生成。运行中/已停止或并发生成返回 409。
- `GET /api/training/batches/{id}/archive/files/{name}`：下载校验通过的 JSON/CSV/SVG/PNG；`all.zip` 包含全部默认图集、数据和 Manifest。
- `POST /api/training/batches/{id}/figure?format=svg|png`：kind 为 overall/matrix/recall/precision/samples/features；接受 metric、model、sort、page（0 基）、search、errors、matrix_mode（percent/count）、width（260–1800）。SVG 返回 svg、width、height 和绘图行列数据；PNG 返回图像。筛选图不覆盖默认归档。

归档位于 Run 数据库同级的 `batches/<batch_id>/`，格式为 `batch-comparison-archive-v1`，包含比较快照、默认 SVG/PNG 与 CSV、带 SHA-256/大小的 manifest.json；绘图版本为 comparison-figures-v1。完整 Sample_ID 图集按 50 个分图保存。

Worker 在终态提交之外尝试归档；归档失败不能改写训练成功状态，另存不可下载的错误标记。跨进程文件锁阻止重复生成，临时目录完成后在 SQLite 发布保护下替换；STOP/DELETE 阻止迟到发布并清理该批次归档。部分失败只保存可比较结果；不可比较时保存原因，不画排行。历史批次首次打开时由前端显式 POST 补建，无需重训。

默认归档不包含旧重复一致性图、私有模型文件或服务器路径；配置详情保留逐折选优信息。历史缺特征工程产物时不补造图像。
