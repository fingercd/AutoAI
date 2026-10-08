# 2026-10-08 开放三个传统分类模型

## 目标与实现边界

按用户确认开放已有 sPLS-DA、PCA-LDA、PCA-SVM，新版网页和技能严格入口由六模型扩展到九模型。复用现有分类器，不增加依赖，不改变未声明实验版本的兼容 API 行为。本轮只做代码、文档和静态核对，未执行数据训练或运行测试，不能据此声称模型效果已经验收。

## 新版候选参数

记 m 为当前选参训练记录数，p 为变换后特征数，成分数上限为 min(m-1,p)。完整模式同时考虑 Train/Valid 审计模型的训练维度，不用测试数据计算上限。

| 模型 | 快速模式 | 完整模式 |
|---|---|---|
| sPLS-DA | (成分数, keepX)=(1,25)/(2,50)/(3,100) | 复用既有成分数 1/2/3/5/8/10 × keepX 25/50/100/300 网格 |
| PCA-LDA | 成分数 1/3/5 | 成分数 1/2/3/5/8/10/15/20/30/40/50 |
| PCA-SVM | 成分数 min(5,上限)，C=0.1/1/10 | 同 PCA-LDA 的成分数 × C=0.01/0.1/1/10/100 |

超限成分候选过滤；keepX 截到 p，完整网格按截断后值去重。快速 sPLS 候选成分数不同，截断 keepX 后仍是不同参数组合。PCA-SVM 固定线性核和现有概率估计方式；它不是 RBF-SVM 的别名。候选为空或数值拟合失败时保留明确原因，不用测试成绩补选。

快速模式用当前验证集 Balanced Accuracy 选优，同分保留候选顺序，最多三次候选拟合加一次 Train+Valid 重训。完整模式继续用分组五折平均 Balanced Accuracy 选优，每类搜索样品数下限保持五个独立 Sample_ID。标准化、模型 PCA 和稀疏变量选择均在每次允许的训练折内拟合。

## 特征与接口

- sPLS-DA 支持 full、bin_5/10/20、pca_90/95/99，合计七方案。
- PCA-LDA、PCA-SVM 已内置 PCA，仅支持 full 和三种 Binning，合计四方案；外层 PCA 标记 not_applicable，未选适用方案在快速模式标记 not_run。
- `/api/models.models[*].supported_feature_schemes` 和 `training_scheme.supported_feature_schemes` 共用同一函数生成；前者用于页面交集和数量显示，后者用于请求组装校验。原 cnn_supported、traditional_scheme_count、cnn_scheme_count 继续保留，通用数量表示原七方案族与 CNN 四方案族，具体模型数量由能力数组获取。
- 仍使用 experiment_version=word-0904，结果和 Manifest 结构版本不变；Worker 能力升级为 run-artifact-manifest-v2-quick-training-v2，阻止新旧执行器混用。无需数据库迁移或历史结果重训。
- 比较页、图集和预测 Excel 同步九模型展示名称。四种评估口径、单 Run/Batch 结果、刷新恢复和停止流程沿用现有实现。

## 后续运行验收场景

用户明确要求测试时，覆盖三模型 quick/full、四种划分、单模型与混合批次、方案交集限制、小维度参数过滤、选参隔离 Test、结果/图集/Excel 导出，以及旧模型与历史结果兼容。真实数据对比使用固定数据和 seed 并记录实际参数与代码版本；本轮未提供此项性能证据。

## 2026-10-08 后续测试记录

用户随后明确要求用根目录 `测试.csv` 测试新增模型。固定数据 SHA-256 为 `c0f4233b7d77a0cb221c86e8509e580d4feac994f2edb12341399a204c8031af`：72 条测量、24 个独立 Sample_ID、4 类、7500 个特征，每个样品三次测量。使用 quick、全特征、zscore、seed=42；实际 Train/Valid/Test 为 16/4/4 个样品，三模型划分完全一致，各拟合四次。

| 模型 | 验证选定参数 | Test Balanced Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| sPLS-DA | 成分数 3，keepX=100 | 0.833333 | 0.803571 |
| PCA-LDA | PCA 成分数 5 | 0.833333 | 0.803571 |
| PCA-SVM | PCA 成分数 5，线性核，C=0.1 | 0.750000 | 0.651786 |

结果投影、分区隔离、42 个固定文件的 SHA-256/大小、两表各 72 行的预测 Excel 和 28 个 PNG/SVG 导出检查通过。status.json 按可变状态文件契约豁免哈希校验。真实测试仅四个独立样品，不能据此宣称某模型普遍最好；完整比较和四种评估方式的覆盖来自合成数据回归，不是该真实数据的完整实验。

真实产物为本聊天 outputs/new-models-real-20261008-140338；配置、代码差异、数据版本、环境和命令均已保存。三条 SwanLab 指标与配置已回读核验：[sPLS-DA](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/8zywelzd)、[PCA-LDA](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/4aw0auu3)、[PCA-SVM](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/15k1v39b)。

首次整套回归仅旧六模型目录断言失败，更新为九模型清单后重跑 `backend/tests`：595 passed、9 skipped，83.54 秒；compileall 与 diff --check 通过。回归记录 [SwanLab](https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/z6yro5b1) 已上传并回读。跳过项与现有环境、可选路径有关，原因保留在 pytest 日志和 JUnit 中；警告按原样保留，不把它们当作新模型失败。
