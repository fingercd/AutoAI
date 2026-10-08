# 实验策略

模型由 Agent 推荐，来源是实时 `/api/models`。严格入口支持当前 Word 0904 九类公开模型：PLS-DA、sPLS-DA、PCA-LDA、Elastic Net、SVM、PCA-SVM、Random Forest、XGBoost、1D-CNN。其余隐藏模型的后端兼容能力继续保留，但不进入本技能严格提交路径。

每个模型的 `supported_feature_schemes` 列出新版可用方案。多模型快速训练只能选择交集中的方案；PCA-LDA、PCA-SVM 已内置折内 PCA，不能叠加外层 PCA，只支持 full 和 bin_5/10/20；CNN 同样仅支持这四种方案。其余六类传统模型支持全部七种方案。模型内部 PCA 不在数据准备阶段执行。

推荐依据用真实数据统计说明：例如小样品、高维曲线可建议传统模型做基线；CNN 要说明数据规模与计算成本。数据统计不足以证明某模型最佳；不设置一个总被选中的固定模型，也不为了推荐训练额外模型。

默认明确使用 `experiment_version=word-0904`、`training_profile=quick`、`feature_scheme=full`、`normalization=zscore`、seed=42。快速传统模型最多三候选，验证集选优后 Train+Valid 重训，普通流程通常四次拟合。CNN 使用后端调度、早停和最多 200 epochs。

完整比较主动启用 `training_profile=full`，feature_scheme 保持 full 作为请求占位，后端分别运行模型适用方案：六类传统模型七方案，PCA-LDA、PCA-SVM 和 CNN 四方案。传统每个搜索池每类至少五个独立 Sample_ID。未知配置和新版候选策略不支持的手工覆盖会被严格入口拒绝。

sPLS-DA 快速候选为 (成分数, keepX)=(1,25)/(2,50)/(3,100)，keepX 不超过当前特征数；PCA-LDA 成分数为 1/3/5；PCA-SVM 使用不超过 5 的可行成分数和 C=0.1/1/10，固定线性核。成分数不超过当前训练记录数减一或特征数，超限候选过滤。参数详情见项目 docs/plans/2026-10-08-public-traditional-models.md，不向严格入口提交手工参数覆盖。

|模式|默认比例|主指标|
|---|---|---|
|stratified_holdout|8:1:1|direct Test|
|leave_one_sample_id_cv|每折其余样品 8:2|pooled OOF|
|external_test_holdout|主数据 8:2|direct external Test|
|leave_one_sample_id_cv_with_external_test|主数据留一，再全主数据重训|direct external Test；OOF 审计|

自定义比例为整数，合计 10；两段模式 split_test=0。所有划分按 Sample_ID 整组进行，显示独立样品数和测量条数。主/外部 Test 必须逐点同轴，标签兼容且 Sample_ID 不交叉。

预检通过表示当前数据与配置具备执行条件，不保证所有候选数值拟合成功，也不预测效果或精确运行时长。完整训练中部分方案可以失败并保留原因。Test 不参与候选、方案、checkpoint 选择，预处理中的标准化/PCA 不提前在全数据拟合。

新实验使用新任务目录。用户根据已有 Test 结果调整方案时，记录该 Test 已经参与对话选择；不能再把反复挑出的 Test 成绩称为未使用过的独立验证。
