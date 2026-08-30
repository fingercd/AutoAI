# Small-sample Benchmark v1

本清单冻结首轮 Agent/后端工程评测数据，不把模型或 AutoML 平台误称为 Benchmark。

## 任务

- `molecular_biology_promoters`：正式主任务，106 x 57，类别 53/53。PMLB 将核苷酸类别编码为 0/1/2/3，只能用于端到端鲁棒性测试，不能据此宣称完成最优类别特征建模。
- `haberman`：正式辅助任务，306 x 3，类别 225/81。用于验证不平衡证据是否会改变 class-balance、模型选择或停止行为；数据只作算法基准，不作临床结论。
- `parity5`：32 x 5，只用于下载、适配、训练、反馈和结果 Manifest smoke，不进入科学平均排名。

数据文件不进入 Git。版本、SHA-256、许可与规模以同目录 JSON Manifest 为准。

## 两条不可混用的评测口径

1. AutoAI 原生闭环固定 `stratified_holdout` 和 seed，主报 macro-F1，辅报 balanced accuracy、失败率、wall-clock、LLM/API 调用和 model-fit 数。
2. 如需与 TabMini 发布结果比较，另跑固定 3-fold ROC-AUC runner；不得把 AutoAI 的 8:1:1 holdout 分数称为复现 TabMini。

PMLB 没有真实 `Sample_ID`。适配时只能给每行稳定 row ID，不能把这些任务直接用于 `leave_one_sample_id_cv`。Raman/HPLC 领域任务必须保留真实公共轴与 Sample_ID 分组，单列 `domain_benchmark`，不与 PMLB 分数求平均。

首轮同预算平台对照为 AutoGluon Tabular 与 FLAML。MLE-bench Lite 需要 Kaggle 凭据且约 158 GB，本版不纳入轻量门禁。
