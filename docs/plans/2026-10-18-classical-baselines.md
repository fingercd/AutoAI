# 补齐用户指定分类模型

## 范围与语义

用户最后给出老师清单，以该清单取代中途“保留十五项”方案。界面机器学习十项：PLS-DA、Shrinkage-LDA、Lasso、Ridge、Elastic Net、SVM、KNN、Random Forest、Extra Trees、XGBoost；下方保留 1D-CNN、CNN-SE、ResNet 三个深度模型，共十三项。sPLS-DA、PCA-LDA、PCA-SVM 仅隐藏，不删除后端或历史结果；后端能力目录共二十二项，DSCARNet 继续隐藏。

本平台仅分类：Lasso/Ridge 分别表示 L1/L2 正则化逻辑回归，不使用 sklearn Lasso/Ridge 回归器或对连续输出做四舍五入；Elastic Net 继续使用原有逻辑回归实现。三者输出类别概率，实际 penalty/solver/C 写入参数与归档。

## 训练与调度

- Shrinkage-LDA：真实 LinearDiscriminantAnalysis(solver=lsqr, shrinkage=auto)，自动 Ledoit-Wolf 收缩只拟合当前训练折，不是 PCA-LDA，不内置 PCA。自动收缩不额外搜索参数，quick/full 每个方案一个候选；完整模式仍在训练池内按五折评估。

- Lasso/Ridge：liblinear + 显式 OneVsRestClassifier，max_iter=5000，quick C=0.1/1/10；full C=0.01/0.1/1/10/100。不再使用初验的 multinomial SAGA；两者目标函数不同，不要求概率或分数数值一致，选型只看训练/验证收敛与耗时，不按外层 Test 分数切换。
- Elastic Net 保留 SAGA 和原 C/l1_ratio 候选；真实初验发现原 1000 迭代预算未收敛，调整 max_iter=10000、tol=1e-3，参数明确写入结果。此为数值求解停止精度/预算变化，必须用固定数据、相同划分验证；不把更宽松 tol 宣称为更精确或模型性能提升。
- KNN：真实 KNeighborsClassifier，quick k=1/3/5、uniform、Minkowski p=2；full k=1/3/5/7/9 × uniform/distance × p=1/2。k 不超过当前内层训练测量条数，不窥探验证或测试标签。
- Extra Trees：真实 ExtraTreesClassifier，bootstrap=False；quick 300 trees × depth=None/5/10，leaf=1、max_features=sqrt；full 再加入 leaf=3、max_features=0.3。复用现有按 Worker 预算限制线程，seed 固定。
- quick 最多三候选，验证 Balanced Accuracy 选优，Train+Valid 重训；full 沿用分组五折搜索。标准化/PCA 等仅在允许的训练折拟合；四种评估和 pooled OOF 规则不变。
- 原六种特征 BIN5/10/20、PCA5/10/20 保持。五个新增模型支持全部六种。界面全选十三模型共 69 有效组合、9 不适用；24 样品留一共 1656 外层任务，单批上限 100 可容纳。兼容 API 的全部十六个已接通版本模型仍可请求，合计 81 有效组合、15 不适用。
- Web/Worker 版本 run-artifact-manifest-v2-classic-baselines-v1；绘图 comparison-figures-v9。

## 验证要求

专项覆盖真实类型、概率、三种正则化区别、k 上限、线程预算、参数校验、四种划分、LOO 组隔离、组合计数和导出；整套 backend/tests、Node、compileall 与语法检查。真实测试.csv 在独立目录运行并全部使用 SwanLab 在线记录和云端回读，不覆盖用户产物。不据本轮功能验收宣称新增模型效果更好。

## 最终验证记录（2026-10-18）

- 全量 backend/tests：656 passed、9 skipped，321.68 秒；针对模型/目录/布局的中间专项 73 项通过。相关 Node 测试 32 项通过，compileall、内联及模块 JS 语法、git diff --check 通过。
- 数据为测试.csv，72 测量、24 独立 Sample_ID、4 类、7500 特征。保留 SHA-256、commit、未提交 diff、配置、seed、运行命令、环境及日志。隔离目录不进 Git。
- 最终有效真实验收包括六个新增/调整模型（Shrinkage-LDA、Lasso、Ridge、Elastic Net、KNN、Extra Trees）：每个模型分别完成六特征 holdout 和 BIN5 的 24 折 LOO，共 42 条 Run、180 外层折；不是全部 1656 组合外层任务的实跑。SwanLab 在线记录全部云端回读成功，最终这 42 条记录均无 ConvergenceWarning。
- 初验 SAGA 的 Lasso/Ridge 14 条 Run 均出现收敛告警；切换 liblinear/OvR 后的对应 14 条为零。相同数据、划分、seed、C 候选和每 Worker 五线程上限下，execute 区间时长之和为 932.639 秒与 164.216 秒。并发背景负载不同，且 OvR 改变多分类目标，因此这不是独占耗时基准，也不代表概率数值相同或准确率提升。
- Elastic Net 初验的 BIN5 holdout 使用原 1000 迭代预算触发未收敛；保持 SAGA/Elastic Net 及 C 候选不变，max_iter=10000、tol=1e-3 后，六 holdout 与一次 24 折 LOO 共七条 Run 均无收敛告警。更宽松 tol 不代表更高数值精度；未来其他数据/参数仍可能需要诊断。
- 汇总证据：outputs/teacher-models-final-validation-20261018/report.json。对应原始证据在 classical-baselines-validation-20261018、classical-baselines-liblinear-validation-20261018、shrinkage-lda-validation-20261018、elastic-numeric-validation-20261018 下。初验不作为最终模型结果冒充验收。
- Shrinkage-LDA LOO：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/aw1omti0
- Elastic Net LOO：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/hans21lf
- 本地 8877 Web/四个 Worker 在原队列无 queued/running 时同步更新，25 条历史 Run 保留。API 验证机器学习十项、深度三项均可用；实际浏览器全选十三模型、六特征、24 样品 CV 显示 1656，检查后恢复 PLS-DA + BIN5，未点开始训练。
