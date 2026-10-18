# 恢复分类模型与调整六种特征处理

## 本轮范围

- 用户要求恢复有价值的后端模型，排除 DSCARNet；随后明确增加 CNN-SE、ResNet。
- 经典界面共十一项：PLS-DA、sPLS-DA、PCA-LDA、Elastic Net、SVM、PCA-SVM、Random Forest、XGBoost、1D-CNN、1D CNN-SE、1D ResNet。
- 学生特征六选项为 BIN5/10/20、固定 PCA5/10/20，默认 BIN5。旧 BIN3/6/12、PCA15 可读取及兼容执行，但不显示在新表单。

## 训练策略

- 复用真实 CNNSE1DDocumentV2、ResNet1DDocumentV2，不用普通 CNN 替代。沿用已有网络定义，使用当前折 N/L 分档的显式 channels/kernels/pools/dropout，AdamW、CrossEntropyLoss（含二分类两输出）、调度、早停、最高 200 epochs；标准化只拟合训练折。
- 现有普通 CNN 不改结构。新增网络的类名写入 profile；模型权重、特征变换器、逐折参数和预测沿用 Manifest。
- sPLS-DA quick 组件数 1/3/5、keepX=min(50,特征数)；PCA-LDA quick 组件数 5/10/20；PCA-SVM quick PCA=min(10,训练条数,特征数)、线性核 C=0.1/1/10。候选数最多三，按验证 Balanced Accuracy 选择，再 Train+Valid 重训。完整模式复用已有各模型完整网格。
- CNN/CNN-SE/ResNet 及内置 PCA 的 PCA-LDA/PCA-SVM 不叠加外层 PCA，批次显式记不适用，单请求拒绝，不静默替换方案。
- 全选十一模型×六方案共 51 个有效组合、15 个不适用组合；24 个 Sample_ID 留一时共 1224 个外层任务。组合上限 100，避免旧 50 上限阻挡全选。
- 保留四种评估模式、分组隔离、逐折选参、pooled OOF 和原有 Worker 并行调度。

## 验证安排

专项测试覆盖模型真实网络类型、二分类输出、四种评估模式、特征选择、组合数量；完整 backend/tests、Node 和 compileall 回归。真实测试.csv 在隔离队列验证，全部通过 SwanLab 在线记录和云端回读，不覆盖用户历史结果。正常调度/早停的真实验证与短轮次合成单元测试分开记录；本轮不宣称新增网络优于旧模型。

## 2026-10-18 真实数据流程记录

- 数据：测试.csv，72 条测量、24 个 Sample_ID、4 类、7500 特征；原始 SHA-256、每 Run seed/config/代码 diff/环境/命令均保存在 outputs/restored-models-validation-20261018，不进入 Git。
- 使用隔离 SQLite、最多四个 Worker 与每进程五线程预算。五个恢复模型各跑 BIN5；另以 PLS-DA 分别跑六个新特征，共 11 次 holdout 训练，11 次成功且云端回读一致。两批结果 comparable；不是 24 折真实全组合验收，不作性能提升结论。
- CNN-SE 正常早停于 36 epochs，最佳 checkpoint 为 epoch 16；ResNet 正常早停于 53 epochs，最佳 checkpoint 为 epoch 33。两者均设置最多 200 epochs、既有早停与调度，没有人工缩短轮次。
- CNN-SE 云端：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/fof9cd8p
- ResNet 云端：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/d50nzdls
- 全部 11 条 URL、指标与核验状态：outputs/restored-models-validation-20261018/report.json。
- 全量回归 610 passed、9 skipped（198.25 秒）；前端相关 Node 测试 32 项通过。最终 Worker 版本更新为 run-artifact-manifest-v2-restored-models-v1 后另跑 Worker/目录/恢复模型专项 38 项通过，compileall、JS 语法和 diff 检查通过。
- Web/四个 Worker 在原队列无运行及排队任务时同步更新，继续使用原存储，历史结果未重训或删除。绘图版本 comparison-figures-v8；新旧特征名和模型名用于派生图像及导出。
