# 深度模型固定全特征

## 用户确认范围

深度学习不再执行外部 BIN 或 PCA。三个公开网络 CNN/CNN-SE/ResNet 均使用全部原始特征列；保留训练折拟合的标准化与网络自身卷积/池化，不改变网络定义、优化器、调度、早停和 epoch 上限。传统机器学习仍执行选择的六种特征处理。

## 实现约束

- 能力目录 forced_feature_schemes 公布三个深度模型强制 full，前端不维护另一套深度模型名单。
- 新 word-0904 单 Run、显式及隐式特征批次、直接训练入口均归一化深度 feature_scheme=full。一个深度模型不因多选特征产生重复 Run；仅深度模型时不要求选择机器学习特征。
- 深度 full 比较模式也只训练一次，不再暗中运行旧七方案。历史完成结果仍按其原记录的 BIN/PCA 展示，不重训、不重标记。
- Batch 保存 requested_feature_schemes 与实际 feature_schemes；实际列表包含深度 full。新增 deep_full_features 标记仅用于新批次的公式/结果说明，历史投影保持原语义。
- 全选十三模型、六方案为 10×6+3=63 个 Run；24 样品留一共 1512 外层任务。发现 repository 原硬上限仍为 50，与路由的 100 不一致，统一为 100，并通过真实队列创建 63 条记录的单元测试防止回归。
- 结果矩阵保留真正的全特征行。深度结果只出现在 full 分组，明确命名“全特征（深度学习）”；机器学习 BIN/PCA 行不补造深度分数，不填零。预测 Excel 和归档继续按实际模型×特征 Run 导出。
- Worker 版本 run-artifact-manifest-v2-deep-full-features-v1；绘图 comparison-figures-v10。Web/Worker 在无 queued/running 的窗口同步更新，不中断活跃训练。

## 验证范围

单元覆盖三个深度模型×六特征×quick/full 的归一化、机器/深度混合、只选 PCA 的深度批次、63 条队列创建、1512 计数、混合结果分组和缺失格原因、旧结果保留；训练覆盖四种评估模式和 full profile。短轮次合成训练只作契约检查。

真实测试.csv 使用隔离队列，正常最多 200 epochs/早停，在线 SwanLab 与云端回读；混合批次包含 PLS-DA 六特征和三个全特征深度模型，另验证全特征 CNN 的 24 折 LOO。不按 Test 成绩改变用户指定的全特征策略。运行时交付须完整 backend/tests、相关 Node、compileall、JS 语法与差异检查通过。

## 2026-10-19 验证完成

- 新边界/队列/投影专项 41 项通过，相关训练专项 69 项通过；完整 backend/tests 最终 696 passed、9 skipped（236.99 秒）。相关 Node 测试 32 项通过，compileall、内联/模块 JS 语法、git diff --check 通过。首轮仅一项旧字符串断言仍要求“模型×特征”的旧理论组合文案，按新 ML×特征+DL×full 公式更新后重跑全量通过。
- 真实测试.csv：72 条测量、24 个独立 Sample_ID、4 类、7500 特征。隔离目录 outputs/deep-full-validation-20261019，保留数据 hash、commit、dirty diff、配置、seed、命令、环境与日志，不进入 Git。
- 混合 holdout 为 PLS-DA 六方案 + CNN/CNN-SE/ResNet 各一条 full，9 Run；另将 BIN10/PCA5 作为请求特征，CNN 只建立一条 full LOO Run，完成 24 折。总计 10 Run、33 外层折，全部成功、SwanLab 全部云端回读一致。
- 四条深度 Run 的所有折均为 full，input_features=output_features=7500、pca_components=null、profile.L=7500。三条 holdout 的实际 epochs 为 36/68/50；LOO 的 history 共 859 行，配置均为最多 200 epochs + 正常调度/早停，未用短轮次替代真实验证。
- 两批比较均 comparable、归档 ready，深度结果只在 full 分组，命名“全特征（深度学习）”；六个 ML 分组不复制深度结果。详细验证见该目录 report.json、verification.json 和逐批 comparison.json。
- CNN 全特征 LOO：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/0t5fufvs
- CNN-SE 全特征：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/3mx19epm
- ResNet 全特征：https://swanlab.cn/@ryfrcd693/AutoAI-Validation/runs/e3b0r6oz
- 原 8877 服务检查时已停止，原数据库 25 条 Run 均 succeeded、无 queued/running；使用原存储启动新 Web + 4 个匹配 Worker，历史产物保留。浏览器实际全选显示 1512，三个深度模型且机器学习特征全不选时显示 72；检查后回到 PLS-DA + BIN5，未开始新训练。
- ui-all-models.png、ui-deep-only.png 为实际浏览器截图。任务数量减少不等于固定倍数加速，全特征单次计算可能比 BIN 输入更重；本轮不宣称预测精度改善或固定速度提升。
