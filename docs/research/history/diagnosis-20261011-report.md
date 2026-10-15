> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

# AutoAI 有界诊断 v5：执行与结果报告

日期：2026-10-11。任务来自用户听完三步方案后明确要求独立对话执行的交办。第一轮 full-v4 的源码快照、科学配置、选择和正式成绩保持冻结。

**结论：原提示下，完全相同请求稳定，但只改变候选排列会改变模型选择，位置依赖获得了受控证据。明确性能目标显著降低了位置依赖，却主要收敛到 XGBoost 等常见模型；选择更稳定不能直接等同于更高泛化表现。补客观信息后仍有部分排列敏感性。本轮只在四个开发上下文筛查机制，不提供新的正式胜出结论。**

## 执行范围与协议

新协议：autoai-diagnostic-v5；源码摘要 `f9a53e6bb4efbbc18edc0957560aa95d3f6d3fa67ec856933ec3248c94140bd1`；本地服务器 Git tag `diagnosis-20261011-v5-f9a53e6bb4ef`。tag 只固定 Git 基准，必须结合未提交补丁和不可变源码快照还原。执行均在 Pan 根目录的 pan/agent；没有新建分支、worktree、提交或推送，也没有改生产发布。

采用共同入口 backend/app/probability_evaluation.py：Valid 与 Test 均按真实 Sample_ID 平均类别概率，再 argmax，按样本组计算 BA。精确并列按冻结类别顺序取第一类；候选 BA 差不超过 1e-12 时取原候选顺序最早者。单条样本也走相同入口。原生类别仅保留为敏感性信息。普通训练、历史 v4 的恢复和结果读取继续使用原行为；这次接入范围是隔离 v5 诊断矩阵，不宣称已修改通用训练服务。

传统模型仍只在 Train 拟合候选，按新 Valid BA 选参数，再在 Train+Valid 重训；深度模型的每个候选仍沿用最低 Valid loss 权重、原早停，候选间按新 Valid BA 选择。默认预算只取第一套参数；预算三只用旧后端固定候选表，未增加或由 LLM 生成参数。

| 开发上下文 | 轮次 | 合法模型 | Train 行/组 | Valid 行/组 | Test 行/组 | 轴类型 |
|---|---|---:|---:|---:|---:|---|
| Beef | 01 | 13 | 24/24 | 6/6 | 30/30 | time_series |
| Coffee | 01 | 13 | 23/23 | 5/5 | 28/28 | time_series |
| appendicitis | 01 | 7 | 68/68 | 16/16 | 22/22 | feature_index |
| glass2 | 01 | 7 | 104/104 | 26/26 | 33/33 | feature_index |

四上下文在查看新成绩前按属性和合法性登记，未扩大为 15 开发任务或 114 正式任务。memory、五条静态知识卡、检索和反馈行动均关闭。原始 Qwen3.5-9B、GPU4、temperature=0、top_p=1、max_tokens=1024、enable_thinking=false 均保持不变。资产四个权重分片重新核验 SHA256，匹配第一轮登记。

## 产物盘点与必要补训

旧 predictions.csv 实际只有 Test 逐行概率，Valid 审计只保存汇总指标。四上下文的 147 条缓存索引中实际权重均已清理。因此 114 个独立候选的 Valid 概率不可直接恢复，需重新训练候选；旧最终 Test 概率可在数据、划分、参数和训练代码相同且最终所选参数匹配时复用。

初次盘点误将 57 个旧已选候选计作 Valid 可恢复，覆盖检查在任何训练发生前阻止了复用。错误登记保存在 artifact_inventory-superseded.json，原因与修正见 inventory-correction.json。纠正后补训上限为 114+24=138 次，低于授权 168 次；没有因此改科学配置。

实际训练 **114 次**：{'traditional_candidate': 66, 'deep_candidate': 48}。全部完成；新最终传统重训 0 次，48 个传统预算结果复用同参数旧最终 Test 概率。深度模型的 Test 概率来自新候选的原早停权重。两档完整矩阵共 40 模型上下文、80 结果、114 候选评价。

训练仅使用空闲 GPU5，FP32、batch8、seed42、最多200 epochs及原早停，CPU 4线程。GPU1/2/3已有其它工作，不占用、不停止。新权重通过既有 WeightStore 写入预约；新额度扣除了旧冻结权重字节数，合计不超过50,000,000,000字节，未清理旧结果。

新口径改变候选赢家：**1/80**。具体模型/预算及原生与概率 BA 差异见 matrix_results.csv 与 candidate_validation_scores.csv。保留新旧选择，未回写正式成绩。

复训变化需单独解释：114候选中，只有Beef/TCN默认、Coffee/ResNet第三候选、Coffee/TCN第三候选的原生Valid BA与旧记录不同。只有Coffee/TCN预算三改变赢家；它的新原生与新概率BA相同，不能将这一赢家变化归因于聚合规则。现有记录无法完全分开GPU训练非确定性、CPU线程等数值运行差异的贡献；没有为追求旧分数额外训练或改种子。证据见 validation/refit-reproducibility.json。

## 顺序与提示诊断

40 个循环排列让每个合法模型各居首一次，同时改变 user 列表和 system schema 枚举。原版本额外每个上下文重复2次相同 wire request，合计48次；另两个版本各40次，复用原40次作对照。128次都有独立服务 response_id，并与服务请求日志逐项匹配；不是 Session 幂等缓存。

| 版本 | 轮换请求 | 选首项 | 选最早常见模型 | 与该版本原顺序选择一致率（四上下文等权） | 选择的开发 Test BA | 均匀随机合法模型 BA |
|---|---:|---:|---:|---:|---:|---:|
| 原提示 | 40 | 16/40 | 31/40 | 44.23% | 84.643% | 72.922% |
| 只明确性能目标 | 40 | 3/40 | 8/40 | 75.00% | 84.175% | 72.922% |
| 目标＋客观信息 | 40 | 4/40 | 8/40 | 67.03% | 78.332% | 72.922% |

常见模型集合预先采用 LR、SVM、RF、XGBoost，与既有复盘一致。一致率以每个版本原顺序（旋转0）的选择为参照，不混用不同版本。表中性能先在每个上下文对所有排列等权，再对四上下文等权；不是40个独立任务，也不是114正式任务的成绩。均匀随机列来自完整合法模型矩阵的精确平均，没有额外随机请求或训练。

相同输入的8次额外重复全部与首次选择一致；四上下文各3次请求的模型身份均一致。原提示下各上下文选择分布如下：

| 上下文 | 原提示分布 | 目标版分布 | 目标＋信息版分布 |
|---|---|---|---|
| Beef | xgboost:4, logistic_regression:5, svm:1, random_forest:2, pca_lda:1 | xgboost:13 | xgboost:12, cnn_transformer1d:1 |
| Coffee | logistic_regression:6, xgboost:1, random_forest:5, svm:1 | xgboost:13 | xgboost:8, cnn_transformer1d:3, cnn1d_se:1, resnet1d:1 |
| appendicitis | random_forest:3, logistic_regression:3, xgboost:1 | random_forest:5, xgboost:2 | xgboost:3, random_forest:4 |
| glass2 | random_forest:4, xgboost:1, svm:1, logistic_regression:1 | random_forest:2, xgboost:5 | xgboost:5, pca_lda:1, random_forest:1 |

目标版仅追加明确未见样本 BA 的选择目的、说明顺序不代表优先级，其余 wire 请求保持一致。信息版再将旧8:1:1描述替换为真实 Train/Valid 行和组规模、评价单位、可信轴类型，并补各模型结构定义；预算1仍只展示实际默认参数。没有历史赢家、Test信息、知识卡、memory或参数提议。

每条请求都保存实际 wire JSON、SHA256、原始HTTP响应、原始文本、修复操作、模型身份、provider response_id、开始/结束时间和token；核验了枚举与列表顺序一致，登记字段之外未发生输入变化。提示原文和结构定义见 approval/protocol.json。

## 成本与可靠性

128次请求总 token **239,759**（输入 233,665，输出 6,094），未知用量0；请求累计 263.884 秒，中位数 2.047 秒。Finalize0、网络重试0、语义拒绝0。原始JSON直接合法 1/128，127/128 经已存在的最小序列化修复接受；修复不改变 model_id。

开始时一次健康探测受 shell 代理影响失败，尚未触发生成；改用进程内直连loopback，无服务重启、模型替换或真实请求重试。

训练完成 114 次，累计候选 epochs 4915，fit累计用时 205.123 秒，矩阵阶段墙钟 208.722 秒。新权重 11,150,452 字节。各fit的开始/结束、epochs、dtype和权重摘要见 fit-events/ 与 weights 配额账本。本轮按当前用户授权继续跳过SwanLab，全部本地记录保留。

## 代码与真实验收

- 最终相关回归测试45项通过（119.78秒）；其中7项针对性测试覆盖：单条、重复测量、精确并列、覆盖不一致、组标签不一致、非有限/非法概率，以及80份已有真实Test预测与旧聚合语义完全一致。
- 使用已有 Meat 真实重复测量预测补充检查：60条观测归为30个真实样本组，每组2条；只读验收证据见 validation/real-repeated-group-check.json。
- 80份旧产物的预测和配置SHA256与Manifest核对；原有backend训练实现逐文件与冻结v4源码比较，没有已有文件差异。
- 四个真实开发上下文完成114候选训练及80结果验收；既有数据、划分、候选参数、seed和模型实现得到保留。
- 128次真实原始9B决策请求与服务日志匹配；格式可靠性与决策/位置可靠性分别记录。

三个提示版本四上下文等权开发Valid BA分别为73.139%、65.656%、64.309%；开发Test BA分别为84.643%、84.175%、78.332%。目标版稳定性提升而平均分数没有提升，信息版的本轮读数下降。不能仅因输入更详细就认定版本更好，也不把该信息版直接作为新默认。

## 原因判断与后续边界

1. **候选排列影响选择有受控支持。** 相同输入重复稳定，单独改变列表和枚举排列后发生模型身份切换，因此不能再仅归为服务随机波动。由于列表和枚举一起改变，本轮不能区分二者各自贡献。
2. **明确目标能降低位置依赖，但仍有模型先验集中。** Beef/Coffee所有排列均选XGBoost；这是一致性改善，不证明模型已利用各任务细节作最佳选择。补信息仍出现多个模型和残余排列依赖，效能以本报告开发矩阵读数为准。
3. **评价口径统一属于选择语义修复。** SVM原生标签与概率标签差异的敏感性保留，但不能根据已看过的Test换主口径。新口径和新赢家只属于v5，不能替换v4正式成绩。
4. **不自动实施LM参数提议或更强模型实验。** 值得先保留明确目标版本作为后续受控对照；下一轮若比较更强登记模型，应用同一四上下文、同一排列和输入，检验是否真正改善选择与稳定性。自由生成参数目前证据不足：本轮预算1只验证选模，未验证三套参数展示或LM生成参数。新科学配置须另获用户确认。

已看过第一轮完整正式Test。本轮是开发机制诊断，四上下文规模很小，排列之间高度相关；不计算把40个排列当独立任务的显著性，不将旧正式数据称为全新确认集，也不作论文效果胜出结论。

训练执行源码摘要为7c3ace3fa3c167a25dad26b4a59d65b9fd1a69b1a1c49d15f8033e85be4a0a42；最终交付源码另记录了将验收测试改为可移植临时夹具的更新。执行代码未改变，旧执行快照与tag保留。

## 文件导航

- approval/protocol.json、contexts.json、request_plan.json：协议、批准范围、四上下文、128请求与排列。
- approval/source-freeze.json、source-snapshots/、initial-tracked.patch、initial-status.txt、version-tag.json：源码、既有改动和版本证据。
- lm/、analysis/requests.csv、lm_summary.json：逐请求原始输入输出与位置分析。
- matrix/、fits.jsonl、fit-events/、matrix_results.csv、candidate_validation_scores.csv：必要训练与两档矩阵。
- analysis/final_summary.json、choice_performance.csv：精确成本、选择表现与原因判断数据。
- validation/、evaluation-tests.log：实现、真实样本分组、服务日志与产物完整性验收。

服务器目录：/users/fotile/AutoAI/Pan/work/diagnosis-20261011-v5。最终本地交付：D:/PythonProject/AutoAgent/outputs/diagnosis_20261011_v5/final_delivery。
