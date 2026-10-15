# Benchmark 选模提示与数据输入

2026-10-11 用户要求：修正评价口径；采用明确性能目标的客观提示；不加入额外模型架构/数据类型说明；先解释输入构成，再决定下一轮实验。当日仅完成实现与离线验证；2026-10-12 用户随后批准全量重跑，目前已进入真实正式评测。

## 实际消息构成

一次选模请求有 system 和 user 两条消息。system 是固定规则；user 是后端根据当前 Train 分区和已冻结模型目录填入的 JSON，并非整份原始数据。

system 保留合法动作和结构化输出约束，选模阶段增加已经诊断过的目标说明：

> Select the legal model expected to achieve the highest balanced accuracy on unseen samples. Candidate display order does not indicate priority. Compare candidates using the supplied task evidence.

含义：根据提供的任务证据，在合法模型中选择预期未见样本 Balanced Accuracy 最高者；排列顺序不代表优先级。没有加入特定模型推荐、架构解说、历史赢家或新统计特征。

实际生成位置是 `backend/app/benchmark_llm.py::BenchmarkLLMAdapter._build_request`，不依靠修改普通 LLM 适配器来间接改变最终请求。目标版标识为 `benchmark-decision-objective-v2`，原提示为 `benchmark-decision-v1`。最后的结束请求不会收到选模目标说明。

## user JSON 中有什么

| 内容 | 当前实际字段/含义 |
|---|---|
| 任务 | classification、balanced_accuracy、seed |
| Train 规模 | observation_count、sample_group_count、feature_count |
| 类别分布 | 各类别的记录数、样本组数、比例；类别为编码，不是类别语义说明 |
| 重复测量 | 重复样本组数量、每组次数最小/最大/中位数、重复次数是否一致 |
| 数据质量 | 零方差列数、重复列/冗余列数、重复向量及标签冲突数量 |
| 风险标记 | small_sample、high_dimension、imbalance 等后端生成的风险代码 |
| 合法模型 | model_id、固定预处理、class_balance、默认 model_params |
| 执行信息 | 原有架构/策略版本标识、候选数量说明、search_mode、max_trials |

没有输入各特征均值、标准差、分位数、相关矩阵、原始曲线或完整数据表，也没有显式 axis_kind、数据域或额外模型架构描述。训练阶段 zscore 需要计算均值与尺度，这不表示 LM 已收到它们。group_size_median 是每个样本测量次数的中位数，不是特征值的中位数。

当前合法模型集会受后端能力和数据适用性筛选，所以模型列表可能间接反映任务差异，但它不等于一个显式的数据类型字段。

原 benchmark 把传输兼容格式中的 8:1:1 写进 LM 输入，不能代表真实冻结划分。新目标版移除了这个占位 evaluation_config，不新增数据描述。Train 统计和风险仍严格来自 Train，Test 数据/成绩不进入选择请求。

真实 Coffee 轮次01的请求预览：23条Train记录、23个样本、286个特征，类别0有11个样本、类别1有12个样本；提供13个合法模型。它是根据已有冻结上下文离线生成的消息预览，没有发给模型。

## 参数由谁决定

LM 仍只返回 model_id。默认参数由冻结模型目录决定，最多三套配置由后端固定候选表生成，再按 Valid 选择；目标版没有引入 LM 提参数能力，也不扩展参数空间。memory 与静态知识继续关闭。

## 评价口径接入

新完整 benchmark 版本 `autoai-classification-benchmark-v5` 同时绑定目标提示与 `sample-id-mean-probability-v1`。它与已经完成的四任务隔离诊断 `autoai-diagnostic-v5` 是不同的协议标识。

新入口在传统和深度候选比较时使用样本组平均概率 BA，最终 Train/Valid/Test 指标和验证审计也采用同一组级入口。每候选仍按原训练流程拟合；深度候选内部的最低 Valid loss checkpoint 与早停规则不变。候选逐行 Valid 概率和组级结果保存到 candidate_validation/，方便后续重算。predictions.csv 保留逐行原生标签和概率作为审计，benchmark 的组级结果由共同概率聚合函数生成。

v3/v4 继续映射到旧提示和旧候选评价语义；不能静默重算旧成绩。本轮范围、预算与实际源码已在启动前冻结，执行根目录为 `work/benchmark-20261012-full-v5`；实际请求的语义记录已核对目标提示版本。未切换生产服务，普通训练入口未自动改为本次 benchmark 规则。
