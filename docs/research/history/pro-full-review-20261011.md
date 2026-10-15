> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

我沿用前面已经完成的附件检查、代码核对和敏感性计算，不重新展开完整重算。memory 与五条静态知识卡继续关闭；以下只做**整体复盘、问题定性和最小诊断规划**。

先给结论：

> 当前没有证据证明“LLM 方法无效”，但当前 benchmark 还有一个已经确认的评价协议风险：**Valid 阶段用于选参/选模型的决策规则，与 Test 阶段最终评价规则不一致，导致 Agent 收益估计被污染。**
>
> 在修正这个问题之前，不应该进入 memory 或更复杂 Agent 闭环，否则记忆会吸收错误反馈。

---

# 一、当前已经证实的问题（无需重新验证）

## 1. SVM 存在 Valid/Test decision rule 不一致（已确认）

这个是目前最明确的问题。

当前：

### Valid

`training.py::_evaluate_traditional_model`

使用：

```python
model.predict()
```

计算：

\[
BA_{valid}
\]

即：

```text
训练模型
↓
native class prediction
↓
Balanced Accuracy
```

---

### Test

`benchmark_driver.py`

使用：

```text
predict_proba()
↓
概率平均
↓
argmax
```

计算：

\[
BA_{test}
\]

即：

```text
训练模型
↓
probability decision
↓
class prediction
```

---

对于 SVM：

由于：

```python
probability=True
```

存在：

```text
predict()
≠
predict_proba().argmax()
```

这个不是理论问题，而是已经从结果中确认：

- 正式 SVM 结果：560 条
- 345 条存在类别预测差异

ToeSegmentation2：

native:

```
BA = 62.0283%
```

概率口径：

```
BA = 37.9717%
```

这是同一个 predictions.csv 的事后分析，不涉及重新训练。

因此：

## 已确认：

> SVM 的模型行为在选择阶段和评价阶段不是同一个分类器。

---

## 2. 当前 A0/A1 收益结论受到这个问题影响（已确认）

已有敏感性分析：

原口径：

```
A0-R0 = -0.5380 pp

A1-R1 = -0.1306 pp
```

替换 SVM decision rule 后：

```
A0-R0 = -0.2686 pp

A1-R1 = +0.3085 pp
```

注意：

这不是新的成绩。

它只是说明：

> 当前 Agent 收益方向对评价口径敏感。

因此目前不能写：

“Agent 没有效果”。

正确写法应该是：

> 在当前冻结评价协议下，Agent 与随机基线的差异需要在统一预测决策规则后重新确认。

---

# 二、整体复盘：当前 Agent 到底发生了什么？

## 1. 三套参数是谁产生的？

这个必须明确。

当前不是：

> LLM 自己生成完整超参。

实际流程：

```
数据输入
 ↓
backend 根据模型目录生成合法候选
 ↓
固定搜索空间 / recipe
 ↓
LLM 选择 recipe 或模型
 ↓
backend 执行对应配置
```

搜索策略代码中已经明确：

- 候选空间有限
- 参数域提前冻结
- Agent 不允许临时创造新的超参

例如：

`search_policy.py`

定义：

- PLS components
- SVM C
- RF 参数组合
- XGBoost 参数组合
- 深度学习 learning rate 等

这些候选由系统生成。

所以：

当前 LLM 的主要贡献不是：

> “发现一个新的 C=17.83”

而是：

> “在有限合法建模空间中选择一个更合适的方案。”

这个定位其实更适合小样本场景。

---

# 三、模型偏好目前说明什么？

根据已有 benchmark 分析：

当前 Agent 不是随机输出。

它存在明显模型选择偏好。

但这里要注意：

## 偏好 ≠ 优势

例如：

如果 LLM 经常选择：

- SVM
- PLS-DA
- Logistic Regression

这可能说明：

它理解：

- 高维
- 小样本
- 共线性

这些条件。

但也可能说明：

prompt 或候选排序有偏。

所以：

现在不能解释为：

> Agent 学到了正确模型适配规律。

只能说：

> LLM 的决策分布不是均匀随机。

---

# 四、候选哈希排序问题

这个是第二类需要关注的问题。

当前 recipe / candidate 有：

- recipe_id
- digest
- 固定排序

搜索计划中：

候选顺序冻结，并且 baseline candidate 放在前面。

风险：

如果 LLM 对候选顺序敏感：

例如：

```
candidate 1
candidate 2
candidate 3
```

和：

```
candidate 3
candidate 1
candidate 2
```

得到不同选择。

那么所谓：

“LLM 根据证据选择”

可能部分变成：

“LLM 根据列表位置选择”。

这不是大问题，但需要一个小诊断。

---

# 五、输入证据 / Prompt 当前应该怎么看？

当前 LLM 输入不是原始数据。

它看到的是：

```
Train evidence
+
candidate recipes
+
knowledge（当前关闭）
+
constraints
```

其中 Train evidence 包含：

- observation count
- sample group count
- feature count
- class distribution
- repeated measurement
- duplicate / variance 风险等。

所以当前论文真正的 claim 应该是：

不是：

> LLM 看懂光谱，然后选择模型。

而是：

> LLM 基于结构化训练证据和受约束候选空间进行建模决策。

这其实更符合 KBS/ESWA。

---

# 六、A1 与 A0 的收益应该重新拆解

现在不要只看：

```
Agent vs Random
```

因为 Agent 的收益来源至少有三个：

---

## 来源 1：模型选择能力

对应：

A0 vs R0

含义：

固定默认参数情况下：

> LLM 是否选择了更好的模型。

---

## 来源 2：搜索能力

对应：

A1 vs A0

含义：

同一个 LLM 选择下：

> 开启搜索是否增加收益。

---

## 来源 3：模型+搜索联合收益

对应：

A1 vs R0

含义：

完整 pipeline 是否比无智能 baseline 更好。

---

因此现在不要只看：

```
A1-R1
```

应该拆成：

```
模型选择收益：
A0-R0

搜索收益：
A1-A0

整体收益：
A1-R0
```

另外：

R1 是随机模型 + 搜索。

它实际上回答：

> 如果模型选择很差，但给足 HPO，能不能补回来？

这个非常重要。

---

# 七、固定 SVM 问题

你提到“固定 SVM”。

这里需要谨慎。

现在不能因为 SVM 出问题，就认为：

> SVM 不应该参与。

正确处理：

## 当前确认：

SVM 评价口径需要统一。

## 未确认：

SVM 是否真的导致 Agent 失败。

因为：

如果修正后：

- SVM 变强
- Agent 选择 SVM 更多

可能反而提高 Agent。


所以：

不要删除 SVM。

---

# 八、关于 predict 与概率平均：上一条需要修正说明

你指出得对。

上一条：

> “统一 predict；重复测量概率平均不冲突”

这个表述不够严谨。

这里必须重新说明。

---

## 两种合法方案

### 方案 A：全部 native prediction

流程：

Valid：

```
predict()
↓
sample-level majority vote
↓
BA
```

Test：

```
predict()
↓
sample-level majority vote
↓
BA
```

完全一致。

---

### 方案 B：全部 probability aggregation

流程：

Valid：

```
predict_proba()
↓
repeat-level probability average
↓
argmax
↓
sample prediction
↓
BA
```

Test：

```
predict_proba()
↓
repeat-level probability average
↓
argmax
↓
sample prediction
↓
BA
```

完全一致。

---

## 不允许：

Valid：

```
predict()
```

Test：

```
probability average + argmax
```

这是当前问题。

---

所以最终统一原则：

> 重复测量融合方式属于预测协议的一部分，必须同时应用于 Valid、Test、OOF。不能只修改 Test。

---

# 九、重复测量如何保持合理？

这里建议不要简单取消 probability。

因为重复测量天然有：

```
sample
 ├ spectrum1
 ├ spectrum2
 └ spectrum3
```

如果目标是 Sample 分类：

合理流程：

```
每条 spectrum 输出预测
          ↓
同 Sample_ID 聚合
          ↓
sample label
```

这个聚合可以是：

## native vote

或者：

## probability mean

但必须：

整个 pipeline 一致。

---

我目前倾向：

第一版 benchmark：

## 主结果：

native prediction + majority vote

原因：

- 简单
- 稳定
- 无额外 calibration

## 附加：

probability aggregation

作为 uncertainty / sensitivity analysis。

但这只是建议，不是已经批准的正式方案。

---

# 十、当前最小诊断（不要扩大训练）

按照优先级，我建议只做三项。

---

# Diagnostic 1（最高优先）：冻结 Prediction Protocol

目的：

解决当前最大问题。

不训练。

只确定：

二选一：

### Protocol-N

```
predict
+
majority vote
```

或者：

### Protocol-P

```
predict_proba
+
probability mean
+
argmax
```

然后重新生成：

- Valid metric
- Test metric
- A0
- A1
- R0
- R1


只是重新计算，不重新训练。

---

# Diagnostic 2：候选排序敏感性

目的：

确认 LLM 是不是被 candidate 顺序影响。

做法：

同一个输入：

版本 1：

```
recipe A
recipe B
recipe C
```

版本 2：

随机打乱：

```
recipe C
recipe A
recipe B
```

比较：

- 选择一致率
- 模型分布变化

不用训练。

---

# Diagnostic 3：A1/A0/R1/R0 收益重新分解

在 Protocol 冻结后：

重新报告：

```
A0-R0
A1-A0
A1-R0
A1-R1
```

并分：

- 全任务
- SVM 任务
- 非 SVM 任务

目的：

判断：

Agent 问题来自：

1. LLM 选择？
2. HPO？
3. 某个模型评价污染？

---

# 十一、当前不要做的事情

现在不建议：

❌ 开 memory

原因：

memory 会学习当前错误评价信号。

---

❌ 加五条知识卡

原因：

知识收益还没有 baseline。

---

❌ 换 GPT/Claude/更大 LM

原因：

当前主要问题不是模型能力，而是实验协议。

---

❌ 做更多 Agent 化

原因：

如果基础选择收益还没测清楚，增加自主循环只会增加变量。

---

# 十二、当前论文路线重新排序

我认为现在最合理：

## Phase 1（现在）

冻结：

- prediction protocol
- benchmark protocol
- A/R baseline

↓

## Phase 2

证明：

LLM 选择是否有价值

↓

## Phase 3

加入：

memory

↓

## Phase 4

加入：

feedback → replanning

↓

## Phase 5

外部 ML Agent 对比


---

最后总结一句：

**目前最大的风险不是“LLM 不行”，而是 benchmark 还没有完全成为一个可以证明 Agent 价值的测量系统。SVM 发现说明评价链路需要先冻结。修正后，如果 A1/R1 的方向仍稳定，再讨论模型偏好、记忆和 Agentic 闭环才有意义。现在优先做三个最小诊断：统一预测协议、候选排序敏感性、收益来源分解。不要扩大训练。**
