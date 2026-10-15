> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

我先把这项只读核对纳入当前复盘。按照你的要求：

- **不重新训练**
- **不打开 memory**
- **不使用五条静态知识卡**
- **不扩大训练规模**
- 只讨论诊断、协议和后续统一方案

我认为这次发现是一个**真实的评测协议问题**，但需要准确命名：

> 它不是“训练过程错误”，而是**模型选择阶段与最终评价阶段的预测决策规则不一致（decision rule mismatch）**。

而且这个问题比单纯的 SVM bug 更重要，因为它直接影响 Agent benchmark 的可信性。

---

# 一、当前是否构成“选参 / 最终评价口径不一致”？

## 结论：是，但需要拆成两个层面

你现在的链路实际上存在：

```
训练
 ↓
Valid BA 计算
 ↓
选择模型 / 选择超参
 ↓
Test预测
 ↓
Test BA计算
```

理论要求：

> Valid 阶段用于选择的模型行为，应该和 Test 阶段评价时的模型行为一致。

但是现在：

## Valid：

`training.py::_evaluate_traditional_model`

使用：

```python
model.predict()
```

得到：

```text
class label
```

计算：

```text
Balanced Accuracy
```

即：

\[
BA_{valid}=BA(y_{valid}, predict(X_{valid}))
\]

---

## Test：

benchmark_driver:

```python
probability average
        ↓
argmax
        ↓
predicted label
```

也就是：

\[
BA_{test}=BA(y_{test}, argmax(\frac{1}{K}\sum p_k))
\]

对于 SVM：

因为：

```python
probability=True
```

所以：

- `predict()`
- `predict_proba().argmax()`

理论上可能不同。

你现在发现：

> 560 条正式 SVM 结果中，345 条 pred_label 与概率 argmax 不同。

这说明不是偶然数值误差，而是两个决策规则产生了不同分类器。

---

# 二、为什么 SVM 会出现这种情况？

这是 SVM 一个比较典型的问题。

对于 sklearn SVC：

```python
predict()
```

使用的是：

\[
sign(f(x))
\]

也就是 decision function。

而：

```python
predict_proba()
```

不是简单把 decision_function 转概率。

它经过：

- Platt scaling
- probability calibration

然后再：

\[
argmax(P(y|x))
\]


所以：

```
decision boundary
        ≠
probability calibrated boundary
```

是可能发生的。

尤其：

- 小样本
- 类别不平衡
- calibration 样本不足

时更加明显。

你现在 ToeSegmentation2：

native:

```
BA=62.0283%
```

probability argmax:

```
BA=37.9717%
```

刚好互补性质非常明显。

这不是模型重新训练导致，而是**同一个训练结果下，两个决策规则产生两个分类器。**

---

# 三、这个问题影响论文哪里？

影响最大的是：

## 1. 影响 SVM 的绝对性能

这个显然。

现在 SVM 的 test score 实际评价的是：

> calibrated probability decision rule

但是 Valid 选择的是：

> native predict decision rule

所以：

你选择的是 A：

```
SVM_native_valid
```

最后报告的是：

```
SVM_probability_test
```

这两个不是同一个东西。

---

## 2. 更严重：影响 Agent vs Random baseline

因为你的 A0/A1 和 R0/R1 都包含 SVM。

现在：

敏感性分析：

```
A0-R0

-0.5380
↓
-0.2686


A1-R1

-0.1306
↓
+0.3085
```

这说明：

Agent 结论对评价口径敏感。

这里不能说：

> Agent 已经提升。

也不能说：

> 原结果完全错误。

正确表述应该是：

> 当前 benchmark 存在 classifier decision rule inconsistency，对包含 SVM 的比较结果造成影响，需要统一评价协议后重新锁定正式结果。


这个属于 benchmark protocol 修正，不属于模型优化。

---

# 四、后续应该怎么统一？

我建议不要简单选择“全部 probability”或者“全部 predict”。

应该先定义论文里的**预测协议（prediction protocol）**。

有三个方案。

---

# 方案 A（我最推荐）：统一使用 predict() 作为最终分类规则

即：

所有模型：

```
fit
 ↓
predict
 ↓
BA/F1
```

包括：

- Valid
- Test
- OOF
- CV


优点：

1. 最符合当前模型训练选择逻辑

2. 不引入额外 calibration

3. 对小样本更加稳定


尤其你的论文重点不是概率预测，而是：

> 小样本自动建模选择


那么分类标签正确率比概率质量更重要。

---

缺点：

不能展示可靠概率。

但是可以补：

- calibration error
- reliability diagram

作为附加，不参与主 benchmark。

---

# 方案 B：全部改成 probability argmax

也可以。

但是需要：

Valid：

也改：

```python
predict_proba()
argmax
```

然后：

Test：

保持现在。

这样一致。

---

问题：

小样本下 calibration 本身就是一个额外学习过程。

对于 SVM：

probability=True

内部会额外做 CV calibration。

意味着：

你的搜索成本增加。

而且：

对于极小样本：

概率校准可能增加方差。

---

# 方案 C：根据模型选择不同规则

不推荐。

例如：

SVM predict

RF probability

CNN probability


这会导致：

benchmark 的比较对象不是同一种 decision rule。

论文审稿人容易问：

> 为什么不同模型评价标准不同？

---

所以我的建议：

## 主结果：

统一 predict()

## 附加实验：

概率输出作为 uncertainty analysis。

---

# 五、重复测量如何保持合理？

这里更重要。

你的数据有：

- 多光谱
- 同一样本重复测量
- Sample_ID


所以必须区分：

## 训练评价单位

和：

## 输出预测单位


---

现在你的测试可能存在：

```
Sample A
 ├── spectrum1
 ├── spectrum2
 └── spectrum3
```

如果直接：

```
每条 spectrum predict
↓
计算 BA
```

实际上评价的是：

> 光谱级分类

不是：

> 样本级分类


---

如果论文目标是：

“根据光谱判断样本类别”

我更建议：

## sample-level aggregation


例如：

测试阶段：

```
每条光谱预测
        ↓
同 Sample_ID 聚合
        ↓
sample prediction
        ↓
BA
```


聚合方式提前固定：

例如：

### label majority vote

或者：

### probability mean + argmax


注意：

这里 probability mean 是合理的。

因为它发生在：

> 同一个 Sample_ID 内多个重复测量融合

不是：

> 单个样本内部改变分类器


这是两个不同问题。


---

所以最终建议：

### 模型决策：

统一：

```
model.predict()
```


### 重复测量融合：

允许：

```
sample-level aggregation
```

例如：

```
repeat spectra
        ↓
probability average
        ↓
argmax
```


这不会和 SVM 当前问题冲突。

---

# 六、下一步不用扩大训练，只做三个诊断

按照你的要求，我建议不要马上重跑全部 benchmark。

先做：

---

## Diagnostic 1：统计影响范围

不用训练。

从已有 predictions.csv：

统计：

每个模型：

```
native prediction
vs
evaluation prediction
```

差异：

- SVM
- RF
- XGBoost
- CNN

确认是不是 SVM 特有。

预计：

SVM 最大。

---

## Diagnostic 2：冻结新的评价协议

写一个正式协议：

例如：

```
Prediction Protocol v1

All classifiers:
    inference = model.predict()

Sample aggregation:
    same Sample_ID predictions are aggregated by majority vote

Primary metrics:
    Balanced Accuracy
    Macro-F1

Probability outputs:
    only used for uncertainty analysis
```

然后所有 benchmark 结果重新生成。

---

## Diagnostic 3：检查 Agent 结论稳定性

只看已有结果重新计算：

比较：

```
old protocol

new protocol
```

看：

- A0-R0
- A1-R1

变化多少。


如果：

```
方向一致
```

说明只是数值修正。

如果：

```
方向改变
```

说明之前 benchmark 结论不稳定，需要重新解释。


---

# 七、对论文影响判断

我的判断：

## 好消息：

这个问题现在发现，比投稿后被审稿人发现好很多。

因为它不是：

- 数据泄漏
- 测试集污染
- 训练错误

而是：

> evaluation protocol inconsistency


机器学习论文中很常见。

---

## 但是必须修

因为你的论文核心是：

> Agent 自动选择模型

那么模型比较协议必须非常严格。

否则审稿人可能问：

> 你证明的是 Agent 选择能力，还是某个预测后处理规则差异？


这个问题必须提前消掉。

---

# 八、对当前路线的小调整

我建议当前路线变成：

## Step 0（现在）
✅ 修正评价协议

↓

## Step 1
benchmark v1

↓

## Step 2
A0/A1/R0/R1/HPO

↓

## Step 3
memory

↓

## Step 4
feedback replanning

---

不要现在进入 memory。

因为 memory 学到的是：

> 过去实验结果

如果 benchmark 评价规则还没冻结：

memory 会学习错误信号。

这点非常重要。

---

最后给一个简短判断：

**这次发现不是推翻当前方向，而是说明 benchmark protocol 需要先冻结。SVM 的 predict 与 probability argmax 不一致确实构成选参阶段和最终评价阶段的不一致。后续最稳妥方案是统一模型决策规则（推荐 predict），把概率平均限制在重复测量融合层，而不是单模型分类决策层。先完成这个协议修正，再进入 Agent 选择、HPO、memory，否则后面的所有收益分析都会受到污染。**
