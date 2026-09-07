# AutoAI 超长 Goal：原生 Agent 多方向实践与范式收敛计划

> 计划日期：2026-07-24\
> 执行环境：从 `D:\PythonProject\AutoAI` 当前 working tree 创建的独立 Codex worktree\
> 主执行模型：`gpt-5.6-sol`\
> 推理强度：`ultra`\
> 工作方式：在新 AutoAI 任务中显式创建长期 Goal，持续实现、实验、证伪和收敛\
> 原始项目保护：禁止在 `D:\PythonProject\AutoAI` 原目录进行实验性代码修改；所有探索只能发生在隔离 worktree

---

## 1. 一句话目标

在不预设“多 Agent 一定有效”的前提下，围绕 AutoAI 至少真实实现并评测 12 个 Agent/科学 AutoML 方向，通过统一 benchmark、强基线、代码、测试、运行轨迹、失败证据和成本数据，最终收敛出一个可复现、可证伪、能明确说明适用边界的原生 Agent 范式。

---

## 2. 完成定义

新任务只有同时满足以下条件才能把 Goal 标记为 complete：

1. 已从当前 working tree 建立独立实验环境，且未修改原始 `D:\PythonProject\AutoAI`。
2. 已读取项目的 `AGENTS.md`、`CONTEXT.md`、`README.md`、相关契约和原始深度调研报告。
3. 当前 AutoAI 固定工作流已完成基线测试和至少一次可运行闭环；如果本地数据或环境阻塞，已留下可复现的失败证据并使用合法替代 fixture。
4. 建成统一 research harness，而不是每个方向使用不可比较的一次性脚本。
5. 至少 12 个必做方向均有：
   - 预注册假设；
   - 可运行代码或最小原型；
   - 针对性测试；
   - 至少一次实际实验；
   - 原始结果；
   - 与至少一个强基线比较；
   - `promote / hold / kill` 结论；
   - 局限和可信度等级。
6. 至少 3 个一维科学信号域进入评测；至少覆盖分类和回归。缺乏公开真实数据时，可用合成数据验证机制，但不得把合成结果冒充真实外部有效性。
7. 建成 clean/invalid 配对的反事实有效性 benchmark，并做到 source-disjoint 或 dataset-disjoint 隔离。
8. 统一报告：
   - Valid Completion Rate；
   - Unsafe Acceptance Rate；
   - Repair Success Rate；
   - Correct Abstention / False Rejection；
   - clean-case utility；
   - cost-to-first-valid-run；
   - replay/证书结果。
9. Rule-only、固定 AutoAI、单 Agent、无 verifier、无 replan 和完整范式均进入基线。
10. 至少完成一轮组合故障测试，而不只处理单字段错误。
11. 最终范式由证据选择，不因计划预先推荐某方向而强行得出正面结论。
12. 形成最终报告、实验矩阵、方向排名、失败档案、可信度说明和后续论文路线。

不能因为某个模型 API 不可用、某个方向失败或结果为负面就提前完成 Goal。负面结果是有效产出；应记录、换用可复现替代方案并继续其他方向。只有整个目标完成，或按照 Goal 工具规则在多轮确认后真正不可推进，才允许结束。

---

## 3. 核心研究原则

### 3.1 实践优先

每个方向必须落到以下证据链：

```text
假设
→ 可运行实现
→ 测试
→ 固定输入和预算
→ 原始运行记录
→ 指标与置信区间
→ 与基线比较
→ 失败分析
→ 继续/暂停/淘汰
```

以下内容不算“完成一个方向”：

- 只写设计文档；
- 只做文献综述；
- 只写 prompt；
- 只添加一个类或接口但不运行；
- 只展示一个成功案例；
- 只报告模型分数而没有有效性指标；
- 用更多 token 或更多尝试与低预算基线比较；
- 用合成数据声称真实科研泛化；
- 由同一个 LLM 自己判断自己是否科学正确。

### 3.2 先证伪，再优化

每个方向在实现前必须写明：

- 它为什么可能有效；
- 最强替代解释是什么；
- 什么结果会否定它；
- 最低可接受效应；
- 预算和停止条件。

### 3.3 有效性优先

统一字典序目标：

\[
\max_{\pi}
\left(
\mathrm{ValidityPass},
\mathrm{ScientificUtility},
-\mathrm{Cost}
\right)
\]

未通过 blocking validity rules 的 F1、Accuracy、R² 不得计为成功。

### 3.4 确定性验证

LLM 可以规划、解释和选择合法 repair，但不能：

- 修改 verifier；
-读取锁定测试标签；
-签发最终 Validity Certificate；
-依据自然语言自评代替 deterministic oracle；
-把系统外 test peeking 表述成已被证明不存在。

### 3.5 公平预算

比较 Agent 架构时至少固定：

- 模型和版本；
- 可用工具；
- 数据可见性；
- test 权限；
- 总 token 或强模型调用上限；
- 工具调用上限；
- wall time / compute budget；
- 随机种子；
- retry 上限。

同时报告等预算成功率和等成功率成本。

---

## 4. 隔离、目录和产物约定

### 4.1 隔离原则

- 原项目：`D:\PythonProject\AutoAI`
- 实验项目：由 Codex 为新任务创建的独立 worktree
- 新任务可以在 worktree 中大幅修改代码、创建分支和实验文件。
- 禁止回写、移动、删除或清理原项目中的文件。
- 禁止读取或输出 `.env`、`auth.json`、token、密钥、认证缓存和 `.sandbox-secrets`。
- 不把数据、模型、缓存、二进制和压缩包提交进 Git。
- 不自动 push 或创建 PR，除非用户之后明确要求。

### 4.2 建议目录

```text
research_lab/
├── README.md
├── config/
│   ├── directions.yaml
│   ├── budgets.yaml
│   ├── benchmark_splits.yaml
│   └── model_providers.example.yaml
├── core/
│   ├── contracts.py
│   ├── evidence.py
│   ├── tool_registry.py
│   ├── policies.py
│   ├── verifier.py
│   ├── controller.py
│   ├── repair.py
│   ├── certificate.py
│   ├── replay.py
│   └── metrics.py
├── adapters/
│   ├── autoai.py
│   ├── raman.py
│   ├── infrared_nir.py
│   ├── chromatography.py
│   └── generic_1d.py
├── benchmarks/
│   ├── episode_schema.py
│   ├── fixtures/
│   ├── mutations/
│   ├── oracle.py
│   ├── hidden_runner.py
│   └── source_split.py
├── directions/
│   ├── d01_contract_verifier/
│   ├── d02_rule_repair/
│   ├── d03_agent_loop/
│   ├── d04_preprocess_search/
│   ├── d05_model_search/
│   ├── d06_domain_transfer/
│   ├── d07_ood_abstention/
│   ├── d08_counterfactual_benchmark/
│   ├── d09_memory/
│   ├── d10_adaptive_routing/
│   ├── d11_multi_agent/
│   └── d12_evidence_rag/
├── stretch/
│   ├── d13_mcp/
│   ├── d14_tree_search/
│   └── d15_hybrid_bo/
├── runners/
│   ├── run_baselines.py
│   ├── run_direction.py
│   ├── run_matrix.py
│   └── summarize.py
├── tests/
├── journal/
│   ├── decision_ledger.md
│   ├── failures.md
│   └── checkpoints/
└── schemas/
```

临时数据、日志和运行缓存放 `work/agent_goal_exploration/`。\
最终交付物放 `outputs/agent_goal_exploration/`。\
需要长期保留、可审查但不含大文件的摘要放 `docs/research/agent_goal_exploration/`。

### 4.3 最终交付物

```text
outputs/agent_goal_exploration/
├── final_report.md
├── executive_decision.md
├── direction_scorecard.csv
├── experiment_matrix.csv
├── baseline_results.json
├── ablation_results.json
├── validity_results.json
├── cost_results.json
├── limitations_and_failures.md
├── recommended_paradigm.md
├── paper_readiness_assessment.md
└── artifact_manifest.json
```

---

## 5. 阶段 0：建立不可移动的事实基线

### 任务 0.1：项目和环境盘点

**涉及文件或模块：**

- 读取：`AGENTS.md`
- 读取：`CONTEXT.md`
- 读取：`README.md`
- 读取：`docs/frontend_backend_handoff.md`
- 读取：`docs/run_result_contract.md`
- 读取：`backend/app/training.py`
- 读取：`backend/app/models/registry.py`
- 读取：`backend/app/routers/preprocess.py`
- 读取原始报告：`D:\PythonProject\AutoAI\outputs\未来Agent开发方向_深度调研报告.md`

**实施内容：**

- 确认当前代码版本、Python、Node、CPU/GPU、可用依赖和测试状态；
- 只检查外部模型 provider 是否“可用”，不得输出或读取凭据值；
- 列出可合法使用的数据集、文件格式、规模和许可状态；
- 标记哪些现有数据只可本地使用，哪些可以进入论文复现；
- 记录当前工作树快照和初始测试结果。

**验收标准：**

- `research_lab/journal/checkpoints/00_inventory.md` 包含事实、未知项和风险；
- 每项结论都有命令、代码位置或运行结果支撑；
- 没有读取或复制敏感文件。

### 任务 0.2：冻结当前 AutoAI 基线

**实施内容：**

- 运行项目规定的 smoke、完整测试和 compileall；
- 如果根目录存在可合法使用的本地数据，使用轻量模型跑一次真实分类闭环；
- 新增回归之前，记录当前“不支持正式回归”的事实；
- 采集运行时间、指标、artifact、split 和错误；
- 创建 `AutoAIBaselineAdapter`，使统一 harness 能调用现有固定流程。

**验证：**

```powershell
$env:PYTHONPATH='<WORKTREE>'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest '<WORKTREE>\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest '<WORKTREE>\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall '<WORKTREE>\backend\app' -q
```

**验收标准：**

- 有机器可读 baseline；
- 失败也必须有完整日志和复现命令；
- 后续方向不能偷偷使用不同数据或更高预算。

---

## 6. 阶段 1：统一研究基座

### 任务 1.1：Episode、Contract、Evidence 和 Result schema

**产出接口：**

- `EpisodeSpec`
- `ScientificContract`
- `ToolSpec`
- `EvidenceItem`
- `VerificationFinding`
- `ExperimentBudget`
- `DirectionResult`
- `ValidityCertificate`

**关键要求：**

- 使用 Pydantic 或等价强类型 schema；
- 所有结果可 JSON 序列化；
- schema 有版本；
- 输入、contract、split、manifest 和确定性产物有 SHA-256；
- 非确定训练和 LLM 输出使用版本、seed 和预注册容差，不承诺字节级一致；
- finding 区分 `pass / warning / fail / unknown`；
- severity 区分 `informational / review / blocking`。

### 任务 1.2：统一工具和策略接口

```text
observe(state) -> Observation
plan(observation, budget) -> ProposedAction[]
policy_gate(action, contract) -> Allowed | Rejected
execute(action) -> ToolResult
verify(state, evidence) -> VerificationFinding[]
repair(findings, state, budget) -> ProposedAction[]
certify(state, findings) -> ValidityCertificate
```

工具必须声明：

- 输入/输出 schema；
- requires / guarantees / invalidates；
- read/write capability；
- deterministic 类型；
- 预算；
- evidence。

### 任务 1.3：统一运行器

运行器必须支持：

- 单方向；
- 全矩阵；
- 固定 seed；
- 并发但不破坏可复现；
- checkpoint/resume；
- 预算耗尽；
- 原始 JSONL 轨迹；
- 汇总 CSV；
- 失败不吞异常；
- 不因一个方向失败而终止整个 Goal。

### 任务 1.4：预注册模板

每个方向在运行前写：

```yaml
direction_id:
hypothesis:
strongest_alternative:
primary_metric:
secondary_metrics:
minimum_effect:
datasets:
baselines:
budget:
seeds:
kill_criteria:
known_limitations:
```

预注册文件的哈希进入结果，避免看完 private test 后改成功标准。

---

## 7. 必做方向 D01：Scientific Contract + Deterministic Verifier

### 假设

将样本实体、轴、split、fit/transform、test 隔离和 artifact 规则显式化，可以显著降低无效运行被接受，同时保持较低 clean-case 误拒绝。

### 实现

- Contract schema；
- schema、axis、lineage、split、protocol、reproducibility 五类 verifier；
- findings 和证据；
- Validity Certificate v1；
- verifier 单元测试；
- 至少 20 个 clean/invalid fixtures。

### 必测故障

- group overlap；
- 已确认技术重复跨集合；
- label leakage；
- full-data scaler/PCA/feature selection；
- test-driven selection；
- 轴反向、重复、单位不一致；
- 不可行 split；
- artifact/manifest 缺失；
- replay 证据缺失；
- 可疑近重复仅 warning，不能自动 blocking。

### 基线

- 无 verifier；
- 当前 AutoAI 零散校验；
- rule-only verifier。

### Promote 条件

- Unsafe Acceptance Rate 显著降低；
- clean False Rejection 在预注册范围内；
- 每个 blocking finding 有机器可复核证据。

### Kill 条件

- 规则只能由 LLM 主观判断；
- 大量 clean case 被阻断；
- certificate 无法独立复核。

---

## 8. 必做方向 D02：Rule-only Repair Engine

### 目的

建立最强非 Agent 对照，避免把规则系统的能力错误归因给 LLM。

### 实现

- finding → repair template；
- 确定性优先级；
- 最短 repair chain；
- 不可修复时 abstain；
- 无自然语言推理。

### 实验

- 单故障；
- 两种组合故障；
- repair 冲突；
- repair 成本不同；
- 缺元数据。

### 判定

如果 rule-only 在绝大多数任务上达到或超过 Agent，最终范式应以规则系统为核心，并把 LLM 降级为交互层。这是允许且可信的最终结论。

---

## 9. 必做方向 D03：Single-Agent Planner–Executor–Verifier Loop

### 假设

当任务需要理解目标、跨工具编排并在多个合法 repair 中选择时，受约束单 Agent 能超过固定 repair template。

### 实现

- provider-agnostic planner interface；
- 若没有可用外部 LLM，不读取秘密，先实现 replay planner 和 deterministic test planner；
- 可用时运行 live LLM 试验并完整记录 model/version/prompt/budget；
- policy gate；
- verifier-conditioned replan；
- retry、resume、budget、abstain；
- 不能执行任意未注册 Python。

### 基线

- D02 rule-only；
- 单次 tool caller；
- planner + executor 无 verifier；
- planner + verifier 无 replan；
-完整 loop。

### Promote 条件

- 在复杂、组合、上下文依赖 repair 上，同预算 VCR 超过 D02；
- 提升不是来自更多尝试；
- Unsafe Acceptance 不增加。

### Kill/降级条件

- D02 同样有效且更便宜；
- Agent 经常建议非法 action；
- 结果高度依赖单一 prompt；
- 没有 live provider 时不得声称已验证 LLM 泛化。

---

## 10. 必做方向 D04：有效性约束的预处理搜索

### 假设

Agent 可根据 modality、任务、数据状态和验证反馈选择预处理，但只有在所有拟合型变换严格位于训练折内时才有意义。

### 候选操作

- baseline correction；
- smoothing；
- normalization；
- derivatives；
- scaling；
- PCA；
- peak alignment；
- resampling。

### 约束

- modality-specific contract；
- fit/transform 分离；
- test 不参与选择；
- 原始轴和变换证据保留；
- 未声明的插值、外推和单位换算必须拒绝。

### 基线

- 无预处理；
- 固定专家 pipeline；
- grid/random search；
- nested CV；
- Agent search。

### 指标

- 有效 run utility；
- 搜索成本；
- 选择稳定性；
- invalid proposal rate；
- 外部域表现。

### Kill 条件

- Agent 只做更昂贵的随机搜索；
- 选择结果在 source-held-out 不稳定；
- 提升来自 test leakage。

---

## 11. 必做方向 D05：Agentic Model Selection and Tuning

### 假设

Agent 能利用数据规模、类数、信号长度、资源和解释需求缩小模型空间，但在小样本上未必优于简单模型和成熟 AutoML。

### 模型

- Logistic/Elastic Net；
- PLS-DA / PLSR；
- PCA-LDA；
- SVM/SVR；
- Random Forest；
- XGBoost；
- MLP；
- 现有 1D 深度网络。

### 基线

- 当前 fixed catalog；
- 简单 Logistic/PLS；
- AutoGluon 或可用成熟 AutoML；
- random/grid/Bayesian search；
- Agent。

### 必须报告

- 同预算；
- 选择模型复杂度；
-有效 utility；
- 过拟合与方差；
- wall time；
- 是否因更多候选获得优势。

### Kill 条件

- 简单基线相当或更强；
- Agent 只增加成本；
- 选择不稳定。

---

## 12. 必做方向 D06：跨仪器/批次域偏移与校准转移

### 假设

有效性 Agent 在看到训练域和外测域差异时，应先诊断可识别性和合法适配条件，而不是自动重采样后宣称可用。

### 实现

- domain metadata；
- axis/unit compatibility；
- shift diagnostics；
- domain-held-out split；
- 允许的 calibration transfer；
- 无法识别时 abstain。

### 方法候选

- 无适配；
- direct standardization；
- piecewise direct standardization；
- CORAL；
- domain-invariant representation；
- 预注册的 axis mapping。

### 关键限制

- 目标域标签的使用规则必须显式；
- 未声明的单位换算、外推、peak alignment 不能自动进行；
- batch 与 label 完全混杂时必须报告不可识别。

### Promote 条件

- 跨域有效性和性能均改善；
- 不通过隐藏目标标签选方法；
- 至少一个真实或公开跨域数据案例。

---

## 13. 必做方向 D07：OOD、Uncertainty 与 Selective Abstention

### 假设

科学 Agent 的可靠性不仅是分类正确，还包括识别未知样本、域外输入和证据不足，并选择性拒绝。

### 实现

- OOD detectors；
- prediction uncertainty；
- conformal/coverage-risk（若适用）；
- abstention policy；
- certificate 中记录拒绝依据。

### 基线

- 总是预测；
- softmax confidence；
- 简单距离/密度；
- ensemble；
- Agent-informed abstention。

### 指标

- AUROC/AUPR；
- coverage-risk；
- correct abstention；
- false rejection；
- valid utility at fixed coverage。

### Kill 条件

- Agent 自然语言“感觉不确定”代替数值证据；
- 拒绝策略无法校准；
- OOD 数据与训练数据同源。

---

## 14. 必做方向 D08：Counterfactual Hidden Validity Benchmark

### 假设

按来源隔离、只改变一个有效性条件的 clean/invalid 配对，可以可靠测量 Agent 是否真正识别和修复科学错误。

### 实现

- EpisodeSpec；
- fault mutation library；
- deterministic oracle；
- source/dataset-disjoint split；
- hidden seed；
- containerizable evaluator；
- compound faults；
- benchmark card。

### 必须满足的因果约束

clean/invalid 配对固定：

- 原始数据；
- 任务目标；
- 候选工具；
- 预算；
- 模型可用性；
- 预测难度。

单故障 episode 只改变被测 contract 条件。组合故障按预注册设计构造并单独报告交互。

### 最低规模

- 3 个 signal domains；
- 15 个独立来源；
- 60 个 hidden episodes；
- 8 个 fault families；
- 每个主要故障至少包含多个来源。

### 理想规模

- 5 个 domains；
- 25–40 个来源；
- 150–250 episodes；
- 分类、回归、OOD；
- 真实跨仪器 challenge。

---

## 15. 必做方向 D09：Episodic/Semantic Memory

### 假设

经过治理的历史 repair 经验可能提高未见来源的 first-valid-plan，但也可能造成错误复用、污染和 test 泄漏。

### 实现

- working memory；
- episodic memory；
- method/semantic memory；
- memory policy；
- provenance；
- redaction；
- expiry；
- negative-transfer logging。

### 不得存入

- locked test 标签；
- test-driven 模型选择；
- hidden episode 答案；
- 未授权原始数据；
- 无来源方法结论。

### 实验

- 相同来源不同 session；
- 未见来源；
- 相似但规则不同的陷阱任务；
- poisoned/wrong memory；
- no-memory 对照。

### Promote 条件

- source-held-out first-valid-plan 或 repair 提升；
- token/工具调用降低；
- 无 Unsafe Acceptance 增加；
- 负迁移可检测。

### Kill 条件

- 只在重复同一数据集时有效；
- 存在 test/benchmark 泄漏；
- 负迁移抵消收益。

---

## 16. 必做方向 D10：Adaptive Model Routing / Cost-aware Scheduling

### 假设

简单 schema 和单规则故障可由便宜策略处理，复杂冲突才需要强模型；合理路由可能降低 cost-to-valid。

### 比较

- rule-only；
- 固定便宜模型；
- 固定强模型；
- heuristic routing；
- learned/rule routing；
- oracle routing。

### 指标

- VCR；
- Unsafe Acceptance；
- cost-to-first-valid；
- token；
- latency；
-强模型调用数；
- Pareto frontier。

### Promote 条件

Routing 至少支配一个固定策略，或在相同 VCR 下显著更便宜。

### Kill 条件

- 只是把失败任务全部转给强模型；
- 成本下降伴随有效性明显下降；
- 没有超过简单阈值规则。

---

## 17. 必做方向 D11：Single Controller vs Multi-Agent

### 假设

角色分工只在信息或能力真正独立时可能有价值；同一模型的角色扮演可能只是增加采样和 token。

### 架构

- 单 controller；
- planner + repair 两角色；
- planner + evidence retriever；
- planner + critic；
-完整多角色方案。

Verifier 始终是确定性程序，不计作“会思考的 Agent”。

### 公平性

- 同模型；
- 同总 token；
- 同工具；
- 同 retry；
- 同 episode；
- 同可见 evidence。

### Promote 条件

- 复杂任务 VCR/repair 有独立提升；
- CI 排除零；
- 提升大于成本；
- 轨迹证明来自信息分工，不是更多采样。

### Kill 条件

- 单 Agent 同预算相当或更优；
- 多 Agent 只增加延迟；
- 角色输出高度冗余。

---

## 18. 必做方向 D12：Evidence-grounded Method RAG

### 假设

带来源和版本的方法知识可以帮助 Agent理解 modality 约束和候选方法，但无治理的 RAG 会引入 prompt injection、错误方法和过时结论。

### 实现

- 只收录可追溯官方文档、论文和项目方法说明；
- chunk 带 DOI/URL、版本、访问日期；
- retrieval 输出与 action policy 分离；
- 外部文本视为不可信数据；
- 方法建议必须映射到已注册工具；
- citation completeness；
- 不允许 RAG 内容修改 system policy。

### 实验

- 无 RAG；
- 通用 RAG；
- 结构化 method cards；
- 含冲突来源；
- 含恶意提示的文档；
- 未见 modality。

### Promote 条件

- 提高合法计划或 repair；
- 引用可核实；
- 不增加 prompt-injection 成功率；
- 不直接复制 benchmark 答案。

### Kill 条件

- 仅增加文本长度；
- 引入错误方法；
- 不能证明来源；
- 使 Agent 越权。

---

## 19. Stretch D13：MCP Tool Boundary

### 目标

验证 MCP 作为工具互操作边界的工程价值，不把它包装成算法创新。

### 实现

- 选择 3–5 个只读/受限工具暴露为 MCP；
- 输入/输出 schema；
- capability scope；
- untrusted tool metadata；
- 与本地直接调用比较。

### 判定

只评价：

- 接入成本；
- schema 一致性；
-可替换性；
-延迟；
-安全边界；
- replay。

论文中最多作为实现细节。

---

## 20. Stretch D14：Verifier-pruned Tree Search

### 假设

对预处理、模型和 repair 的组合空间进行树搜索，并用 verifier 提前剪枝，可能比线性 ReAct 更稳定。

### 实现

- 节点：有效状态；
- 边：合法 action；
- verifier pruning；
- utility/cost priority；
- beam/A*/MCTS 中选择最小充分方案；
- 与线性 planner、公平预算随机搜索比较。

### Kill 条件

- 优势仅来自更多评估；
- 搜索成本过高；
- 简单 rule planner 相当。

---

## 21. Stretch D15：LLM Prior + Deterministic Bayesian Optimization

### 假设

LLM 适合解释目标、提出搜索空间和先验，但数值后验与 acquisition 应由 BO/bandit 负责。

### 场景

- 预处理超参数；
- 校准转移样本选择；
- HPLC 条件的模拟优化；
- 主动标注。

### 对照

- random search；
- GP-BO；
- linear/contextual bandit；
- LLM-only；
- LLM prior + BO；
- 打乱数值反馈的 LLM。

### 关键证伪

如果打乱实验反馈后 LLM 行为不变，则不能声称它在使用数值反馈。

---

## 22. 数据策略

### 22.1 三层数据

1. **机制 fixture**

   小、确定、可快速测试 verifier 和 repair。

2. **公开科学数据**

   至少 Raman/IR/NIR、色谱/时间信号、另一种一维信号；保存许可、版本、URL 和哈希。

3. **本地真实案例**

   只在许可范围内使用现有 Raman/HPLC 数据；不复制进 Git；不把内部结果冒充公开 benchmark。

### 22.2 分类与回归

- 分类至少包含多类和不平衡场景；
- 回归包含小样本和外部校准；
- split 以原始实体分组；
- source-held-out 不能由同一数据集 mutation 代替。

### 22.3 数据不足时

如果公开数据下载、许可或网络阻塞：

- 先使用可追溯 fixture 完成机制验证；
- 记录真实数据缺口；
- 不标记外部泛化已完成；
- 继续实现其他方向；
- 在最终可信度中降级。

---

## 23. 统一基线矩阵

| ID | 系统 | LLM | Verifier | Replan | Memory | Multi-Agent |
|---|---|---|---|---|---|---|
| B0 | 当前 AutoAI | 否 | 当前零散规则 | 否 | 否 | 否 |
| B1 | 强固定 AutoML | 否 | 否/外置 | 否 | 否 | 否 |
| B2 | Rule-only validator + repair | 否 | 是 | 模板 | 否 | 否 |
| B3 | 单次 tool caller | 是 | 否 | 否 | 否 | 否 |
| B4 | Planner + executor | 是 | 否 | 是 | 否 | 否 |
| B5 | Planner + verifier | 是 | 是 | 否 | 否 | 否 |
| B6 | Full single-controller | 是 | 是 | 是 | 否 | 否 |
| B7 | Full + memory | 是 | 是 | 是 | 是 | 否 |
| B8 | Full + routing | 混合 | 是 | 是 | 可选 | 否 |
| B9 | Multi-Agent | 是 | 是 | 是 | 可选 | 是 |

如果外部 LLM provider 不可用：

- B3–B9 的 live LLM 结果标记为 blocked/unverified；
- 仍实现 provider interface、replay policy、测试和离线轨迹；
- 不用 deterministic mock 的结果代替真实 LLM 结论；
- Goal 继续推进能完成的机制、数据、规则和基线工作。

---

## 24. 统一指标和置信度

### 24.1 主指标

- Valid Completion Rate；
- Unsafe Acceptance Rate；
- Repair Success Rate；
- Correct Abstention Rate；
- False Rejection Rate；
- Cost-to-First-Valid-Run。

### 24.2 科学效用

只在 valid runs 上：

- 分类：macro-F1、balanced accuracy、per-class recall、ECE；
- 回归：RMSE、MAE、R²、bias、RMSEP；
- OOD：AUROC/AUPR、coverage-risk；
- 域外：source/instrument/batch-held-out。

### 24.3 复现

- 输入、contract、split、manifest、确定性工具输出哈希一致；
- 非确定训练/LLM 输出按预注册容差；
- certificate 可独立复核；
- replay 成功；
- 失败可重现。

### 24.4 统计

- 至少 3 seeds；条件允许时 5；
- episode-level bootstrap 95% CI；
- paired clean/invalid 使用配对检验；
- 成败使用 McNemar；
- 连续指标使用 paired bootstrap/Wilcoxon；
- 多来源使用 hierarchical bootstrap；
- 报告 effect size；
- primary endpoint 预注册；
- 全部失败 run 保留。

### 24.5 可信度等级

每个方向结论标记：

- **A：高可信**——真实多来源、强基线、同预算、统计稳定、外部验证；
- **B：中可信**——多数据集但外部验证或模型重复有限；
- **C：初步**——fixture/合成或单一来源；
- **D：不可判断**——被环境、凭据或数据阻塞；
- **F：已证伪/淘汰**——未超过基线或产生不可接受风险。

---

## 25. 决策与淘汰机制

每完成一个方向，在 `research_lab/journal/decision_ledger.md` 追加：

```markdown
## Dxx

- 预注册哈希：
- 实现提交/文件：
- 数据与来源：
- 运行命令：
- 原始结果：
- 主要效应与 CI：
- 最强基线：
- 失败：
- 替代解释：
- 可信度：
- 决策：PROMOTE / HOLD / KILL
- 下一步：
```

最终范式只能由 PROMOTE 组件组成。HOLD 可以进入未来研究，KILL 必须在最终报告中保留失败原因，不能悄悄删除。

---

## 26. 阶段性检查点

### Checkpoint A：基线与基座

必须具备：

- 当前测试证据；
- baseline adapter；
-统一 schema；
- runner；
- 预注册；
- 最小 fixture。

### Checkpoint B：可靠性核心

完成 D01、D02、D03、D08：

- contract；
- verifier；
- rule repair；
- Agent loop；
- benchmark；
- certificate。

此时先判断“Agent 是否超过规则”。若没有，不能停止；继续其他方向验证其边界。

### Checkpoint C：科学建模

完成 D04–D07：

- 预处理；
- 模型；
- 跨域；
- OOD；
- 分类和回归。

### Checkpoint D：Agent 架构变量

完成 D09–D12：

- memory；
- routing；
- multi-agent；
- evidence RAG。

### Checkpoint E：Stretch 与收敛

根据证据实施 D13–D15；至少完成其中一个，或者用前序证据明确说明不值得投入并实现最小证伪原型。

### Checkpoint F：最终复跑

- 冻结代码和配置；
- 清空非必要缓存；
- 从 manifest 重跑主结果；
- 独立核对 certificate；
- 汇总方向矩阵；
- 写最终范式。

---

## 27. 最终范式候选，不是预设答案

当前最值得验证的候选是：

```text
Contract Compiler
→ Single Planner/Controller
→ Policy Gate
→ Typed Deterministic Tools
→ Deterministic Verifier
→ Restricted Repair/Replan
→ Validity Certificate + Replay
```

但最终可能得到以下任何结论：

1. **规则核心范式**

   Rule-only 已足够；LLM 只做交互和复杂目标解析。

2. **单 Agent + verifier 范式**

   Agent 在组合故障和上下文 repair 上带来同预算增益。

3. **分层路由范式**

   大多数任务由规则/小模型解决，复杂任务升级强模型。

4. **Memory-enhanced 范式**

   只有在未见来源前迁移证据成立时。

5. **Multi-Agent 范式**

   只有同预算消融证明信息分工的独立价值时。

6. **Agent 不适合当前问题**

   固定 AutoML/规则系统更可靠、更便宜。这同样是可信研究结论。

新任务不得预先选定结论。

---

## 28. 最终报告必须回答的问题

1. 哪 12 个方向实际实现了什么？
2. 每个方向运行了哪些数据和命令？
3. 哪些结果是正面、负面或不可判断？
4. 哪些结论来自真实数据，哪些仅来自 fixture/合成数据？
5. Agent 相对 rule-only 的独立价值是什么？
6. 有效性提升是否以 clean utility、成本或误拒绝为代价？
7. 多 Agent、memory、routing 是否值得保留？
8. 哪些能力只是工程实现，不能写成论文创新？
9. 最终范式的适用条件和禁用条件是什么？
10. 复现者如何从零重跑？
11. 哪些证据足以支持二区论文，哪些还不够？
12. 最推荐的论文题目、贡献、图表、benchmark 和目标期刊是什么？

---

## 29. 新任务的第一轮动作

新任务启动后必须按顺序执行：

1. 调用 `create_goal`，目标文本使用本计划第 1 节和完成定义，不设置 token budget。
2. 读取本计划全文。
3. 读取项目规则和原始深度调研报告。
4. 确认当前路径是独立 worktree，而不是 `D:\PythonProject\AutoAI`。
5. 运行 `git status --short`，记录起点。
6. 创建 `research_lab/journal/checkpoints/00_start.md`。
7. 开始阶段 0 项目盘点和 baseline，不再次停留在写计划。
8. 第一轮 commentary 明确报告 Goal 已创建、worktree 路径和正在运行的首个验证命令。

---

## 30. 计划验收

本计划覆盖用户要求：

- 先制作计划；
- 另开 AutoAI 任务；
- 使用 `gpt-5.6-sol`；
- 使用 `ultra` 推理；
- 新任务进入显式 Goal 模式；
- 从当前代码建立独立目录；
- 至少实践 10 个方向，实际规定 12 个必做和 3 个 stretch；
- 最终给出可行性、原因、可信度和真正有效范式；
- 不以时间成本为提前停止理由；
- 保护原项目和用户数据。
