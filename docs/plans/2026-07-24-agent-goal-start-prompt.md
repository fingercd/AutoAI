# AutoAI Agent Goal 新任务启动 Prompt

这是用户明确授权的长期 Goal 模式实践任务。你运行在从 AutoAI 当前 working tree 创建的独立 Codex worktree 中，主模型应为 `gpt-5.6-sol`，推理强度为 `ultra`。

你的第一项工具动作必须是调用 `create_goal`，不要设置 token budget。Goal objective 使用：

> 在独立 AutoAI worktree 中真实实现、运行和评测至少 12 个科学 AutoML/原生 Agent 方向，建立统一 benchmark、强基线、有效性指标、失败档案和可信度分级，最终以实践证据收敛出一个真正有效、可复现、可证伪且适合论文的 Agent 范式；不得只做调研或设计，不得在未完成代码、测试、实验、比较和最终综合前标记 Goal complete。

创建 Goal 后立即继续，不要停在复述计划或向用户再次确认。

## 必读内容

完整读取：

1. `AGENTS.md`
2. `CONTEXT.md`
3. `README.md`
4. `docs/plans/2026-07-24-agent-goal-exploration.md`
5. `docs/run_result_contract.md`
6. `docs/frontend_backend_handoff.md`
7. 原始深度调研报告：`D:\PythonProject\AutoAI\outputs\未来Agent开发方向_深度调研报告.md`

## 隔离要求

1. 先确认当前工作目录不是 `D:\PythonProject\AutoAI`，而是 Codex 创建的独立 worktree。
2. 只允许在当前 worktree 修改和实验。
3. 禁止回写、移动、删除或清理原始 `D:\PythonProject\AutoAI` 中的任何文件。
4. 禁止读取或输出 `.env`、`auth.json`、token、密钥、认证缓存和 `.sandbox-secrets`。
5. 数据、模型、缓存、压缩包和二进制不进入 Git。
6. 不 push、不创建 PR，除非用户以后明确要求。

## 实践要求

严格执行 `docs/plans/2026-07-24-agent-goal-exploration.md`，其中 12 个方向全部必做：

1. Scientific Contract + Deterministic Verifier
2. Rule-only Repair Engine
3. Single-Agent Planner–Executor–Verifier Loop
4. 有效性约束预处理搜索
5. Agentic Model Selection and Tuning
6. 跨仪器/批次域偏移与校准转移
7. OOD、Uncertainty 与 Selective Abstention
8. Counterfactual Hidden Validity Benchmark
9. Episodic/Semantic Memory
10. Adaptive Model Routing / Cost-aware Scheduling
11. Single Controller vs Multi-Agent
12. Evidence-grounded Method RAG

并尽量实践三个 stretch：

13. MCP Tool Boundary
14. Verifier-pruned Tree Search
15. LLM Prior + Deterministic Bayesian Optimization

“实践一个方向”必须同时包含：

- 预注册假设和 kill criteria；
- 可运行代码或最小原型；
- 针对性测试；
- 实际运行；
- 原始结果；
- 强基线；
- 同预算比较；
- 失败证据；
- `PROMOTE / HOLD / KILL`；
- A/B/C/D/F 可信度等级。

只写文档、prompt、接口或文献综述不算完成。

## 首轮必须完成

1. `create_goal`。
2. 确认隔离 worktree 绝对路径。
3. 运行 `git status --short`。
4. 创建 `research_lab/journal/checkpoints/00_start.md`。
5. 盘点 Python、依赖、CPU/GPU、可用数据和测试。
6. 运行 AutoAI smoke test；条件允许时开始完整测试。
7. 建立 baseline 结果文件。
8. 开始创建统一 research harness，不要再次停下来等待用户。

推荐 Python：

```text
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe
```

## 研究纪律

- 任何无效 run 的 F1/R² 都不算成功。
- LLM 不能签发 Validity Certificate；使用确定性 verifier。
- Rule-only 是必须击败或承认更优的强基线。
- 多 Agent、memory、routing、MCP 不是预设贡献。
- 没有 live LLM provider 时，不读取凭据；实现 provider interface 和可复现替代实验，并把 live 结论标为未验证。
- 某个方向失败时记录负面结果并继续其他方向。
- 不因工作量大、结果负面或时间较长而提前结束 Goal。
- 只有计划中的完成定义全部满足才能调用 `update_goal(status="complete")`。
- 若出现真正阻塞，先穷尽其他可独立推进方向；只有符合 Goal 工具的多轮阻塞规则时才标记 blocked。

## 最终交付

写入：

```text
outputs/agent_goal_exploration/
```

至少包括：

- `final_report.md`
- `executive_decision.md`
- `direction_scorecard.csv`
- `experiment_matrix.csv`
- `limitations_and_failures.md`
- `recommended_paradigm.md`
- `paper_readiness_assessment.md`
- `artifact_manifest.json`

最终报告必须明确：

- 哪些方向真正有效；
- 哪些被证伪；
- 每个结论的证据和可信度；
- Agent 相对规则系统的独立价值；
- 最终范式的适用与禁用条件；
- 距离二区论文还缺什么。

现在先调用 `create_goal`，然后立即进入阶段 0 的真实执行。
