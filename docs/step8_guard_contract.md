# 第八步训练检查协议

本地工程实现，待架构独立验收。检查事实由 `backend/app/runs/guard.py` 聚合，
共享 `training.prepare_training_inputs` 返回的实际数组和划分；不创建第二套训练算法或用量账本。

## 版本与开关

新配方任务采用 `agent-recipes-revision-v6`、`agent-state-v8`，保留 Observation v2。
`extensions.guard` 为严格的 `agent-guard-projection-v1`，仅含报告引用、阶段、状态、
资格和有限检查原因；不含 Test 数值、样品、路径、原异常或预算数量。
旧 revision/checkpoint 按原 wire 读取，缺报告不补造历史 passed。

CLI 新任务默认 `--fail-fast-guard on`；`off` 只跳过提交前额外的无拟合准备检查。
数据与配置硬校验、worker pre-fit、发布和候选复查始终执行。
Session/State 冻结 `training-guard-policy-v1`、`training-guard-rules-v1`、
规则源码 digest 和开关，恢复不能换策略。旧未执行搜索计划遇源码摘要变化明确拒绝，
不放宽 `search_policy` 的完整摘要验证。规则源码变化也不能复用旧 Guard 策略执行。
人工 HTTP 提交同样创建 Guard Run，默认 admission off，不要求 Agent Session/预算/LLM。
holdout、Sample_ID CV 和 external_test 保留原评估及 Train-only 拟合口径。

## 阶段与可信绑定

`training-guard-report-v1` 的阶段为 admission、pre_fit、pre_publish、publication、
observation、finalize。每份报告绑定 Run、Principal scope 摘要、有效配置、数据摘要、
策略、claim fence，以及阶段适用的 Manifest 和已发生用量摘要。
report_id 是内容摘要；创建时间不参与身份。重复检查保留原报告，复查失败产生新事实。
passed、failed、pending、unavailable、disabled、not_applicable 是不同状态；
只有发布或候选检查通过才可以 eligible。只有 admission 可以 disabled。

报告表 `run_guard_reports_v1` 与 Run 共用 Runs SQLite。Run 新增
`guard_policy_json` 和 `publication_report_id`；策略不可移除或替换。
报告按 scope 查询，完整内部绑定不进入 Agent 安全投影。
报告写入失败不能按成功继续。历史 Run 的 guard_policy 为空，保持旧身份。

## 执行与发布

worker 重新读取有大小限制的数据字节，对同一份字节验 SHA、解析和准备，
提前检查通过不豁免 worker 复验。准备阶段复用既有数据、划分、模型、处理和有限搜索验证。
确定无效输入在标准化及模型 fit 前失败；不自动重切、换模型或重建搜索计划。

训练完成后先按服务端 catalog/模型/模式要求检查必要产物。所有非 volatile 条目均
流式校验 SHA-256 和大小，包括私有模型；不反序列化模型。JSON 解析使用被校验的
同一字节，单 JSON 上限 64 MiB、Manifest 上限 4 MiB、检查时间上限 60 秒。
每 MiB 检查执行权。Manifest 自报 required/volatile 不能豁免服务端要求。

指标要求有限、非 bool、范围合法且选择指标存在；正常零分及低分有效。
实际 config、架构、划分、processing 审计、搜索计划/候选/选参及 ledger 用量必须一致。
子进程 pre_publish 后才能写 Manifest，但这仍不是 Run 成功。
预算父进程必须确认整个子进程树退出、settle 后复查，且子报告绑定必须与父报告相同。
父检查与最终状态事务处于 LeaseGuard 内，继续执行 claim/lease/deadline 约束。
publication 报告插入、关联和 succeeded 转换在 Runs DB 同事务完成，失败一起回滚。

worker 合同为 `training-worker-guard-v1`。SQL trigger 同步支持旧预算 Run 被
budget-v1/guard-v1 worker 领取，新 Guard Run 仅允许 guard-v1。旧 worker 无法仅靠
直接打开数据库绕过该限制。迁移幂等，预算 trigger 更新有事务写锁。

## 失败、候选与历史事实

admission 不预约训练额度。已绑定尝试不因 pre-fit 失败返还 experiment；
fit/epoch 已 entered 的事实沿第七步账本结算。退出未知保留 unknown hold，禁止发布；
检查不新增免费重试或重规划。Guard 子异常只传注册 code/stage/report_id。

Observation、Session best、Finalize 使用同一 `assess_candidate`，每次核验当前产物。
发布后同大小损坏也会令候选 unavailable；历史 succeeded、selected_run_id、费用仍保留。
新协议 Finalize 保存当前 Guard 快照，并在 Agent 事务内复核 scope、策略和结算状态。
Finalize 与应用删除共享按 Run 的跨进程文件锁；Finalize/terminate 仍由 Session 事务互斥。
重复 Finalize 必须重新通过当前检查。Graph confirm 检查选定实验当前是否还有有效分数；
离线收集器读取 Test 前也重新读取 Session，历史锁定本身不能授权损坏结果。
本地 checkpoint 的只读 status 是历史快照，不表示已经在线重新检查产物。

## 验证与边界

定向测试在 `test_step8_guard.py`、`test_step8_guard_integration.py`、
`test_step8_failures.py`、`test_step8_selection_safety.py`、`agent_poc/tests/test_step8_graph.py`，并运行整个 backend/agent 测试集。
故障注入与真实模型/真实监督证据分开记录；确定性 provider 不代表真实 LLM。
Windows Job Object 专项必须在 Windows 实际运行。Linux 监督沿既有平台边界，
服务器恢复后另验，S7-S01–S7-S08 不因此关闭。
