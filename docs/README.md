# Pan 文档导航

> 最近核对：2026-09-29，源码基准 019f1cc。用户当前要求决定任务目标；代码和接口用于确认现状；历史计划不授予操作权限，也不覆盖当前规则。

## 先读的入口

- [AGENTS](../AGENTS.md)：唯一开发目录/分支、Git、授权和验证规则。
- [CONTEXT](../CONTEXT.md)：有日期和 SHA 的当前阶段、协议、发布与候选状态。
- [README](../README.md)：当前能力、数据格式和使用方式。
- [CLAUDE](../CLAUDE.md)：共用上述规则的协作入口。
- [Git 与发布规范](github_publish_policy.md)：仅在服务器 Pan 根目录核对、提交及推送 pan/agent；操作须单独授权。
- [部署说明](../deploy/server_deploy.md) 与 [服务控制](service_control.md)：当前 release、storage 和受控启停。

## 现行接口和阶段契约

- [前后端接口](frontend_backend_handoff.md)、[结果页](run_result_contract.md)、[历史 Manifest](legacy_manifest_compatibility.md)。
- [Agent API](agent_api_contract.md)：旧版兼容与 Session/Experiment、持久绑定及输入边界。
- [全模型及配方](agent_step2_contract.md)：能力、证据、知识和逐阶段协议。
- [State/CLI](langgraph_state_contract.md)、[编排与恢复](langgraph_orchestration.md)。
- [第五步处理消融](step5_processing_ablation.md)、[第七步预算](step7_budget_contract.md)、[第八步训练检查](step8_guard_contract.md)。
- [第九步反馈诊断](step9_diagnosis_contract.md)、[第九步隔离备份验收](step9_backup_acceptance.md)。

当前新 recipe 任务为 revision-v7 / State-v9，诊断默认启用且建议不执行，仍为单实验流程。旧协议章节必须按其版本阅读。模型注册与实现状态见代码目录，当前环境可用性见 GET /api/models，不以历史数量替代接口。

Agent 的工具和后端生成输入不含 Test；通用结果页仍显示既有 Test。这里说明既有边界，不提出额外的 Test 开放时序改造。

## 架构与历史材料

- [稳定 Run 架构决策](adr/0001-stable-run-architecture.md)：保留仍有效的架构依据，开发操作遵守现行 AGENTS。
- [早期开发计划](../AutoAI_开发计划.md)：历史路线，不是当前阶段或操作指令。
- `plans/`：按文件日期保留的历史实施记录；其中建分支、推送和部署命令不能直接执行。当前十三步顺序与已实现状态见 CONTEXT。

接口字段、服务状态和历史实验证据分别维护，不把“文档里写了”“代码已进入主线”“服务已发布”和“论文效果已证实”混成一个状态。
