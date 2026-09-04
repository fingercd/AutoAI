# SpecAutoAI 文档导航

> 最近核对：2026-07-16。本文用于区分当前契约、架构记录、部署说明和历史材料；发生冲突时，优先级为自动化测试与接口实现、`AGENTS.md`、`CONTEXT.md`、`README.md`、当前接口契约，最后才是历史计划和审查快照。

## 当前维护文档

- `../README.md`：安装、启动、功能范围和对外项目说明。
- `../CONTEXT.md`：面向维护者的当前实现速查表。
- `../AGENTS.md`：协作、修改边界、模型口径和验证要求。
- `frontend_backend_handoff.md`：前端调用后端时必须遵守的请求、响应和 artifact 契约。
- `run_result_contract.md`：独立建模结果页使用的 `run-result-v1` 字段、状态、指标聚合和下载描述。
- `agent_api_contract.md`：Agent Session/Experiment、durable submission mapping、显式 reservation reconciliation、Observation Test 防火墙及 LLM Tool Client 契约。
- `../deploy/server_deploy.md`：校园集群 Web + worker 部署说明。
- `github_publish_policy.md`：可以进入 Git/GitHub 的内容边界。
- `adr/0001-stable-run-architecture.md`：SQLite Run 状态机、worker、lease 与 Manifest 的架构决策。

## 历史或内部材料

- `../AutoAI_开发计划.md`：早期 1–9 步路线，保留原始语境，不代表当前 React/Vite、Redis/RQ 或模型实现。
- `hplc-pipeline-review-prompt.md`：2026-07-06 前后的 HPLC 提交审查快照，里面的文件位置、测试数量和旧响应字段不能替代当前代码。
- `plans/`：内部实施计划。计划描述预期工作，不自动等同已经交付的功能。

## 快速核对入口

- 模型目录：`GET /api/models`；目标 15 项，当前环境通常 14 项可用。
- 训练创建：`POST /api/training/runs`；只持久化 queued Run，训练由独立 worker 执行。
- 建模结果：`GET /api/training/runs/{run_id}/result`；前端 URL 为 `#/results?run_id=...`。
- 轻量记录：`GET /api/training/runs?projection=summary&limit=20&cursor=...`。
- 主色谱预处理：`POST /api/preprocess/hplc`；旧 `chromatography` 仅做简单范围截取。
- Run 下载：`GET /api/training/runs/{run_id}/artifact/{name}`；必须通过 Principal、成功状态、显式 catalog 和完整性校验。
- 服务器认证：`AUTOAI_DEPLOYMENT_MODE=server` + Bearer token；CORS 禁止通配来源。
- 回归测试：`python -m pytest backend/tests -q`，并运行 `python -m compileall backend/app -q`。
