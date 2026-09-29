# ADR 0001: Stable Run Architecture

> **决策记录说明（2026-09-29）：** 本 ADR 于 2026-07-16 核对，以下 Run 状态机、独立 worker、lease 与 Manifest 的架构决策仍作为该领域的有效依据；它不是开发目录、Git 或部署流程指南。执行工作以 [AGENTS.md](../../AGENTS.md)、[CONTEXT.md](../../CONTEXT.md)、[README.md](../../README.md) 为准。当前主线为 `019f1cc`；服务器唯一开发目录为 `/users/fotile/AutoAI/Pan`，固定 `pan/agent`，不得新建分支、worktree、fork 或可开发复制。开发用 Git 命令仅在服务器 Pan 根目录执行；本次文档整理不提交、推送或部署，GitHub 其他分支保留。

- Status: Accepted
- Scope: Run persistence, worker execution, state projection and artifact publication
- Last reviewed: 2026-07-16

## Decision

- SQLite RunRepository is the command-side authority.
- A separate local worker claims queued runs using a lease.
- status.json is a legacy projection, never a state-machine source.
- A Run becomes succeeded only after its Manifest is committed; Manifest v2 uses an explicit public catalog and integrity metadata.
- Local requests use an empty Principal. Server deployment requires Bearer authentication, and owner/tenant are injected by the server rather than accepted in request bodies.
- Server-scoped Run/Dataset reads and mutations are filtered by Principal; unowned historical Runs require an explicit audited migration.
- `GET /api/training/runs/{run_id}/result` is the versioned result-page projection; raw artifacts remain the audit source.
- Worker heartbeats are persisted separately from Run state and exposed only as an anonymous health summary.
- At the time of this decision, the architecture release preserved master model behavior. Later classification-v2 model work is governed by its independent model plan and acceptance tests; it does not change this ADR's Run-state guarantees.

## Consequences

- Restart recovery is possible without Redis/Celery.
- Existing clients receive legacy status strings through an adapter.
- Generic Run-directory downloads are removed.
- A corrupt or partial result can be reported without treating status.json as authoritative or returning server paths.
