> 范围：本页主体记录 `/api/agent` 的 v1 wire，供旧客户端与历史 Session 读取。当前新 CLI 配方任务使用 revision v7 / State v9，仍通过 `/api/agent/v2`；新增字段、冻结和恢复分派见 [第二步契约](agent_step2_contract.md) 与 [第九步诊断契约](step9_diagnosis_contract.md)。维护与运行范围以根目录 `AGENTS.md`、`CONTEXT.md` 为准。

# Agent API 接口契约（agent-session-v1）

Guard 从 revision v6 引入，当前 revision v7 继续使用；候选复查见 [第八步协议](step8_guard_contract.md)。本页 v1 wire 保持不变。

Agent API 是 LLM/Orchestrator 与稳定训练引擎之间的有限适配层。它只允许创建冻结的 Session、提交受限实验、读取 Validation Observation 和 Finalize；不允许读取服务器文件、数据库、Test 指标、预测、解释性结果或 artifact。它不实现自动选模、Prompt、Memory、诊断、重规划和论文创新算法。

## 版本与端点

成功响应均含 `contract_version="agent-session-v1"`；实验反馈另含 `observation_version="agent-observation-v1"`。

元数据增量使用 `metadata_version="agent-metadata-v1"`；Session/Observation 主版本保持不变，六个工具请求不增加客户端可写指纹字段。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/agent/health` | 能力、模型与模块状态 |
| POST | `/api/agent/sessions` | 创建冻结 Session |
| GET | `/api/agent/sessions/{session_id}` | 读取安全 Session 摘要 |
| POST | `/api/agent/sessions/{session_id}/experiments` | 预约并创建真实 queued Run |
| GET | `/api/agent/sessions/{session_id}/experiments/{run_id}/feedback` | 读取安全 Observation |
| POST | `/api/agent/sessions/{session_id}/reconcile` | 显式恢复 stale reservation 与补偿记录 |
| POST | `/api/agent/sessions/{session_id}/finalize` | 锁定一个完整成功 Run |

请求模型全部 `additionalProperties=false`。Session 固定 `dataset_id`、`seed`、8:1:1 分组评估、允许模型和 `max_runs`；Experiment 只能选择 `model_type`、`normalization`、`class_balance`、同 Session 的父 Run 和简短理由。请求体不接受 owner、tenant、路径或独立测试集字段。

当前 Agent 模型集为 `logistic_regression`、`svm`、`random_forest`。可选 `client_request_id` 长度为 1–128，仅允许字母、数字、点、下划线、冒号和连字符。

## 冻结数据与安全有效配置

新 Session 通过当前 Principal 的 `DatasetRepository.resolve` 和 `verify_integrity` 取得 SHA-256，与 Session 一起持久化。创建响应及检查响应的 `locked_config` 新增：

```json
{
  "metadata_version": "agent-metadata-v1",
  "dataset_sha256": "64位小写十六进制 SHA-256",
  "dataset_fingerprint_status": "ready"
}
```

已有 DB 只新增可空列，不回填历史指纹。历史 Session 明确返回 `dataset_sha256=null,dataset_fingerprint_status="unavailable"`；不会以当前文件内容冒充创建时冻结内容。同一 Session 创建 request 重放优先读取原持久化记录，即使源文件后来不再可用也返回原冻结元数据，内容冲突仍拒绝。

新 Session 的实验提交把冻结摘要作为内部 `expected_dataset_sha256` 传给共享 Run Submission Service；它对实际准备写入 Run 的 Dataset 快照执行比较，在创建 queued Run 前拒绝不一致。此内部字段不接受 HTTP 客户端写入。已绑定 Run 的提交重放、Session 检查、Observation 和 Finalize 也会核对持久 Run 的 Dataset ID 与摘要；源数据变动不会刷新 Session 指纹，也不会产生不同数据的新 Run。

实验提交（含重放）、`Session.experiments[]` 和 Observation 同级新增相同投影：

```json
{
  "metadata_version": "agent-metadata-v1",
  "dataset_sha256": "64位小写十六进制 SHA-256，缺失时为 null",
  "dataset_fingerprint_status": "ready",
  "effective_config_status": "ready",
  "effective_config": {
    "model_type": "logistic_regression",
    "normalization": "zscore",
    "class_balance": "none",
    "seed": 42,
    "feature_selection_enabled": false,
    "evaluation_config": {
      "mode": "stratified_holdout",
      "train_weight": 8,
      "validation_weight": 1,
      "heldout_weight": 1
    }
  }
}
```

该配置只从已校验并持久化的 `RunRecord.config` 白名单投影，不按 `effective_action` 再次推导，不直接透传 legacy 配置。摘要来自该 Run 的快照；它与历史 Session 是否曾冻结摘要是两个事实。未绑定 reservation 的配置为 `pending/null`；没有可用 Run 配置的旧记录为 `unavailable/null`。完整类型由 `backend/app/agent/metadata.py` 定义，值不符合枚举或比例约束时不公开不可信配置。路径、展示文件名、内部划分字段、预测和产物均不进入投影。

本增量不公开内部 reservation/submission key、多维成本或真实分组划分指纹。编排器可对确认的安全 evaluation 配置求 `evaluation_config_fingerprint`；实际 `split_fingerprint` 保持 null 和 unavailable，不能把比例与 seed 的摘要当作 Sample_ID 分配摘要。

## 状态机

Session：

| 当前状态 | 操作 | 下一状态 |
|---|---|---|
| open | 创建实验 | open |
| open | finalize 完整成功 Run | finalized |
| finalized | 相同 Run 再次 finalize | finalized（幂等） |
| finalized | 新实验或更换 Run | 拒绝 |

Experiment reservation：

| 当前状态 | 事件 | 下一状态/补偿 |
|---|---|---|
| reserved | queued Run 创建并绑定成功 | bound |
| reserved | Run 创建失败 | released，返还预算 |
| reserved | 绑定失败且 Run 从未开始、取消成功 | released，返还预算 |
| reserved | Run 已开始或取消失败 | compensation_required，保留预算 |
| bound | Worker 执行 | Run 依次 queued → running → succeeded/failed/cancelled |
| released | 相同 `client_request_id` 重放 | 保持 released，返回 409 `agent_request_released` |
| released | 新 `client_request_id` 或无 ID、配置相同 | 新建 reservation，attempt 按历史最大值递增 |

并发预约在 Agent SQLite 的 `BEGIN IMMEDIATE` 事务内完成 Session scope、状态、活动实验、`max_runs`、配置哈希、attempt 和 reservation 检查。进程内锁不是正确性依赖。

`reservation_id` 同时是永久 `submission_key`，一经创建永不复用。`released` 行是不可变审计历史：不会恢复为 `reserved`，不会清空 `run_id`、failure/reconciliation 字段或删除旧 mapping。需要以相同配置重新训练时，调用方必须使用新的 `client_request_id`（或提交一个没有 ID 的新请求），服务端创建新的 reservation、submission key 和 queued Run；活动配置 collision、单活动 Run 与 `max_runs` 规则不变。

## Durable submission mapping 与显式对账

新建 reservation 使用 `protocol_version="agent-reservation-reconciliation-v1"`，并把服务端生成的 `reservation_id` 作为内部 `submission_key`。`RunSubmissionService` 用主/独立测试 Dataset ID 与 SHA-256、校验后的 effective config 和 `submission_source` 计算稳定 SHA-256；展示名称、路径、时间戳和随机 Run ID 不参与哈希。

Runs DB 在创建 queued Run 的同一个 `BEGIN IMMEDIATE` 事务内写入内部表 `run_submission_keys_v1`。相同 key、hash 和 Principal 返回原 Run；hash 或 Principal 冲突时 fail closed。人工 `/api/training/runs` 不接收 key，也不写该表；mapping 不进入 Run 列表、状态、结果或 artifact 响应。

`POST /api/agent/sessions/{session_id}/reconcile` 使用服务端固定 stale 阈值，请求体只能为空对象或省略。调用方不能指定 Run、reservation、Principal、阈值或目标状态。它不会在 GET、启动或后台线程中隐式执行，也不属于 LLM 常规 Tool Schema。

| reservation / Run 证据 | 决策 | resolution_code |
|---|---|---|
| 新协议 stale reserved，无 mapping | released，返还预算 | `stale_without_run` |
| 新协议 stale reserved，同 Principal mapping 与 Run 完整 | bound 原 Run | `recovered_binding` |
| 旧协议或时间戳不可验证 | 保持占用并进入人工核对 | `legacy_protocol_unknown` / `reservation_timestamp_invalid` |
| compensation_required + queued 且 started_at 为空 | 原子取消后 released | `cancelled_before_start` |
| compensation_required + running | 保持占用 | `run_still_active` |
| compensation_required + succeeded/failed | bound，Run 终态不变 | `recovered_terminal_binding` |
| compensation_required + cancelled 且从未开始 | released | `cancelled_before_start` |
| compensation_required + cancelled 且已开始 | bound，保留预算 | `recovered_cancelled_binding` |
| scope 冲突、mapping/Run 缺失或取消不确定 | 保持占用并人工核对 | 稳定的 fail-closed resolution code |

Agent reservation 额外记录 `last_reconciled_at`、`reconcile_attempt_count` 和 `resolution_code`。并发对账依赖 `state + updated_at` 条件更新；失败方重读最终状态，不会把 bound 再释放或重复计数为恢复成功。Runs DB 不可用时不修改 Agent 状态；取消前先完成 Agent DB 审计写入，无法写入时不改变 Run 终态。

## Observation 与 Test 防火墙

Observation 只含状态、安全进度、有效动作、剩余预算和以下 Validation 有限标量：`accuracy`、`balanced_accuracy`、`macro_precision`、`macro_recall`、`macro_f1`、`weighted_f1`。

Validation 只有在 Run 为 `succeeded`、Manifest 完整，且 `metrics.json` 的大小与 SHA-256 通过校验后才为 `ready`。只读取 `metrics.valid`，绝不从 `test`、`pooled_test`、fold mean 或其他位置回退。NaN、Infinity、布尔值会被丢弃。

Observation 黑名单包括 Test 指标/样本数、预测、混淆矩阵、分类报告、解释性、artifact 名称/URL、服务器路径、Traceback 和原始异常。`extensions` 只能由服务端 available 模块写入；当前固定为空对象。

Finalize 要求 Run 属于当前 Principal 与 Session、状态成功、Manifest 完整并有 selection metric。Finalize 只锁定选择，不把 Test 回流给 Agent。人类结果页和独立 evaluator 继续使用既有 `run-result-v1` 权限边界。

## 错误结构

```json
{"detail":{"code":"agent_invalid_action","message":"安全说明","retryable":false,"allowed_actions":[]}}
```

| HTTP | code | 含义 |
|---:|---|---|
| 404 | `agent_session_not_found` | Session 不存在或不可见 |
| 404 | `agent_experiment_not_found` | Experiment 不存在或不可见 |
| 404 | `dataset_unavailable` | Dataset 不存在或不可访问 |
| 409 | `agent_session_finalized` | Session 已锁定 |
| 409 | `agent_duplicate_config` | 配置重复 |
| 409 | `agent_active_run_exists` | 存在活动实验 |
| 409 | `agent_run_budget_exhausted` | 预算用尽 |
| 409 | `agent_idempotency_conflict` | 同 request id 对应不同 payload |
| 409 | `agent_request_released` | 同 request id 的历史尝试已结束，禁止静默重放 |
| 409 | `agent_submission_key_conflict` | durable submission key 内容或作用域冲突 |
| 409 | `agent_submission_mapping_invalid` | durable mapping 目标不完整 |
| 409 | `agent_dataset_fingerprint_mismatch` | 实际 Run 快照与冻结 Session 不一致，拒绝提交或确认 |
| 409 | `agent_metadata_invalid` | 数据指纹格式或 Session 元数据版本不兼容，需要核对 |
| 422 | `agent_module_unavailable` | 请求了不可用模块 |
| 422 | `agent_invalid_action` | 动作不符合 Session/Run 状态 |
| 503 | `worker_contract_mismatch` | 活跃 Worker 契约不兼容 |
| 503 | `agent_reconciliation_unavailable` | 对账所需存储暂时不可用，可重试 |

404 不区分“真实不存在”和“属于其他 Principal”。公开错误不含绝对路径、数据库信息或堆栈。

## 能力与未来模块

此处说明 v1 基础适配层：基础五项能力可用，v1 不接受后续模块。能力状态按协议修订返回；当前 revision v7 的 `knowledge`、`dynamic_preprocessing`、`bounded_hpo`、`fail_fast_guard`、`feedback_diagnosis` 已有相应协议，预算由冻结的 `budget-policy-v1` 约束。`limited_replanning`、`uncertainty_selection`、`case_memory`、`constrained_code_evolution` 仍不可用。不能用 v1 的模块状态推断当前 v7 的能力。

未来模块应通过 capability registry、版本化请求字段和 Observation `extensions` 接入，不应修改共享 Run Submission Service、Worker、训练算法或 `run-result-v1`。

## Tool Client 调用顺序

1. `inspect_ml_capabilities()`
2. `start_ml_session(...)`
3. `submit_ml_experiment(...)`
4. `observe_ml_experiment(...)`，按 `retry_after_seconds` 轮询
5. 根据服务端 `allowed_actions` 决定继续实验或 finalize
6. `finalize_ml_session(...)`

GET 可有界重试；POST 只有携带稳定 `client_request_id`（Finalize 的同 Run 幂等语义除外）才可重试。客户端遇到 contract version 不匹配必须 fail closed。Bearer token 只保存在客户端内存，不写日志、Trace 或异常文本。


## 第九步反馈诊断（当前配方任务）

新 recipe 默认采用 revision-v7 / State-v9，可用 `--feedback-diagnosis off` 关闭。
诊断使用私有同源 Train/Valid 审计与当前 Guard 证据；建议只读、不执行。成功在 Finalize 前解释并复验资格，
失败先确认后端关闭再解释。诊断最多两次 work 物理调用，未知费用不清零，终态历史读取零网络。
详见 [第九步诊断契约](step9_diagnosis_contract.md)。
当前主线已包含该协议；是否在具体服务生效仍以实际部署版本为准。模型目录有 14 个目标、13 个实现，DSCARNet 已退役，CNN-Mamba 未实现；其余模型的 `available` 还取决于运行环境。
