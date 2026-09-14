> 版本说明：本页原 v1 契约作为兼容基线保留。新任务默认采用训练证据配方 profile 与 agent-state-v3；直接模型选择 v2 和旧 v1 检查点保持原版本。当前扩展与恢复分派见 [Agent 契约](agent_step2_contract.md)。

# LangGraph 任务 State 契约

第一步协议：`agent-state-v1`，图协议：`agent-single-experiment-v1`。实现位于
`agent_poc/orchestration/state.py`。本协议执行一个科学实验；完整 State 为后续模块保留
严格类型，但不表示 Evidence、配方、知识、案例、搜索、诊断、重规划或统一预算已实现。

当前上下文投影为 `agent-context-step1-v1`，Prompt 为 `agent-decision-step1-v2`，
LLM HTTP 配置为 `agent-llm-http-v1`。运行入口在恢复前核对这三个实际实现版本。

## 序列化与可用性

`StateModel` 是严格 Pydantic 模型，21 个顶层块分别对应独立的严格模型。所有模型
`extra=forbid`、`strict=True`，禁止 NaN/Infinity；分类指标还必须在 `[0,1]`。
`GraphState` 是 21 个 JSON channel 的 TypedDict，其值只允许递归 JSON primitive。
写 checkpoint 前使用 `model_dump(mode="json")`，读回后先执行
`StateModel.model_validate`。客户端、认证信息、SDK/模型实例及任意 Python 对象不属于 State。

| 值 | 含义 |
|---|---|
| `unavailable` | 实现或权威来源尚未提供；空值不能推断为零 |
| `disabled` | 当前任务策略关闭该模块 |
| `pending` | 已规划或已发起，尚未取得完整事实 |
| `ready` | 权威来源已经提供并通过当前契约校验 |
| `invalid` | 已取得的信息未通过有效性校验 |

模块策略同时记录 `enabled=false` 和 `implementation_status=unavailable`，模块输出块
记录 `status=disabled`。接口保留了将来结构，当前模型明确拒绝启用不可用模块，以及向
关闭模块填充统计、匹配结果、诊断或新实验。初态标量结果使用 `null`，记录使用空集合；
已知本地调用次数为零。没有 provider usage 时 token 数保持未知。

用量记录兼容新增可空 `total_tokens` 和 `token_status=partial`。仅返回 total 的 provider
用量保存在 journal 与 `budget.usage`，不猜测输入/输出分摊；本步不计算价格或科研成本。
旧 checkpoint 缺少 total 字段时按 null 读取，启动指纹和协议版本不变；旧 journal 在写事务
内幂等加列，历史总量与新增状态列保持 null，旧记录维持原 token_status 投影，不重写已确认
用量。新的结算只允许 dispatched 记录转为结果，重复 finish 不重复计量或覆盖结果。

时间为 UTC Unix 秒（有限非负数），不是本地显示时间字符串。公开 ID 是长度 1–128 的
单段标识，允许字母、数字、点、下划线、冒号和连字符，禁止路径分隔符与 `..`。
SHA-256 使用 64 位小写十六进制文本。版本与枚举不兼容、未知字段及错误类型均失败关闭。

## 21 个状态块

以下字段以 Python 模型为机器可校验的权威定义；未另注的输出均可空或初始为空集合。

| 状态块 / 模型 | 字段及含义 | 权威来源与允许写入方 |
|---|---|---|
| `identity` / `IdentityState` | `task_id/thread_id/session_id/current_experiment_id`；Session/Experiment 的 `operation_id/request_id` 与 Finalize operation；启动配置、后端目标、受信调用作用域摘要 | 编排器产生任务、请求与 operation ID；Session ID 取自后端。初始化、prepare 和确认响应节点写入。绑定后的 ID 不可改 |
| `lifecycle` / `LifecycleState` | `status/stage/next_action/started_at/ended_at/reason_code`；status 区分 initializing、running、waiting、recovering、completed、failed、cancelled、timed_out、needs_attention | 编排器路由节点；不覆盖后端 Session/Run 状态。开始时间冻结 |
| `task` / `TaskState` | dataset ID 与 SHA-256/状态；classification；选择指标；用户允许模型；seed；评估配置与配置指纹；真实 split 指纹/状态；固定 normalization/class_balance | 可信启动配置；Session 响应只允许绑定一次数据指纹。其他字段冻结。模型 ID 是开放的受控标识，不永久写死三模型 |
| `versions` / `VersionsState` | State、Graph、API、Observation、metadata；能力、Evidence、配方、搜索、规则、知识、案例的 `VersionRef`；context projection、Prompt、LLM 配置版本与摘要 | 初始化时冻结。State/Graph/API/Observation/metadata 使用当前版本 Literal；未实现模块为 `version=null/status=unavailable`。模型名称不代替 LLM 配置版本 |
| `module_policy` / `ModulePolicyState` | 12 个明确模块开关；`ablation_id`；source role、case read/write、context projection | 初始化后冻结；第一步全部研究模块关闭，case read/write 为 false |
| `capabilities` / `CapabilitiesState` | status、来源、快照摘要、观测时间；模型、工具和模块列表；`eligible_models`；每模型保留参数与兼容规则结构 | inspect capabilities 节点，来源仅 Agent API。eligible models 必须同时属于用户允许集合和服务端 ready/available 模型集合，不能混入人工结果页能力 |
| `evidence` / `EvidenceState` | status、实现状态、算法版本、有效性、`TrainStatistics`、`RiskItem`、摘要与引用 | 未来 Train Evidence 模块；第一步禁止生成输出。统计项含观测数、独立样品数、维度、类别统计、重复测量及异常特征计数；风险与客观统计分开 |
| `recipes` / `RecipesState` | direct action/recipe selection 分型；目录引用、版本、摘要；合法 RecipeReference 列表与按 use ID 合并的使用记录 | 未来配方模块；第一步仅 direct_action，无配方目录。配方引用保留模型、处理、搜索与兼容关系引用 |
| `knowledge` / `KnowledgeState` | 先验版本；匹配 entry ID/版本、来源引用、依据、适用条件、简短理由 | 未来知识匹配模块；第一步 disabled/unavailable，匹配为空 |
| `memory` / `MemoryState` | 冻结快照版本/摘要；召回 case ID/版本、依据；发布 operation ID、已发布案例 ID、写入状态 | 未来案例服务；第一步关闭。不是本任务 history，也不是 LangGraph checkpoint |
| `decision` / `DecisionState` | status、decision ID、direct_action/recipe_selection/search_selection/finalize 类型；action 或 recipe/search ID 或 selected Run；依据、简短理由、校验状态/原因、tool/response ID | LLM 选择和 finalize 提案的确定性校验节点；只保留通过校验的结构化提案，不保存模型思维链或原始回复 |
| `execution` / `ExecutionState` | experiment ID、purpose；reservation/submission key 与来源状态；稳定提交 request ID、完整安全请求及摘要；唯一权威 `run_id`、effective action/config、metadata/数据指纹、进度、Run 状态、终止原因 | prepare experiment 固定请求；submit/inspect/observe 写入后端投影。当前 API 未暴露的 reservation/key 保持 null/unavailable；科学实验与网络重放不同 |
| `budget` / `BudgetState` | 截止时间、每 operation 网络尝试/输出修复上限；runs、LLM/API calls、三种 token、wall time、model fits、epochs、resource seconds；usage 引用记录 | runs 以服务端 remaining_runs 为准；本地调用以 durable call journal 为准；token 以 provider usage 为准。未来训练成本仍 unavailable；`control_status=disabled`，不宣称已实现第七步预算系统 |
| `feedback` / `FeedbackState` | Observation 版本、状态、Validation 状态与六个有限标量、selection score；安全错误代码、进度、允许动作、完整性、下一次等待秒数；诊断/成本接口 | observe 节点的字段白名单。ready Validation 要求 Run succeeded、integrity ready。selection score 必须等于冻结指标的实际值 |
| `guard` / `GuardState` | pre/post check ID、kind、规则版本、passed/failed/pending、原因与依据；候选资格及原因列表 | 确定性契约、动作、Manifest、指标、scope/config 校验。`research_guard_status=disabled`，不冒称科研 Guard 已完成 |
| `diagnosis` / `DiagnosisState` | 已确认事实、问题类型、假设、依据引用、不确定性、限制与调整建议 | 未来诊断模块；第一步无输出，普通错误代码不进入诊断事实 |
| `replanning` / `ReplanningState` | trigger；父/子实验引用；变更项；次数/深度上限与用量；停止原因 | 未来重规划模块；第一步次数及深度明确为已知零，不产生科学重试 |
| `candidates` / `CandidatesState` | 候选 ID、同 Session Run/experiment 引用、valid/invalid/pending、Validation 指标与分数、不确定性及成本接口；推荐记录 | observe 后的有效性检查与 LLM finalize 校验节点。第一步最多一个候选；低分包括 0 不构成无效；未知不确定性为 null |
| `history` / `HistoryState` | event ID、类型、唯一递增逻辑序号、时间、decision/operation/experiment/Run 引用、安全记录引用 | 各节点生成稳定事件 ID；恢复合并去重。这里只保存索引，不保存完整日志/原始响应 |
| `recovery` / `RecoveryState` | 按 operation ID 的网络、修复、对账累计次数；pending operation；最后确认后端状态；下次唤醒；对账状态/结果/原因；人工核对标记 | prepare/recovery/inspect 节点，次数由 durable journal 汇总，不能重启归零。reconcile 不带业务 payload |
| `finalization` / `FinalizationState` | selected Run、理由、pending/confirmed/unselected、后端 Session 状态、locked_at、终止原因 | finalize 与随后 inspect 确认。confirmed 必须与 recovery 中最后回读的后端 finalized/selected Run 一致；否则不允许写 locked_at |

## 嵌套结构与安全边界

`EvaluationConfig` 只使用 `mode/train_weight/validation_weight/heldout_weight`，第一步
为 grouped holdout 8:1:1。该配置的规范化 SHA-256 是评估配置指纹；真实 Sample_ID
划分指纹保持 `null/unavailable`，不以配置哈希冒充实际划分。

`EffectiveConfig` 仅包含 model_type、normalization、class_balance、seed、
feature_selection_enabled 和上述安全 evaluation_config。它来自后端白名单 metadata，
不能复制内部配置字典。Run 的数据 SHA-256 若已返回，必须与冻结 task 指纹一致。

`SessionRequest`、`ExperimentRequest`、`FinalizeRequest` 为不同严格模型。
`PendingOperation.kind` 决定允许的 content 类型，content 指纹按完整规范化内容核验。
Session content 使用安全评估字段；Client 调用边界进行确定性 HTTP 参数转换。
Experiment content 固定 session/model/处理/rationale/client_request_id，恢复不得再次
调用 LLM 生成理由。只读操作和 reconcile 的 content 必须为 null。

`BudgetDimension` 每项都定义 `limit/reserved/actual/unknown_pending/remaining/unit/source/measurement_status`。
`unknown_pending` 是尚未核对的操作数；`actual` 是已知累计量，不能把未知消耗记为零。
`UsageRecord` 区分 pending/confirmed/unknown 调用状态及 pending/known/partial/unknown/not_applicable
token 状态，token 数可空；输入、输出、缓存及 provider 总 token 独立记录。限额和时间冻结，持久化执行
入口仍负责在实际网络调用前原子计数与拦截；State 类型不替代 durable journal 的执行控制。

`ValidationMetrics` 只定义 accuracy、balanced_accuracy、macro_precision、macro_recall、
macro_f1、weighted_f1。`SafeError` 仅包含安全 code 与 retryable；进度只容纳 stage、percent、
current、total。没有服务器路径、预测、完整结果、产物地址或原始异常的结构入口。
SafeText 还拒绝明显路径、URL、认证标记及越界结果词；它是第二道检查，不能代替调用方
对白名单字段的选择。Pydantic 公开格式化错误隐藏输入值，运行入口仍需使用安全错误分类。

后端路径、受信调用作用域和 LLM 配置在 State 中仅存 SHA-256；连接与凭据由进程内运行
依赖提供。恢复必须另外验证目标服务与调用者作用域；仅凭一个摘要并不能认证调用者。

## 初始化、冻结与 patch

`new_state(...) -> GraphState` 一次产生 21 个完整初态块，要求 dataset ID、用户允许模型、
后端摘要、调用者作用域摘要和 LLM 配置摘要；可以指定 task/thread ID、seed、指标、调用
上限、修复上限、总时限与版本。State 可表示全部未来模型名称；当前可选集合仍由 Agent
能力和用户集合交集决定。

`identity.startup_config_fingerprint` 覆盖 task 配置、versions、module_policy、预算限额/
来源/单位/截止时间以及后端/调用者摘要；`identity.runtime_config_fingerprint` 绑定 API
请求超时上限，且同样纳入启动摘要。State 反序列化时重算并核对，防止恢复参数意外
漂移。task 数据指纹及其状态是在创建 Session 后才取得的可信元数据，单独允许 pending
到 ready 或 unavailable 的一次绑定，不包括在创建前的配置摘要中；已绑定后不能更换。

`apply_patch(state, patch) -> GraphState` 先校验原状态，递归合并局部字段，再校验结果和
跨版本/冻结约束。它不原地修改传入 state。普通快照列表覆盖，不能追加成多个互相矛盾
的“当前能力”或“当前反馈”。新 pending operation ID 整体替换请求结构；相同 ID 只可
推进状态，kind、request ID、tool、content 和 content 指纹不可更改。

| 按稳定 ID 合并的集合 | 重放与冲突规则 |
|---|---|
| history.events、recipes.uses、candidates.items、candidates.recommendations | 同 ID 同内容去重；同 ID 不同内容拒绝。事件 sequence 必须唯一递增 |
| recovery.attempts | 按 operation ID 合并计数；网络/修复/对账次数只能增加，不能因恢复减少 |
| budget.usage | 同 ID 的 operation/kind/calls 不可变；pending/unknown 可推进为结算事实；confirmed 记录不可重写或回退 |
| guard.checks | pending 检查可写入最终结果；check ID、phase、kind、version 不变；已完成检查同 ID 冲突拒绝 |

除此之外，task、versions、module_policy、运行限额、deadline、开始时间以及已绑定身份
不可改变。Experiment 的完整请求、摘要、Run ID 绑定后不可变，已确认后端锁定不可变。
LLM/API 实际累计计数不能减少。

## 终止与核对

编排 `completed` 与后端 Run succeeded、Session finalized 是三个事实。只有经过后端
inspect 确认同一选择，才允许 `finalization.status=confirmed` 和 locked_at。
无候选可写编排 failed/cancelled/timed_out 与 `finalization.status=unselected`，并如实保留
后端 Session open。超时且 Run 仍 queued/running 时保留 execution 与 pending operation，
使用 reason code/needs_human_review 表达待核对；不能表示 worker 已经停止。

State 不承担定时唤醒；运行入口依据 recovery.next_wake_at 自动等待或在新进程 resume。
SQLite checkpoint 是单机持久化，不是多机 HA，也不是跨任务案例记忆。

## 验证

`agent_poc/tests/test_orchestration_state.py` 验证完整 21 块及严格字段、JSON 往返、未知版本、
冻结配置/摘要、数据指纹一次绑定、请求重放、事件去重、未知 usage 结算、次数与期限跨恢复、
未实现模块拒绝伪输出、安全哨兵、正常低分候选、后端确认锁定与无候选终止。
图执行、跨进程锁、Client/LLM 防火墙及真实服务联调需要同时参考对应模块测试与交付证据，
不能用本状态单元测试代替真实闭环验收。

## 单机持久化运行入口

`python -m agent_poc.orchestration` 提供 `start / resume / status`。启动后端网页或
`run_v2` 不会自动启动 LLM 服务。后端 API、独立训练 worker、已部署的 LLM HTTP 服务
是三个运行依赖；下列 LLM 地址和 served model ID 必须用实际部署值替换，不能把示例
当成已经验证可用的服务。本入口不安装模型、不自动启动服务器模型服务。

依赖安装使用 `agent_poc/requirements.txt` 与实际验证约束
`agent_poc/constraints-verified.txt`。本次工作区隔离解释器为
`work/langgraph-step1/venv/Scripts/python.exe`，正式运行可使用安装了这些依赖的 Python。

```powershell
python -m pip install -r agent_poc/requirements.txt -c agent_poc/constraints-verified.txt

python -m agent_poc.orchestration start --dataset-id DATASET_ID --thread-id example-session-1 --allowed-models logistic_regression,svm,random_forest --backend-url http://127.0.0.1:8000 --principal-scope local-user --llm-base-url http://LLM_HOST:LLM_PORT/v1 --llm-model SERVED_MODEL_ID --protocol json_action --storage storage/orchestration --wait

python -m agent_poc.orchestration status --thread-id example-session-1 --storage storage/orchestration

python -m agent_poc.orchestration resume --thread-id example-session-1 --backend-url http://127.0.0.1:8000 --principal-scope local-user --llm-base-url http://LLM_HOST:LLM_PORT/v1 --llm-model SERVED_MODEL_ID --protocol json_action --storage storage/orchestration --wait
```

`DATASET_ID` 必须来自该后端已登记且当前调用者可访问的数据集，不是服务器文件路径。
若实际服务支持原生工具调用，显式使用 `--protocol native_tools`；否则使用已实现的
`json_action` 协议，并在真实联调证据中如实记录。默认 `start` 每次推进到后端等待或
明确终态即返回；`--wait` 由 runner 按 next_wake_at 自动等待并继续，不消耗额外 LLM 调用。
每次等待最多 sleep 一秒，可 Ctrl+C 中断。中断保留 checkpoint，未向后端请求取消训练；
返回 JSON 包含 thread ID 供下一进程恢复。省略 thread ID 时由可信编排器生成。

SQLite checkpoint 使用独立 `storage/orchestration/checkpoints.sqlite`，调用计量使用
`calls.sqlite`。两个连接均使用 WAL/FULL；SqliteSaver 首次访问创建自身 schema，连接在
入口退出时关闭。每个无环图 tick 以 `durability="sync"` 调用；崩溃后有 pending node 时
传 `None` 恢复，已结束 tick 则传空输入启动下一 tick。不会调大 recursion limit 来轮询。
JSON serializer 禁止任意对象恢复与 pickle/msgpack 回退。

同一 thread 的 start/resume 全程持有 OS 文件锁；第二个执行者立即拒绝，进程被杀后
锁由 OS 释放。不同 thread 使用同一 SQLite 存储时仍由 SQLite 事务隔离。`status` 不要求
后端或 LLM 配置、不发起网络调用、不持有执行锁，以只读 SQLite 事务读取最新已提交的
一致快照；在途尚未 checkpoint 的状态不保证可见。查询不驱动图或修改用量/history。终态 resume 核对
身份和配置后也只返回已有状态。`--full` 可显示完整已校验 State，默认仅输出白名单摘要。

| 配置 | CLI / 环境变量 |
|---|---|
| 后端地址 | `--backend-url` / `AUTOAI_BASE_URL`，默认本机 8000 |
| 调用者作用域标签 | `--principal-scope` / `AUTOAI_PRINCIPAL_SCOPE`，必须明确提供 |
| 后端凭据 | 仅 `AUTOAI_API_TOKEN`，无 token 命令行参数 |
| LLM 地址、served model、协议 | `--llm-base-url`、`--llm-model`、`--protocol` / `AUTOAI_LLM_BASE_URL`、`AUTOAI_LLM_MODEL`、`AUTOAI_LLM_PROTOCOL` |
| LLM 凭据 | 仅 `AUTOAI_LLM_TOKEN` |
| LLM timeout、输出 token/响应字节上限 | `--llm-timeout`、`--llm-max-tokens`、`--llm-max-response-bytes` / 对应 `AUTOAI_LLM_TIMEOUT`、`AUTOAI_LLM_MAX_TOKENS`、`AUTOAI_LLM_MAX_RESPONSE_BYTES` |
| 生成参数 | `--temperature`、`--top-p` / `AUTOAI_LLM_TEMPERATURE`、`AUTOAI_LLM_TOP_P` |
| API 请求超时上限 | `--api-timeout` / `AUTOAI_API_TIMEOUT`，默认 10 秒，恢复须一致 |

入口只读这些已存在的进程环境变量，不读取 `.env`、认证缓存或文件中的凭据。后端地址
规范化后哈希；调用者摘要使用 token 作 HMAC key，把已验证 endpoint 与明确 scope 标签
作为消息。token、scope 明文和 URL 不写入 State。token 或 scope 变更时恢复失败关闭，
不会把另一个调用者的缓存接入当前任务；本机无 token 场景只绑定明确的本地 scope 标签，
不把标签当成服务端身份认证。

resume 不接受 dataset、seed、模型集合或预算替换参数。LLM 端点/模型/协议/生成配置、
实际 Prompt/投影版本、调用者和 API timeout 必须匹配原冻结摘要。软件包在加载图依赖前
强制关闭 `LANGSMITH_TRACING`、`LANGCHAIN_TRACING_V2` 和 `LANGCHAIN_TRACING`；驱动期间
再次使用禁 tracing 上下文及空 callbacks，防止环境配置把完整 State 自动发往远程服务。

## CLI 退出码

| 退出码 | 含义 |
|---|---|
| 0 | start/resume 已 completed，或非 --wait 正常返回 waiting/暂停；status 成功读取任何合法状态 |
| 1 | start/resume 返回 failed、timed_out、cancelled 或 needs_attention，无论是否 --wait |
| 2 | 参数、配置、执行锁、存储或查询错误，输出安全错误码 |
| 130 | start/resume 收到 Ctrl+C；保留 checkpoint，不自动取消后端 Run |

status 退出 0 表示查询成功，不表示实验成功；自动化脚本需读取 JSON 的 lifecycle.status。

## 已验证部署拓扑

历史真实 LLM 验收使用服务器 Qwen3-4B + 本地 HTTP 后端及独立 worker + 合成分组数据。
临时推理服务已停止，18001 不是常驻接口。此结果证明工程闭环，不证明服务器整体部署
或领域科研效果。第二轮仅本地修复与回归，目标服务器部署检查未执行。
