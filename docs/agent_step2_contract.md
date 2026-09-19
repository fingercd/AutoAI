# 全模型单实验协议与训练证据配方

第二步直接动作任务使用 `/api/agent/v2`、`agent-session-v2`、`agent-observation-v2`、`agent-metadata-v2` 和 `agent-state-v2`。图仍为 `agent-single-experiment-v1`，每个 Session 只运行一次分类实验。旧客户端、旧检查点及人工训练保留原路径与语义。第三步新任务默认协商下文的训练证据配方 profile。

## 模型与配置

`backend/app/model_catalog.py` 是模型 ID、别名、实现状态、执行家族及依赖的轻量来源。`model_config.py` 描述实际开放的固定控制；训练器继续拥有原有搜索网格、选择指标及网络 profile。Web 不导入 AggMap 进行编译探测，`available` 仅代表当前进程的准入探测，不表示训练已经验收。

- 传统搜索模型 PLS-DA、PCA-LDA、LR、SVM 不提供虚假固定覆盖。
- Random Forest 开放树数 50–1000 与搜索次数 1–18，默认 200/10；保留 OOB 选参。
- XGBoost 仅开放非负有限 gamma，默认 0；其余参数维持原 12 组候选。
- 深度模型共享现有 epochs、batch size、learning rate、weight decay、scheduler 和 early stopping 参数；默认值复用原训练策略。
- DSCARNet 已退役；新任务返回 `model_retired`。历史冻结快照保留原参数和摘要，已绑定 Run / durable mapping 在验证原请求身份后回读；旧未提交任务不可继续。CNN-Mamba 保持版本未实现，不因装包而开放。
- v2 只接受 canonical ID、严格类型及本模型字段；bool 不充当整数，null、非有限数与未知字段拒绝。

Session 请求可包含 `model_configs: {model_id: {固定覆盖}}`。服务端解析默认值，冻结能力语义及每模型配置。`allowed_models` 是操作者上限；编排先与当前可用集合取交集，并记录排除原因。Session 内固定 zscore、none class balance、8:1:1 group holdout、关闭可解释性特征选择。

Experiment 的 `model_params` 省略时消费 Session 冻结对象；显式提供必须是完整且相同的对象。LLM 只选择模型；固定参数显示在上下文 `fixed_model_params` 中，由适配器注入提案，再由 Graph、Client 和服务器核对。参数不从 rationale 解析。

## 冻结、提交与恢复

Session 表增加可空协议与能力快照列；旧行仍视为 v1。预约保存 compiled config 和完整科学摘要，Run 创建复用既有提交服务、持久 mapping、预算和绑定状态机。

v2 重放先校验冻结请求，再读取已绑定 Run。mapping 已存在而预约尚未绑定时，按保存的提交配置核对并绑定原 Run，不重复创建；compensation 仍通过显式空请求 reconciliation 处理。新实验在预约前检查模型参数、策略摘要及当前可用性。无关模型新增不会使已有 Session 失效。

请求摘要包含协议与 rationale；科学摘要排除 request ID、rationale、展示名称与路径，包含冻结数据、划分、模型、固定参数及策略。接口仍给 16 位 config_hash，预约内部保存完整 SHA-256。省略默认与显式默认在 v2 等价；v1 序列化和摘要保持原样。

`effective_action` 是授权动作；`effective_config` 来自持久化 Run 的提交配置，标明 `config_stage=submission`；`resolved_execution` 只读取经过 Manifest 校验的执行审计白名单，不含指标、样品统计、路径、预测或 Test。进度只保留安全状态与真实 epoch 数；未知项不伪造。

## 检查点与 Prompt

版本类型与初始化统一在 `state.py`；`state_v2.py` 仅保留旧导入的薄适配，不复制 Graph。新版本使用 `agent-context-step2-v1` / `agent-decision-step2-v1`。读取检查点后选择版本类型和 LLM 摘要计算规则；旧 checkpoint 不被升级、补参数或刷新请求 ID。

Prompt 相对 `agent-decision-step1-v2` 只增加固定参数说明、候选投影及绑定逻辑，不改变模型偏好、推理策略或提供额外训练/测试证据。JSON action 与 native tools 使用同一候选和校验；真实本地 Qwen 验收使用 JSON action，native tools 只有协议测试证据。

## 启动与验收

操作者可显式选择新增模型并冻结参数，例如在现有环境变量提供服务地址、Principal 和 LLM 配置后：

```bash
python -m agent_poc.orchestration start --dataset-id DATASET_ID \
  --allowed-models cnn1d,cnn_transformer1d \
  --model-configs '{"cnn1d":{"epochs":2},"cnn_transformer1d":{"epochs":2}}' --wait
```

CLI 省略 allowed-models 时保留旧三模型上限。新任务默认采用下文配方 profile；第二步命令添加 `--execution-profile direct_action`。显式传入旧 Client 的程序调用保留相应版本任务创建。

独立验收驱动为 `scripts/agent_step2_acceptance.py`。它要求新建隔离根目录，在所有业务导入前绑定 Web/worker 的全部路径，使用 loopback/server 模式及仅存于进程内存的随机 token。它不改变平台默认 storage 路径，不触碰历史数据库。报告含逐模型时间线、配置、Validation、Finalize、Manifest 和运行绑定。数据为工程合成夹具，不构成领域科研结论。


## 独立审查后的修正（2026-09-14）

- Graph 在每个 HTTP 执行入口前恢复检查点内的能力 Schema 和冻结 Session，不依赖 prepare 重跑。恢复只使用已经校验的快照，不请求当前 health 覆盖它。
- v2 Client 的 submit 不再隐式 GET；独立调用者应先显式 inspect Session，Graph 从可信检查点恢复本地绑定。实际每个 HTTP 都经过原 journal、调用上限和 deadline。
- durable Run/mapping 已存在时，并发 bind 的条件更新失败会按 scope 重读；仅相同预约已绑定同一 Run 时返回幂等成功。
- scheduler_factor 的合法区间为 `0 < factor < 1`。深度配置策略升级到 `agent-model-config-v2`，有限参数 Schema 增加 exclusive_maximum。旧 v1 快照保持原序列化与摘要；仅明确识别的旧调度边界策略、且固定值满足当前有效区间时允许继续未提交任务。已有 Run 的重放仍优先返回原 Run，不重建配置。
- model_configs 的键必须属于 allowed_models，允许模型必须来自服务端目录；未知参数在排除不可用候选之前校验。已知不可用模型的合法配置仍按协议记录排除原因，拼写错误不会被过滤掉。


## 第三步：冻结训练证据与有限配方

沿用 `/api/agent/v2`，通过 `X-AutoAI-Agent-Revision: agent-recipes-revision-v1` 协商扩展。新 Session 请求同时提供 `execution_profile=train-evidence-recipes-v1`、`protocol_revision=agent-recipes-revision-v1` 和 `modules=[train_evidence, legal_recipes]`。新 profile 的读取、提交、观察、恢复和 Finalize 均需要该修订头；旧严格客户端访问时显式返回版本不兼容。旧行与旧响应不补入这些字段。

准备入口属于普通后端。`evaluation_plan.py` 使用原分层分组算法生成不可变计划，按 Principal scope 存在 Run 数据库中；固定原文件 SHA-256、轴、种子、评估配置、分区及完整摘要。对外仅公开安全引用，不公开行号或样品分组。worker 在拟合之前再次核对文件、计划与配置，并消费保存的索引。新人工 grouped holdout 请求也经过同一准备入口；CV 和独立测试集保留原执行路径。

`train_evidence.py` 仅接收原始 float32 TrainView，不能访问 Dataset、数据库或网络。统计包括观测数与样品组数、类别分布、重复测量、常量列、重复列、重复向量及标签冲突。重复项用哈希分桶后精确比较；汇总使用确定性规则。Validation/Test 值不参与统计。证据风险只作说明，不按风险筛模型；确实无法拟合的硬维度条件会留下明确排除原因。

`recipes.py` 按冻结能力与固定参数为每个可执行允许模型编译一个配方，继续引用原有搜索策略，不复制搜索网格。配方包含预处理、结构版本、固定配置、模型策略与所选模型的执行依赖摘要；无关模型新增不使它失效。完整配方摘要排除 rationale、展示文本、请求/Session/Run ID 和路径。目录摘要绑定数据、计划、证据、允许模型及其冻结策略。当前源码绑定跟踪 Python AST 的静态符号及相对导入；新增动态导入或模块属性间接调用时必须同步扩展绑定并测试。

Session 与准备结果原子冻结。Experiment 只接受 `recipe_id`、`recipe_digest`、`catalog_digest`、`rationale`、`client_request_id`；不能混入自由模型参数。普通人工请求不能伪造 `execution_recipe_digest`、`execution_catalog_digest`、`execution_evidence_digest`、`execution_search_digest`。已有 Run / durable mapping 仍优先幂等回读；没有执行记录时才检查当前模型可用性和搜索策略漂移。后端返回配置前核对持久 Run 与冻结配方的绑定。

新检查点为 `agent-state-v3`，仍使用同一套 21 组 State、Graph、journal 和恢复入口；旧 v1/v2 序列化与历史摘要保持不变。Evidence/recipes 是已准备内容；LLM 只能看到安全统计、风险、有限配方及随后真实 Validation。JSON action 与 native tools 使用等价的 recipe_id 约束。`--hide-evidence-context` 和 `--hide-risk-context` 只隐藏 LLM 上下文，后端仍计算并冻结相同证据，不改变候选或后端行为。

```bash
python -m agent_poc.orchestration start --dataset-id DATASET_ID \
  --allowed-models logistic_regression,svm,random_forest --wait
python scripts/agent_step2_acceptance.py --backend-only --root /tmp/autoai-step3-native-UNIQUE
python scripts/agent_step2_acceptance.py --recipes \
  --root /tmp/autoai-step3-UNIQUE --llm-url http://127.0.0.1:18762/v1
```

准备限制由 `AUTOAI_PREPARATION_SECONDS`、`AUTOAI_PREPARATION_MAX_BYTES`、`AUTOAI_PREPARATION_BLOCK_SIZE` 控制。超时、资源不足、数据或计划不合法会显式失败，不将缺失证据记为零。验收驱动复用原 Web/worker 生命周期；`--backend-only` 场景不启动 Agent/LLM，原生请求不携带配方或计划引用，并强制检查 Agent 表无记录。另在新隔离目录完成真实 LLM 传统/深度/恢复场景，并用独立普通 HTTP 进程提交相同配方配置与计划；比较实际分区、执行审计和模型 profile，传统模型额外以固定容差比较指标与预测。普通提交不新增 Agent Session 或预约；worker 与普通请求进程阻断 Agent/LLM 导入。Web 仍托管原 Agent 路由。产物完整性、退出清理和运行版本均进入验收记录。


## 第四步：静态知识与引用

新 CLI 的配方任务使用 `agent-state-v4`，同一 v2 API 协商 `agent-recipes-revision-v2`。`--knowledge on|off` 映射唯一的 `module_policy.knowledge`，默认 off；直接动作模式不允许 on。开启请求按固定顺序发送 `modules=[train_evidence,legal_recipes,knowledge]`，关闭只发送前两个。旧 revision-v1 的请求、响应、请求 hash 和 checkpoint 不增字段。

知识仅提供可质疑建议，不改变 recipe/catalog、搜索、训练或科学实验摘要。`backend/app/knowledge_data/common_modeling_v1.json` 冻结五条通用知识及 scikit-learn 1.5.2 来源；不从文件名或 source_role 猜测测量领域。条件仅使用 Train 统计、风险和合法模型集合，样本规模采用独立 Sample_ID 组数。匹配版本为 `knowledge-match-v1`，投影为 `knowledge-projection-v1`，快照为 `knowledge-snapshot-v1`。条目顺序规范化后计算 SHA-256；优先级只决定展示顺序，不是科学置信度。

开启时 Session 同事务保存完整发布知识集、匹配、实际投影及 Evidence/plan/catalog/dataset 来源绑定。知识文件仅在创建新开启 Session 时装载；`AUTOAI_KNOWLEDGE_FILE` 是部署配置，HTTP 不接受知识路径。进程缓存不可变发布对象；更新发布使用新路径或显式清除加载缓存并重启部署进程，旧 Session 继续使用已存快照。关闭不装载文件；无匹配为 ready+空集合。装载失败显式报错；损坏/未知存储快照不能当作无匹配。

持久模块声明包含 train_evidence、legal_recipes 或 knowledge 时，记录解码必须存在冻结准备包。整包为 SQL NULL、JSON null 或空串时，读取、创建请求回放及实验提交统一返回不可重试的 409 / agent_preparation_failed，不降级 direct、不重建快照，也不新增预约或 Run。真正旧 direct Session 不含这些模块，继续兼容空准备包。

响应只给安全知识摘要和投影，不返回完整发布知识集。最多展示6条、canonical JSON UTF-8最多16 KiB；按稳定顺序逐完整条目或完整冲突组选择，保留建议、依据与局限，记录省略计数。关闭直接 evidence/risks 展示时，知识不显示对应具体统计/风险码；条目本身仍可能间接传达 Train 信息，完全移除信息须同时关闭 knowledge。

新 submit 参数必须含 `knowledge_refs` 数组，允许空、最多6个、不得重复，只能引用本次实际展示 ID。JSON action 顶层保持 tool_name/arguments/rationale，native tools 使用同一规则；Finalize 参数保持原样。引用版本由冻结投影解析，写入 decision.evidence_refs 并绑定 projection_digest。HTTP 服务再次校验，并将 `knowledge-decision-v1` 写入预约独立的 `decision_metadata_json`；不混入 effective_action/compiled_config/scientific_digest。客户端还核对提交回执的引用集合、版本和摘要与实际请求一致。

```bash
python -m agent_poc.orchestration start --dataset-id DATASET_ID   --allowed-models logistic_regression,svm --knowledge on --wait
python scripts/agent_step2_acceptance.py --recipes --knowledge-ablation   --root /tmp/autoai-step4-UNIQUE --llm-url http://127.0.0.1:18762/v1 --llm-model qwen3-4b
```

受控协议测试与真实 LLM 验收分开报告。知识消融驱动固定数据、种子和候选执行 on/off，要求真实 on 实际引用已展示知识；分别为选中配方运行普通后端配对，检查参数、实际划分、搜索审计、指标/预测容差及 Manifest。失败状态与调用计量保留，不能用脚本选择器冒充真实 LLM。

## 同域表达与轻量消融（2026-09-15）

当前实现集中在无版本后缀业务模块；v1/v2 请求类、State v1-v4 与真实 wire 字面量仍为历史解码边界。客户端能力、执行响应、准备包和知识分别由 `capabilities.py`、`execution_contracts.py`、`preparation.py`、`knowledge.py` 承载。两个 Agent URL 前缀在同一个 router 模块内调用同一 Service。

新 CLI 配方任务默认冻结 `decision_mode=recipe_id`，也可选 `structured_config`。该字段是 `agent-recipes-revision-v2` 的显式可选扩展；缺字段的历史请求、State、启动摘要和 hash 保持原样，语义按 recipe_id。显式模式仅用于当前知识协议，模式进入 Session、任务启动摘要和 prepared 内容，resume 不提供更改入口。`restricted_strategy_pool` 始终表示硬有限域。

structured_config 的 LLM 提案包含 `session_id, model_id, normalization, class_balance, model_params, knowledge_refs, rationale`，不要求配方 ID。HTTP 请求使用同名执行字段（无 session_id，另有 client_request_id）；参数必须完整匹配冻结成员，normalization 固定 zscore，class_balance 固定 none。Graph 内部将完整表达绑定到规范 recipe_id，再由 Client 按冻结模式生成原始 HTTP 请求。服务端两表达解析成同一个 ExperimentCommand，复用预约、worker、恢复和 Finalize；科学摘要相同，原始请求 hash 可不同。旧 direct_action 不参与该对照。

两表达知识引用规则相同：K-off 只能空引用；K-on 仅能引用冻结投影中提供的条目。E/R 只控制各自展示，硬合法域不变；K-on 可间接提供 Train 信息，不能将 E/R-off 称为完全无 Train 信息。软 Evidence 筛选为 not_applicable，动态处理为 unavailable。

### 单行实验命令与最小记录

`python scripts/agent_ablation.py` 针对已运行的服务执行一行并保存到 `work/ablation-records/<experiment-id>/record.json`。相同 ID 再次运行沿用冻结配置和原 checkpoint/journal；科研重复必须使用新 ID。Linux 上以 OS 文件锁避免并发重复。普通训练 API 没有幂等键：基线在 POST 前持久化 attempt_started，绑定 Run ID 后才继续；丢响应或在绑定前崩溃写 submission_uncertain 并停止，必须人工核对，禁止自动再次 POST。

以下示例假定 Web/worker 已运行、数据已上传，`DATASET_ID` 为上传返回值；Bearer 仅由环境注入，不写入命令参数、文件或日志。LLM 模型和协议显式指定。固定基线必须先执行，再执行 Agent 对照。

```bash
PY=/users/fotile/work/autoai-server-acceptance-20260912/venv/bin/python
COMMON=(--backend-url http://127.0.0.1:18771 --dataset-id "$DATASET_ID" --scope-key step3-acceptance --seed 42 --allowed-models logistic_regression svm)
LLM=(--llm-url http://127.0.0.1:18762/v1 --llm-model qwen3-4b --protocol json_action)
$PY scripts/agent_ablation.py "${COMMON[@]}" --experiment-id fixed-lr-001 --kind baseline
$PY scripts/agent_ablation.py "${COMMON[@]}" "${LLM[@]}" --experiment-id recipe-001 --kind agent --decision-mode recipe_id --knowledge off
$PY scripts/agent_ablation.py "${COMMON[@]}" "${LLM[@]}" --experiment-id structured-001 --kind agent --decision-mode structured_config --knowledge off
$PY scripts/agent_ablation.py "${COMMON[@]}" "${LLM[@]}" --experiment-id plain-001 --kind agent --decision-mode structured_config --knowledge off --hide-evidence-context --hide-risk-context
```

Evidence 关闭：在对应新实验命令增加 `--hide-evidence-context`；风险关闭：增加 `--hide-risk-context`；知识对照：仅切换 `--knowledge on/off`。保持其余配置、数据、seed、候选、固定参数和预算一致。Plain Agent 标为“同域结构化 Plain Agent”。每模型固定参数通过 `--model-configs <JSON文件>` 在 Session 前提供，不能看结果后更改同一个实验。

完整隔离工程联调复用原验收脚本（真实本地 LLM 服务须已就绪）：

```bash
$PY scripts/agent_step2_acceptance.py --recipes --light-ablation --root /tmp/autoai-step4-ablation-UNIQUE --port 18771 --llm-url http://127.0.0.1:18762/v1 --llm-model qwen3-4b
```

该命令创建自有 Web/worker、确定性合成分类数据，依次执行固定 LR、recipe Agent、structured Agent，并核对同数据/实际划分/固定参数以及终态不重复提交。它是实际 HTTP/worker/LLM 工程联调，不能称为真实业务数据验收或科研贡献证明。原 `--recipes` 中 Agent 选后普通 API 重放在记录中标为 `backend-equivalence`，不是独立科研基线。

记录包括实验/配置摘要、源码摘要、数据/实际划分摘要、scope 绑定、seed、候选固定参数、E/R/K、表达模式、Prompt/LLM 摘要、Session/thread/Run、状态原因、validation、调用/token/耗时及产物引用。不可计量字段标 unknown。Test 仅在固定基线完成或 Agent Finalize 后离线提取汇总；不进入 Prompt、基线选择或逐样品公开投影。

### 独立核验补正：源码与计量覆盖范围

消融 CLI 每次执行均获取当前 `head / dirty_diff_sha256 / source_digest`，在网络调用前写入持久 attempt。恢复要求与 record.source 的起始绑定完全一致；变化时记录 `rejected_source_change` 并退出，不覆盖历史来源，也不发送 HTTP 请求。此绑定只代表 CLI 本地源码，每个 attempt 的 Web/worker 版本默认 `unknown`，不能据此声称远端代码相同；隔离验收清单中的独立 Web/worker 证据仍单独保留。

每次 attempt 在 I/O 前写 running，正常 finally 写 settled 和本次 request-hook 计数、耗时。下次发现 running 表明上次没有结算，转 interrupted/incomplete。已知计数和耗时保留在 measurement.known_http_calls / known_elapsed_seconds；有中断或旧记录未核实区间时，累计 record_http_calls、普通基线 api_calls、elapsed_seconds 为 `unknown`，不把缺失区间计为零。Agent 的执行调用数仍由原 Graph journal 计量；recording client 的 HTTP 计量另列 scope。

旧记录没有持久 attempt 标记，首次通过源码检查的恢复会保留旧测量值并标 legacy unverified；不能反推旧运行完整性。严格源码冻结也适用于修复前创建的记录：它们不会由新代码继续执行。同源且已绑定 Run 的恢复只轮询原 Run；POST 绑定前丢响应仍停在 submission_uncertain，需人工核对，绝不自动重提。该修正不改变普通 API 幂等协议或训练流程。


## 正文向量知识检索（2026-09-19）

新 knowledge-on Session 使用 `knowledge-snapshot-rag-v1` / `body-cosine-topk-v1`；旧 `knowledge-snapshot-v1` 只按原规则和原摘要读取，不重新检索。knowledge-off 不读取卡库、不加载 transformers。on 的模型或发布包故障返回 503 `agent_knowledge_unavailable`，没有合法候选则是 ready 空集，不回退条件匹配。

`knowledge_query` 是可选 Session 请求字段：`{"query_mode":"train_template","user_text":null,"domain":null}`。省略时使用 Train 模板且不改变旧请求的幂等摘要。`user_text` 模式必须显式给非空文本；domain 只接收已确认领域，不由文件名推断。查询只使用 Train 的组数、观测数、特征数、类别组数和重复测量；原查询与相似度不进入 Agent 上下文。E/R-off 仍可能通过检索结果间接传递 Train 信息。

模型固定为 BAAI/bge-small-zh-v1.5 revision `7999e1d3359715c523056ef9478215996d62a620`。正文独立编码，不加前缀；查询使用固定中文前缀。CPU float32、CLS pooling、eval/no_grad、L2 归一化，超过 512 token 拒绝而不截断。发布包包含 cards.json、embeddings.npy、index_manifest.json；校验 hash、行映射、维度及单位范数后原子切换 current.json。新版本使用新目录，已有 Session 使用冻结内容恢复。

安装可选 `backend/requirements-knowledge.txt`，不将其加入普通后端强制依赖。离线发布示例：

```bash
python scripts/build_knowledge_index.py --cards backend/app/knowledge_data/modeling_cards.json --model work/models/bge-small-zh-v1.5 --output work/knowledge --download-model
export AUTOAI_EMBEDDING_MODEL="$PWD/work/models/bge-small-zh-v1.5"
export AUTOAI_KNOWLEDGE_BUNDLE="$PWD/work/knowledge"
```

只有显式 `--download-model` 才访问公开模型站点，运行服务不下载模型。只发布 status=published 的卡片；同正文复用向量，但每行仍绑定 ID/version/body 摘要。当前迁移了 5 条原有知识，未达到正式 20–50 条规模；不将合成测试卡计入正式库。

检索仅做一次 NumPy 精确点积，默认 Top-3，硬上限 6；按分数降序和 ID 打破同分。当前 tau=null，不能保证无关查询返回空集。只按 scope、任务、确认领域及合法模型关联过滤；正文保留前提和限制，不将相似度解释成科学置信度。

CLI 支持 `--query-mode train_template|user_text`、`--query-text` 和 `--confirmed-domain`。新 RAG（`--knowledge on`）启动必须提供 `--llm-tokenizer /path/to/local/model --llm-context-window 32768`，也可设置 `AUTOAI_LLM_TOKENIZER` / `AUTOAI_LLM_CONTEXT_WINDOW`；消融脚本对应参数为 `--llm-tokenizer` / `--context-window`。启动前验证本地 tokenizer 文件、chat template 和实际计数能力，缺失返回 `rag_prompt_budget_configuration_required`，不可用返回 `rag_prompt_budget_configuration_invalid`，均不创建 Session、不请求 LLM。注入适配器须绑定同一配置。历史检查点仍按原配置指纹恢复，不追补新启动约束。配置后使用实际 chat template 对完整消息、工具 schema 和生成预留计数，必要时整卡减少，不截断正文或模型目录。完整实际展示上下文与 digest 写入现有 proposal journal，重放必须一致。预算不足是终止性 `llm_context_too_long`。

`knowledge_refs` 可以为空；JSON action 和 native tools、recipe_id 和 structured_config 均使用实际提供的 ID/version。查询、检索分数、Embedding 版本不进入训练有效配置或科学摘要。`scripts.agent_ablation.summarize_plan` 仅离线汇总 Finalize 后的指标，失败保留计划分母；成功子集均值与完整均值分别报告。
