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
