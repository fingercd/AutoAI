# 第二步：全模型单实验协议

新 CLI 任务默认使用 `/api/agent/v2`、`agent-session-v2`、`agent-observation-v2`、`agent-metadata-v2` 和 `agent-state-v2`。图仍为 `agent-single-experiment-v1`，每个 Session 只运行一次分类实验。旧客户端、旧检查点及人工训练保留原路径与语义；不新增第三步研究模块。

## 模型与配置

`backend/app/model_catalog.py` 是模型 ID、别名、实现状态、执行家族及依赖的轻量来源。`model_config.py` 描述实际开放的固定控制；训练器继续拥有原有搜索网格、选择指标及网络 profile。Web 不导入 AggMap 进行编译探测，`available` 仅代表当前进程的准入探测，不表示训练已经验收。

- 传统搜索模型 PLS-DA、PCA-LDA、LR、SVM 不提供虚假固定覆盖。
- Random Forest 开放树数 50–1000 与搜索次数 1–18，默认 200/10；保留 OOB 选参。
- XGBoost 仅开放非负有限 gamma，默认 0；其余参数维持原 12 组候选。
- 深度模型共享现有 epochs、batch size、learning rate、weight decay、scheduler 和 early stopping 参数；默认值复用原训练策略。
- DSCARNet 额外开放 sar/car/dual，默认 dual；缺少 AggMap 时不可提交。CNN-Mamba 保持版本未实现，不因装包而开放。
- v2 只接受 canonical ID、严格类型及本模型字段；bool 不充当整数，null、非有限数与未知字段拒绝。

Session 请求可包含 `model_configs: {model_id: {固定覆盖}}`。服务端解析默认值，冻结能力语义及每模型配置。`allowed_models` 是操作者上限；编排先与当前可用集合取交集，并记录排除原因。Session 内固定 zscore、none class balance、8:1:1 group holdout、关闭可解释性特征选择。

Experiment 的 `model_params` 省略时消费 Session 冻结对象；显式提供必须是完整且相同的对象。LLM 只选择模型；固定参数显示在上下文 `fixed_model_params` 中，由适配器注入提案，再由 Graph、Client 和服务器核对。参数不从 rationale 解析。

## 冻结、提交与恢复

Session 表增加可空协议与能力快照列；旧行仍视为 v1。预约保存 compiled config 和完整科学摘要，Run 创建复用既有提交服务、持久 mapping、预算和绑定状态机。

v2 重放先校验冻结请求，再读取已绑定 Run。mapping 已存在而预约尚未绑定时，按保存的提交配置核对并绑定原 Run，不重复创建；compensation 仍通过显式空请求 reconciliation 处理。新实验在预约前检查模型参数、策略摘要及当前可用性。无关模型新增不会使已有 Session 失效。

请求摘要包含协议与 rationale；科学摘要排除 request ID、rationale、展示名称与路径，包含冻结数据、划分、模型、固定参数及策略。接口仍给 16 位 config_hash，预约内部保存完整 SHA-256。省略默认与显式默认在 v2 等价；v1 序列化和摘要保持原样。

`effective_action` 是授权动作；`effective_config` 来自持久化 Run 的提交配置，标明 `config_stage=submission`；`resolved_execution` 只读取经过 Manifest 校验的执行审计白名单，不含指标、样品统计、路径、预测或 Test。进度只保留安全状态与真实 epoch 数；未知项不伪造。

## 检查点与 Prompt

`state_v2.py` 扩展现有 State 块，不复制 Graph。新版本使用 `agent-context-step2-v1` / `agent-decision-step2-v1`。读取检查点后选择版本类型和 LLM 摘要计算规则；旧 checkpoint 不被升级、补参数或刷新请求 ID。

Prompt 相对 `agent-decision-step1-v2` 只增加固定参数说明、候选投影及绑定逻辑，不改变模型偏好、推理策略或提供额外训练/测试证据。JSON action 与 native tools 使用同一候选和校验；真实本地 Qwen 验收使用 JSON action，native tools 只有协议测试证据。

## 启动与验收

操作者可显式选择新增模型并冻结参数，例如在现有环境变量提供服务地址、Principal 和 LLM 配置后：

```bash
python -m agent_poc.orchestration start --dataset-id DATASET_ID \
  --allowed-models cnn1d,cnn_transformer1d \
  --model-configs '{"cnn1d":{"epochs":2},"cnn_transformer1d":{"epochs":2}}' --wait
```

CLI 省略 allowed-models 时保留旧三模型上限，以兼容既有命令；新任务协议仍为 v2。显式传入 v1 Client 的程序调用保留 v1 任务创建；普通新运行默认 v2。

独立验收驱动为 `scripts/agent_step2_acceptance.py`。它要求新建隔离根目录，在所有业务导入前绑定 Web/worker 的全部路径，使用 loopback/server 模式及仅存于进程内存的随机 token。它不改变平台默认 storage 路径，不触碰历史数据库。报告含逐模型时间线、配置、Validation、Finalize、Manifest 和运行绑定。数据为工程合成夹具，不构成领域科研结论。


## 独立审查后的修正（2026-09-14）

- Graph 在每个 HTTP 执行入口前恢复检查点内的能力 Schema 和冻结 Session，不依赖 prepare 重跑。恢复只使用已经校验的快照，不请求当前 health 覆盖它。
- v2 Client 的 submit 不再隐式 GET；独立调用者应先显式 inspect Session，Graph 从可信检查点恢复本地绑定。实际每个 HTTP 都经过原 journal、调用上限和 deadline。
- durable Run/mapping 已存在时，并发 bind 的条件更新失败会按 scope 重读；仅相同预约已绑定同一 Run 时返回幂等成功。
- scheduler_factor 的合法区间为 `0 < factor < 1`。深度配置策略升级到 `agent-model-config-v2`，有限参数 Schema 增加 exclusive_maximum。旧 v1 快照保持原序列化与摘要；仅明确识别的旧调度边界策略、且固定值满足当前有效区间时允许继续未提交任务。已有 Run 的重放仍优先返回原 Run，不重建配置。
- model_configs 的键必须属于 allowed_models，允许模型必须来自服务端目录；未知参数在排除不可用候选之前校验。已知不可用模型的合法配置仍按协议记录排除原因，拼写错误不会被过滤掉。
