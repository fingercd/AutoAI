# AutoAI 三模型本地 LLM 与 Agent POC 服务器实施计划

> 日期：2026-08-20\
> 计划对象：当前 `master`（基线提交 `ab8362e`）及本次上传的工作树快照\
> 最终目标：`Qwen3.5-9B`、`Qwen3.5-27B`、`Qwen3.8-27B` 均通过同一 OpenAI-compatible 客户端驱动 AutoAI，完成真实 Dataset 的 Session → Experiment → Worker → Validation Feedback → Finalize 闭环。

## 0. 执行规则与唯一 Todo List

任何 Codex/Agent/人工执行者开始工作时，都必须先复制下面四项作为唯一主进度表；只能在该 Task 的代码、操作和验收证据全部完成后打勾。不得把“命令执行过”当成“验收通过”。

- [ ] **Task 1：冻结当前代码与服务器部署基线，建立可复现的 AutoAI 部署目录**
- [ ] **Task 2：在 node2/node3 完成三套 Qwen 模型的统一量化、vLLM 服务和 OpenAI-compatible 验收**
- [ ] **Task 3：按当前真实后端契约实现薄 Agent Adapter 与低耦合 Agent POC**
- [ ] **Task 4：用三个 backbone 分别完成 AutoAI 真实端到端闭环，并形成统一验收报告**

执行边界：

- 默认由一个主执行者按 Task 1 → 4 顺序推进，不强制子代理、分支、提交或推送。
- 不读取或上传 `.env`、`auth.json`、token、密钥、认证缓存、`.sandbox-secrets`。
- 不上传 `storage/`、`data.csv`、模型、训练产物、缓存、压缩包或本地环境。
- 不抢占他人的 GPU；每次启动前重新检查 GPU 进程和显存。GPU 不足属于资源等待，不得通过杀进程或改用他人显卡绕过。
- Web、Worker、Agent Adapter 与模型服务只能通过 HTTP/JSON 交互；LLM 不直接访问 SQLite、`RunRepository`、Worker、文件系统、Shell 或 Test 产物。
- 搜索阶段只向 Agent 暴露 Validation。Test 只允许在 Finalize 后用于一次最终报告，不能回流到下一次决策。

## 1. 已核实事实与关键修正

### 1.1 三个模型的正确标识

本阶段固定为：

| key | 官方 Hugging Face ID | 角色 | 初始服务端口 | 初始 TP |
|---|---|---|---:|---:|
| `qwen35_9b` | `Qwen/Qwen3.5-9B` | 快速开发、协议 smoke | 8101 | 1 |
| `qwen35_27b` | `Qwen/Qwen3.5-27B` | 中等规模主 backbone | 8102 | 2 |
| `qwen38_27b` | `Qwen/Qwen3.8-27B` | 新一代 Agent backbone | 8103 | 2 |

第三个模型必须从旧草案的 `Qwen/Qwen3.6-35B-A3B` 改为 **`Qwen/Qwen3.8-27B`**。官方仓库已确认存在，模型卡声明 27B language model、原生 262,144 context，并兼容 vLLM。官方仓库当前约 55.6 GB；下载与部署必须固定实际 revision，不使用名字相近的第三方仓库。

官方来源：

- <https://huggingface.co/Qwen/Qwen3.5-9B>
- <https://huggingface.co/Qwen/Qwen3.5-27B>
- <https://huggingface.co/Qwen/Qwen3.8-27B>

### 1.2 当前 AutoAI 代码的真实状态

当前 FastAPI 在 `backend/app/main.py` 只注册 `catalog`、`auth`、`datasets`、`preprocess`、`runs` 五组路由。已存在并可复用：

- `POST /api/datasets/upload`：上传并登记受 Principal 约束的 Dataset。
- `POST /api/training/runs`：只创建 SQLite queued Run，不直接训练。
- `GET /api/training/runs/{run_id}`：读取 Run 状态。
- `GET /api/training/runs/{run_id}/result`：读取完整 `run-result-v1`。
- `backend.app.runs.worker`：独立领取 queued Run、执行训练并写 Manifest。
- `GET /api/models`：15 项分类模型目录，当前环境通常 14 项可训练。

当前代码**不存在**：

- `POST /api/agent/sessions`
- `POST /api/agent/sessions/{session_id}/experiments`
- `GET /api/agent/sessions/{session_id}/experiments/{run_id}/feedback`
- `GET /api/agent/sessions/{session_id}`
- `POST /api/agent/sessions/{session_id}/finalize`
- `GET /api/agent/capabilities`
- `GET /api/datasets/{dataset_id}/readiness`

因此不能按旧草案假设“V1 Adapter 已完成”。Task 3 必须先实现薄 Adapter，并用测试证明它不会泄露 Test/artifact/server path。

### 1.3 当前服务器事实（2026-08-20 实测）

网络拓扑：Windows 笔记本 → Tailscale 跳板机 → node2/node3；SSH config 的 ProxyJump 可用。

| 项目 | node2 | node3 |
|---|---|---|
| 主机名 | `ibnode2` | `ibnode3` |
| GPU | 8 × V100 32 GB | 8 × A100 PCIe 40 GB |
| 主要用途 | 联网下载/环境准备 | 模型推理、AutoAI 训练与最终验收 |
| `/users` | NFS：`ibnode3:/users` | 本机 XFS `/users` |
| `/users` 可用空间 | 约 1.3 TB（83% 已用） | 同一份数据 |
| Python/Conda | `/users/fotile/miniconda3`，Python 3.9 base | 同一共享安装 |
| Git/rsync | 可用 | 可用 |

node3 当前 GPU 0–7 均有计算进程；GPU 3、6、7 接近满载，当前不能直接部署三个模型。node2 多数 V100 也有其他用户进程。最终同时在线按初始方案需要 5 张 A100；若无法一次获得 5 张卡，先做逐模型端到端验收，三端点同时在线验收必须等到明确的 5 卡窗口。

node2 可解析 Hugging Face 且存在 `HTTP_PROXY`/`HTTPS_PROXY`，但本次 30 秒 HTTPS 探针没有完成，完整下载前必须解决/确认代理链路；不能把“环境变量存在”当成网络可用。

## 2. 目标部署拓扑

统一根目录：`/users/fotile/AutoAI`。该路径位于 node2/node3 共享的 `/users`，所有 checkpoint 只下载/量化一次。

```text
/users/fotile/AutoAI/
├── deployments/
│   └── 20260820-poc-bootstrap/
│       └── source/                  # 当前工作树快照；不包含 storage/data/secret
├── current -> deployments/<release>/source   # Task 1 校验后才创建/切换
├── shared/
│   ├── models/
│   │   └── qwen/
│   │       ├── qwen3.5-9b-int8/
│   │       ├── qwen3.5-27b-int8/
│   │       └── qwen3.8-27b-int8/
│   ├── hf-cache/
│   ├── storage/                     # AutoAI SQLite、uploads、runs；独立于代码发布
│   ├── traces/                      # Agent session trace
│   └── manifests/                   # model revision、量化、hash、环境清单
├── envs/
│   ├── autoai-app/                  # AutoAI Web/Worker，Python 3.12
│   └── qwen-serving/                # vLLM/量化工具，避免污染训练环境
├── runtime/                         # PID、端口、非敏感 runtime mapping
├── logs/
└── incoming/                        # 上传包；校验后可保留，禁止含敏感数据
```

服务拓扑：

```text
本地浏览器/POC
  └─ SSH tunnel
      ├─ 8000 -> node3 AutoAI FastAPI
      ├─ 8101 -> node3 Qwen3.5-9B vLLM
      ├─ 8102 -> node3 Qwen3.5-27B vLLM
      └─ 8103 -> node3 Qwen3.8-27B vLLM

AutoAI FastAPI :8000 ── SQLite queue ── AutoAI Worker
Agent POC ── HTTP ── Thin Agent Adapter ── existing Dataset/Run service
Agent POC ── OpenAI client ── selected vLLM endpoint
```

默认只监听 `127.0.0.1`，通过 SSH tunnel 访问，沿用 local mode；如未来改成 `0.0.0.0`，必须启用 server mode、至少 32 字符 token、明确 CORS 与 TLS 终止。本 POC 不把 token 写入配置、命令行、日志或仓库。

## Task 1：冻结代码与服务器部署基线

**交付物：** 一个可复现、可校验、不会覆盖用户数据的 AutoAI 源码部署；服务器目录、环境和资源事实写入清单。

**涉及文件或模块：**

- 当前工作树全部代码/文档，但排除 `.git/`、`storage/`、`work/`、`outputs/`、`data.csv`、缓存、模型与秘密文件。
- 创建服务器目录 `/users/fotile/AutoAI/...`。
- 创建：`shared/manifests/source-20260820-poc-bootstrap.txt`（提交、dirty 文件、包 hash、文件数）。
- 后续修改：`deploy/server_deploy.md`（从历史 `~/AutoAI-v2` 统一到新的共享发布布局）。
- 后续创建：`deploy/poc/check_cluster.sh`、`deploy/poc/start_autoai.sh`、`deploy/poc/stop_autoai.sh`。

**Todo：**

- [ ] 保存 `git rev-parse HEAD`、`git status --short`，明确本包同时包含用户未提交改动，不能声称是纯提交产物。
- [ ] 创建 `/users/fotile/AutoAI/{deployments,shared/{models/qwen,hf-cache,storage,traces,manifests},envs,runtime,logs,incoming}`。
- [ ] 生成排除敏感/大文件的压缩包，计算 SHA-256 后用 `scp` 上传。
- [ ] 在 `deployments/20260820-poc-bootstrap/source` 解压；远端再次计算包 SHA-256，必须与本地一致。
- [ ] 远端检查不得出现 `.env`、`auth.json`、`.sandbox-secrets`、`storage`、`data.csv`、`.git`、模型扩展名或 Python cache。
- [ ] 统计源文件数、大小并运行 `git status` 对照清单；只有校验通过才允许创建 `current` 软链接。
- [ ] 创建 Python 3.12 的路径式环境 `envs/autoai-app`，在 node2 安装依赖，在 node3 验证 `torch.cuda.is_available()`；不要复用 base Python 3.9。
- [ ] 将 AutoAI runtime storage 指向 `shared/storage`。实现时优先新增受测试保护的 `AUTOAI_STORAGE_DIR`，不要把部署绝对路径写死进 `backend/app/paths.py`。

**关键命令：**

```bash
# node3：资源与目录
nvidia-smi
df -h /users
mkdir -p /users/fotile/AutoAI/{deployments,shared,envs,runtime,logs,incoming}

# node2：共享环境（执行前确认代理与磁盘）
conda create -y -p /users/fotile/AutoAI/envs/autoai-app python=3.12
conda run -p /users/fotile/AutoAI/envs/autoai-app \
  python -m pip install -r /users/fotile/AutoAI/current/backend/requirements.txt \
  -c /users/fotile/AutoAI/current/backend/constraints-verified.txt

# node3：导入验收
conda run -p /users/fotile/AutoAI/envs/autoai-app \
  python -c "from backend.app.main import app; print(app.title)"
conda run -p /users/fotile/AutoAI/envs/autoai-app \
  python -c "from backend.app.runs.worker import RunWorker; print(RunWorker.__name__)"
```

**验收标准：**

- 远端源码与上传清单一致，无秘密、数据、训练产物或模型文件。
- `current` 指向一个具体 release，不直接指向可变的 home 目录。
- Web 与 Worker 使用同一 release、同一 Python 3.12 环境和同一 `shared/storage`。
- 不使用当前 `deploy/install_on_node2.sh` 的默认 Python 3.11，也不使用 `run_on_node3.sh` 的默认 GPU 6；两者在更新前只能作为历史参考。

## Task 2：三模型下载、统一量化与 vLLM 服务

**交付物：** 三个固定 revision、同一量化口径、同一 vLLM 版本的 OpenAI-compatible endpoint；每个模型均有 machine-readable manifest 与 smoke 结果。

**涉及文件或模块：**

- 创建：`agent_poc/config/models.toml`（只记录 model key、served name、endpoint、TP、context；不记录 secret）。
- 创建：`deploy/poc/download_model.sh`（node2，固定 HF ID/revision、断点续传）。
- 创建：`deploy/poc/quantize_model.py`（统一 W8A8 校准入口，保存 calibration/config/revision）。
- 创建：`deploy/poc/serve_model.sh`（node3，统一 vLLM 参数模板）。
- 创建：`deploy/poc/smoke_openai.py`（`/v1/models`、中英文、JSON schema、重试与 latency）。
- 生成：`shared/manifests/<model-key>.json`。

**量化决策门：**

本计划保持原约束：目标为统一 INT8/W8A8，不混用 4-bit。新一代 hybrid DeltaNet/vision 架构能否被当前 vLLM + LLM Compressor 对三个 checkpoint **一致、可靠**支持，必须用实际版本和量化 smoke 验证，不能仅凭 A100 支持 INT8 推断。

统一顺序：

1. 查官方模型卡与 vLLM recipe，记录推荐的最低 vLLM/Transformers 版本。
2. node2 只下载 `config.json`、tokenizer、generation config 和权重索引，确认 ID、revision、文件总量。
3. 先对 `Qwen3.5-9B` 做小校准集 W8A8 dry run 与完整量化。
4. 验证 logits 有限、JSON 能正常闭合、中文/英文输出正常，再量化两个 27B。
5. 若任一模型的关键层不支持 W8A8、量化后无法稳定加载或结构化输出明显失效，立即停止并写 blocker：具体算子、框架版本、错误日志、可行替代和公平性影响。未经决策不得把一个模型改成 W4A16/INT4。

**Todo：**

- [ ] node2 用 IPv4/代理做 `config.json` 200 探针；记录 DNS、HTTP code、时延，不输出代理值。
- [ ] 创建 `envs/qwen-serving`，固定 Python、PyTorch、CUDA-compatible vLLM、Transformers、OpenAI SDK、LLM Compressor 版本；运行 `pip freeze` 到 manifest。
- [ ] 分别执行 metadata-only 下载并固定 HF commit SHA；确认许可和模型类型。
- [ ] 使用 `HF_HOME=/users/fotile/AutoAI/shared/hf-cache`，权重目标只能是 `shared/models/qwen`。
- [ ] 按 9B → 3.5-27B → 3.8-27B 顺序下载、量化、校验；支持断点续传，不重复下载到 node3。
- [ ] 每个模型先以 `--language-model-only`（若该版本 vLLM recipe 支持）和 `--max-model-len 32768` 做文本 Agent POC，关闭不需要的 vision encoder，给 KV cache 留余量。
- [ ] 记录每个服务的 `served_model_name`、revision、quantization、TP、GPU mapping、vLLM version、启动参数、加载峰值显存、空闲显存、首 token latency、输出速度。
- [ ] 每个 endpoint 通过 `/v1/models`、中英文 completion、严格 JSON decision、连续 20 次请求与一次 32K 边界 smoke。
- [ ] 三端点同时在线前重新检查 5 张 A100 均为空闲且使用者确认；端口 8101–8103 必须未占用。

**统一启动参数基线（以实际 recipe 支持为准）：**

```bash
CUDA_VISIBLE_DEVICES=<allocated_gpu_ids> \
HF_HOME=/users/fotile/AutoAI/shared/hf-cache \
conda run -p /users/fotile/AutoAI/envs/qwen-serving \
vllm serve <quantized_checkpoint> \
  --host 127.0.0.1 \
  --port <8101|8102|8103> \
  --served-model-name <qwen35_9b|qwen35_27b|qwen38_27b> \
  --tensor-parallel-size <1|2|2> \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser qwen3
```

不在计划阶段硬编码 `--quantization` 值；必须以量化产物的 config 与所锁定 vLLM recipe 为准，避免重复量化或错误 backend。

**验收标准：**

- 三个 manifest 均记录官方 HF ID 与不可变 revision。
- 三个服务只改变配置即可由同一 OpenAI client 调用。
- 三者量化口径一致；不存在静默 4-bit、第三方未知 checkpoint 或模型别名冒充。
- 20 次 structured-output 请求各自成功率 100%；失败必须有原始响应和解析错误 trace。
- 三服务部署不会占用为 AutoAI Worker 预留的 GPU；最终布局至少保留 1 张卡给真实训练。

## Task 3：实现薄 Agent Adapter 与低耦合 Agent POC

**交付物：** 后端只增加受控 Agent HTTP 契约；Agent 决策循环位于独立 `agent_poc/`，不直接依赖后端 Python 对象。

**后端涉及文件：**

- 创建：`backend/app/agent/contracts.py`：Session、Experiment、Feedback、Finalize 的 Pydantic 契约。
- 创建：`backend/app/agent/repository.py`：单独 `storage/agent.sqlite3`，保存 session、experiment 绑定和 budget；按 Principal scope。
- 创建：`backend/app/agent/service.py`：调用既有 Dataset/Run service，并从 `run-result-v1` 生成 Validation-only feedback。
- 创建：`backend/app/routers/agent.py`：五个 V1 endpoint 与可选 capability endpoint。
- 修改：`backend/app/main.py`：只做 router 注册。
- 修改：`backend/app/paths.py`：增加可配置 storage 根与 Agent DB/trace 路径。
- 测试：`backend/tests/test_agent_api.py`、`backend/tests/test_agent_feedback_security.py`。

**POC 涉及文件：**

- 创建：`agent_poc/main.py`：CLI 入口，参数为 `--model-key --dataset-id --max-runs --selection-metric`。
- 创建：`agent_poc/config/models.toml`、`agent_poc/config/agent.toml`。
- 创建：`agent_poc/clients/llm_client.py`：唯一 OpenAI-compatible client。
- 创建：`agent_poc/clients/autoai_client.py`：唯一 AutoAI HTTP client，含 timeout/retry/poll。
- 创建：`agent_poc/schemas.py`：`RUN_EXPERIMENT | FINALIZE | REQUEST_HUMAN` 判别联合。
- 创建：`agent_poc/prompts.py`：版本化 system/user prompt；明确 Validation-only、budget、禁止重复配置。
- 创建：`agent_poc/policy.py`：解析、校验、fallback、重复动作检查、max-runs 硬门禁。
- 创建：`agent_poc/state.py`：不变 state model；不把 Test 放入 observation。
- 创建：`agent_poc/trace.py`：JSONL 原子追加、脱敏、schema_version。
- 测试：`agent_poc/tests/`：fake LLM、fake AutoAI、超时、非法 JSON、重复实验、失败 Run、Finalize。

**Agent V1 契约：**

```text
POST /api/agent/sessions
POST /api/agent/sessions/{session_id}/experiments
GET  /api/agent/sessions/{session_id}/experiments/{run_id}/feedback
GET  /api/agent/sessions/{session_id}
POST /api/agent/sessions/{session_id}/finalize
```

`feedback` 仅允许返回：

- session/run 标识和 queued/running/succeeded/failed/cancelled 状态；
- experiment config；
- Validation 的 selection metric、有限的 secondary metrics、fold 审计摘要；
- 脱敏错误、remaining budget、是否可继续。

严禁返回：Test、predictions、confusion matrix、classification report、artifact URL/目录、模型文件、服务器绝对路径、完整 `metrics.json`。

第一轮 capability 可以是 Agent Adapter 内部的版本化白名单，只开放：

```json
{
  "models": ["logistic_regression", "svm", "random_forest"],
  "normalization": ["zscore", "minmax", "area", "none"],
  "class_balance": ["none", "class_weight"],
  "selection_metric": "valid_balanced_accuracy"
}
```

不能直接把 `TrainingRunRequest.config` 当前 53 个兼容字段全部暴露给 LLM。真实超参数在基础闭环成功后单独扩展。

**Todo：**

- [ ] 先写契约测试，证明五个 endpoint 当前不存在，再以最小改动实现。
- [ ] Session 创建时绑定 Principal、dataset_id、allowed models、max_runs、seed、metric；客户端不能传 owner/tenant。
- [ ] Experiment Adapter 复用 `TrainingSpec.validated()` 和现有 queued Run 路径，不复制训练配置校验或直接写 Run DB。
- [ ] Feedback 以服务层白名单投影 Validation；增加专门测试，用含 Test/artifact/server path 的 fixture 证明响应不会泄漏。
- [ ] Finalize 校验 selected run 属于该 session 且已 succeeded；Finalize 后拒绝新 experiment，Test 只在最终 report 阶段读取。
- [ ] LLM 输出先经 Pydantic 严格校验；非法 JSON 最多做一次 repair request，仍失败则 `REQUEST_HUMAN`，不默认提交实验。
- [ ] 训练等待期间只轮询 AutoAI，不重复调用 LLM；使用有限退避和总 timeout。
- [ ] 配置 hash 去重；同一 session 不允许提交完全相同 experiment。
- [ ] max_runs 由 Python 硬限制，LLM 无法越过；达到上限只能 Finalize 或 Request Human。
- [ ] Trace 记录 prompt version、model revision、quantization、LLM latency/token、raw response、parsed decision、HTTP request ID、run_id、Validation feedback 和最终选择，不记录 token/secret/Test observation。

**第一轮 POC 固定参数：**

```text
allowed_models = [logistic_regression]
max_runs = 1
temperature = 0
seed = 42
selection_metric = valid_balanced_accuracy
poll_interval_seconds = 5
run_timeout_seconds = 3600
```

**验收标准：**

- 单元测试覆盖 session 生命周期、归属隔离、budget、去重和信息泄漏。
- fake LLM 下闭环可重复，非法 JSON/HTTP 5xx/Worker 超时不会产生失控 Run。
- 真实 9B 模型可产生合法 decision，并通过 Adapter 创建一个 logistic regression Run。
- Agent POC 不 import `backend.app.runs.repository`、`worker` 或 `training`；代码搜索与测试共同证明只走 HTTP。

## Task 4：三个 backbone 的 AutoAI 端到端验收

**交付物：** `outputs/agent_poc_acceptance/` 下的 manifest、逐模型 trace、汇总 JSON/Markdown 与复现命令；三个模型均完成真实 AutoAI 闭环。

**前置条件：**

- Task 1–3 全部完成。
- 一份不进 Git 的真实 `wide-feature-v2` Dataset 已上传，获得 `dataset_id`；每类至少 3 个 `Sample_ID`，满足 `stratified_holdout`。
- AutoAI `/health` 为 `status=ok`、`worker.available=true`、`worker.compatible=true`。
- 对应模型 endpoint `/v1/models` 健康。
- AutoAI Worker 使用独立空闲 GPU；传统 logistic regression smoke 可用 CPU，但最终至少补一次 GPU 模型训练证明资源隔离。

**Todo：**

- [ ] 运行后端完整测试：`python -m pytest backend/tests -q`。
- [ ] 运行编译与 POC 测试：`python -m compileall backend/app agent_poc -q`、`python -m pytest agent_poc/tests -q`。
- [ ] 用同一 dataset/prompt/tools/seed/budget/temperature/metric，依次对 `qwen35_9b`、`qwen35_27b`、`qwen38_27b` 运行 `max_runs=1` 闭环。
- [ ] 每个闭环必须观测：LLM `RUN_EXPERIMENT` → 202 queued → Worker running → succeeded → Validation-only feedback → LLM/Orchestrator Finalize。
- [ ] 每个 backbone 连续重复 3 次，统计 structured JSON 成功率、闭环成功率、LLM latency、实验配置、Validation 和总时长。
- [ ] 第一轮稳定后再把 allowed models 扩为 logistic regression/SVM/random forest、`max_runs=3`，检查模型是否重复配置、是否使用反馈、是否按 budget Finalize。
- [ ] Finalize 后由独立 evaluator 一次性读取 Test；不得把 Test 写回 Agent trace 的 observation 字段。
- [ ] 三端点同时在线时再跑一次并发健康检查，确认没有端口冲突、OOM、GPU 互抢或 AutoAI Worker 饥饿。
- [ ] 生成公平比较表，只改变 LLM backbone；任何 prompt、tool schema、后端 commit、dataset hash、split/seed 或量化差异都必须标红为无效比较。

**逐模型验收命令形态：**

```bash
conda run -p /users/fotile/AutoAI/envs/autoai-app \
python -m agent_poc.main \
  --model-key qwen35_9b \
  --dataset-id <dataset_id> \
  --max-runs 1 \
  --selection-metric valid_balanced_accuracy

# 只改 --model-key：qwen35_27b / qwen38_27b
```

**最终通过矩阵：**

| 验收项 | 3.5-9B | 3.5-27B | 3.8-27B |
|---|---:|---:|---:|
| 固定官方 revision、统一 W8A8 manifest | 必须 | 必须 | 必须 |
| `/v1/models` 与 chat completion | 必须 | 必须 | 必须 |
| 严格 AgentDecision JSON，20/20 | 必须 | 必须 | 必须 |
| Session/Experiment/Worker/Feedback/Finalize | 必须 | 必须 | 必须 |
| 搜索观察无 Test/artifact/path 泄漏 | 必须 | 必须 | 必须 |
| 同一 Dataset 的真实 AutoAI succeeded Run | 必须 | 必须 | 必须 |
| 可复现 trace 和最终 report | 必须 | 必须 | 必须 |

只有三列全部通过，才可以说“完整验收：三个模型都可以在 AutoAI 上跑通”。仅模型能聊天、仅 API 200、仅 9B 闭环或只跑 mock 都不算完成。

## 3. 风险、阻塞条件与回滚

### 当前已知阻塞

1. **GPU 资源：** node3 当前没有满足部署需求的空闲 5 卡窗口。不得停止他人进程。先完成代码、环境、metadata 与 9B sequential smoke；同时在线验收等待明确资源。
2. **node2 Hugging Face 网络：** 代理变量存在但 30 秒探针未成功。下载前必须获得 HTTP 200；排查 IPv4、代理出口、CA、DNS、Xet/LFS 和超时，禁止立即改用未知镜像。
3. **统一 W8A8：** Qwen3.8 发布时间很新，量化/serving support 必须实测。任一模型失败时暂停公平比较并提交 blocker，不混用 4-bit。
4. **V1 Agent Adapter 缺失：** 当前代码与旧草案不一致，Task 3 是必需开发，不是可选优化。

### 回滚设计

- 代码 release 使用时间戳目录；回滚只切换 `current` 到上一 release，并同时重启 Web/Worker。
- `shared/storage` 与 release 解耦；任何 schema 改动前先停止 Web/Worker并备份 SQLite 和相关 Run 目录。
- 模型 checkpoint 按 revision 独立目录，不原地覆盖；失败量化写新目录并保留 manifest，不删除官方 cache。
- 停止服务只处理 `/users/fotile/AutoAI/runtime` 记录且命令行与启动时间匹配的本用户 PID，不使用宽泛 `pkill python`。

## 4. 本轮停止点

本轮工作只负责：核实模型、检查服务器、建立部署目录、上传当前代码快照并交付这份实施计划。不会在 GPU 全部被占用时启动 vLLM，不会下载 55 GB 以上权重，不会安装/改动服务器环境，也不会修改现有 AutoAI 业务代码。后续执行严格从 Task 1 未完成项继续。
