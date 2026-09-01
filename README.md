# SpecAutoAI

面向拉曼、色谱和 HPLC 曲线的小样本预处理、分类建模与 Agent 实验平台。

SpecAutoAI 把一条完整链路串起来：原始曲线上传 → 生成统一建模表 → 选择模型训练 → 查看 Validation/Test 结果 → 下载受完整性保护的产物。后端是 FastAPI，训练由独立 worker 从 SQLite Run 队列领取；Agent 通过 API 控制实验，不依赖前端。

> 当前版本聚焦**分类任务**。`Label` 即使是数字也按类别处理；PLSR、SVR 等回归入口暂不开放。

## 你可以用它做什么

- **处理光谱数据**：拉曼支持行号/X 轴范围截取和基线校正；HPLC 支持行号/时间范围、公共轴检测和可选线性插值。
- **训练分类模型**：传统机器学习、PCA-MLP 和 1D 深度模型使用统一的 Run、评估和产物契约。
- **做可解释性分析**：传统模型使用真实类别 Log-loss 窗口遮挡；卷积模型使用 Grad-CAM-like；DSCARNet 使用 SAR/CAR 双通路 2D 映射后回投到 1D。
- **让 Agent 做受控实验**：Agent 只能通过健康检查、Session、Experiment、Validation feedback 和 Finalize API 工作；它不能直接读取服务器文件或 Test 指标。
- **复现实验并比较成本**：冻结 Benchmark、固定 seed、记录 LLM/API/retry/model-fit/wall-clock，报告不制造一个无法解释的“总分”。

## 整体流程

```text
原始 CSV
   │
   ▼
预处理（公共轴 + wide-feature-v2）
   │
   ▼
FastAPI ── SQLite Run Queue ── 独立 Worker ── Manifest-backed artifacts
   │                                  │
   │                                  └─ train / valid / test
   │
   ├─ 浏览器（可选）
   └─ Agent POC ── Qwen API ── Validation feedback ── Finalize
                                      │
                                      └─ Finalize 之后才允许独立 Test evaluator
```

Agent 的决策链和前端是两条独立入口：不启动前端也可以运行 Agent；不启用 Agent 模块时，原有训练 API 仍按兼容路径工作。

## Agent 实验层

Agent V1 的模块全部是 Session 级开关。默认关闭，打开后配置会被锁定，实验中不能自行改写。

| 模块 | 作用 | 状态/依赖 |
|---|---|---|
| `evidence_card` | 生成 train-only 数据证据卡 | ready |
| `dynamic_preprocessing` | 根据证据卡选择有限预处理策略 | ready；需要 evidence card + restricted pool |
| `restricted_strategy_pool` | 使用服务端 canonical proposal 目录 | ready |
| `bounded_hpo` | 固定候选上限和可审计选优口径 | ready |
| `fail_fast_guard` | 在训练前后检查数据、划分、成本和结果契约 | ready |
| `constrained_code_evolution` | 隔离生成并验证受约束候选代码 | experimental；需要 Guard |
| `feedback_diagnosis` | 只根据 train/valid 生成安全诊断 | ready；需要 Guard |
| `limited_replanning` | 失败后最多一层、有限次数的 canonical REPLAN | ready；需要诊断 + restricted pool |
| `uncertainty_selection` | 用 Validation 独立组不确定性、保守分数和 plateau 规则决定是否停止 | ready；需要诊断 |
| `case_memory` | Principal 隔离的 Case Bank / Failure Ledger + 静态 Prior | ready；需要 Evidence Card + 诊断 |
| `budget_control` | 原子 model-fit 预留，以及 LLM/API/retry/wall-clock 硬门禁 | ready；需要 bounded HPO + Guard |

几个重要边界：

- Agent 只看到 Validation 白名单标量和安全摘要；Test 指标、预测、混淆矩阵、artifact 路径不会进入 Prompt、Trace 或案例记忆。
- `case_write=true` 必须显式启用 `case_memory`；`source_role=benchmark` 永远禁止写入案例库。
- `budget_control=true` 必须显式提供五个上限：`max_model_fits`、`max_llm_calls`、`max_api_calls`、`max_wall_clock_seconds`、`max_retry_attempts`。
- 失败诊断缺失、动作不被允许或后端预算契约损坏时，Agent fail closed，不会偷偷继续或强行 Finalize。

## 模型与评估

能力目录公开 15 个目标分类模型，当前环境稳定可训练 14 个：

```text
pls_da                 pca_lda                 logistic_regression
svm                    random_forest           xgboost
pca_mlp                cnn1d                   cnn1d_se
resnet1d               inception1d              tcn1d
cnn_transformer1d      cnn_mamba1d              dscarnet
```

`cnn_mamba1d` 因 `mamba-ssm` 依赖不可用而明确标记 unavailable，不会静默替换成近似网络。训练入口支持三种评估口径：

- `stratified_holdout`：按 `Sample_ID` 整组划分，目标 8:1:1，并保证 Train/Valid/Test 每类至少一个独立组。
- `leave_one_sample_id_cv`：每折留一个 `Sample_ID` 作 Test，其余按 8:2 划 Train/Valid；Test 主指标使用 pooled OOF。
- `external_test_holdout`：主数据 8:2，独立测试集作为唯一 Test。

标准化、PCA、AggMap、超参选择和 early stopping 只拟合当前训练集。CV 的 pooled、fold mean 和 fold std 会明确分开，不把 fold mean 冒充主指标。

## 建模数据格式

推荐使用 `wide-feature-v2`：

```csv
Index,Label,Sample_ID,Name,0,0.0066675556740898788,...,50
1,A,S001,curve-001.csv,0.12,0.15,...,0.08
2,A,S001,curve-002.csv,0.11,0.16,...,0.09
```

- 前四列必须固定为 `Index,Label,Sample_ID,Name`。
- 第五列起是有限、唯一、严格递增的真实坐标；单元格是有限标量强度。
- `Sample_ID` 是重复测量分组，同一组不能出现多个 `Label`。
- 主数据和独立 Test 必须逐点同轴。
- `wide-feature-v1` 仍可读取；旧六列数组/JSON、`linspace-v1` 和 `linspace-slice-v1` 会明确拒绝，不做有损自动迁移。
- Excel 单文件最多 16,384 列；v2 最多 16,380 个特征。

训练数据、模型、运行产物和本地 `data.csv` 都不应提交到 Git。

## 安装

已验证 Python 3.12。先按机器环境安装匹配 CUDA 的 PyTorch，再安装项目依赖：

```bash
python -m pip install -r backend/requirements.txt
```

开发和测试：

```bash
python -m pip install -r backend/requirements-dev.txt \
  -c backend/constraints-verified.txt
```

DSCARNet/AggMap 是可选链路，按项目约束安装：

```bash
python -m pip install -r backend/requirements-dscarnet.txt \
  -c backend/constraints-verified.txt
python -m pip install aggmap==1.2.1 --no-deps
```

AggMap 1.2.1 的旧元数据可能让 `pip check` 报告 `tensorflow-gpu`、`lapjv` 等冲突；本项目使用的是 SAR/CAR 映射和 SciPy 兼容层，不调用 TensorFlow AggModel。

## 启动后端

一键启动（Web + worker）：

```bash
python run.py
```

只启动 Web 不会执行训练，queued Run 需要另起 worker：

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
python -m backend.app.runs.worker
```

常用地址：

- Web：<http://127.0.0.1:8000/>
- v2 工作台：<http://127.0.0.1:8000/v2>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

对外部署必须显式启用 server 模式、配置至少 32 字符的 Bearer token 和明确的 CORS 来源。不要把 token 放进 URL、命令参数、日志或仓库；浏览器 token 只保存在当前标签页的 `sessionStorage`。

```bash
export AUTOAI_DEPLOYMENT_MODE=server
export AUTOAI_API_TOKEN="由 secret manager 注入"
export AUTOAI_PRINCIPAL_ID=server-admin
export AUTOAI_TENANT_ID=default
export AUTOAI_ALLOWED_ORIGINS=https://autoai.example.edu
python run.py --server --host 0.0.0.0 --no-browser
```

## API-only Agent

Agent POC 的直接入口：

```bash
# 检查 Agent 后端和全部已登记模型
python -m agent_poc.main --health-check

# 检查某个模型的实际推理
python -m agent_poc.main \
  --health-check \
  --model-key qwen35_9b \
  --probe-inference
```

当前登记的 Qwen runtime 是：

| key | 模型 | 默认端口 |
|---|---|---:|
| `qwen35_9b` | Qwen3.5-9B | 8101 |
| `qwen35_27b` | Qwen3.5-27B | 8102 |
| `qwen38_27b` | Qwen3.8-27B | 8103 |

服务器没有 Qwen3.8-9B 资产；不要在配置中虚构该模型。首次推理可能包含 CUDA 图/内核预热，健康探针第一次超时后应复测。

Agent 生产 API 只有五类动作：

```text
GET  /api/agent/health
POST /api/agent/sessions
POST /api/agent/sessions/{session_id}/experiments
GET  /api/agent/sessions/{session_id}/experiments/{run_id}/feedback
POST /api/agent/sessions/{session_id}/finalize
```

训练请求只创建 queued Run，不在 HTTP 请求线程里启动训练。Session 创建后，数据集、模型集合、评估口径、模块开关和上下文策略都会锁定。

## 冻结 Benchmark

Benchmark 文件和校验清单位于 `docs/benchmarks/`；冻结数据不进 Git。当前轻量套件包含：

| 数据集 | 角色 | 规模 | 用途 |
|---|---|---:|---|
| `molecular_biology_promoters` | formal primary | 106 × 57，53/53 | 正式主任务 |
| `haberman` | formal auxiliary | 306 × 3，225/81 | 不平衡与稳健性 |
| `parity5` | pipeline smoke only | 32 × 5，16/16 | 下载、适配、闭环 smoke |

首轮 AutoAI 口径是 8:1:1 `stratified_holdout`，主指标 Macro-F1，辅报 balanced accuracy、失败率、wall-clock、LLM/API/retry/model-fit。`parity5` 不进入正式汇总；如需和 TabMini 发布结果比较，必须另跑固定 3-fold ROC-AUC，不能混称为同一 Benchmark。

验证冻结文件：

```bash
python -m agent_poc.benchmark.main verify \
  --benchmark-root /path/to/benchmark \
  --policy-manifest docs/benchmarks/small-sample-benchmark-v1.json
```

运行一条真实 trial：

```bash
python -m agent_poc.benchmark.main run \
  --benchmark-root /path/to/benchmark \
  --policy-manifest docs/benchmarks/small-sample-benchmark-v1.json \
  --dataset-name parity5 \
  --model-key qwen35_9b \
  --models-config agent_poc/config/models.toml \
  --agent-config agent_poc/config/agent.toml \
  --autoai-base-url http://127.0.0.1:8000 \
  --output-dir outputs/benchmark-parity5
```

Benchmark runner 要求干净的 Git checkout，并自动把当前 `git rev-parse HEAD` 写入 Trace 和报告。它只在 Agent Finalize 且 Trace 审计通过后访问结果接口；未 Finalize、失败或需要人工介入时，Test evaluator 调用次数必须为 0。

MLE-bench Lite 需要 Kaggle 凭据且约 158 GB，不属于本轻量门禁。

## 安全与可复现性

- **身份隔离**：Dataset、Run、Session 和案例记忆都按 owner/tenant 约束访问。
- **状态原子性**：Run claim、lease、reservation、Finalize 和 selected Case 使用 SQLite 事务。
- **结果完整性**：成功 Run 必须先提交带 SHA-256/大小校验的 Manifest。
- **产物最小暴露**：模型权重、私有 joblib、路径和原始错误不进入 Agent 可见响应；下载也受白名单保护。
- **Test 防火墙**：Test 只在最终选择之后由独立 evaluator 读取，结果不回流 Prompt、Trace 或 Case Bank。
- **确定性**：seed、配置 hash、canonical proposal、代码 SHA 和 Benchmark 输入 SHA 都进入审计链。

## 目录结构

```text
backend/app/                 FastAPI、预处理、训练、Run 队列、Guard 和模型
backend/tests/               后端契约、训练、迁移、安全和集成测试
agent_poc/                   API-only Agent、Qwen client、预算、Prior、Trace
agent_poc/benchmark/         Benchmark manifest、适配器、runner、evaluator、report
deploy/                      服务器启动、模型服务和快照校验脚本
docs/                        接口契约、结果契约、Benchmark 和 ADR
static/                      经典入口与 v2 工作台（Agent 不依赖它们）
storage/                     本地上传、SQLite 和训练产物；不进 Git
run.py                       Web + worker 公共启动器
```

## 验证

```bash
python -m pytest backend/tests -q
python -m pytest agent_poc/tests -q
python -m compileall backend/app agent_poc -q
python run.py --help
```

改动训练、评估、预处理或 Agent 训练请求逻辑后，还应在存在本地 `data.csv` 时完成一次轻量真实闭环；`data.csv` 只能作为本地验证数据，不能提交。

## 常见问题

### Run 一直是 queued

检查 `/health` 的 `worker.available` 和 `worker.compatible`。Web 进程只负责入队；没有匹配版本的 worker 时不会强行执行。

### Agent 返回 degraded

`/api/agent/health?probe=models` 会逐个报告 runtime 可达性。Agent API/数据库 ready 不等于每个 Qwen 端口都在线；逐个启动并预热后再做 inference probe。

### 为什么看不到 Test 指标

这是设计边界。Agent 只用 Validation 做选择；Test 由最终 evaluator 在 Finalize 之后读取，避免调参过程泄漏。

### 为什么 `cnn_mamba1d` 不训练

它依赖当前环境没有的 `mamba-ssm`。系统会返回 unavailable，而不是偷偷换成另一个网络。

### 如何迁移历史 server Run

先使用 migration 的 `--dry-run` 检查 owner/tenant 绑定数量，确认无误后再执行正式迁移。不要在服务启动时自动重绑历史记录。

## 进一步阅读

- [前后端接口契约](docs/frontend_backend_handoff.md)
- [run-result-v1 结果契约](docs/run_result_contract.md)
- [小样本 Benchmark 清单](docs/benchmarks/small-sample-benchmark-v1.md)
- [服务器部署说明](deploy/server_deploy.md)
- [架构决策记录](docs/adr/)

历史 `AutoAI_开发计划.md` 只用于了解早期路线；当前实现以 `AGENTS.md`、`CONTEXT.md`、本 README 和 `docs/` 契约为准。
