# SpecAutoAI 服务器部署说明

> 最近核对：2026-07-16。仓库内 node3 脚本仍只管理 Web 进程，生产训练必须另行托管 `backend.app.runs.worker`。所有对外监听必须启用 Bearer 认证；SSH tunnel 的 loopback 部署可继续使用 local 模式。

目标是在校园集群中运行 FastAPI Web 服务和独立训练 worker，并通过 SSH 隧道从本地访问。当前已验证的解释器是 Python 3.12.12；集群 GPU/CUDA 组合必须按实际驱动选择 PyTorch。

## 1. 目录与访问约定

```text
项目目录：~/AutoAI-v2
服务端口：8000
推荐访问：SSH tunnel
```

本地隧道示例：

```powershell
ssh -N -L 8000:127.0.0.1:8000 node3
```

浏览器打开 <http://127.0.0.1:8000/>。

此方式服务端只监听 loopback，不直接暴露网络端口，可保持默认 `AUTOAI_DEPLOYMENT_MODE=local`。

具体登录入口、跳板机、端口和计算节点属于部署环境配置，不应硬编码进公开脚本或提交凭据。

## 2. 获取私密仓库

在已配置 GitHub 凭据的机器上：

```bash
git clone https://github.com/fingercd/AutoAI-v2.git ~/AutoAI-v2
cd ~/AutoAI-v2
```

仓库不包含 `data.csv`、上传数据、模型或历史运行产物。不要把本地附件或内部计划打入部署包。

## 3. Python 3.12 与依赖

在可联网节点创建环境：

```bash
conda create -y -n autoai-v2 python=3.12
conda activate autoai-v2
python -m pip install --upgrade pip
```

CPU 或已单独安装正确 CUDA PyTorch 的环境：

```bash
python -m pip install -r backend/requirements.txt -c backend/constraints-verified.txt
```

需要 DSCARNet 时：

```bash
python -m pip install -r backend/requirements-dscarnet.txt -c backend/constraints-verified.txt
python -m pip install aggmap==1.2.1 --no-deps
```

CUDA 版 PyTorch 应先按官方渠道安装；随后安装 requirements 时，pip 会保留满足版本范围的现有 torch。

基本导入检查：

```bash
python -c "from backend.app.main import app; print(app.title)"
python -c "from backend.app.runs.worker import RunWorker; print(RunWorker.__name__)"
```

`deploy/install_on_node2.sh` 是旧集群辅助脚本，默认环境名和 Python 版本可能不符合上述验证基线；使用前应显式审核或优先执行本节手动命令。

## 4. 启动 Web 与 worker

训练 HTTP 请求只创建 SQLite 中的 queued Run。FastAPI 不使用 BackgroundTasks 执行训练；必须同时运行独立 worker。

终端 1（Web）：

```bash
cd ~/AutoAI-v2
conda activate autoai-v2
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

终端 2（worker）：

```bash
cd ~/AutoAI-v2
conda activate autoai-v2
python -m backend.app.runs.worker
```

Web 与 Worker 必须来自同一提交/部署包。升级时先停止二者，再更新代码并同时启动；不能只替换静态文件或只重启 Web。

### 对外监听与认证

直接监听 `0.0.0.0` 前，先把随机令牌存入受权限保护的 secret 文件或进程管理器，不要写入仓库、命令行参数或日志。例如：

```bash
export AUTOAI_DEPLOYMENT_MODE=server
export AUTOAI_API_TOKEN="$(< /secure/path/autoai_api_token)"
export AUTOAI_PRINCIPAL_ID=server-admin
export AUTOAI_TENANT_ID=default
export AUTOAI_ALLOWED_ORIGINS=https://autoai.example.edu
python run.py --server --host 0.0.0.0 --no-browser --no-worker
```

- `AUTOAI_API_TOKEN` 至少 32 个字符。
- `AUTOAI_ALLOWED_ORIGINS` 可逗号分隔，禁止 `*`；同源部署可以不设置。
- `/`、静态文件、`/health` 和 `/api/auth/config` 可匿名访问；API、OpenAPI 和文档需要 Bearer token。
- 浏览器 token 只保存在当前标签页 sessionStorage，不进入 URL。
- server 训练请求必须先上传 Dataset，再使用 `dataset_id`/`test_dataset_id`；不接受 `data_path`。

Bearer 令牌等同于访问凭据，不能通过公网明文 HTTP 传输。对外生产部署必须在反向代理或网关终止 HTTPS/TLS，并建议在代理层设置请求限流、访问日志脱敏和异常请求告警；人员、设备或泄露风险变化时应轮换令牌。本轮不引入账号系统，仍以单一服务端 Principal 边界为准。

上述命令使用 `--no-worker`，适用于 Web 与 GPU worker 分别由进程管理器托管的部署；若希望 `run.py` 在同一生命周期监督一个本机 worker，可以去掉该参数。

仓库的 `deploy/run_on_node3.sh` 默认监听 `127.0.0.1`。如果通过 `HOST=0.0.0.0` 对外启动，脚本会强制设置 server 模式，并在 token 缺失或过短时立即退出，且不会打印 token。

仓库中的 `deploy/run_on_node3.sh` 与 `deploy/start_persistent_node3.sh` 目前只管理 Web 进程，且默认 `APP_DIR=$HOME/AutoAI`。在 v2 中使用时必须显式传入：

```bash
APP_DIR="$HOME/AutoAI-v2" HOST=127.0.0.1 bash deploy/run_on_node3.sh
```

worker 仍需由另一个终端、SGE 作业或集群进程管理器独立托管。不要仅启动 Web 后就进行训练验收。

## 5. GPU 选择

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=<空闲GPU编号>
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Web 与 worker 可使用相同环境；GPU 环境变量应设置在 worker 进程上。

## 6. SGE

如果服务必须通过 SGE：

```bash
cd ~/AutoAI-v2
qsub deploy/qsub_autoai_node3.sh
qstat
```

提交前审核脚本中的目录、队列、GPU 和日志参数，并确保另有 worker 作业。仓库脚本是环境模板，不替代集群管理员规范。

## 7. 验收

```bash
curl http://127.0.0.1:8000/health
```

server 模式验证认证会话：

```bash
curl -H "Authorization: Bearer $AUTOAI_API_TOKEN" \
  http://127.0.0.1:8000/api/auth/session
```

验收清单：

1. `/health` 返回 `status=ok`、`deployment_mode`、`contracts.run_result=run-result-v1`、`contracts.artifact_manifest=run-artifact-manifest-v2`；`worker.available=true` 且 `worker.compatible=true`。
2. local 模式首页和 `/docs` 可访问；server 模式 `/docs` 无 token 为 401、有效 token 为 200。
3. 上传一份本地建模 CSV，返回数据摘要和稳定 `dataset_id`；server 响应不暴露 `dataset_path`。
4. 创建训练任务后，状态从 queued/running 进入 succeeded；如果一直 queued，检查 `/health` 和独立 worker。
5. 训练完成后进入 `/#/results?run_id=...`，刷新和 Web 重启后仍能恢复同一结果。
6. 页面显示直接 Test 或 pooled OOF 主指标，以及 Train/Valid/Test 三组混淆矩阵、各类别指标、预测分布和单样品解释；传统模型没有训练曲线，当前不显示虚假 ROC/AUC/PR。
7. 能逐项下载 catalog 允许的 predictions、metrics、无路径 config 等文件；模型 pickle/PT 和 joblib 保持私有。
8. 拉曼与 HPLC 预处理能下载统一 CSV；HPLC 曲线同时包含 `raw_y` 与 `processed_y`。

如果创建训练返回 503 `worker_contract_mismatch`，说明至少一个活跃 Worker 没有当前契约版本。停止所有旧 Worker，并与 Web 从同一版本重新启动；不要绕过门禁直接写 queued Run。

不要依赖固定的样本数、类别名、GPU 编号或模型文件扩展名作为部署成功标准。

## 8. 持久化与备份

运行状态位于 `storage/`，包括上传、SQLite 数据库、日志和 Run artifacts；该目录不进 Git。升级代码前应单独备份所需数据，并停止 Web 与所有 Worker；升级后先通过 `/health` 确认契约一致，再允许创建训练。不要把 `storage/` 复制回 Git 仓库。

从旧 local 版本切换 server 前，先在停止 Web/worker 且完成备份后预览历史绑定：

```bash
python -m backend.app.runs.migration --dry-run \
  --owner-id server-admin --tenant-id default --rebind-unowned
```

确认数量后去掉 `--dry-run`。该命令幂等，不移动或改写 Run 目录，也不会由服务启动自动执行。参数必须与部署使用的 Principal ID/Tenant ID 一致。
