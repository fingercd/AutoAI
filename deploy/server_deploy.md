# SpecAutoAI 服务器部署说明

> 最近核对：2026-07-15。仓库内 node3 脚本仍只管理 Web 进程，生产训练必须另行托管 `backend.app.runs.worker`。

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

需要对局域网直接监听时才将 host 改为 `0.0.0.0`，并先确认集群安全策略。

仓库中的 `deploy/run_on_node3.sh` 与 `deploy/start_persistent_node3.sh` 目前只管理 Web 进程，且默认 `APP_DIR=$HOME/AutoAI`。在 v2 中使用时必须显式传入：

```bash
APP_DIR="$HOME/AutoAI-v2" bash deploy/run_on_node3.sh
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

验收清单：

1. `/health` 返回 `{"status":"ok"}`，首页和 `/docs` 可访问。
2. 上传一份本地建模 CSV，返回数据摘要和稳定 `dataset_id`。
3. 创建训练任务后，状态从 queued/running 进入 success；如果一直 queued，检查 worker。
4. 页面显示 train/valid/test 指标、macro-F1、混淆矩阵和训练/解释性图表。
5. 能下载 predictions、metrics 和对应模型 artifact。
6. 拉曼与 HPLC 预处理能下载统一 CSV；HPLC 曲线同时包含 `raw_y` 与 `processed_y`。

不要依赖固定的样本数、类别名、GPU 编号或模型文件扩展名作为部署成功标准。

## 8. 持久化与备份

运行状态位于 `storage/`，包括上传、SQLite 数据库、日志和 Run artifacts；该目录不进 Git。升级代码前应单独备份所需数据，并确保 Web 与 worker 停止或使用一致版本。不要把 `storage/` 复制回 Git 仓库。
