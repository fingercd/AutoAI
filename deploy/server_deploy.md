# AutoAI Server Deployment Notes

目标：把本地项目部署到校园集群，使用服务器 Miniconda/Anaconda 环境，在 `node3` 常驻运行 Web 服务；本地用户通过 SSH 隧道访问网页并提交训练。

> 当前状态（2026-07-06）：主服务仍是 FastAPI + `static/index.html` 同源托管。训练任务使用 FastAPI `BackgroundTasks` 和 `storage/runs/{run_id}/status.json` 记录状态，尚未切换为独立队列 worker。色谱主页面默认调用 HPLC 三步预处理；Transformer 已注册；预处理页已有曲线预览和 HPLC 原始/处理后对比。

## 1. 服务器约定

```text
登录入口：ibmnode / 10.49.16.5:8822
默认运行节点：node3
联网安装节点：node2
项目目录：~/AutoAI
服务端口：8000
推荐访问：SSH tunnel
```

不建议第一版直接改 Nginx、防火墙或系统服务。集群内 compute node 往往是内网地址，本地电脑通常不能直接访问 `node3:8000`，所以推荐隧道：

```powershell
ssh -N -L 8000:127.0.0.1:8000 node3
```

然后本地浏览器打开：

```text
http://127.0.0.1:8000/
```

## 2. 上传项目

从本地项目根目录打包，建议排除训练输出：

```powershell
Compress-Archive -Path backend,static,deploy,docs,data.csv,README.md,CONTEXT.md,AGENTS.md,AutoAI_开发计划.md -DestinationPath autoai_deploy.zip -Force
scp autoai_deploy.zip ibmnode:~/autoai_deploy.zip
```

登录后解压：

```bash
mkdir -p ~/AutoAI
unzip -o ~/autoai_deploy.zip -d ~/AutoAI
cd ~/AutoAI
```

## 3. 在 node2 安装依赖

如果 home 是共享文件系统，建议在联网的 node2 执行：

```bash
ssh node2
cd ~/AutoAI
bash deploy/install_on_node2.sh
```

该脚本会创建或复用 `autoai` conda 环境，并安装：

```text
fastapi
uvicorn
python-multipart
pandas
numpy
scikit-learn
torch
matplotlib
rampy
aggmap 及 DSCARNet 所需兼容依赖
```

DSCARNet 依赖 `aggmap==1.2.1`，当前建议按 `backend/requirements.txt` 中说明使用 `--no-deps` 安装 aggmap，再安装兼容依赖，避免旧版 PyPI 元数据拉取不合适的包。

## 4. 在 node3 选择空闲 GPU

```bash
ssh node3
nvidia-smi
```

若 GPU 6 空闲：

```bash
export CUDA_VISIBLE_DEVICES=6
```

若 GPU 7 空闲：

```bash
export CUDA_VISIBLE_DEVICES=7
```

## 5. 在 node3 启动服务

临时前台运行：

```bash
cd ~/AutoAI
bash deploy/run_on_node3.sh
```

常驻后台运行：

```bash
cd ~/AutoAI
bash deploy/start_persistent_node3.sh
```

查看日志：

```bash
tail -f ~/AutoAI/storage/logs/autoai.out.log
tail -f ~/AutoAI/storage/logs/autoai.err.log
```

停止服务：

```bash
cd ~/AutoAI
bash deploy/stop_persistent_node3.sh
```

## 6. 通过 SGE 提交

如果集群要求服务也通过 SGE 提交：

```bash
cd ~/AutoAI
qsub deploy/qsub_autoai_node3.sh
qstat
```

## 7. 本地访问验收

在本地 PowerShell 开隧道：

```powershell
.\deploy\local_tunnel.ps1
```

或直接：

```powershell
ssh -N -L 8000:127.0.0.1:8000 node3
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

验收标准：

1. 首页能打开。
2. `data.csv` 自动加载并显示 90 条样本、Fe/Si 各 45 条。
3. 点击“开始训练”后状态从 `pending/running` 变为 `success`。
4. 页面显示 test accuracy / macro F1。
5. 能下载 `predictions.csv`、`metrics.json`、`model.pt`。
6. 能下载 `feature_importance.json/csv` 和 `sample_feature_importance.json/csv`；如果模型是 MLP，页面应显示不支持解释性分析而不是报错。
7. 拉曼/色谱预处理上传后能下载统一格式 CSV；色谱默认 HPLC 流程应显示原始/处理后曲线和共同时间轴信息。

## 8. 当前还需要继续补齐的生产功能

当前版本已经能完成本地/服务器训练闭环，但还需要继续加强：

- 训练任务改为真正的队列 Worker，避免 Web 进程重启丢状态。
- 如确实需要交叉验证，再把 Repeat_index 整组划分扩展为完整多轮留一法；当前默认是 8/1/1 或外部测试集模式。
- 增加 focal loss。
- 增加用户级权限、任务隔离和上传配额。
- 增加服务器端 systemd 或 SGE 长期运行策略的最终确认。
- 若要通过接口下载 DSCARNet joblib 映射文件，需要扩展 artifact 白名单并补路径安全测试。
