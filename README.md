# AutoAI-v2

AutoAI-v2 是面向拉曼与色谱/HPLC 曲线的预处理和分类建模平台。它通过同一个 FastAPI 服务提供网页、数据上传、预处理、训练任务、指标与模型产物下载；训练由独立 worker 从 SQLite Run 队列领取执行。

当前版本只支持分类。`Label` 即使是数字也按类别处理，不提供 PLSR、SVR 等回归入口。

## 正式功能

- 拉曼：按行号或 X 轴范围截取，支持基线校正及两种“截取/校正”顺序。
- HPLC：插值到共同时间轴、逐条减最小值、按真实时间轴面积归一化。
- 分类评估：分层 8:1:1、按 `Repeat_index` 留一交叉验证、独立测试集 holdout。
- 10 个模型：`pls_da`、`svm`、`random_forest`、`xgboost`、`cnn1d`、`transformer1d`、`resnet1d`、`inception1d`、`tcn1d`、`dscarnet`。
- 可解释性：传统模型、PCA-MLP 和 CNN-Transformer 使用真实类别 Log-loss 窗口遮挡；卷积模型使用 Grad-CAM-like；DSCARNet 使用 SAR/CAR 双通路映射和 2D Grad-CAM 回投。

历史 UI 画廊已经从正式产品移除；未跟踪的界面候选不属于本仓库发布内容。

## 环境要求

- 已验证：Python `3.12.12`。
- CPU 环境可直接安装核心依赖。
- NVIDIA CUDA 环境应先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装匹配驱动/CUDA 的 PyTorch，再安装其余依赖。不要依赖通用 requirements 自动猜测 CUDA wheel。
- DSCARNet 额外依赖 AggMap；其余 9 个模型不要求 AggMap。

建议新建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 三种安装方式

### 核心运行环境

普通兼容安装：

```bash
python -m pip install -r backend/requirements.txt
```

使用当前环境验证过的直接依赖版本基线：

```bash
python -m pip install -r backend/requirements.txt -c backend/constraints-verified.txt
```

### 开发和测试环境

```bash
python -m pip install -r backend/requirements-dev.txt -c backend/constraints-verified.txt
```

### DSCARNet 可选环境

AggMap 1.2.1 的 PyPI 元数据包含过时的 `tensorflow-gpu` 和 `lapjv` 依赖，必须分两步安装：

```bash
python -m pip install -r backend/requirements-dscarnet.txt -c backend/constraints-verified.txt
python -m pip install aggmap==1.2.1 --no-deps
```

AutoAI 使用 SciPy 提供 `lapjv` 兼容实现，并且只调用 AggMap 的 SAR/CAR 映射，不使用 TensorFlow AggModel。

## 启动

### 一键启动

`run.py` 默认同时启动网页服务与本地训练 worker，并打开浏览器：

```bash
python run.py
```

常用参数：

```bash
python run.py --help
python run.py --host 0.0.0.0 --port 8000 --no-browser
python run.py --reload
python run.py --no-worker
```

### 手动拆分 Web 与 worker

终端 1：

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

终端 2：

```bash
python -m backend.app.runs.worker
```

只启动 uvicorn 时，训练任务会停留在 queued，直到 worker 启动。

### 正式本地 URL

## 分类模型 v2 契约

当前目录公开 **15 个目标分类模型**，且训练入口仅支持分类：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`、`dscarnet`。`Label` 即使为数字也按类别编码，`Repeat_index` 不改名并等同文档中的 Sample_ID。

新 Run 使用 `architecture_version="docx-classification-v2"`；旧模型类、旧 checkpoint 和旧 artifact 名仅作只读兼容，不把旧权重静默载入 v2 结构。二分类深度模型使用单 logit + `BCEWithLogitsLoss`，多分类使用多 logit + `CrossEntropyLoss`。15 个目标模型是能力目录，不等于当前环境全部可训练：`cnn_mamba1d` 在 Windows Conda 环境中因 `mamba-ssm` 依赖不可用而显示 unavailable；`dscarnet` 支持 SAR、CAR、dual 三种输入模式。

评估策略固定为：`stratified_holdout` 默认 8:1:1；`leave_one_repeat_index_cv` 每次留一个 `Repeat_index` 作 test、其余按 8:2 分 train/valid；`external_test_holdout` 使用主数据 8:2，独立数据作为唯一 test，禁止 CV。传统模型按验证集 balanced accuracy 选优，锁定参数后用 train+valid 重训。深度模型使用 AdamW、batch size 8、最多 200 epochs，并以最低 validation loss 保存最佳权重。

解释性方法矩阵：六个传统模型及 `pca_mlp`、`cnn_transformer1d`、`cnn_mamba1d` 使用真实类别 Log-loss 窗口遮挡，并同时提供类别等权全局结果与单样品结果；五个 1D 卷积模型使用 1D Grad-CAM 并保留输入梯度 sanity check；`dscarnet` 使用模式对应的 2D Grad-CAM 回投。窗口遮挡会将用户请求的窗口数解析为最接近且能整除特征数的窗口数，保证所有窗口等宽；例如 160 个特征请求 100 窗时实际使用 80 窗、每窗 2 点。旧 `feature_importance.*`、`sample_feature_importance.*` 和 `model.pt/model.pkl` 下载名继续兼容。
- 主页面：<http://127.0.0.1:8000/>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## 建模 CSV

建模文件固定需要以下字段：

```text
Index, Name, XXX, Intensity, Label, Repeat_index
```

- `XXX` 与 `Intensity` 是等长数值数组。
- `Label` 必填并始终作为分类类别。
- `Repeat_index` 表示同一样品的重复测量组；同组不得混入多个 `Label`。
- 不同样品的重复次数应一致。

最小工作流程：

1. 在网页上传拉曼/色谱原始 CSV 并完成预处理。
2. 下载统一 CSV，补全 `Label` 与 `Repeat_index`。
3. 将建模 CSV 上传到“AI 建模”。
4. 选择模型与评估口径，创建 queued Run。
5. worker 完成训练后查看 train/valid/test 指标、混淆矩阵、曲线和解释结果。
6. 下载预测、指标和模型 artifact。

仓库不附带真实 `data.csv`。本地验证数据、上传文件、模型和运行结果都位于 Git 管理范围之外。

## 目录结构

```text
backend/app/                 FastAPI、预处理、训练、Run 队列与模型
backend/tests/               自动化测试
backend/requirements*.txt    核心、开发、DSCARNet 依赖与验证约束
static/index.html            正式网页入口
static/js/                   正式前端模块
deploy/                      集群部署脚本与说明
docs/                        接口契约、ADR 和发布规范
storage/                     本地上传、SQLite 与训练产物（不进 Git）
run.py                       一键启动入口
```

## 验证

安装开发依赖后，在仓库根目录运行：

```bash
python -m pytest backend/tests -q
python -m compileall backend/app -q
python run.py --help
python -c "from backend.app.main import app; print(app.title)"
python -c "from backend.app.runs.worker import RunWorker; print(RunWorker.__name__)"
```

服务启动后：

```bash
curl http://127.0.0.1:8000/health
```

应返回：

```json
{"status":"ok"}
```

## 常见问题

### 安装了 CUDA 驱动但 PyTorch 仍使用 CPU

检查 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。若为 `False`，按 PyTorch 官方渠道重新安装与驱动/CUDA 匹配的 wheel，然后再安装 AutoAI 其余依赖。

### AggMap 安装时尝试拉取 tensorflow-gpu

不要直接执行普通的 `pip install aggmap`。先安装 `requirements-dscarnet.txt`，再执行 `python -m pip install aggmap==1.2.1 --no-deps`。

### 任务一直显示 queued

确认独立 worker 正在运行，或改用默认会同时启动 worker 的 `python run.py`。

### 修改代码后浏览器仍显示旧行为

未使用 `--reload` 的服务不会自动加载新代码。停止旧进程并重启，然后刷新浏览器。

### 上传后提示 Label 或 Repeat_index 无效

检查六个必需字段、空值、数组长度，以及同一 `Repeat_index` 是否只对应一个标签。

## 协作与发布

前后端契约见 `docs/frontend_backend_handoff.md`，部署见 `deploy/server_deploy.md`，GitHub 内容策略见 `docs/github_publish_policy.md`。架构决策记录在 `docs/adr/`；`AutoAI_开发计划.md` 仅保留为历史路线资料，不代表当前实现。
