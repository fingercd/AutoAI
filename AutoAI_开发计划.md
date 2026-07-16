# SpecAutoAI 谱学数据预处理与自动建模平台开发计划（历史路线）

> **历史文档，不作为当前实现或安装说明。** 本文件保留早期需求与技术路线原文，其中 React/Vite、BackgroundTasks、Redis/RQ、旧模型清单和旧命令均可能已经失效。当前事实以 `README.md`、`CONTEXT.md`、`AGENTS.md` 和 `docs/frontend_backend_handoff.md` 为准；维护者不得据此直接修改正式主线。

## 当前实现提示（仅帮助识别下文的过时内容）

- 当前没有独立 `frontend/` 主链路，正式页面是 `static/index.html`，由 FastAPI 同源托管；历史 UI 画廊已移除。
- 训练 HTTP 请求只创建 SQLite 中的 queued Run，由独立 worker 通过 claim token 和 lease 执行；FastAPI `BackgroundTasks` 不承担训练。
- 分类评估支持分层 8:1:1、`Sample_ID` 留一交叉验证和独立测试集 holdout。
- 当前能力目录有 15 个目标分类模型，其中本机基线可训练 14 个；准确清单与可用性见 `README.md`。
- 训练产物包含常规配置、指标、预测结果、模型文件和解释性 JSON/CSV；DSCARNet 额外写入 AggMap/PCA 映射元数据与 joblib 文件。
- 下面 1-9 步保留原始规划语境，出现 React/Vite、BackgroundTasks、Redis/RQ、旧模型或旧命令时，均按历史计划理解。

本文档根据 `AutoAI要求.docx`、示例建模数据 `data.csv`、原始拉曼数据目录 `拉曼/`、原始色谱数据目录 `色谱/` 整理。目标是在服务器内网部署一个网页系统，让用户通过浏览器完成数据上传、预处理、自动划分、模型训练、结果查看和结果下载。

本计划覆盖 1-9 步开发，不包含 Docker 安装与 Docker 化部署。原计划曾优先采用 Miniconda + FastAPI + React/Vite + 后台任务队列的方式跑通完整功能；当前实现已调整为 FastAPI 同源托管静态前端。

## 一、已确认需求与样例数据

### 1. 功能边界

系统分为两个大模块：

1. 数据预处理
   - 色谱数据预处理。
   - 拉曼数据预处理。
   - 多文件上传。
   - 原始数据曲线预览。
   - 用户选择保留行范围。
   - 统一整理为建模数据格式。
   - 拉曼数据需要基线校正，默认 `rampy.baseline` 的 `arPLS` 方法。
   - 整理结果可下载，供用户补充 `Label`。

2. AI 建模
   - 上传已预处理且补充标签的数据文件。
   - 数据预览：样本总数、每类样本数。
   - 默认分层划分训练集、验证集、测试集，比例 8:1:1。
   - 如已有独立测试集，默认训练集/验证集按 8:2 划分。
   - 支持基于 `Sample_ID` 的留一法交叉训练验证。
   - 第一阶段以 1D-CNN 分类模型为主。
   - 支持未来加载新模型算法。
   - 支持轻量自动超参数。
   - 支持类别不平衡处理：`class_weight` 或 `focal_loss`，默认 `None`。
   - 训练过程中展示训练状态与训练曲线。
   - 训练完成后下载 train/valid/test 预测结果。

### 2. 示例数据特征

`data.csv` 字段：

```text
Index, Name, XXX, Intensity, Label, Sample_ID
```

已检查结果：

```text
样本数：90
标签类别：Fe 45 条，Si 45 条
每条曲线长度：160
XXX 与 Intensity 均为数组字符串
Sample_ID：每个样品编号约 3 条记录
解析错误：0
```

原始文件：

```text
拉曼 CSV：50 个文件，字段类似 RamanShift,Intensity
色谱 CSV：6 个文件，无表头两列，第一列为 Time，第二列为 Intensity
```

### 3. 技术选型

第一阶段推荐：

```text
后端：FastAPI
训练：PyTorch
数据处理：pandas, numpy, scikit-learn
拉曼基线校正：rampy
任务队列：Redis + RQ，后续可升级 Celery
前端：React + Vite + TypeScript
图表：ECharts 或 Recharts
数据库：SQLite 起步，后续可切 PostgreSQL
部署：Miniconda + systemd + Nginx
```

模型优先级：

```text
第一阶段：1D-CNN
第二阶段：Transformer Encoder
暂不优先：U-Net
```

原因：当前任务是 1D 曲线分类，U-Net 更适合分割、去噪、重建或峰区域提取。若后续需要做拉曼去噪、峰检测、区域分割，可再引入 U-Net 或 U-Net encoder。

## 二、建议项目结构

```text
AutoAI/
  backend/
    app/
      main.py
      core/
        config.py
        paths.py
        database.py
      api/
        preprocessing.py
        datasets.py
        training.py
        runs.py
      services/
        parsers.py
        preprocessing.py
        dataset_builder.py
        training_jobs.py
        result_exporter.py
      ml/
        datasets.py
        splits.py
        models/
          cnn1d.py
          registry.py
        train.py
        evaluate.py
        losses.py
      schemas/
        preprocessing.py
        training.py
        runs.py
      worker.py
    tests/
      test_parsers.py
      test_preprocessing.py
      test_splits.py
      test_train_smoke.py
    requirements.txt
  frontend/
    src/
      api/
      components/
      pages/
        PreprocessChromatography.tsx
        PreprocessRaman.tsx
        DatasetUpload.tsx
        TrainingConfig.tsx
        RunDetail.tsx
        Results.tsx
      styles/
    package.json
  storage/
    uploads/
    preprocessed/
    runs/
    reports/
  sample_data/
    data.csv
    raman/
    chromatography/
  docs/
  AutoAI_开发计划.md
```

## 三、9 步详细实施计划

## 第 1 步：读取需求、确认数据格式、建立开发基线

### 实现

- 从 `AutoAI要求.docx` 整理需求清单。
- 对 `data.csv` 做字段、标签、曲线长度、解析错误统计。
- 对 `拉曼/` 和 `色谱/` 原始 CSV 做格式抽样。
- 建立 `sample_data/`，放入最小可复现样例。
- 建立 `docs/`，保存需求拆解、接口草案和数据格式说明。
- 明确敏感信息处理原则：参考网页账号密码只保留在原始文档，不写入代码、日志、计划或仓库。

### 运行测试

```powershell
python -m backend.app.services.parsers --inspect sample_data/data.csv
python -m pytest backend/tests/test_parsers.py
```

### 如何验证

- 控制台能输出样本数、类别数、每条曲线长度。
- `data.csv` 解析后没有数组长度不一致。
- 能正确识别：
  - 建模 CSV：`XXX` + `Intensity` 数组字段。
  - 拉曼原始 CSV：`RamanShift/Raman_Shift` + `Intensity`。
  - 色谱原始 CSV：无表头两列或 `Time,Intensity`。
- 生成 `docs/data_format.md`，里面说明统一格式。

### 产出

- `docs/requirements_breakdown.md`
- `docs/data_format.md`
- `sample_data/`
- 解析器最小测试用例

## 第 2 步：设计项目架构和运行环境

### 实现

- 创建后端、前端、存储目录。
- 建立 Miniconda 环境说明。
- 建立后端配置文件，统一管理：
  - 上传目录。
  - 预处理结果目录。
  - 训练 run 目录。
  - 最大上传文件大小。
  - 后台任务并发数。
  - 是否使用 GPU。
- 建立 `.env.example`，避免硬编码路径。
- 建立 SQLite 数据库表：
  - `datasets`
  - `preprocess_jobs`
  - `training_runs`
  - `run_artifacts`
- 建立 API 错误返回规范。

### 运行测试

```powershell
conda create -n autoai python=3.11
conda activate autoai
pip install -r backend/requirements.txt
python -m backend.app.main --check-config
python -m pytest backend/tests/test_config.py
```

### 如何验证

- 在新环境中可以安装依赖。
- 后端能读取 `.env` 或默认配置。
- `storage/` 下自动创建必要目录。
- 数据库能初始化。
- 没有任何代码写死 `D:\PythonProject\AutoAI` 这类本地路径。

### 产出

- `backend/requirements.txt`
- `backend/app/core/config.py`
- `backend/app/core/paths.py`
- `backend/app/core/database.py`
- `.env.example`
- `docs/local_setup.md`

## 第 3 步：实现数据解析、预处理和统一格式导出

### 实现

#### 3.1 色谱预处理

- 支持多 CSV 上传。
- 支持无表头两列和有表头两列。
- 第一列识别为 `Time`，第二列识别为 `Intensity`。
- 支持用户输入起止行，例如第 100 行到第 2000 行。
- 每个文件整理为一条样本记录：

```text
Index, Name, XXX, Intensity, Label, Sample_ID
```

- `Name` 默认来自文件名。
- `Label` 默认空。
- `Sample_ID` 支持前端手动填或自动从文件名规则推断。

#### 3.2 拉曼预处理

- 支持多 CSV 上传。
- 第一列识别为 `Raman_Shift` 或 `RamanShift`。
- 第二列识别为 `Intensity`。
- 支持用户输入起止行。
- 调用 `rampy.baseline` 做基线校正。
- 默认方法 `arPLS`。
- 预留方法选择，例如 `poly`、`drPLS`、`als`，具体以 rampy 实际支持为准。
- 输出校正后的 `Intensity`。

#### 3.3 统一格式

- `XXX` 保存横轴数组。
- `Intensity` 保存强度数组。
- 支持导出 CSV。
- 导出前做一致性检查：
  - `XXX` 和 `Intensity` 长度一致。
  - 数组长度不为 0。
  - `Name` 非空。
  - 建模前 `Label` 非空。

### 运行测试

```powershell
python -m pytest backend/tests/test_preprocessing.py
python -m backend.app.services.preprocessing --type raman --input sample_data/raman --start-row 1 --end-row 160 --output storage/preprocessed/raman_demo.csv
python -m backend.app.services.preprocessing --type chromatography --input sample_data/chromatography --start-row 1 --end-row 160 --output storage/preprocessed/chrom_demo.csv
```

### 如何验证

- 预处理后的 CSV 能被 Excel 打开。
- 每个上传文件对应一条统一样本记录。
- 拉曼预览中能同时看到原始曲线和基线校正后曲线。
- 色谱预览中能看到 Time-Intensity 曲线。
- 用户选择不同行范围后，输出数组长度随之变化。
- 错误文件能给出清楚提示，例如列数不足、非数字、空文件。

### 产出

- `backend/app/services/parsers.py`
- `backend/app/services/preprocessing.py`
- `backend/tests/test_preprocessing.py`
- 预处理 API 初版

## 第 4 步：实现训练脚本 CLI 和 1D-CNN 模型

### 实现

- 实现独立命令行训练入口，先不依赖网页。
- 输入：统一格式 CSV。
- 输出：完整 run 目录。

每次训练生成唯一 `run_id`：

```text
storage/runs/{run_id}/
  config.json
  label_map.json
  split.json
  model.pt
  metrics.json
  history.csv
  predictions.csv
  confusion_matrix.png
  loss_curve.png
```

#### 4.1 数据加载

- 解析 `XXX` 和 `Intensity` 数组。
- 标签编码并保存 `label_map.json`。
- 支持标准化：
  - `none`
  - `minmax`
  - `zscore`
  - `area`
- 保存预处理配置，保证推理时一致。

#### 4.2 数据划分

- 默认分层 8:1:1。
- 支持已有独立测试集字段，若后续数据包含 `Split` 或用户上传测试集，则训练/验证按 8:2。
- 支持 `Sample_ID` 留一法：
  - 每轮一个 `Sample_ID` 做测试。
  - 剩余样本再分训练/验证。

#### 4.3 模型策略

- 输入维度小于 500：浅层 1D-CNN，1-2 个卷积层。
- 输入维度 500-3000：中等 1D-CNN，2-3 个卷积层。
- 输入维度大于 3000：更深 CNN + 池化压缩。
- 样本量少于 100：降低通道数、加大 dropout、减少 epoch 默认值。
- 支持 `class_weight`。
- 预留 `focal_loss`。

### 运行测试

```powershell
python -m backend.app.ml.train --data data.csv --model auto_cnn --epochs 3 --output storage/runs/smoke_test
python -m pytest backend/tests/test_splits.py
python -m pytest backend/tests/test_train_smoke.py
```

### 如何验证

- 使用 `data.csv` 能跑完 3 个 epoch 的 smoke test。
- 生成 `metrics.json`、`history.csv`、`predictions.csv`、`model.pt`。
- `predictions.csv` 格式满足需求：

```text
dataset,index,Sample_ID,true_label,pred_label,prob_Fe,prob_Si,...
```

- `label_map.json` 中类别顺序固定。
- 固定随机种子后，重复运行划分结果一致。
- 小样本不会因模型过大快速过拟合到不可用状态。

### 产出

- `backend/app/ml/train.py`
- `backend/app/ml/models/cnn1d.py`
- `backend/app/ml/models/registry.py`
- `backend/app/ml/splits.py`
- `backend/app/ml/evaluate.py`
- `backend/app/ml/losses.py`

## 第 5 步：实现 FastAPI 后端接口

### 实现

建立后端 API：

```text
GET  /health
POST /api/preprocess/raman/upload
POST /api/preprocess/chromatography/upload
POST /api/preprocess/{job_id}/run
GET  /api/preprocess/{job_id}
GET  /api/preprocess/{job_id}/preview
GET  /api/preprocess/{job_id}/download

POST /api/datasets/upload
GET  /api/datasets/{dataset_id}/summary
GET  /api/datasets/{dataset_id}/preview

POST /api/training/runs
GET  /api/training/runs
GET  /api/training/runs/{run_id}
GET  /api/training/runs/{run_id}/logs
GET  /api/training/runs/{run_id}/metrics
GET  /api/training/runs/{run_id}/artifacts/{name}
```

后端职责：

- 接收文件，不直接长期阻塞训练。
- 文件保存到 `storage/uploads/`。
- 校验文件大小、扩展名、列格式。
- 返回清晰错误码和错误信息。
- 记录任务状态：
  - `pending`
  - `running`
  - `success`
  - `failed`
  - `cancelled`

### 运行测试

```powershell
uvicorn backend.app.main:app --reload --port 8000
python -m pytest backend/tests/test_api.py
```

也可用浏览器打开：

```text
http://127.0.0.1:8000/docs
```

### 如何验证

- `/health` 返回正常。
- `/docs` 中能看到所有接口。
- 上传 `data.csv` 后，数据集 summary 返回：
  - 样本总数。
  - 标签类别数。
  - 每类样本数。
  - 曲线长度分布。
- 上传错误格式文件，接口返回可读错误。
- 大文件上传不会让服务崩溃。

### 产出

- `backend/app/main.py`
- `backend/app/api/*.py`
- `backend/app/schemas/*.py`
- `backend/tests/test_api.py`

## 第 6 步：实现前端页面和交互流程

### 实现

前端使用 React + Vite + TypeScript。

页面设计以内部实验工作台为主，不做营销式首页。第一屏直接进入工作流。

#### 6.1 页面结构

```text
左侧导航：
  数据预处理
    色谱预处理
    拉曼预处理
  AI 建模
    数据集上传
    训练配置
    训练任务
    结果查看
```

#### 6.2 色谱预处理页

- 多文件上传。
- 文件列表。
- 起止行输入。
- 曲线预览。
- 整理结果预览。
- 下载按钮。

#### 6.3 拉曼预处理页

- 多文件上传。
- 起止行输入。
- 基线校正方法选择。
- 原始曲线与校正后曲线对比。
- 下载按钮。

#### 6.4 建模页

- 上传预处理后的 CSV。
- 显示样本总数、类别分布、曲线长度。
- 训练配置表单：
  - 模型：Auto / 1D-CNN。
  - 划分方式：分层划分 / Sample_ID 留一法。
  - epoch。
  - batch size。
  - learning rate。
  - normalization。
  - class imbalance 策略。
  - random seed。
- 创建训练任务。

#### 6.5 训练详情页

- 任务状态。
- 实时日志。
- loss / accuracy 曲线。
- 当前 epoch。
- 失败原因。

#### 6.6 结果页

- 指标卡片。
- confusion matrix。
- 训练曲线。
- train/valid/test 预测结果下载。
- 模型文件下载。
- 配置下载。

### 运行测试

```powershell
cd frontend
npm install
npm run dev
npm run lint
npm run build
```

### 如何验证

- 浏览器打开前端页面。
- 能上传多个拉曼或色谱 CSV。
- 曲线图显示正确。
- 页面刷新后仍能通过任务列表找回历史任务。
- 表单校验清楚，例如起止行为空、end 小于 start、未上传文件。
- 所有按钮在 loading 状态下不可重复点击。

### 产出

- `frontend/src/pages/*.tsx`
- `frontend/src/api/*.ts`
- `frontend/src/components/*.tsx`
- `frontend/package.json`

## 第 7 步：接入后台任务队列和训练状态更新

### 实现

训练和预处理都可能耗时，因此需要后台任务。

第一阶段建议使用 Redis + RQ：

```text
FastAPI 创建任务
  -> 写数据库状态 pending
  -> 推入 Redis 队列
Worker 执行任务
  -> 更新 running
  -> 写日志
  -> 保存 artifacts
  -> 更新 success 或 failed
前端轮询状态或 WebSocket 获取状态
```

功能要求：

- 训练任务不阻塞 HTTP 请求。
- 同一时间限制训练并发数。
- 支持任务失败后查看错误栈。
- 支持日志写入文件。
- 支持前端轮询：

```text
GET /api/training/runs/{run_id}
GET /api/training/runs/{run_id}/logs
```

后续可增加：

- 取消任务。
- 重跑任务。
- WebSocket 实时推送。

### 运行测试

```powershell
redis-server
python -m backend.app.worker
uvicorn backend.app.main:app --reload --port 8000
python -m pytest backend/tests/test_training_jobs.py
```

### 如何验证

- 前端点击开始训练后，接口立即返回 `run_id`。
- 训练仍在后台执行。
- 页面可以显示 `pending -> running -> success/failed`。
- Worker 停止时任务不会假装成功。
- 训练失败时前端能看到失败阶段和错误信息。
- 连续提交多个任务时，任务按队列执行，不会同时把 GPU/CPU 打爆。

### 产出

- `backend/app/worker.py`
- `backend/app/services/training_jobs.py`
- `backend/tests/test_training_jobs.py`

## 第 8 步：实现结果展示、报告导出和可追踪实验记录

### 实现

训练完成后，每个 run 必须可复现、可下载、可比较。

#### 8.1 指标

- train/valid/test accuracy。
- macro F1。
- weighted F1。
- precision。
- recall。
- confusion matrix。
- 每类样本数。

#### 8.2 文件

每个 run 目录保存：

```text
config.json
label_map.json
preprocess_config.json
split.json
metrics.json
history.csv
predictions.csv
model.pt
confusion_matrix.png
training_curves.png
report.md
```

#### 8.3 结果下载

按需求导出预测结果：

```text
第1列：dataset，train/valid/test
第2列：index
第3列：Sample_ID
第4列：true_label
第5列：pred_label
第6到N列：各类别预测概率
```

#### 8.4 报告

- 自动生成 Markdown 报告。
- 报告包含：
  - 数据摘要。
  - 训练配置。
  - 划分方式。
  - 模型结构摘要。
  - 指标。
  - 图表路径。
  - 风险提示，例如样本过少、类别不均衡。

### 运行测试

```powershell
python -m backend.app.ml.train --data data.csv --model auto_cnn --epochs 3 --output storage/runs/report_test
python -m pytest backend/tests/test_result_exporter.py
```

### 如何验证

- `predictions.csv` 列顺序符合需求文档。
- `metrics.json` 可以被前端读取。
- `report.md` 可以直接打开阅读。
- 训练配置和标签映射都能从 run 目录找回。
- 不同 run 不会互相覆盖。

### 产出

- `backend/app/services/result_exporter.py`
- `backend/tests/test_result_exporter.py`
- 前端结果页

## 第 9 步：服务器内网部署，不使用 Docker

### 实现

目标：部署在服务器账号下，内网用户通过 URL 访问。

#### 9.1 Miniconda 环境

```bash
conda create -n autoai python=3.11
conda activate autoai
pip install -r backend/requirements.txt
```

#### 9.2 后端服务

使用 `systemd` 管理：

```text
autoai-backend.service
autoai-worker.service
```

后端命令：

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Worker 命令：

```bash
python -m backend.app.worker
```

#### 9.3 前端构建

```bash
cd frontend
npm install
npm run build
```

构建产物由 Nginx 提供静态访问。

#### 9.4 Nginx 反向代理

```text
/          -> frontend/dist
/api/      -> http://127.0.0.1:8000/api/
/docs      -> http://127.0.0.1:8000/docs
```

需要配置：

```text
client_max_body_size
proxy_read_timeout
proxy_connect_timeout
上传目录权限
日志目录权限
```

#### 9.5 内网访问

- 绑定服务器内网 IP。
- 设置防火墙只开放内网端口。
- 如果学校/实验室有统一网关，按网关规则转发。

### 运行测试

```bash
systemctl status autoai-backend
systemctl status autoai-worker
curl http://127.0.0.1:8000/health
curl http://服务器内网IP/health
```

前端测试：

```text
http://服务器内网IP/
```

### 如何验证

- 服务器重启后服务自动恢复。
- 内网电脑可以打开网页。
- 上传文件成功。
- 后端接口正常。
- Worker 正常执行训练任务。
- Nginx 上传大小限制足够。
- 日志能在服务器上定位：
  - Nginx access/error log。
  - backend log。
  - worker log。
  - run 目录日志。

### 产出

- `docs/deploy_miniconda_nginx.md`
- `deploy/autoai-backend.service`
- `deploy/autoai-worker.service`
- `deploy/nginx_autoai.conf`

## 四、关键风险和避免方式

### 1. 数据泄漏

风险：同一个样本的重复测量同时进入训练集和测试集，导致测试结果虚高。

避免：

- 支持基于 `Sample_ID` 的留一法。
- 默认划分时记录 `split.json`。
- 对同一来源文件名或同一重复编号做分组检查。

### 2. 标签顺序错误

风险：训练时 `Fe=0, Si=1`，推理或导出时顺序反了。

避免：

- 每个 run 保存 `label_map.json`。
- 所有预测概率列按 `label_map.json` 固定顺序导出。

### 3. 预处理不一致

风险：训练数据做了标准化/基线校正，预测或复现实验时忘了同样处理。

避免：

- 保存 `preprocess_config.json`。
- 训练入口只读取统一格式。
- 推理入口必须加载训练 run 中的预处理配置。

### 4. 小样本过拟合

风险：当前示例只有 90 条，复杂模型容易虚高。

避免：

- MVP 使用轻量 1D-CNN。
- 小样本自动减少模型复杂度。
- 加 dropout。
- 指标必须同时报告 train/valid/test。
- 报告中提示样本量风险。

### 5. 后端接口阻塞

风险：HTTP 请求直接跑训练，网页超时或服务卡死。

避免：

- 训练通过后台队列执行。
- API 只创建任务并返回 `run_id`。
- 前端轮询状态。

### 6. 上传文件解析失败

风险：不同仪器导出的 CSV 表头、编码、分隔符不同。

避免：

- 自动探测 UTF-8/GBK。
- 自动探测逗号、制表符、分号。
- 错误信息包含文件名和失败行。
- 前端先预览再训练。

### 7. 服务器资源被打满

风险：多人同时提交训练任务，CPU/GPU/内存不足。

避免：

- Worker 并发数默认 1。
- 限制最大文件大小。
- 限制最大 epoch。
- 记录任务队列。
- 后续增加取消任务。

## 五、推荐开发节奏

### 第 1 个里程碑：命令行闭环

目标：不用网页，直接用 `data.csv` 跑完整训练。

包含：

- 数据解析。
- 数据划分。
- 1D-CNN smoke test。
- 结果导出。

验收：

```powershell
python -m backend.app.ml.train --data data.csv --epochs 3 --output storage/runs/mvp_cli
```

能生成完整 run 目录。

### 第 2 个里程碑：后端闭环

目标：通过 FastAPI 上传数据并启动训练。

包含：

- 数据上传 API。
- 数据摘要 API。
- 创建训练任务 API。
- 查询任务状态 API。

验收：

```text
http://127.0.0.1:8000/docs
```

可以手动调用接口完成上传、训练、下载。

### 第 3 个里程碑：网页闭环

目标：通过浏览器完成上传、配置、训练和结果查看。

包含：

- 前端上传页。
- 训练配置页。
- 任务详情页。
- 结果页。

验收：

浏览器中完成一次完整训练，不需要手动运行训练脚本。

### 第 4 个里程碑：服务器内网部署

目标：实验室成员通过内网 URL 使用。

包含：

- Miniconda 环境。
- Nginx。
- systemd。
- 日志。
- 权限。

验收：

其他电脑可以通过内网地址访问并完成一次训练。

## 六、给 Codex 的开发注意事项

每次让 Codex 写代码时，建议按小任务推进：

```text
请先实现第 N 步中的某一个模块。
要求：
1. 实现代码。
2. 添加或更新测试。
3. 运行测试。
4. 说明如何手动验证。
5. 不要修改无关文件。
```

必须反复强调：

- 不要硬编码本机路径。
- 不要把账号密码写入代码。
- 所有训练配置保存为 JSON。
- 每次训练生成唯一 `run_id`。
- 保存 `label_map.json`、`split.json`、`metrics.json`、`predictions.csv`。
- 训练和预处理失败要给出清楚错误。
- 先做可运行 MVP，再扩展 Transformer、U-Net 或复杂 AutoML。

## 七、下一步建议执行的第一个编码任务

建议下一次直接从命令行闭环开始：

```text
实现 backend/app/services/parsers.py：
1. 能解析 data.csv 中的 XXX/Intensity 数组。
2. 能解析拉曼原始 CSV。
3. 能解析色谱原始 CSV。
4. 输出统一的 SpectrumSample 数据结构。
5. 添加 pytest 测试。
6. 用当前示例数据运行测试。
```

这一步完成后，整个项目就有了稳定的数据入口，后面的预处理、训练、API 和前端都会顺很多。
