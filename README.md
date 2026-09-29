# SpecAutoAI

## Word 0904 新版训练入口（当前状态核对：2026-09-29）

经典 AI 建模入口已接通六模型特征方案比较：传统模型在 Train+Valid 内按 Sample_ID 分层五折搜索，全特征、Binning 5/10/20、PCA 90/95/99% 独立比较；CNN 只比较全特征与三种 Binning，按 validation loss 选择。每模型一个 Run，不增加重复次数。

有独立 Test、无独立 Test、留一法及留一法加独立 Test 均支持自定义比例（合计 10）；每个模式单独记住比例。结果页/对比页提供可展开的最佳配置、逐折参数与实际划分数量。特征工程热图使用指标切换器，PNG/SVG/CSV/ZIP 与页面同源；旧结果没有特征产物时不会补造或重训。

对比页“下载全部”旁边提供“下载 Excel”：第一个工作表是 `Index, Label, Sample_ID, 划分` 加每个模型一列的预测类别，第二个工作表布局相同，每个模型单元格是按 `label_map` 类别顺序排列、逗号分隔的六位小数概率文本。原始概率按类别归一化；写成六位小数后总和可能有舍入误差。`stratified_holdout` 与 `external_test_holdout` 用最终拟合模型预测全量记录并标注 `train/valid/test`，独立测试记录标为 `external_test`。普通留一法导出主数据 pooled OOF 行，每条记录只出现一次且划分为 `test`；留一法加独立 Test 时还包含 pooled OOF 主数据行和外部测试行，后者标为 `external_test`。该模式的独立 Test 是主指标，主数据 pooled OOF 仅作审计。历史 Run 没有全量明细时只导出已有 test/OOF 行，不补造。

对比页按“总体表现、分类指标、样品预测、特征方案、参数与划分”组织为标签页；特征方案标签只在后端能力开启时出现。分类预测指标显示各类别召回率与 Precision（精确率）两张热图，精确数值与 Support 折叠查看，单图支持 SVG、PNG 下载与放大，CSV 随归档提供。绘图版本为 `comparison-figures-v4`，过期或缺图的归档可从已有训练产物幂等补建，不重训。

更新代码后，需让 Web 与 Worker 都加载新代码，再刷新页面。不要在训练进行中强制重启。`/api/models` 的 `training_scheme` 字段表示当前 Web 的新版能力；未声明方案版本的 API 请求继续兼容旧行为。

当前训练入口仅支持分类。`cnn_mamba1d` 在依赖不可用时返回 unavailable，不使用替代网络。

SpecAutoAI 是面向拉曼与色谱/HPLC 曲线的预处理和分类建模平台。它通过同一个 FastAPI 服务提供网页、数据上传、预处理、训练任务、指标与模型产物下载；训练由独立 worker 从 SQLite Run 队列领取执行。

当前版本只支持分类。`Label` 即使是数字也按类别处理，不提供 PLSR、SVR 等回归入口。

## 正式功能

- 拉曼：按行号或 X 轴范围截取，固定先截取目标范围，再对截取后的片段执行基线校正。
- HPLC：保留行号/保留时间范围和线性插值开关；选择文件后逐个检测原文件名、点数与时间范围，支持任意一致且不少于 2 的点数，行号上限随当前批次动态变化。点数不一致时列出异常文件和实际点数，X 非严格递增时直接报错。
- 开启插值时，范围只选择固定 0–50 分钟轴上的实际目标点（例如 100–4000 共 3901 点），再按完整源 X 左右邻点线性映射；仪器轴与固定轴仅有一个采样间隔内的边界相位差时使用首尾两点线性延伸。关闭时保留所选原始 X/Y，但批次内轴不一致会拒绝导出。均不消负或做面积归一化。
- 分类评估：以分层 8:1:1 为目标且保证 Train/Valid/Test 各自类别完整、按 `Sample_ID` 留一交叉验证、独立测试集 holdout；批量比较时同一批次固定 `split_seed`，确保所有模型使用同一测试对象。
- 每次训练使用唯一 Run ID；训练完成后通过中央提示框在 3 秒后进入可刷新、可复制链接的独立“建模结果”页，也可立即查看或留在当前页。
- 结果页按 Train、Valid、Test 分层展示混淆矩阵、各类别指标和竖向预测分布；传统模型不显示训练曲线，深度模型曲线包含数值坐标。
- 前端建模页显示 Word 0904 六类：PLS-DA、Elastic Net（logistic_regression）、SVM、Random Forest、XGBoost、1D-CNN。其他模型保留后端兼容能力。
- 多模型批次保持“一模型一 Run”。新版后端支持 experiment_version=word-0904 的七/四种特征方案；历史任务未声明该版本时仍按原训练流程。经典多模型对比页支持响应式四指标条图、墨绿色方形矩阵、正误筛选分页，以及 PNG/SVG/CSV 整套归档下载；v2 和单模型结果页保持原样。没有特征工程结果的历史批次不补造数据。
- 可解释性实现已保留，但当前产品面通过内部 `TEMPORARILY_HIDDEN` 开关完全关闭：新训练不计算或生成解释性 artifact，结果投影仅返回隐藏状态，历史 artifact 也不开放直接下载。

## 双前端入口

特征方案训练与比较已启用；经典页面依据 `/api/models` 的 `training_scheme.enabled` 展示相应方案结果。历史批次缺少特征产物时只说明缺失，不补造或重训。可解释性仍由 `TEMPORARILY_HIDDEN` 关闭，新训练不生成解释性结果，历史解释性 artifact 也不开放下载。经典混淆矩阵放大窗口限制在视口范围内。

仓库同时维护两个受测试保护的原生静态前端，它们共享同一套 FastAPI、鉴权、Dataset/Run API、artifact 白名单和 `run-result-v1` 契约：

- 经典前端：`/`，代码位于 `static/index.html` 与 `static/js/`，继续作为兼容基线。
- v2 独立工作台：`/v2`（重定向到 `/static/v2/index.html`），代码位于 `static/v2/`。它提供工作台、AI 建模、训练记录、建模结果和分页面说明。

v2 是正式纳入仓库的并行前端，不是历史 UI 画廊，也不会替换或破坏经典入口。两套页面均为原生 HTML/CSS/JavaScript，不依赖 React、Vue、Vite 或外部 CDN。新增接口和结果字段应先维护共享契约，不能只适配其中一个前端。

## 环境要求

- 仓库依赖基线记录 Python `3.12.12`；本轮未重新验收运行环境。
- CPU 环境可直接安装核心依赖。
- NVIDIA CUDA 环境应先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装匹配驱动/CUDA 的 PyTorch，再安装其余依赖。不要依赖通用 requirements 自动猜测 CUDA wheel。
- `/api/models` 注册 17 个分类模型；当前可用性由运行环境的依赖探测动态返回，不能把目录总数当作可训练数。DSCARNet 依赖可选安装的 AggMap；`cnn_mamba1d` 需要 `mamba-ssm`。前端仅显示 `ui_visible=true` 的六个模型，隐藏模型在依赖可用时仍保留兼容 API。

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

SpecAutoAI 使用 SciPy 提供 `lapjv` 兼容实现，并且只调用 AggMap 的 SAR/CAR 映射，不使用 TensorFlow AggModel。

AggMap 1.2.1 的包元数据固定依赖多个过时版本，并声明本项目不使用的 `tensorflow-gpu`、`lapjv` 和 `shap`。因此按上述方式安装后，`pip check` 仍会报告 AggMap 的已知元数据冲突；这不表示 SpecAutoAI 使用的 SAR/CAR 映射链路缺少依赖。核心环境不安装 AggMap 时不受此问题影响。

## 启动

### 一键启动

本地提供两个明确的前端启动文件。两者启动的是同一个 FastAPI 服务和同一个训练 worker，区别只在于自动打开哪个页面：

| 启动文件 | 自动打开 | 用途 |
|---|---|---|
| `run_classic.py` | `http://127.0.0.1:8000/` | 经典前端，保留现有操作习惯与兼容入口 |
| `run_v2.py` | `http://127.0.0.1:8000/v2` | 新版 v2 独立工作台 |

推荐按需要选择其中一个：

```bash
python run_classic.py
python run_v2.py
```

两个脚本都支持公共启动参数，例如：

```bash
python run_v2.py --port 9000
python run_classic.py --no-browser
python run_v2.py --reload
```

不要在同一端口同时运行两个启动文件；如服务已经启动，直接在浏览器中访问 `/` 或 `/v2` 即可切换，不需要再启动第二个进程。

`run.py` 继续作为公共兼容启动器，默认打开经典前端：

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

### 服务器模式

对外监听必须启用服务器安全模式，并提供至少 32 个字符的随机 Bearer 令牌。不要把令牌写入命令行参数、脚本、URL 或仓库文件；应由进程管理器、受限环境文件或 secret manager 注入。

Linux 示例：

```bash
export AUTOAI_DEPLOYMENT_MODE=server
export AUTOAI_API_TOKEN="$(< /secure/path/autoai_api_token)"
export AUTOAI_PRINCIPAL_ID=server-admin
export AUTOAI_TENANT_ID=default
export AUTOAI_ALLOWED_ORIGINS=https://autoai.example.edu
python run.py --server --host 0.0.0.0 --no-browser
```

`AUTOAI_ALLOWED_ORIGINS` 可用逗号分隔多个明确来源，禁止 `*`。同源部署不需要额外跨域来源。浏览器首次访问受保护 API 时会要求令牌，令牌只保存在当前标签页的 `sessionStorage` 中。

如果通过 SSH tunnel 访问，推荐让服务继续监听 `127.0.0.1` 并使用 local 模式，无需将端口直接暴露到网络。

### 正式本地 URL

- 经典前端：<http://127.0.0.1:8000/>
- v2 独立工作台：<http://127.0.0.1:8000/v2>
- 经典专属结果页：`http://127.0.0.1:8000/#/results?run_id=<Run ID>`
- v2 专属结果页：`http://127.0.0.1:8000/static/v2/index.html#/results?run_id=<Run ID>`
- 训练记录：<http://127.0.0.1:8000/#/runs>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

专属结果页从 `GET /api/training/runs/{run_id}/result` 读取 `run-result-v1`。留一 Sample_ID CV 的测试主指标使用 pooled OOF；折均值和标准差仅作审计。

## 分类模型 v2 契约

后端保留 17 个分类模型，依赖可用性按环境检测。UI 以 ui_visible 过滤显示六类模型，Label 按类别编码，Sample_ID 用于分组，隐藏模型保留兼容 API。

UI 新 Run 使用 architecture_version=docx-classification-v4-0904，CNN 二分类也使用两类别输出与 CrossEntropyLoss。未指定新版 experiment_version 的兼容 API 保留旧模型与单 logit 契约；旧权重不载入新结构。详情见 docs/plans/2026-09-05-word0904-models-and-scientific-results.md。

评估策略固定为：`stratified_holdout` 以 8:1:1 为目标；如果 10% 对应的样品组不足以覆盖全部类别，Valid 和 Test 会自动提高到每类至少 1 个 `Sample_ID`，Train 同样必须类别完整。因而该模式要求每类至少有 3 个不同 `Sample_ID`，不足时训练会给出明确错误。`leave_one_sample_id_cv` 每次留一个 `Sample_ID` 作 test、其余按 8:2 分 train/valid；`external_test_holdout` 使用主数据 8:2，独立数据作为最终 test。交叉验证的主测试指标由所有折的 OOF 测试预测合并后计算；逐折均值与标准差仅作为审计值保留。传统模型以按 `Sample_ID` 分组的内层 5 折 Balanced Accuracy 选优，标准化和 PCA 均在内层训练折拟合，锁定参数后用外层 train+valid 重训。深度模型使用 AdamW、batch size 8、最多 200 epochs，并以最低 validation loss 保存最佳权重。

可解释性实现矩阵（传统模型窗口遮挡、卷积模型 Grad-CAM、DSCARNet 双通路 2D 回投）保留在后端源码中，但当前 `TEMPORARILY_HIDDEN`：新训练不计算、不写出解释性 artifact，经典与 v2 前端均不显示入口或发起请求，`sample_feature_importance.*`、`feature_importance.*`、`model_feature_visualization.json` 与 `dscarnet_mapping.json` 均不开放下载。`model.pt/model.pkl` 仍可由训练内部生成，但不属于公开下载白名单。

## 建模 CSV

新预处理文件使用 `wide-feature-v2` 宽表。一条曲线占一行，前四列名称与顺序固定；第 5 列起的列名是真实 `XXX` 坐标，单元格是对应的标量 `Intensity`：

```csv
Index,Label,Sample_ID,Name,0,0.0066675556740898788,...,50
1,A,S001,GSGC-001.csv,0.12,0.15,...,0.08
2,A,S001,GSGC-002.csv,0.11,0.16,...,0.09
```

- `Index`、`Label`、`Sample_ID`、`Name` 必须依次位于前四列；`Name` 保存原始文件名，`Label` 必填并始终作为分类类别。
- 第 5 列起的表头必须能解析为有限浮点数，数值唯一且严格递增；生成器使用 float64 可往返文本保存真实坐标。
- 每个特征单元格必须是有限标量强度；预处理输出最多保留 5 位小数，不再把整条数组塞入单元格，也不自适应降低精度。
- `Sample_ID` 表示同一样品的重复测量组；同组不得混入多个 `Label`，不同样品的重复次数应一致。
- 所有行必须共享表头所表示的公共轴。拉曼、简单色谱及关闭插值的 HPLC 在多文件轴不一致时拒绝导出；开启插值的 HPLC 使用公共固定目标轴。
- 独立测试集必须与主数据集具有完全相同的特征坐标及顺序。没有 `Name` 的 `wide-feature-v1` 宽表仍可训练；旧 `Index,Name,XXX,Intensity,Label,Sample_ID` 数组/JSON 文件会被明确拒绝，不做有损自动迁移。
- Excel 工作表最多 16,384 列；扣除四个元数据列后，v2 单个文件最多 16,380 个特征。

一次预处理仍只生成并下载一个统一建模 CSV。响应中的 `output_precision` 以 `format=wide-feature-v2`、`xxx_encoding=column_headers` 描述宽表，报告特征数、总列数和 Excel 兼容性。

最小工作流程：

1. 在网页上传拉曼/色谱原始 CSV 并完成预处理。
2. 下载统一 CSV，补全 `Label` 与 `Sample_ID`。
3. 将建模 CSV 上传到“AI 建模”。
4. 选择一个或多个模型与评估口径；单模型创建一个 queued Run，多模型创建统一 Batch，并为每个模型创建一个 Run。
5. worker 完成后，浏览器中央提示框提供“立即查看结果”和“留在当前页”；未操作时 3 秒后进入该 Run 的专属结果 URL。
6. 在结果页查看 Train/Valid/Test 或 pooled OOF 指标、三分区混淆矩阵、各类别指标、预测分布和训练/参数审计；批次比较页还提供 Accuracy 排名、四指标分组柱图或热力表，以及样品与类别热力图。
7. 在每项真实产物旁下载对应 JSON/CSV；裸 `model.pkl`、`model.pt` 和内部 joblib 本轮不开放。

仓库不附带真实 `data.csv`。本地验证数据、上传文件、模型和运行结果都位于 Git 管理范围之外。

## 目录结构

```text
backend/app/                 FastAPI、预处理、训练、Run 队列与模型
backend/tests/               自动化测试
backend/requirements*.txt    核心、开发、DSCARNet 依赖与验证约束
static/index.html            经典前端入口（兼容基线）
static/js/                   经典前端与共享 API 客户端
static/v2/                   v2 独立工作台、组件和 Node 纯函数测试
deploy/                      集群部署脚本与说明
docs/                        接口契约、ADR 和发布规范
storage/                     本地上传、SQLite 与训练产物（不进 Git）
run.py                       两种前端共享的底层启动器（默认经典前端）
run_classic.py               启动服务并打开经典前端
run_v2.py                    启动服务并打开 v2 工作台
```

## 验证

安装开发依赖后，可先跑快速 smoke，再执行交付门禁：

```bash
python -m pytest backend/tests/test_smoke.py -q
python -m pytest backend/tests -q
python -m compileall backend/app -q
node static/v2/tests/run-tests.mjs
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
{
  "status": "ok",
  "deployment_mode": "local",
  "contracts": {
    "run_result": "run-result-v1",
    "artifact_manifest": "run-artifact-manifest-v2",
    "run_summary": "v1"
  },
  "worker": {
    "available": true,
    "compatible": true,
    "contract_version": "run-artifact-manifest-v2",
    "live_count": 1,
    "last_seen_at": "...",
    "active_run_count": 0
  }
}
```

`status="ok"` 表示 Web 可用；`worker.available=false` 表示当前没有近期心跳，训练会停在 queued。`worker.compatible=false` 表示活跃 Worker 与 Web 的结果产物契约不一致；此时创建训练会返回 503 `worker_contract_mismatch`，应同时重启 Web 与 Worker。健康接口是匿名探针，只返回汇总，不暴露令牌、Principal 或 worker_id。

前端改动还要抽取 `static/index.html` 的内联脚本并用 `node --check --input-type=commonjs` 检查，同时执行 `backend/tests/test_result_frontend_contract.py` 中的 Node 纯函数测试。没有 Playwright/JSDOM 时不强行增加依赖。

## 常见问题

### 安装了 CUDA 驱动但 PyTorch 仍使用 CPU

检查 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。若为 `False`，按 PyTorch 官方渠道重新安装与驱动/CUDA 匹配的 wheel，然后再安装 SpecAutoAI 其余依赖。

### AggMap 安装时尝试拉取 tensorflow-gpu

不要直接执行普通的 `pip install aggmap`。先安装 `requirements-dscarnet.txt`，再执行 `python -m pip install aggmap==1.2.1 --no-deps`。

安装后执行 `pip check` 会按 AggMap 1.2.1 的旧元数据报告 `tensorflow-gpu`、`lapjv`、`shap` 和若干固定旧版本冲突，这是当前兼容安装方式的已知现象。SpecAutoAI 不调用 AggMap 的 TensorFlow AggModel，并为所用映射路径提供 SciPy `lapjv` 兼容层。

### pandas 提示 numexpr 版本过低

`numexpr` 不是 SpecAutoAI 的必需依赖。如果环境中已经安装旧版并触发 pandas 警告，可升级到 pandas 提示的最低版本，或在不被其他项目使用时卸载旧版 `numexpr`；不要仅为消除警告改动 SpecAutoAI 的核心依赖集合。

### 任务一直显示 queued

先查看 `/health` 的 `worker.available` 和 `worker.compatible`。没有 Worker 时确认独立进程正在运行，或改用默认会托管并监督 Worker 的 `python run.py`；版本不兼容时停止旧 Web/Worker，并从同一代码版本重新启动二者。

### 服务器页面提示需要访问令牌

确认服务端使用 `AUTOAI_DEPLOYMENT_MODE=server`，并由管理员安全分发与 `AUTOAI_API_TOKEN` 相同的令牌。浏览器只把令牌保存在当前标签页；刷新可继续使用，关闭标签页后需要重新输入。不要把令牌放在结果链接中。

### 升级到 server 模式后看不到历史 Run

这是所有权隔离的预期行为。先备份 `storage/`，然后执行只读预览：

```bash
python -m backend.app.runs.migration --dry-run \
  --owner-id server-admin --tenant-id default --rebind-unowned
```

确认数量正确后去掉 `--dry-run`。命令幂等，不移动或改写历史 Run 目录，也不会在服务启动时自动执行。owner/tenant 参数应与服务器进程的 `AUTOAI_PRINCIPAL_ID`、`AUTOAI_TENANT_ID` 一致。

### 结果页显示“部分结果”或下载按钮禁用

结果页会分别识别未生成、不适用、Manifest 缺失/损坏和文件完整性失败。不要手动猜下载 URL；保留 Run ID，检查页面原因、`/health` 和服务日志。ROC-AUC、ROC 与 Precision-Recall 当前没有正式训练产物，因此不会绘制虚假图表。

### 修改代码后浏览器仍显示旧行为

未使用 `--reload` 的服务不会自动加载新代码。静态文件会被新请求读取，但 Web/Worker Python 进程仍可能是旧版本；应停止并从同一提交同时重启 Web 与 Worker，再确认 `/health.contracts`、`worker.compatible=true` 和 OpenAPI 中存在 `/api/training/runs/{run_id}/result`。

### 上传后提示 Label 或 Sample_ID 无效

检查前四列是否依次为 `Index,Label,Sample_ID,Name` 且没有空值；第 5 列起的坐标表头是否为有限、唯一、严格递增的数值；所有强度是否为有限标量；同一 `Sample_ID` 是否只对应一个标签。已有 `wide-feature-v1` 仍兼容；旧六列数组/JSON 文件需要重新预处理导出。

## 协作与发布

前后端契约见 `docs/frontend_backend_handoff.md`，结果结构见 `docs/run_result_contract.md`，部署见 `deploy/server_deploy.md`，GitHub 内容策略见 `docs/github_publish_policy.md`。架构决策记录在 `docs/adr/`；`AutoAI_开发计划.md` 仅保留为历史路线资料，不代表当前实现。
