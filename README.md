# SpecAutoAI

面向拉曼与色谱 / HPLC 曲线的**数据预处理、分类建模与结果比较平台**。通过网页完成数据上传、预处理、模型训练和结果导出，无需手动编写训练脚本。

当前仅支持分类：即使 `Label` 为数字，也作为类别处理，不作为连续回归目标。

## 快速开始

依赖基线为 **Python 3.12**。建议在独立环境中安装，在项目根目录执行：

```bash
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`；Linux / macOS 使用 `source .venv/bin/activate`。然后安装并启动：

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt -c backend/constraints-verified.txt
python run.py
```

启动器会打开 [本地网页](http://127.0.0.1:8000/)，并自动托管训练 Worker。`python run_classic.py` 是同一前端的快捷入口。

> 使用 NVIDIA GPU 时，先安装与实际驱动匹配的 PyTorch，再安装其余依赖；通用依赖文件不会自动选择 CUDA 安装包。当前网页模型无需安装可选 AggMap / Mamba 依赖。

常用选项：

```bash
python run.py --port 8877
python run.py --no-browser
python run.py --workers 4 --threads-per-worker 5
python run.py --help
```

默认端口为 **8000**，不是固定使用演示端口 8877。同一端口不要重复启动服务。

## 使用流程

1. **准备数据**：已有建模宽表 CSV 可直接进入「AI 建模」；原始曲线先在「拉曼预处理」或「色谱预处理」中生成建模 CSV。
2. **加载数据集**：检查样品数、类别分布和曲线预览；需要独立测试时另行加载测试集。
3. **选择配置**：勾选模型和机器学习特征处理，可启用「按样品留一交叉验证」。配置区会显示预计实验数。
4. **开始训练**：查看完成数 / 总数、各组合状态，以及外层折、参数搜索或 epoch 的实时进度。
5. **查看结果**：按特征处理比较模型，查看指标、混淆矩阵、样品预测和逐折参数，下载图表、预测 Excel 或整套归档。

### 预处理

| 数据类型 | 处理方式 |
| --- | --- |
| 拉曼 | 按行号或 X 轴范围截取，再做基线校正；页面默认 arPLS |
| 色谱 / HPLC | 检查文件点数与时间范围，按行号或时间范围截取，可选线性插值到公共 0–50 分钟目标轴 |

HPLC 目标点数由批次动态检测，不固定为 4000。关闭插值时保留所选原始轴；多文件轴不一致会拒绝导出。HPLC 不执行消负或面积归一化。

## 模型与特征处理

### 网页可选模型

| 分组 | 模型 |
| --- | --- |
| 机器学习（10 个） | PLS-DA、Shrinkage-LDA、Lasso、Ridge、Elastic Net、SVM、KNN、Random Forest、Extra Trees、XGBoost |
| 深度学习（3 个） | 1D-CNN、1D CNN-SE、1D ResNet |

Lasso / Ridge 分别是 **L1 / L2 正则化逻辑回归分类器**，不是回归预测后取整；Shrinkage-LDA 使用自动收缩 LDA。DSCARNet 不在当前网页列表中。其他历史模型保留后端兼容能力，可用性以 `/api/models` 为准。

### 特征处理规则

- **机器学习**：可多选 BIN5、BIN10、BIN20、PCA5、PCA10、PCA20，每个「模型 × 特征处理」建立独立 Run。
- **BIN**：按宽度 5 / 10 / 20 对相邻特征点取均值，不是压缩到 5 / 10 / 20 个特征。
- **PCA**：固定保留 5 / 10 / 20 个主成分，不是方差百分比。训练数据不足时明确报错，不自动减少主成分数。
- **深度学习**：始终使用全部输入特征，每个模型只建立一个全特征 Run，**不执行外部 BIN / PCA**；保留训练折标准化和网络自身卷积、池化。

网页默认使用 **BIN5、快速训练（quick）、Z-score 和后端默认随机种子**，留一交叉验证需要主动勾选。标准化、PCA 只在允许的训练数据上拟合，测试数据不参与选参或权重选择。后端保留完整搜索能力，但它不是当前网页的默认策略。

### 实验数量

```text
Run 数 = 机器学习模型数 × 特征方案数 + 深度学习模型数
普通留一外层任务数 = Run 数 × 独立 Sample_ID 数
```

例如，全选 10 个机器学习模型、6 种特征处理和 3 个深度模型：

- 不做留一：`10 × 6 + 3 = 63` 个 Run。
- 24 个独立样品做留一：`63 × 24 = 1512` 个外层训练 / 评估任务。

**这不是底层模型拟合次数**：每个外层任务内部还可能尝试多个参数、重训或运行多个 epoch。

## 数据格式与评估

### 建模 CSV

新预处理导出 `wide-feature-v2` 宽表：一行对应一条测量曲线，前四列固定为 `Index,Label,Sample_ID,Name`，后续列名为真实坐标，单元格为强度。例如：

```csv
Index,Label,Sample_ID,Name,0,0.5,1
1,A,S1,sample_1.csv,0.12,0.15,0.08
2,A,S1,sample_2.csv,0.11,0.16,0.09
3,B,S2,sample_3.csv,0.32,0.25,0.18
```

以上仅示意文件结构，不是满足训练划分要求的完整数据集。

- `Label` 是类别，`Sample_ID` 标识独立样品；同一样品的重复测量使用同一个 ID，且标签必须一致。
- 特征坐标必须有限、唯一、严格递增，强度必须为有限标量；所有曲线共享公共轴。
- 独立测试集必须与主数据集的特征坐标及顺序一致。
- 兼容没有 `Name` 列的 `wide-feature-v1`；旧数组 / JSON 单元格格式需要重新预处理。

### 评估口径

所有划分均按 `Sample_ID` 整组进行，重复测量不会跨训练、验证和测试分区。

| 模式 | 数据划分 | 主测试成绩 |
| --- | --- | --- |
| 普通留出 | 主数据目标 Train:Valid:Test = 8:1:1 | 留出 Test |
| 样品留一 | 每次留一个 Sample_ID 作 Test，其余目标 Train:Valid = 8:2 | 全部外层测试预测合并后的 pooled OOF |
| 独立测试 | 主数据目标 Train:Valid = 8:2，独立数据作 Test | 独立 Test |
| 样品留一 + 独立测试 | 主数据执行留一，确定配置后用全部主数据重训 | 独立 Test；主数据 OOF 仅作审计 |

小样本下实际比例可能为保证类别完整而调整。普通三分区留出要求每类至少 3 个独立样品；其他模式和参数搜索也有相应数据量要求，不会拆散样品凑比例。

留一结果没有统一的「最佳参数」，应查看逐折配置；最终成绩使用 **pooled OOF**，不以折均值替代。

## 并行训练与结果展示

**Worker 是训练进程，不等于一个 CPU 核心。** 启动器默认按 CPU 预算托管最多 4 个 Worker，并限制每进程的计算线程，避免过度争抢资源。

- 不同模型 / 特征组合可以并行，同一个 Run 内的留一折依次执行。
- GPU 可用时，同一队列最多同时运行一个深度学习 Run，其余 Worker 可继续执行 CPU 模型。
- 多个深度任务等待同一 GPU 时排队是正常现象；全部任务长期排队则应检查 Worker。
- 更新代码需要让 Web 和 Worker 加载同一版本；**不要在训练进行中强制重启**。

比较页先展示「特征处理 × 模型」总览，再切换某种特征处理查看完整指标。深度模型只在 **全特征（深度学习）** 分组中展示，不把同一成绩复制到各 BIN / PCA 分组。历史结果保留原方案，不自动重训或改标签。

支持总体指标、混淆矩阵、类别指标、样品预测、参数与划分，以及 PNG / SVG 图表、预测 Excel 和批次归档。缺失或不适用的结果保留原因，不填零。当前可解释性入口关闭，也不提供无正式产物支持的 ROC / PR 图表。

## 常见问题与部署

| 问题 | 检查方法 |
| --- | --- |
| 一直显示「等待可用 Worker」 | 查看 `/health` 的 `worker.available`、`worker.compatible`；仅启动 Web 不会执行训练，推荐使用 `python run.py` |
| 更新后仍是旧行为 | 等当前训练结束，从同一代码版本重启 Web 和 Worker，再刷新网页 |
| GPU 未使用 | 运行 `python -c "import torch; print(torch.cuda.is_available())"`，检查 PyTorch 安装与驱动匹配情况 |
| 上传数据被拒绝 | 检查列名、坐标顺序、有限强度、同一 Sample_ID 的标签，以及主 / 独立数据轴是否一致 |
| 结果缺失或下载不可用 | 保留 Run ID，查看页面原因及日志；不要自行拼接文件下载地址 |

默认只监听本机 `127.0.0.1`。服务器推荐通过 SSH 隧道访问；直接对外监听必须启用 `server` 模式，配置至少 32 字符的 `AUTOAI_API_TOKEN`，跨域访问使用明确的允许来源。令牌不得进入 URL、日志或 Git，浏览器仅在当前标签页保存。

常用地址（默认端口）：[网页](http://127.0.0.1:8000/) · [API 文档](http://127.0.0.1:8000/docs) · [健康检查](http://127.0.0.1:8000/health)

## 开发与文档

项目使用 **FastAPI + 原生 HTML / CSS / JavaScript + SQLite 任务队列 + 独立 Worker**，不需要 React、Vite 或 Redis。网页入口为 `static/index.html`，后端入口为 `backend.app.main:app`。

```bash
python -m pip install -r backend/requirements-dev.txt -c backend/constraints-verified.txt
python -m pytest backend/tests -q
python -m compileall backend/app -q
```

数据、训练产物、权重与本地缓存不提交 Git。开发实验的记录与验收要求见 [AGENTS.md](AGENTS.md)。

| 文档 | 用途 |
| --- | --- |
| [前后端契约](docs/frontend_backend_handoff.md) | 接口、能力目录与请求规则 |
| [Run 结果契约](docs/run_result_contract.md) | 单任务结果与产物结构 |
| [模型比较契约](docs/model_comparison_contract.md) | 比较、聚合及下载规则 |
| [服务器部署](deploy/server_deploy.md) | 部署与访问配置；具体路径按实际环境调整 |
| [分类模型实现与验收](docs/plans/2026-10-18-classical-baselines.md) | 当前十种机器学习模型的实现依据 |
| [深度模型全特征规则与验收](docs/plans/2026-10-19-deep-full-features.md) | 深度模型特征策略、任务计数与验证记录 |

本 README 描述截至 **2026-10-19** 的当前行为。历史变更和实验记录保留在专题文档中，不作为当前默认配置。
