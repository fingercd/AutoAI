# AutoAI — 谱学数据预处理与自动建模平台

## 快速启动

```bash
# 首次运行先安装依赖
python -m pip install -r backend/requirements.txt

# 命令行启动（自动打开浏览器）
python run.py

# 自定义端口 / 热重载
python run.py --port 9000 --reload
```

PyCharm：右键 `run.py` → Run 即可。

如果使用本机已有的 pytorch conda 环境，也可以直接：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe run.py
```

等效的手动命令：

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/          → 主页面
http://127.0.0.1:8000/ui        → UI 方案画廊
http://127.0.0.1:8000/docs      → Swagger API 文档
```

## 功能

- **数据预处理**：上传拉曼 / 色谱原始 CSV，范围截取（按行 / 按 X 轴）、生成统一建模 CSV；拉曼支持基线校正，色谱主流程默认使用 HPLC 三步预处理（统一时间轴插值、逐条消负、面积归一化）
- **AI 建模**：上传建模 CSV → 选择分层 holdout、`Repeat_index` 留一交叉验证或独立测试集 holdout → 查看指标、混淆矩阵和预测结果 → 下载模型与结果文件
- **10 类分类模型**：PLS-DA / SVM / Random Forest / XGBoost + 1D-CNN / 1D-Transformer / 1D-ResNet / 1D-Inception / 1D-TCN / DSCARNet
- **可解释性分析**：卷积模型使用 1D Grad-CAM-like，1D-Transformer 使用输入梯度归因，DSCARNet 使用 SAR/CAR 双通路 2D Grad-CAM 并回投到 1D 特征；传统 ML 使用 `baseline_macro_f1 - perturbed_macro_f1` 的窗口重要性
- **分类限定**：当前版本不实现回归任务；PLSR、SVR 是回归变体，文档中保留说明但前端训练选项不启用
- **8 套 UI 方案**：Workbench / Wizard / Dashboard / Console / Minimal Lab / Swiss / Dark Instrument / Warm Paper

## 稳定 Run 架构基线

稳定架构版本固定保留 master 已有的 10 个分类模型及其数学行为。新增模型、网络结构、二分类输出形式、DSCARNet 映射策略或传统模型搜索空间，必须使用独立模型计划并在固定数据集上验收。

训练请求只在 SQLite RunRepository 中创建 `queued` Run；独立本机 worker 通过 claim token 和 lease 执行训练，FastAPI 进程不以内置后台任务承担训练。`status.json` 是兼容投影，Run 成功前必须先原子提交 Manifest。请求不接受 `owner_id` 或 `tenant_id`，未来身份只由服务端 Principal 注入。

本地 worker 可单独启动：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe -m backend.app.runs.worker
```

## 当前前端说明

主工作台是 `static/index.html`。`/ui` 下的多套界面是候选或历史 UI 方案，用于比较设计，不一定代表当前正式交互。

色谱主页面默认提交到 `/api/preprocess/hplc`，并默认开启三步 HPLC 标准流程：`hplc_interpolate=true`、`hplc_subtract_min=true`、`hplc_normalize_area=true`。旧的 `/api/preprocess/chromatography` 仍保留为简单范围截取接口。

模型配置里，“重要性分段数”只对传统 ML 显示并提交，因为传统 ML 使用窗口遮挡/置乱后 `macro-F1` 的下降量做重要性分析。深度模型解释不依赖该分段数。

训练完成后，run 目录会保存 `config.json`、`label_map.json`、`split.json`、`metrics.json`、`cv_metrics.json`、`fold_metrics.csv`、`predictions.csv`、`cv_predictions.csv`、`hyperparameter_search.csv`、模型文件和解释性 JSON/CSV。留一交叉验证时模型文件只对应最后一折，最终报告性能以 CV 汇总为准。DSCARNet 额外保存 AggMap/PCA 映射元数据与 joblib 文件；当前下载接口只开放白名单内的常规 artifact。

DSCARNet 依赖 `aggmap==1.2.1` 及其兼容依赖。`backend/requirements.txt` 中记录了当前 conda 环境推荐的安装方式：`python -m pip install aggmap==1.2.1 --no-deps`，避免被旧 PyPI 元数据拉取不合适的依赖。

## 建模数据划分说明

当前分类评估有三种口径：

- 没有独立测试集，选择普通划分：按标签比例分层抽样，默认 `train : valid : test = 8 : 1 : 1`。
- 没有独立测试集，选择留一交叉验证：每折留 1 个 `Repeat_index` 独立样品组作为 test，其余样品组默认按 `8 : 2` 划分 train/valid。
- 有独立测试集：独立测试集只作为最终 test，主数据默认按 `8 : 2` 划分 train/valid。

同一个 `Repeat_index` 下的所有重复测量会全部进入同一个集合，避免同一样品的重复测量同时出现在训练集和测试集里造成数据泄漏和指标虚高。所有标准化参数、传统模型小范围调参和深度模型 early stopping 都只使用当前训练集/验证集完成。

上传建模 CSV 时需要保证：

- 同一个 `Repeat_index` 内只能对应一个 `Label`。
- 每个 `Repeat_index` 的重复测量次数应保持一致。
- `Label` 永远按分类标签编码；即使是数字，也不会按连续回归值处理。

例如 50 条数据、10 个 `Repeat_index`、每组 5 条重复测量时：

```text
分层 holdout: train 8 组、valid 1 组、test 1 组
留一交叉验证: 10 个外层折，每折 test 1 组，其余 9 组再按 8:2 生成 train/valid
独立测试集: 主数据 train/valid = 8:2，独立测试文件作为 test
```

## 验证

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端 `static/index.html` 是 HTML 内联脚本，不能直接 `node --check static/index.html`。需要先抽取 `<script>` 内容再检查：

```powershell
$html = [System.IO.File]::ReadAllText('D:\PythonProject\AutoAI\static\index.html', [System.Text.Encoding]::UTF8)
$matches = [regex]::Matches($html, '<script\b[^>]*>([\s\S]*?)</script>', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
$script = ($matches | ForEach-Object { $_.Groups[1].Value }) -join "`n"
$script | node --check --input-type=commonjs
```
