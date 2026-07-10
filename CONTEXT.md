# AutoAI 项目入口

## 项目目标

AutoAI 是一个面向拉曼、色谱/HPLC 曲线数据的预处理与自动建模平台。核心流程是上传原始 CSV，做范围截取、拉曼基线校正或 HPLC 标准化预处理，生成统一建模 CSV，再选择模型训练并查看指标、混淆矩阵、预测结果和关键特征解释。

## 当前状态

- 后端使用 FastAPI，前端静态页面由后端一起托管。
- 入口脚本是 `run.py`。
- 主要代码在 `backend/` 和 `static/`，主前端为 `static/index.html`。
- 测试集中在 `backend/tests/test_smoke.py`。
- 已有多个 UI 方案、模型相关改动和较多未跟踪文件，进入前必须先看 `git status --short`。
- 色谱主页面默认走 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 仍是简单范围截取兼容接口。
- `docs/frontend_backend_handoff.md` 是当前前后端接口契约；`AutoAI_开发计划.md` 是历史路线参考，不代表当前主链路。
- HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。
- 本地模式不接受 owner_id 或 tenant_id；未来身份只经服务端 Principal 注入。

## 预处理与接口事实

- 统一建模 CSV 字段为 `Index, Name, XXX, Intensity, Label, Repeat_index`。
- 拉曼预处理支持 `range_mode=row/x_value`、`baseline_order=range_then_baseline/baseline_then_range` 和 `baseline_method`，默认 `arPLS`。
- HPLC 预处理按固定顺序执行：线性插值到共同时间轴、逐条曲线减最小值消负、按真实时间轴梯形积分做面积归一化。
- HPLC 表单字段 `hplc_interpolate`、`hplc_subtract_min`、`hplc_normalize_area` 默认均为 `true`；响应包含 `processed_y`、`common_time`，可能包含 `common_time_path`。
- `/api/files` 只允许下载 `storage/uploads`、`storage/preprocessed` 下的文件；Run artifact 必须通过 Manifest-backed Run 路由下载。

## 模型与可解释性

- 当前建模任务仅支持分类；`Label` 永远按类别名编码。`PLSR`、`SVR` 属于回归变体，本版不出现在可训练模型列表。
- 当前稳定可训练模型目录固定为 master 已有的 10 个分类模型；新增模型、网络结构、二分类输出形式、DSCARNet 映射策略和传统模型搜索空间必须使用独立模型计划，并提供固定数据集上的对比验收。
- 模型算法改动必须使用独立模型计划，并提供固定数据集上的对比验收。
- 传统机器学习模型：`pls_da`、`svm`、`random_forest`、`xgboost`。
- 深度模型：`cnn1d`、`transformer1d`、`resnet1d`、`inception1d`、`tcn1d`、`dscarnet`。
- 三档输入维度按特征数区分：1000-3000、3000-6000、6000-10000；深度模型会按维度档选择轻量网络宽度、patch/stride、膨胀或映射尺寸。
- 分类评估支持三种口径：`stratified_holdout` 为无独立测试集时按标签比例 8:1:1 划分 train/valid/test；`leave_one_repeat_index_cv` 为无独立测试集时按 `Repeat_index` 留一作 test，其余按 8:2 划分 train/valid；`external_test_holdout` 为有独立测试集时主数据 8:2 划分 train/valid、独立测试集作最终 test。每个口径都只用当前训练集拟合标准化、调参或 early stopping。
- `cnn1d`、`resnet1d`、`inception1d`、`tcn1d` 生成 1D Grad-CAM / Grad-CAM-like 单样品解释。
- `transformer1d` 生成输入梯度归因解释。
- `dscarnet` 使用 AggMap/PCA 生成 SAR/CAR 双通路 2D 输入，解释性分析使用双通路 2D Grad-CAM 并回投到 1D 特征。
- 传统 ML 和无卷积模型使用窗口遮挡/置乱后的 `baseline_macro_f1 - perturbed_macro_f1`；正值越大表示 macro-F1 下降越多，该区间越重要，依赖 `feature_window_count`。
- 前端优先展示 `sample_feature_importance.json/csv`：样品曲线、第一重要红色区间、下方热力条和中文色标。
- 聚合解释性产物是 `feature_importance.json/csv`；单样品解释性产物是 `sample_feature_importance.json/csv`。传统模型通常只有聚合重要性，状态可能是 `ready`、`disabled`、`failed`、`unsupported`。
- DSCARNet 额外写入 `dscarnet_mapping.json`、`dscarnet_pca.joblib`、`dscarnet_sar_aggmap.joblib`、`dscarnet_car_aggmap.joblib`，但当前下载接口不对白名单外 joblib 文件开放。

## 运行方式

推荐本机启动：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe run.py
```

等效手动启动：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

打开：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/ui`
- `http://127.0.0.1:8000/docs`

## 验证方式

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端 JS 改动后，优先抽取 `static/index.html` 的内联 `<script>` 再用 Node 检查；不要直接 `node --check static/index.html`。没有 Playwright 时不要强行引入新依赖。

## 不要碰

- 不要删除用户上传数据、模型参考目录、截图或 zip；需要清理时先列清单。
- 不要读取或输出密钥、token、`.env`。
- 不要把 `.omc/state/`、缓存、模型权重、大数据加入 git。
- 不要把 UI 方案、后端接口和训练逻辑混在一次大改里。

## 下一步建议

1. 先整理 `.gitignore` 降低 git 状态噪声，但不要自动删除历史文件。
2. 梳理 `static/ui-*` 多套 UI 方案，明确正式、候选和归档状态。
3. 后续如要开放 DSCARNet joblib 下载，需要先扩展 artifact 白名单并补路径安全测试。
