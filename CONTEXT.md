# SpecAutoAI 项目入口

## 项目目标

SpecAutoAI 是一个面向拉曼、色谱/HPLC 曲线数据的预处理与自动建模平台。核心流程是上传原始 CSV，做范围截取、拉曼基线校正或 HPLC 标准化预处理，生成统一建模 CSV，再选择模型训练并查看指标、混淆矩阵、预测结果和关键特征解释。

## 当前状态

- 后端使用 FastAPI，前端静态页面由后端一起托管。
- 入口脚本是 `run.py`。
- 主要代码在 `backend/` 和 `static/`，主前端为 `static/index.html`。
- 测试集中在 `backend/tests/test_smoke.py`。
- 正式前端只有 `static/index.html` 与 `static/js/`；历史 UI 画廊和未跟踪候选不属于产品快照。
- 色谱主页面默认走 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 仍是简单范围截取兼容接口。
- `docs/frontend_backend_handoff.md` 是当前前后端接口契约；`AutoAI_开发计划.md` 是历史路线参考，不代表当前主链路。
- HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。
- 本地模式不接受 owner_id 或 tenant_id；未来身份只经服务端 Principal 注入。
- Run 状态转换由 SQLite 事务、claim token 和 lease 控制；本机 worker 可用 `python -m backend.app.runs.worker` 独立启动。
- Run 成功前必须先原子提交 `manifest.json`；Run 下载只允许 Manifest 声明的 downloadable artifact，`storage/runs` 不经 `/api/files` 暴露。

## 预处理与接口事实

- 统一建模 CSV 字段为 `Index, Name, XXX, Intensity, Label, Sample_ID`；`XXX` 与 `Intensity` 输出保留 5 位小数，超过 Excel 单元格 32767 字符上限时拒绝导出并提示调整范围。
- 拉曼预处理支持 `range_mode=row/x_value` 和 `baseline_method`，默认 `arPLS`；处理顺序固定为先选择范围，再执行基线校正。
- HPLC 预处理按固定顺序执行：线性插值到共同时间轴、逐条曲线减最小值消负、按真实时间轴梯形积分做面积归一化。
- HPLC 表单字段 `hplc_interpolate`、`hplc_subtract_min`、`hplc_normalize_area` 默认均为 `true`；响应包含 `processed_y`、`common_time`，可能包含 `common_time_path`。
- `/api/files` 只允许下载 `storage/uploads`、`storage/preprocessed` 下的文件；Run artifact 必须通过 Manifest-backed Run 路由下载。

## 模型与可解释性

分类模型 v2 的权威目标为 **15 个目标分类模型**：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`、`dscarnet`。当前仅支持分类；`Sample_ID` 是样品分组的规范字段。新 Run 写入 `architecture_version="docx-classification-v2"`，旧模型权重和 artifact 只读兼容。

15 个目标模型是 catalog 契约，不表示每个依赖在本机都可用。`cnn_mamba1d` 在当前 Windows Conda 环境中因 `mamba-ssm` 依赖不可用而禁用并跳过训练验收；不得回退成近似模型。`dscarnet` 支持 SAR、CAR、dual 三模式；二分类深度模型使用单 logit + `BCEWithLogitsLoss`。

`stratified_holdout` 默认 8:1:1；`leave_one_sample_id_cv` 的外层 test 留一个 `Sample_ID`，其余按 8:2 分 train/valid；`external_test_holdout` 主数据 8:2、独立数据为唯一 test，禁止 CV。交叉验证测试集的 Precision、Recall 和 Macro F1 以全部折 OOF 预测合并后计算，逐折均值/标准差只作审计。传统模型按 valid balanced accuracy 选优，再以 train+valid 重训；深度模型统一 AdamW、batch size 8、最多 200 epochs，并保存最低 validation loss 权重。

解释性矩阵以 `backend/app/training_explainability.py` 为准：六个传统模型及 `pca_mlp`、`cnn_transformer1d`、未来可用的 `cnn_mamba1d` 使用真实类别 Log-loss 窗口遮挡；`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM，并保留输入梯度 sanity check；`dscarnet` 使用 SAR/CAR/dual 模式对应的 2D Grad-CAM 回投。旧 artifact 名保持兼容。

- 当前建模任务仅支持分类；`Label` 永远按类别名编码。`PLSR`、`SVR` 属于回归变体，本版不出现在可训练模型列表。
- 模型算法改动必须使用独立模型计划，并提供固定数据集上的对比验收。
- 六个传统模型：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`。
- 当前八个可训练深度模型：`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`dscarnet`；`transformer1d` 是 `cnn_transformer1d` 的兼容别名。
- 模型 profile 同时按训练样本数 N 和特征数 L 分档：N 为 `<=100`、`101-299`、`>=300`；L 为 `<=1000`、`1001-2999`、`>=3000`。模型输入范围会另行给出警告，但不会把警告阈值误当成 profile 分档。
- 分类评估支持三种口径：`stratified_holdout` 为无独立测试集时按标签比例 8:1:1 划分 train/valid/test；`leave_one_sample_id_cv` 为无独立测试集时按 `Sample_ID` 留一作 test，其余按 8:2 划分 train/valid；`external_test_holdout` 为有独立测试集时主数据 8:2 划分 train/valid、独立测试集作最终 test。每个口径都只用当前训练集拟合标准化、调参、PCA/AggMap 或 early stopping。
- 窗口遮挡的重要性为 `masked_loss - original_loss = log(p_before / p_after)`；请求窗口数会解析为最接近且能整除特征数的等宽窗口数。全局结果先按真实类别分别求均值，再做类别等权聚合。
- 前端优先展示 `sample_feature_importance.json/csv`：样品曲线、第一重要红色区间、下方热力条和中文色标；聚合产物仍通过 `feature_importance.json/csv` 下载。
- DSCARNet 使用仅由当前训练折拟合的 AggMap/PCA 生成 SAR/CAR 2D 输入，额外写入 `dscarnet_mapping.json` 和若干 joblib 映射对象；当前下载接口不开放这些私有 joblib 文件。
- 仓库不包含真实 `data.csv`；该文件仅可作为本地验证数据存在，不得提交。

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
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

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

## 维护提醒

- 后续如要开放 DSCARNet joblib 下载，需要先扩展 artifact 白名单并补路径安全测试。
- 直接使用 uvicorn 不会启动训练 worker；本地一键入口 `run.py` 默认同时启动二者。
