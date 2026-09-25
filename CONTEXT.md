# SpecAutoAI 项目入口

## 项目目标

SpecAutoAI 是一个面向拉曼、色谱/HPLC 曲线数据的预处理与自动建模平台。核心流程是上传原始 CSV，做范围截取、拉曼基线校正或 HPLC 标准化预处理，生成统一建模 CSV，再选择模型训练并查看指标、混淆矩阵、预测结果和关键特征解释。

## 当前状态

- 第八步训练检查：新任务为 State v8 / recipes revision v6，Guard on/off 仅控制额外准入；worker、发布和候选完整性始终检查。当前 worker 合同为 `training-worker-guard-v1`。见 `docs/step8_guard_contract.md`，待架构独立验收。

- 后端使用 FastAPI，前端静态页面由后端一起托管。
- 公共启动器是 `run.py`；本地使用 `run_classic.py` 打开经典前端，使用 `run_v2.py` 打开 v2 工作台。
- 主要代码在 `backend/` 和 `static/`；经典前端为 `static/index.html`，并行 v2 工作台为 `static/v2/index.html`。
- 快速回归在 `backend/tests/test_smoke.py`；Run 结果、artifact、安全、前端纯函数和迁移另有专项测试，交付时运行整个 `backend/tests`。
- 正式产品包含经典前端与 v2 独立工作台。两者均使用原生 Hash 路由、共享 `static/js/api-client.js`、同一 FastAPI API 与 `run-result-v1`；v2 不覆盖经典入口。
- 色谱主页面默认走 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 仍是简单范围截取兼容接口。
- `docs/frontend_backend_handoff.md` 是当前前后端接口契约，`docs/run_result_contract.md` 是 `run-result-v1` 结果结构；`AutoAI_开发计划.md` 是历史路线参考，不代表当前主链路。
- HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。
- 默认 local 模式只面向本机；server 模式必须配置 `AUTOAI_DEPLOYMENT_MODE=server` 与至少 32 字符的 `AUTOAI_API_TOKEN`，Bearer 身份只经服务端 Principal 注入。
- server CORS 只接受 `AUTOAI_ALLOWED_ORIGINS` 的明确来源，禁止 `*`；浏览器令牌只进当前标签页 sessionStorage。
- server Principal 只能访问同 owner/tenant 的 Dataset 与 Run；历史 owner 为空 Run 默认不可见，使用显式 migration dry-run/rebind。
- Run 状态转换由 SQLite 事务、claim token 和 lease 控制；本机 worker 可用 `python -m backend.app.runs.worker` 独立启动。
- `run.py` 默认托管 worker，并在意外退出时有限退避重启；`/health` 返回 Web 契约版本与匿名 worker 心跳/兼容性摘要。活跃旧 Worker 会使创建训练返回 503，避免新 Web 被旧 Worker 抢占任务。
- 独立结果页使用 `#/results?run_id=...`，刷新后从 `GET /api/training/runs/{run_id}/result` 恢复。
- `run-result-v1` 明确区分 direct、pooled OOF、fold mean 与 fold std；`analysis.splits.train/valid/test` 提供三分区混淆矩阵、分类报告和预测分布。传统模型不生成或展示 epoch history。
- Run 成功前必须提交 Manifest；新 Manifest 使用显式 catalog 和 SHA-256/大小校验。模型 pickle/PT 和 joblib 私有；无路径 `config.json` 才可下载。
- Agent API 已冻结为 `agent-session-v1`，Observation 为 `agent-observation-v1`。人工训练与 Agent 实验共用 `RunSubmissionService`；Agent reservation、幂等键和预算状态持久化在独立 `agent.sqlite3`，Runs DB 使用内部 durable submission mapping 原子关联 reservation 与 Run，并由显式 Principal-scoped reconciliation 恢复崩溃窗口；Test 数据不进入 Observation、Session 或对账响应。

## 预处理与接口事实

- 新预处理统一输出 `wide-feature-v2`：前四列固定为 `Index, Label, Sample_ID, Name`，`Name` 写入原始文件名，第 5 列起的列名是 float64 可往返的真实 `XXX` 坐标，单元格是标量 `Intensity`。坐标必须有限、唯一、严格递增，强度必须有限；预处理强度最多保留 5 位小数且不自适应降精度。训练读取继续兼容没有 `Name` 的 `wide-feature-v1`。
- 所有曲线必须共享表头表示的公共轴，主数据集与独立测试集也必须逐点同轴。旧六列数组/JSON、`linspace-v1`、`linspace-slice-v1` 文件不再可训练，也不自动迁移。
- Excel 总列数上限为 16,384，v2 扣除四个元数据列后最多 16,380 个特征。`output_precision` 固定报告 `format=wide-feature-v2`、`xxx_encoding=column_headers`、`xxx_precision=float64-roundtrip`、强度位数、特征/总列数和 Excel 兼容性。
- 拉曼预处理支持 `range_mode=row/x_value` 和 `baseline_method`，默认 `arPLS`；处理顺序固定为先选择范围，再执行基线校正。
- HPLC 固定轴由 `HplcGridConfig` 配置，覆盖 0–50 分钟，点数来自当前批次检测出的公共点数。选择文件后逐一显示原文件名、点数、时间范围和状态；任意一致且不少于 2 的点数均可处理。不一致时以唯一众数为期望值，列出异常文件及实际点数后拒绝；没有唯一众数时列出全部点数组。行号为 1 基、首尾包含，终止行留空时使用动态完整点数。
- HPLC 保留范围选择与插值开关：开启时范围选择固定目标轴的对应切片，再用完整源曲线左右邻点线性映射；第 n 点真实时间按完整网格的 `(n-1)/(point_count-1)` 位置计算。边界仅允许一个采样间隔内线性延伸。关闭时保留所选原始 X/Y，时间范围应用于每条原始轴，但多文件所选轴必须完全一致，否则拒绝导出。均不消负或做面积归一化。
- HPLC 的实际目标/原始公共轴逐点写入宽表特征表头；`common_time`/`hplc_axis` 继续描述预览所用的真实轴。成功请求只生成一个宽表建模 CSV 并只返回主 `download_url`；不生成逐点 `_xxx.csv`，也不返回 `xxx_download_url`/`xxx_rows`。
- `/api/files` 只允许下载 `storage/uploads`、`storage/preprocessed` 下的文件；Run artifact 必须通过 Manifest-backed Run 路由下载。
- server 模式训练请求必须使用 `dataset_id`/`test_dataset_id`，不接受 `data_path` 或默认 `data.csv` 回退。

## 模型与可解释性

分类模型 v2 的权威目标为 **14 个目标分类模型**：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`。当前仅支持分类；`Sample_ID` 是样品分组的规范字段。新 Run 写入 `architecture_version="docx-classification-v2"`，旧模型权重和 artifact 只读兼容。

14 个目标模型是 catalog 契约，不表示每个依赖在本机都可用。`cnn_mamba1d` 在当前 Windows Conda 环境中因 `mamba-ssm` 依赖不可用而禁用并跳过训练验收；不得回退成近似模型。DSCARNet 已退役，新请求及别名均被拒绝；二分类深度模型使用单 logit + `BCEWithLogitsLoss`。

`stratified_holdout` 以 8:1:1 为目标；当 10% 的组数不足以覆盖全部类别时，Valid/Test 自动提高到每类至少 1 个 `Sample_ID`，Train 同样保留全部类别，因此每类至少需要 3 个不同 `Sample_ID`，不足时明确拒绝。`leave_one_sample_id_cv` 的外层 test 留一个 `Sample_ID`，其余按 8:2 分 train/valid；`external_test_holdout` 主数据 8:2、独立数据为唯一 test，禁止 CV。交叉验证测试集的 Precision、Recall 和 Macro F1 以全部折 OOF 预测合并后计算，逐折均值/标准差只作审计。传统模型按 valid balanced accuracy 选优，再以 train+valid 重训；深度模型统一 AdamW、batch size 8、最多 200 epochs，并保存最低 validation loss 权重。

训练记录使用轻量 `projection=summary`；每项可带 `test_macro_f1`。该字段只在成功、结果完整且指标文件通过 Manifest 大小/SHA-256 校验时读取；holdout 取 `metrics.test.macro_f1`，CV 取 `cv_summary.pooled_test.macro_f1`，不可用时为 `null`，不得回退到 fold mean。

解释性矩阵以 `backend/app/training_explainability.py` 为准：六个传统模型及 `pca_mlp`、`cnn_transformer1d`、未来可用的 `cnn_mamba1d` 使用真实类别 Log-loss 窗口遮挡；`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM，并保留输入梯度 sanity check；历史 `dscarnet` 的 SAR/CAR/dual 解释产物保持只读兼容。旧 artifact 名保持兼容。

- 当前建模任务仅支持分类；`Label` 永远按类别名编码。`PLSR`、`SVR` 属于回归变体，本版不出现在可训练模型列表。
- 模型算法改动必须使用独立模型计划，并提供固定数据集上的对比验收。
- 六个传统模型：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`。
- 当前七个可训练深度模型：`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`；`transformer1d` 是 `cnn_transformer1d` 的兼容别名。
- 模型 profile 同时按训练样本数 N 和特征数 L 分档：N 为 `<=100`、`101-299`、`>=300`；L 为 `<=1000`、`1001-2999`、`>=3000`。模型输入范围会另行给出警告，但不会把警告阈值误当成 profile 分档。
- 分类评估支持三种口径：`stratified_holdout` 为无独立测试集时以 8:1:1 为目标划分 train/valid/test，并强制三个集合都包含全部类别；Valid/Test 的每类 1 个 `Sample_ID` 是高于比例的硬下限。`leave_one_sample_id_cv` 为无独立测试集时按 `Sample_ID` 留一作 test，其余按 8:2 划分 train/valid；`external_test_holdout` 为有独立测试集时主数据 8:2 划分 train/valid、独立测试集作最终 test。每个口径都只用当前训练集拟合标准化、调参、PCA 或 early stopping。
- 窗口遮挡的重要性为 `masked_loss - original_loss = log(p_before / p_after)`；请求窗口数会解析为最接近且能整除特征数的等宽窗口数。
- 新训练只生成并展示 `sample_feature_importance.json/csv`：样品曲线、第一重要红色区间、下方热力条、中文色标和 Top 区间，不再生成全局重要性。历史 `feature_importance.json/csv` 仅保留原 Manifest、Principal 和完整性约束下的直接下载兼容，不进入新 catalog 或结果页。
- 历史 DSCARNet 的 AggMap/PCA SAR/CAR 2D 结果可能包含 `dscarnet_mapping.json` 和若干 joblib 映射对象；当前下载接口不开放这些私有 joblib 文件。
- 当前没有正式 ROC-AUC、ROC 曲线或 Precision-Recall 曲线产物；结果 API 明确返回 unavailable，前端不绘制虚假图表。
- 仓库不包含真实 `data.csv`；该文件仅可作为本地验证数据存在，不得提交。

## 运行方式

推荐本机启动：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe run_classic.py
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe run_v2.py
```

二选一运行即可；两者共享端口、后端和 worker，不能同时占用默认 `8000` 端口。服务已启动时直接访问 `/` 与 `/v2` 切换。

等效手动启动：

```powershell
C:\Users\lenovo\anaconda3\envs\pytorch\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

打开：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/v2`
- `http://127.0.0.1:8000/#/results?run_id=<Run ID>`
- `http://127.0.0.1:8000/static/v2/index.html#/results?run_id=<Run ID>`
- `http://127.0.0.1:8000/#/runs`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

对外部署必须使用 `python run.py --server --host 0.0.0.0 --no-browser`，并由环境安全注入 token。SSH tunnel + `127.0.0.1` 可继续使用 local 模式。

## 验证方式

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
node 'D:\PythonProject\AutoAI\static\v2\tests\run-tests.mjs'
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
- 服务器升级前先备份 `storage/`；历史绑定先执行 `python -m backend.app.runs.migration --dry-run --owner-id ... --tenant-id ... --rebind-unowned`，确认后去掉 `--dry-run`。


## 第四步知识协议

新 CLI 配方任务使用 agent-state-v4 / agent-recipes-revision-v2，唯一知识开关为 module_policy.knowledge，CLI --knowledge 默认 off。新开启任务使用 knowledge-snapshot-rag-v1 / body-cosine-topk-v1：固定 BAAI/bge-small-zh-v1.5，只编码短卡正文；Train Evidence 模板或显式 user_text 查询进行一次精确 Top-3 检索。发布包、查询、结果、投影及来源绑定随 Session 冻结；知识引用独立记录，不改变 recipe/catalog/scientific digest 或训练执行。恢复只消费已有快照，不加载最新知识；旧 knowledge-snapshot-v1 保留原摘要只读兼容。v1/v2/v3 wire、幂等及启动摘要保留，详细规则见 docs/agent_step2_contract.md。

当前正式材料只有 5 张来源卡，20–50 张规模与冻结人工检索质量尚未验收；tau 未启用，无关查询也可能返回卡片。knowledge-off 无需加载 Embedding；on 的发布或模型故障明确失败。生产调用应配置实际 LLM tokenizer 与上下文窗口，整卡减少后仍超限则失败；tokenizer 库必须支持所用模型的文件格式，可将 BGE 服务依赖与 LLM 客户端依赖隔离。真实对照使用原 scripts/agent_ablation.py，失败保留分母，指标仅在 Finalize 后离线读取。

## 第四步 RAG 启动预算补修（2026-09-19）

新 RAG on 在创建 Session 前必须通过本地 tokenizer 与 context window 校验；CLI 的 `--llm-tokenizer` / `--llm-context-window` 或对应环境变量不得同时缺失。旧检查点解码、配置指纹和恢复保持原契约。整卡裁剪、引用范围、实际展示上下文和 journal 摘要继续使用现有实现。原 40 次实验显式配置预算，源码绑定保留，不因本次启动校验重跑。正式知识仍为 5 张，20–50 张规模与人工相关性验收未完成。
