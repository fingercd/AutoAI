# SpecAutoAI Agent 工作规则

当前实现核对日期：2026-09-29。功能事实依据当前工作区代码、专项测试及本项目近期已确认需求；历史计划和验收记录按日期与适用范围使用。

## 默认协作规则

- 默认使用中文回复，除非用户明确要求英文。
- 修改代码或文档前先运行 `git status --short`，只处理本次任务相关文件，不回滚用户已有改动。
- 本文件统一维护项目协作规则，`CLAUDE.md` 引用本文件。按任务读取 `CONTEXT.md`、`README.md` 和相关契约；文档、代码注释与当前实现不一致时，用源码、测试和最新已确认要求核对，不直接复制历史结论。
- 当前任务的授权持续有效，但过去任务中的提交、推送、训练或服务重启授权不自动延续到新任务。未经当前任务明确授权，不扩大为 commit、push、发布或其他外部写入；只读请求保持只读。
- 界面修改限定在请求涉及的页面、控件和样式范围；不顺带重做其他布局、训练策略或重复次数。对比页样式限定在 `#view-comparison`，避免改动全局样式影响其他页面。
- 不读取、不输出 `.env`、`auth.json`、token、密钥、认证缓存、`.sandbox-secrets` 等敏感文件。
- 临时分析、脚本草稿和缓存放 `work/`；最终交付物放 `outputs/`。
- Git 只保留代码、配置和文档；数据、训练产物、缓存、视频、模型、压缩包、二进制工具不进 Git。
- 提交前只 stage 本次任务相关代码/文档，禁止在项目根目录盲用 `git add -A` 把数据或产物扫进索引。
- 不主动移动或删除历史大文件、用户上传数据、模型参考目录、截图或 zip；需要清理时先列清单并说明风险。

## 项目当前形态

- 后端是 FastAPI，ASGI 入口为 `backend.app.main:app`；公共启动器 `run.py` 默认打开经典前端。`run_classic.py` 打开 `/`，`run_v2.py` 打开 `/v2`，二者共享后端、端口与 Worker，不应在同一端口重复启动服务。
- 经典入口为 `static/index.html`，v2 独立工作台为 `static/v2/index.html`，由后端静态托管；两者共享 `static/js/api-client.js` 与 `run-result-v1`。经典入口和共享脚本仍受字符串契约测试保护，局部修改须保持这些契约，不默认重写整套前端。
- 快速回归在 `backend/tests/test_smoke.py`；结果接口、artifact、安全、前端纯函数、迁移和启动器另有专项测试。运行时代码交付运行整个 `backend/tests`；纯文档修改按下方文档检查规则验证。
- 推荐本机 Python 为 `C:\Users\lenovo\anaconda3\envs\pytorch\python.exe`。
- 接口与结果以 [前后端契约](docs/frontend_backend_handoff.md)、[Run 结果契约](docs/run_result_contract.md) 和 [模型比较契约](docs/model_comparison_contract.md) 为导航。`AutoAI_开发计划.md` 仅是历史路线，不能据此引入 React/Vite 或 Redis/RQ；当前 Run 队列实际使用 SQLite。
- 色谱主界面默认使用 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 只作为简单截取兼容接口保留。
- 训练 HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。独立 Worker 从 SQLite 队列领取任务，状态转换由事务、claim token 和 lease 控制。
- `run.py` 默认托管并监督 Worker；仅启动 uvicorn 时还需单独运行 `python -m backend.app.runs.worker`。排查 queued 或更新未生效时核对 Web/Worker 是否加载同一代码、`/health` 的 worker 可用性与契约兼容性，以及 `/api/models` 响应中的 `training_scheme`；浏览器刷新不能更新 Python 进程。不在训练进行中主动强制重启服务。
- local 模式只面向本机；server 模式必须配置 `AUTOAI_DEPLOYMENT_MODE=server`、至少 32 字符的 `AUTOAI_API_TOKEN` 和明确 CORS 来源。请求体不接受 owner_id/tenant_id，身份只经服务端 Principal 注入。
- server 模式训练只接受 `dataset_id`/`test_dataset_id`，不接受浏览器传入 data_path，也不回退根目录 `data.csv`。浏览器 token 只能进当前标签页 sessionStorage，不得写入 URL、localStorage、日志或仓库。
- Dataset 的访问及 Run/Batch 的读取、取消、删除和下载必须按 Principal scope；历史 owner 为空 Run 在 server 默认不可见，只能通过显式 migration dry-run/rebind。
- 前端使用原生 Hash 路由；专属结果 URL 为 `#/results?run_id=...`，导航顺序固定为“AI 建模 → 建模结果 → 训练记录”。不引入 React/Vue/Vite。

## 预处理规则

- 新预处理统一输出 `wide-feature-v2` 宽表：前四列固定为 `Index, Label, Sample_ID, Name`，`Name` 保留原始文件名；第 5 列起的列名是真实、有限、唯一、严格递增的 `XXX` 坐标，单元格是有限标量 `Intensity`。建模读取继续兼容没有 `Name` 的 `wide-feature-v1`。
- 批次内所有曲线必须共享公共轴；独立测试集也必须与主数据逐点同轴。旧六列数组/JSON、`linspace-v1`、`linspace-slice-v1` 文件不再可训练，不得自动取首条轴迁移。
- Excel 总列数上限为 16,384，v2 扣除四个元数据列后最多 16,380 个特征。表头坐标使用 float64 可往返文本；强度最多 5 位小数且 `adaptive=false`，不得为适应单元格字符限制静默降精度或删点。
- 拉曼支持行号或 X 轴范围截取，处理顺序固定为先选择数据范围，再执行基线校正。
- HPLC 固定轴业务值集中在 `HplcGridConfig`，算法函数只接收配置；时间范围为 0–50 分钟，点数由当前批次检测出的公共点数动态构造，不得在算法或前端中写死。
- HPLC 保留行号/X 轴范围选择和 `hplc_interpolate` 开关；选择文件后先解析每个文件并展示原文件名、点数、时间范围与状态。批次点数一致时支持任意不少于 2 的点数；不一致时，以唯一众数作为期望点数并列出全部异常文件及其实际点数，众数并列时列出全部分组并拒绝预处理。源 X 必须严格递增。行号为 1 基、首尾包含，起止都必须位于 `1..point_count`，终止行留空才按检测出的完整点数处理，禁止静默截断越界值。
- 开启 HPLC 插值时，范围选择作用于固定目标轴：第 1–4000 行输出 4000 个目标点，第 100–4000 行输出 3901 个目标点；第 n 点的真实时间为 `start_minutes + (n - 1) * (stop_minutes - start_minutes) / (point_count - 1)`。强度使用完整源曲线的左右邻点做 float64 线性映射，边界相位差不超过一个采样间隔时允许用首尾两点线性延伸。关闭时导出所选原始 X/Y，但多文件所选轴不一致必须拒绝。两种模式都不执行消负或面积归一化。
- HPLC 的实际固定/原始公共轴逐点写入宽表特征表头。一次预处理只生成一个宽表主 CSV，不生成 `_xxx.csv` 或返回第二下载地址；`output_precision` 使用 `format=wide-feature-v2`、`xxx_encoding=column_headers` 和 `xxx_precision=float64-roundtrip`。

## 模型与特征方案训练

- 当前建模任务仅支持分类；`Label` 即使为数字也按类别名编码，不作为连续回归目标。`PLSR`、`SVR` 是回归变体，本版训练入口不启用。
- `/api/models` 保留 17 个后端模型，建模 UI 按 `ui_visible` 仅显示六项：`pls_da`、`logistic_regression`（Elastic Net）、`svm`、`random_forest`、`xgboost`、`cnn1d`。其余十一项以 `ui_visible=false` 隐藏，不得删除后端能力或用近似模型替代。
- 可用性按当前环境检测，不能把 UI 隐藏等同于不可训练。`cnn_mamba1d` 缺少 `mamba-ssm` 时必须返回 `available=false`；DSCARNet 需可选 AggMap 依赖。隐藏模型在依赖可用时仍接受兼容 API；`transformer1d` 继续作为 `cnn_transformer1d` 的兼容别名。
- 经典新训练以 `/api/models` 返回的 `training_scheme` 为唯一能力来源，显式发送 `experiment_version=word-0904`，对应架构 `docx-classification-v4-0904`。前端不维护独立启停常量；未声明版本的兼容 API 保留原语义，历史权重不直接载入新结构。v2 保持自身已有请求流程，不默认套用经典新版策略。
- 当前 `feature_policy.FEATURE_ENGINEERING_ENABLED=True`，特征方案训练已经恢复；此前暂时停用的记录不代表当前状态。五类传统模型比较全特征、Binning 5/10/20、PCA 90/95/99% 七方案；CNN 只比较全特征和三种 Binning，PCA 标为不适用。
- 新版传统模型每方案在 Train+Valid 合并池内按独立 `Sample_ID` 分组分层五折搜索；标准化、PCA 等变换只在各内层训练折拟合。按五折平均 Balanced Accuracy 选择参数与方案，并列采用固定候选顺序；锁定后在 Train+Valid 重训，保留 Test 只用于评估。旧手动搜索预算不能截断完整网格。
- 每次五折搜索的池中每类至少需要 5 个独立 `Sample_ID`，不足时报告具体类别和数量，不静默减折。留一法每个外层折独立拟合、搜索和选优；无效候选或失败方案记录原因，不用 Test 分数补选。
- 经典新版 CNN 使用后端公布的默认策略：AdamW、最多 200 epochs、batch size 8、学习率调度与早停；每方案保存最低 validation loss checkpoint，再按 validation loss 选方案。二分类也使用两类别输出与 CrossEntropyLoss。经典请求不混入与固定策略冲突的旧手动搜索参数。
- 算法、网络结构、二分类输出形式、DSCARNet 映射或搜索空间的改动须有独立模型计划与固定数据集对比验收。细节查 [0904 模型计划](docs/plans/2026-09-05-word0904-models-and-scientific-results.md) 和 [后续接通记录](docs/plans/2026-09-07-word0904-training-connection.md)，并核对当前实现；旧计划中的 scientific-results 前端与短轮次验收记录不能当成当前界面或完整验收。

## 数据划分与评估口径

| 评估模式 | 默认划分 | 主测试指标 |
|---|---|---|
| `stratified_holdout` | 主数据 Train:Valid:Test = 8:1:1 | 留出 Test 的 direct 指标 |
| `leave_one_sample_id_cv` | 每折留一个 `Sample_ID` 作 Test，其余 Train:Valid = 8:2 | 所有外层 Test 预测合并后的 pooled OOF |
| `external_test_holdout` | 主数据 Train:Valid = 8:2，独立数据作 Test | 独立 Test 的 direct 指标 |
| `leave_one_sample_id_cv_with_external_test` | 主数据执行留一法，选定最终配置后用全部主数据重训 | direct external Test；主数据 pooled OOF 仅作审计 |

- 经典页面允许自定义正整数比例，参与划分的比例合计为 10；两段模式不提交内部 `split_test`，后端规范化为 0。各模式分别记忆比例和保留留一法勾选状态，刷新或重新渲染不重置；加载独立 Test 不得自动取消留一法或覆盖比例。
- 所有拆分按 `Sample_ID` 整组进行，区分独立样品数和测量条数；同组不得跨分区或混入不同标签。展示实际样品数、测量条数及比例，小样本无法精确达到目标比例时说明原因，不拆散重复测量凑比例。
- 三分区 holdout 要求 Train/Valid/Test 各自类别完整，每类至少 3 个不同 `Sample_ID`，必要时提高 Valid/Test 的实际占比；新版传统模型还须满足各搜索池的五折条件。
- 标准化、PCA/AggMap、参数及 checkpoint 选择只使用当前允许的训练或选参数据；Test 不参与拟合和选优。独立 Test 完全不进入主数据 CV；CNN 外部最终重训使用主数据选定的 epoch 数和实际自定义划分比例。
- 普通留一法没有统一最佳参数，展示逐折配置；总体成绩取 pooled OOF，fold mean/std 只作审计。传统模型最终重训的 Test 指标与训练期 Train/Valid 审计明确区分，不能把重训见过的数据称为独立验证。

## 结果、批次与可解释性

- 单 Run 结果以 `GET /api/training/runs/{run_id}/result` 的 `run-result-v1` 为准；批次创建与比较沿用 `training-batch-v1`、`model-comparison-v1`。新 Batch 固定每模型一个 Run，`repeat_count=1`；不增加重复实验或重复一致性图。历史重复批次默认展示每模型第一条成功完整 Run，按单次结果切换，不合并为均值矩阵。
- 多模型进度按 Batch 及子 Run 恢复，刷新后不能丢失其他模型或只显示某个单 Run；保留方案、外层折、搜索或 epoch 的真实进度。停止后重置当前训练面板、停止轮询与自动跳转；停止/删除只清理目标 Run 或 Batch 的产物与归档，不影响其他任务。
- 最佳配置、逐折参数、留出样品和实际划分数量按需展开。`feature_experiments.json/csv` 经 Manifest 校验后可投影和下载；普通留一法不虚构全局最佳星号。
- 模型比较必须核对主/独立数据快照、评估与聚合口径、划分 digest、测试样品集合及标签映射一致，且来源产物通过 Manifest 校验；不一致则说明不可比较，不生成排行或热图。缺失、不适用、部分完成和失败保留原因，不填零，不自动重训历史 Run。
- 经典比较页使用 `static/js/comparison-page.js/css` 和后端 `comparison_figures.py`；图像展示及 PNG/SVG 导出复用同一绘图函数，CSV 与 ZIP 对应同一数据。特征热图为七方案 × 所选模型，默认 Balanced Accuracy，可切换 Accuracy、Macro-F1、Weighted-F1；选优标记只来自验证/CV。类别 Recall 与 Precision 热图分别读取对应投影字段，精确数值与 Support 可展开查看；当前图集绘图版本为 `comparison-figures-v4`。
- `GET /api/training/batches/{batch_id}/predictions.xlsx` 导出预测类别与预测概率两个工作表。概率按 `label_map` 顺序写为逗号分隔六位小数文本，舍入后不保证总和严格等于 1。普通留一法导出主数据 pooled OOF；留一法加独立 Test 同时导出主数据 OOF（`test`）和外部预测（`external_test`），其中外部 Test 是主指标、OOF 仅为审计。缺少新全量明细的历史 Run 仅回退导出现有测试预测，不重算或补造。
- 批次归档支持自动保存和幂等补建，历史比较从训练记录的“查看所属对比”进入；已有真实产物但图集过期或缺图时可原子补建派生归档，缺少训练产物时不补造数据。部分失败只保存可比较结果；归档失败不改写成功 Run，停止/删除阻止迟到归档发布。下载沿用 Principal、Manifest 与公开文件白名单。
- `GET /api/training/runs?projection=summary` 可返回 `batch_id` 和可空 `test_macro_f1`。仅成功且 Manifest 完整、指标文件通过大小/SHA-256 校验时读取：普通留一法取 pooled OOF，holdout 及留一法加独立 Test 取 direct Test；不能回退为 fold mean，缺失或不一致时返回 `null`。
- 新 Manifest 使用显式 catalog 与 SHA-256/大小校验；`model.pkl`、`model.pt`、joblib 和 `status.json` 不属于新结果页公开下载。`config.json` 只有在不含服务器路径时才可下载。传统模型不生成或展示 epoch history；当前无正式 ROC-AUC、ROC 或 Precision-Recall 产物，不伪造指标或空图。
- 特征方案比较已启用，但可解释性仍执行源码中标注 `TEMPORARILY_HIDDEN` 的策略：训练入口强制 `feature_selection_enabled=false`，不计算或生成解释性结果，两套前端不渲染或请求入口。历史 `feature_importance.*`、`sample_feature_importance.*`、`model_feature_visualization.json` 和 `dscarnet_mapping.json` 也不开放下载；旧 Manifest 不得形成旁路。
- `training_explainability.py` 中窗口遮挡、1D Grad-CAM/Grad-CAM-like、输入梯度 sanity check 与 DSCARNet 回投实现继续保留。DSCARNet 训练仍使用当前训练折拟合的 AggMap/PCA SAR/CAR 双通路 2D 映射，可内部生成映射 JSON/joblib；不要把它当普通 1D CNN。恢复解释性或开放映射下载须有明确需求，并同步接口和测试。

## 验证命令

纯文档修改检查事实、引用、UTF-8 编码和目标文件的 `git diff --check -- <文件>`，无需启动服务、训练或完整运行时测试。运行时代码改动先执行最相关专项测试，交付前运行整个 `backend/tests` 与 compileall；测试和验收使用隔离临时目录，避免影响现有训练与用户产物。

除单元/冒烟测试外，凡是改动训练、评估、模型、预处理或前端训练请求逻辑，交付前在项目根目录存在本地 `data.csv` 时还需要跑一次真实数据流程验证；`data.csv` 是本地验证数据，不能提交进 Git。优先选择轻量模型完成一次训练/评估闭环；如果因为耗时、环境、数据缺失或数据状态无法运行，必须在交付说明中明确写出未运行原因。

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端 `static/index.html` 内联脚本改动后，先抽取 `<script>` 内容，再用 `node --check --input-type=commonjs` 做语法解析，并运行相关 Node 纯函数/契约测试；v2 或共享模块改动还需运行 `node static/v2/tests/run-tests.mjs`。没有 Playwright 时不要强行引入新依赖。

训练入口改动重点覆盖四种划分、自定义比例、逐折防泄露与 Test 口径；多模型改动保留进度、刷新恢复、停止重置、比较和归档回归。新版完整训练验收要求传统模型完整网格与 CNN 正常调度/早停流程，不以读取历史结果或仅跑两轮 CNN 代替；合成数据只证明相应功能，不代表真实数据性能。2026-09-07 核对时根目录无 `data.csv`，后续任务仍须重新检查，不能擅自改用上传数据启动训练。

## 交付说明

- 每轮完成后说明：改了什么、如何验证、遗留风险，以及确有必要的下一步建议。历史测试成绩注明日期和适用代码，不当成本轮验证；区分合成数据验收与真实用户数据验收。
- 如果用户说“git一下”，先展示本次改动范围，再只提交相关文件。

<!-- gitnexus:start -->
## GitNexus 按需使用

GitNexus 的 `AutoAI` 索引是辅助证据，不是普通修改或交付的强制门禁。范围明确的问题优先使用定向阅读、`rg`、测试、日志和真实接口响应；小型 UI 文案、局部样式、单文件修复和文档修改默认不查询图谱。

- 任务确实涉及跨模块调用链、公共接口影响、大型重构或普通搜索难以确认的依赖时，再按需使用 `query`、`context`、`impact` 或 `trace`，先确认目标仓库和源码范围。
- 查询保持精简：默认 `include_content=false`、`limit<=3`、`max_symbols<=8`（仅用于支持这些参数的工具）。结果明显无关或符号缺失时转回源码，不连续扩展图查询。
- `impact` 按完整改动簇执行，默认每簇一次，不逐函数、私有辅助方法或测试重复分析；公共核心接口或高风险共享路径才单独检查。发现 HIGH/CRITICAL 风险时说明实际影响和验证安排，不忽略风险。
- `detect_changes` 在大型多文件改动收尾或明确要求提交且图谱有助于核对范围时按需执行；普通小改、只读诊断、纯文档修改和未提交交付不要求运行。图谱结果不能替代 diff 与测试。
- 不自动刷新索引。只有索引明确过期、当前任务确实依赖图谱且允许相关写入时才运行 `analyze`，每任务最多一次；不默认读取全部 processes、clusters 或 schema。
- 工具不可用、索引过期或符号未收录时说明局限，并用源码和测试继续。仅当用户明确把图谱或索引状态列为验收条件且无法满足时，才将其作为阻塞；不能把未命中理解为没有影响。
- 按实际任务读取环境中对应 GitNexus 技能，不固定已过期的技能路径或索引统计。工具生成的强制模板与本节冲突时，以本节已确认的项目策略为准，保持 `CLAUDE.md` 统一引用，避免重新引入重复门禁。

<!-- gitnexus:end -->
