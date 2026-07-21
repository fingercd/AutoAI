# SpecAutoAI Agent 工作规则

## 默认协作规则

- 默认使用中文回复，除非用户明确要求英文。
- 修改代码或文档前先运行 `git status --short`，只处理本次任务相关文件，不回滚用户已有改动。
- 当前本项目根目录的主要上下文以 `AGENTS.md`、`CONTEXT.md` 和 `README.md` 为准。
- 不读取、不输出 `.env`、`auth.json`、token、密钥、认证缓存、`.sandbox-secrets` 等敏感文件。
- 临时分析、脚本草稿和缓存放 `work/`；最终交付物放 `outputs/`。
- Git 只保留代码、配置和文档；数据、训练产物、缓存、视频、模型、压缩包、二进制工具不进 Git。
- 提交前只 stage 本次任务相关代码/文档，禁止在项目根目录盲用 `git add -A` 把数据或产物扫进索引。
- 不主动移动或删除历史大文件、用户上传数据、模型参考目录、截图或 zip；需要清理时先列清单并说明风险。

## 项目当前形态

- 后端是 FastAPI，入口为 `run.py` 或 `backend.app.main:app`。
- 前端主入口是 `static/index.html`，由后端静态托管；正式快照不包含历史 UI 画廊或候选方案。
- 并行新前端 `static/v2/index.html` 是独立工作台入口，复用 `static/js/api-client.js` 与 `run-result-v1` 契约；旧 `static/index.html` 与 `static/js/*` 仍受既有字符串契约测试保护，默认不就地重写。
- 快速回归在 `backend/tests/test_smoke.py`；结果接口、artifact、安全、前端纯函数、迁移和启动器另有专项测试，交付前运行整个 `backend/tests`。
- 推荐本机 Python 为 `C:\Users\lenovo\anaconda3\envs\pytorch\python.exe`。
- `docs/frontend_backend_handoff.md` 是当前前后端接口契约，`docs/run_result_contract.md` 是结果页契约；`AutoAI_开发计划.md` 是历史开发计划，不能把里面的 React/Vite、Redis/RQ、SQLite 等早期路线当成当前实现。
- 色谱主界面默认使用 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 只作为简单截取兼容接口保留。
- 训练 HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。
- local 模式只面向本机；server 模式必须配置 `AUTOAI_DEPLOYMENT_MODE=server`、至少 32 字符的 `AUTOAI_API_TOKEN` 和明确 CORS 来源。请求体不接受 owner_id/tenant_id，身份只经服务端 Principal 注入。
- server 模式训练只接受 `dataset_id`/`test_dataset_id`，不接受浏览器传入 data_path，也不回退根目录 `data.csv`。浏览器 token 只能进当前标签页 sessionStorage，不得写入 URL、localStorage、日志或仓库。
- Run 读取、取消、删除和下载必须按 Principal scope；历史 owner 为空 Run 在 server 默认不可见，只能通过显式 migration dry-run/rebind。
- 前端使用原生 Hash 路由；专属结果 URL 为 `#/results?run_id=...`，导航顺序固定为“AI 建模 → 建模结果 → 训练记录”。不引入 React/Vue/Vite。

## 预处理规则

- 统一建模 CSV 固定为 `Index, Name, XXX, Intensity, Label, Sample_ID`。
- 拉曼支持行号或 X 轴范围截取，处理顺序固定为先选择数据范围，再执行基线校正。
- HPLC 固定轴业务值集中在 `HplcGridConfig`，算法函数只接收配置；默认配置为 0–50 分钟、7500 点（包含首尾端点），不得在算法函数体内散落硬编码。
- HPLC 保留行号/X 轴范围选择和 `hplc_interpolate` 开关；每个原始文件仍必须解析出配置要求的完整点数且 X 严格递增。行号为 1 基、首尾包含，起止都必须位于 `1..point_count`，终止行留空才按完整点数处理，禁止把 7501、9000 等越界值静默截到末尾。
- 开启 HPLC 插值时，范围选择作用于固定目标轴：第 1–4000 行输出 4000 个目标点，第 100–4000 行输出 3901 个目标点；第 n 点的真实时间为 `start_minutes + (n - 1) * (stop_minutes - start_minutes) / (point_count - 1)`。强度使用完整源曲线的左右邻点做 float64 线性映射，边界相位差不超过一个采样间隔时允许用首尾两点线性延伸。关闭时导出所选原始 X/Y，轴不一致只警告。两种模式都不执行消负或面积归一化。
- 开启插值的新 HPLC CSV 使用可逆 `linspace-slice-v1` 描述完整分钟网格和本次 `offset/length`，避免 Excel 单元格上限；读取器继续兼容普通数组和历史 `linspace-v1`。关闭时 `XXX` 保存原数组。`Intensity` 继续使用最高 Excel 安全统一精度。一次 HPLC 预处理只生成一个六列主 CSV，不再生成 `_xxx.csv` 或返回第二下载地址。

## 建模与可解释性规则

- 当前建模任务仅支持分类；`Label` 即使为数字也按类别名编码，不作为连续回归目标。`PLSR`、`SVR` 是回归变体，本版训练入口不启用。
- 分类模型 v2 的能力目录固定公开 15 个目标模型：`pls_da`、`pca_lda`、`logistic_regression`、`svm`、`random_forest`、`xgboost`、`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`cnn_mamba1d`、`dscarnet`。
- 当前环境稳定可训练其中 14 个；`cnn_mamba1d` 因 `mamba-ssm` 依赖不可用，只在能力目录中返回 `available=false`，不得用近似网络静默替代。新增模型、网络结构、二分类输出形式、DSCARNet 映射策略和传统模型搜索空间必须使用独立模型计划，并提供固定数据集上的对比验收。
- 支持三种分类评估口径：无独立测试集时可选 `stratified_holdout`（按标签比例 8:1:1 划分 train/valid/test）或 `leave_one_sample_id_cv`（每折留 1 个 `Sample_ID` 作 test，其余按 8:2 划分 train/valid）；有独立测试集时使用 `external_test_holdout`（主数据 8:2 划分 train/valid，独立测试集作最终 test）。交叉验证测试集的主指标必须用所有折 OOF 预测合并计算。
- 所有标准化参数只由当前训练集拟合，并应用于同一评估口径下的验证集和测试集。
- 当前可训练深度模型：`pca_mlp`、`cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d`、`cnn_transformer1d`、`dscarnet`，均支持 test 集单样品可解释性分析；`transformer1d` 是 `cnn_transformer1d` 的兼容别名。
- `cnn1d`、`cnn1d_se`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM / Grad-CAM-like，并保留输入梯度 sanity check。
- 其中 `dscarnet` 当前实际使用 AggMap/PCA 的 SAR/CAR 双通路 2D 映射，再用双通路 2D Grad-CAM 回投到 1D 特征；不要把它当普通 1D CNN 解释。
- 六个传统机器学习模型及 `pca_mlp`、`cnn_transformer1d`（以及未来可用的 `cnn_mamba1d`）使用窗口遮挡后的真实类别 Log-loss 增量做重要性分析：`masked_loss - original_loss = log(p_before / p_after)`；全局结果按真实类别等权聚合，正值表示遮挡后真实类别置信度受损。
- 传统模型和无卷积深度模型同时生成 `feature_importance.json/csv` 与 `sample_feature_importance.json/csv`；当前正式前端优先展示单样品结果，旧下载接口保持可用。
- DSCARNet 会额外写入 `dscarnet_mapping.json` 和 AggMap/PCA joblib 文件；当前 artifact 下载白名单不开放这些 joblib 文件，除非同步更新接口和测试。
- 结果页以 `GET /api/training/runs/{run_id}/result` 的 `run-result-v1` 为准；CV 必须明确区分 pooled OOF、fold mean 和 fold std。
- `GET /api/training/runs?projection=summary` 可返回可空 `test_macro_f1`；仅成功且 Manifest 完整的 Run 读取测试主指标，CV 必须取 pooled test，不能取 fold mean。
- 新 Manifest 使用显式 catalog 与 SHA-256/大小校验；`model.pkl`、`model.pt`、joblib 和 `status.json` 不在新结果页下载白名单。`config.json` 只有在不含服务器路径时才可下载。
- 当前没有正式 ROC-AUC、ROC 或 Precision-Recall 产物；不得在前端伪造指标或空图。

## 验证命令

除单元/冒烟测试外，凡是改动训练、评估、模型、预处理或前端训练请求逻辑，交付前在项目根目录存在本地 `data.csv` 时还需要跑一次真实数据流程验证；`data.csv` 是本地验证数据，不能提交进 Git。优先选择轻量模型完成一次训练/评估闭环；如果因为耗时、环境、数据缺失或数据状态无法运行，必须在交付说明中明确写出未运行原因。

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
```

前端 `static/index.html` 内联脚本改动后，先抽取 `<script>` 内容，再用 `node --check --input-type=commonjs` 做语法解析；没有 Playwright 时不要强行引入新依赖。

## 交付说明

- 每轮完成后说明：改了什么、如何验证、遗留风险、下一步建议。
- 如果用户说“git一下”，先展示本次改动范围，再只提交相关文件。

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

本项目已经初始化 GitNexus，但它是按需使用的辅助工具，不是修改代码、诊断问题、阶段验收或交付前的默认门禁。优先使用定向文件阅读、`rg`、测试、运行日志和真实接口响应解决范围明确的问题。

## 使用条件与限额

- 仅在任务确实依赖跨模块调用链、公共接口影响面、大型重构或难以通过普通搜索确认的依赖关系时使用 GitNexus。
- `query` / `context` 用于陌生且跨模块的执行流；默认 `include_content=false`、`limit<=3`、`max_symbols<=8`。第一次结果明显不相关或目标未找到时，停止继续扩展图查询，改用 `rg` 和定向文件阅读。
- `impact` 按一个完整“改动簇”最多执行一次，不对每个函数、私有辅助方法或测试函数逐一执行。只有公共核心接口或高风险共享路径需要单独检查。
- `detect_changes` 仅在大型多文件改动完成后或用户明确要求提交前执行一次；普通小改、只读诊断、文档修改和未提交交付不要求运行。
- `analyze` 只有在索引明确过期、当前任务确实依赖图谱且旧索引会影响结论时才运行；每个任务最多一次，禁止按 Task、阶段或文件重复刷新。
- GitNexus 查询或索引失败通常不阻塞主体任务；只有用户明确把图谱、影响分析或索引状态列为交付物/验收条件时才视为阻塞。
- 运行时故障、服务启动、前端加载、HTTP 接口、数据库内容和服务器资源问题优先以真实运行证据为准，不用 GitNexus 结果替代日志、请求或测试。
- 不因项目已初始化 GitNexus 就自动读取全部 processes、clusters、schema，或连续调用 `query`、`context`、`impact` 做重复验证。

## 适用示例

- 修改训练状态机、模型注册公共入口或跨前后端契约时，可对相关改动簇做一次影响分析。
- 大范围重命名、抽取或移动公共 symbol 时，可使用图谱辅助确认调用者；仍需以测试和代码审查作为最终证据。
- 小型 UI 文案、局部样式、单文件 bug、测试断言、文档与配置调整默认不使用 GitNexus。

<!-- gitnexus:end -->
