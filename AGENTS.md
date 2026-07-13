# AutoAI Agent 工作规则

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
- 主要回归测试在 `backend/tests/test_smoke.py`。
- 推荐本机 Python 为 `C:\Users\lenovo\anaconda3\envs\pytorch\python.exe`。
- `docs/frontend_backend_handoff.md` 是当前前后端接口契约；`AutoAI_开发计划.md` 是历史开发计划，不能把里面的 React/Vite、Redis/RQ、SQLite 等早期路线当成当前实现。
- 色谱主界面默认使用 `/api/preprocess/hplc`，旧 `/api/preprocess/chromatography` 只作为简单截取兼容接口保留。
- 训练 HTTP 请求只创建 queued Run，不直接启动训练；BackgroundTasks 不承担训练执行。
- 本地模式不接受 owner_id 或 tenant_id；未来身份只经服务端 Principal 注入。

## 预处理规则

- 统一建模 CSV 固定为 `Index, Name, XXX, Intensity, Label, Repeat_index`。
- 拉曼支持行号或 X 轴范围截取，并支持 `range_then_baseline` / `baseline_then_range` 两种基线校正顺序。
- HPLC 标准流程默认启用，顺序固定为：线性插值到共同时间轴、逐条减最小值消负、按真实时间轴面积归一化。
- HPLC 表单字段 `hplc_interpolate`、`hplc_subtract_min`、`hplc_normalize_area` 默认均为 `true`；响应曲线使用 `raw_y` + `processed_y`。

## 建模与可解释性规则

- 当前建模任务仅支持分类；`Label` 即使为数字也按类别名编码，不作为连续回归目标。`PLSR`、`SVR` 是回归变体，本版训练入口不启用。
- 当前 10 类分类模型固定为：`pls_da`、`svm`、`random_forest`、`xgboost`、`cnn1d`、`transformer1d`、`resnet1d`、`inception1d`、`tcn1d`、`dscarnet`。
- 当前稳定可训练模型固定为正式基线已有的 10 个分类模型；新增模型、网络结构、二分类输出形式、DSCARNet 映射策略和传统模型搜索空间必须使用独立模型计划，并提供固定数据集上的对比验收。
- 支持三种分类评估口径：无独立测试集时可选 `stratified_holdout`（按标签比例 8:1:1 划分 train/valid/test）或 `leave_one_repeat_index_cv`（每折留 1 个 `Repeat_index` 作 test，其余按 8:2 划分 train/valid）；有独立测试集时使用 `external_test_holdout`（主数据 8:2 划分 train/valid，独立测试集作最终 test）。
- 所有标准化参数只由当前训练集拟合，并应用于同一评估口径下的验证集和测试集。
- 深度学习模型：`cnn1d`、`transformer1d`、`resnet1d`、`inception1d`、`tcn1d`、`dscarnet` 支持 test 集单样品可解释性分析。
- `cnn1d`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM / Grad-CAM-like。
- 其中 `dscarnet` 当前实际使用 AggMap/PCA 的 SAR/CAR 双通路 2D 映射，再用双通路 2D Grad-CAM 回投到 1D 特征；不要把它当普通 1D CNN 解释。
- `transformer1d` 使用输入梯度归因，避免直接套标准 CNN Grad-CAM。
- 六个传统机器学习模型及 `pca_mlp`、`cnn_transformer1d` 使用窗口遮挡后的真实类别 Log-loss 增量做重要性分析：`masked_loss - original_loss = log(p_before / p_after)`；全局结果按真实类别等权聚合，正值表示遮挡后真实类别置信度受损。
- 传统模型和无卷积深度模型同时生成 `feature_importance.json/csv` 与 `sample_feature_importance.json/csv`，前端默认展示全局并允许切换单样品。旧下载接口应保持可用。
- DSCARNet 会额外写入 `dscarnet_mapping.json` 和 AggMap/PCA joblib 文件；当前 artifact 下载白名单不开放这些 joblib 文件，除非同步更新接口和测试。

## 验证命令

除单元/冒烟测试外，凡是改动训练、评估、模型、预处理或前端训练请求逻辑，交付前在项目根目录存在本地 `data.csv` 时还需要跑一次真实数据流程验证；`data.csv` 是本地验证数据，不能提交进 Git。优先选择轻量模型完成一次训练/评估闭环；如果因为耗时、环境、数据缺失或数据状态无法运行，必须在交付说明中明确写出未运行原因。

```powershell
Set-Location -LiteralPath 'D:\PythonProject\AutoAI'
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests\test_smoke.py' -q
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
