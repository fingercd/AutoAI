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
- 前端主入口是 `static/index.html`，由后端静态托管；`static/ui-*.html` 和多套 CSS/JS 是 UI 方案或历史候选。
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
- 当前稳定可训练模型目录固定为 master 已有的 10 个分类模型；新增模型、网络结构、二分类输出形式、DSCARNet 映射策略和传统模型搜索空间必须使用独立模型计划，并提供固定数据集上的对比验收。
- 支持三种分类评估口径：无独立测试集时可选 `stratified_holdout`（按标签比例 8:1:1 划分 train/valid/test）或 `leave_one_repeat_index_cv`（每折留 1 个 `Repeat_index` 作 test，其余按 8:2 划分 train/valid）；有独立测试集时使用 `external_test_holdout`（主数据 8:2 划分 train/valid，独立测试集作最终 test）。
- 所有标准化参数只由当前训练集拟合，并应用于同一评估口径下的验证集和测试集。
- 深度学习模型：`cnn1d`、`transformer1d`、`resnet1d`、`inception1d`、`tcn1d`、`dscarnet` 支持 test 集单样品可解释性分析。
- `cnn1d`、`resnet1d`、`inception1d`、`tcn1d` 使用 1D Grad-CAM / Grad-CAM-like。
- 其中 `dscarnet` 当前实际使用 AggMap/PCA 的 SAR/CAR 双通路 2D 映射，再用双通路 2D Grad-CAM 回投到 1D 特征；不要把它当普通 1D CNN 解释。
- `transformer1d` 使用输入梯度归因，避免直接套标准 CNN Grad-CAM。
- 传统机器学习模型 `pls_da`、`random_forest`、`svm`、`xgboost` 使用窗口遮挡/置乱后的 `baseline_macro_f1 - perturbed_macro_f1` 做重要性分析；正值越大表示该区间被遮挡后 macro-F1 下降越多，区间越重要。
- 深度模型优先生成 `sample_feature_importance.json/csv` 供前端展示；传统模型优先展示 `feature_importance.json/csv` 的全局窗口重要性。旧下载接口应保持可用。
- DSCARNet 会额外写入 `dscarnet_mapping.json` 和 AggMap/PCA joblib 文件；当前 artifact 下载白名单不开放这些 joblib 文件，除非同步更新接口和测试。

## 验证命令

除单元/冒烟测试外，凡是改动训练、评估、模型、预处理或前端训练请求逻辑，交付前还需要用项目根目录的本地 `data.csv` 跑一次真实数据流程验证；`data.csv` 是本地验证数据，不能提交进 Git。优先选择轻量模型完成一次训练/评估闭环；如果因为耗时、环境、数据缺失或数据状态无法运行，必须在交付说明中明确写出未运行原因。

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

This project is indexed by GitNexus as **AutoAI** (1682 symbols, 3825 relationships, 132 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/AutoAI/context` | Codebase overview, check index freshness |
| `gitnexus://repo/AutoAI/clusters` | All functional areas |
| `gitnexus://repo/AutoAI/processes` | All execution flows |
| `gitnexus://repo/AutoAI/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
