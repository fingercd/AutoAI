# 全项目文档校准与代码注释完善计划

## 目标与成功标准

通读 SpecAutoAI 当前纳入项目范围的后端、前端、测试、部署脚本、依赖约束和 Markdown，以代码、路由契约、模型注册表与自动化测试为事实来源，消除当前有效文档之间的冲突；随后为生产代码补充有维护价值的详细注释，使后续开发者能理解架构边界、数据流、状态机、算法选择和兼容约束，而不改变现有运行行为。

完成标准：

- 当前有效文档对运行入口、预处理流程、Run 队列、模型目录、评估策略、解释性方法、部署和验证命令的描述与代码一致。
- 历史路线、内部评审材料和 ADR 的时间属性被明确保留，不把历史方案改写成当前实现。
- 生产代码的重要模块、公共接口、非显然算法和安全边界有模块说明、docstring 或就地注释；避免逐行复述代码和给简单包装器制造噪声。
- 注释编辑不改变公共接口、请求/响应结构、模型算法、状态转换或前端行为。
- Python 编译、后端测试、前端脚本语法检查全部通过；如根目录存在本地 `data.csv`，再完成一次轻量真实训练闭环，否则在交付中记录未执行原因。

## 范围与事实优先级

### 纳入阅读与校准

- 根目录入口与上下文：`run.py`、`AGENTS.md`、`CONTEXT.md`、`README.md`、`CLAUDE.md`。
- 后端生产代码：`backend/app/` 下的 FastAPI 路由、数据集仓储、Run 状态机与 worker、预处理、训练、模型和解释性实现。
- 前端：`static/index.html`、`static/js/api-client.js`、`static/js/training-store.js`。
- 验证与约束：`backend/tests/`、`backend/requirements*.txt`、`backend/constraints-verified.txt`。
- 部署与长期文档：`deploy/`、`docs/frontend_backend_handoff.md`、`docs/github_publish_policy.md`、`docs/adr/`。
- 历史/计划材料：`AutoAI_开发计划.md`、`docs/hplc-pipeline-review-prompt.md`、`docs/plans/`，用于判断是否需要标注状态，不作为当前实现的权威来源。

### 排除与保护

- 不读取 `.env`、`auth.json`、token、密钥、认证缓存和 `.sandbox-secrets`。
- 不读取或修改 `storage/` 中的上传、数据库、模型和运行产物；真实流程验证只使用项目约定允许的本地数据入口。
- 保留用户现有未跟踪文件 `docs/plans/2026-07-14-raman-csv-excel-safe.md`，除非代码核对证明本轮必须同步且修改内容可与用户计划清晰区分。
- 不修改算法、接口或状态机；若阅读中发现代码缺陷，只记录为遗留风险，不借“加注释”扩大为功能重构。

事实冲突按以下顺序裁决：可执行测试与请求契约 → 当前生产代码与注册表 → 当前架构 ADR → README/CONTEXT/交接文档 → 历史开发计划和评审材料。

## 任务 1：建立项目事实清单

**涉及模块：** 全仓只读巡检，重点为 `backend/app/main.py`、`backend/app/routers/`、`backend/app/runs/`、`backend/app/models/registry.py`、`backend/app/training.py`、`backend/app/parsers.py`、`backend/app/hplc.py`、`static/index.html` 与相关测试。

**实施内容：**

- 按入口到下游的顺序梳理 Web 启动、路由装配、数据上传、拉曼/HPLC 预处理、数据集持久化、Run 创建、worker claim/lease、训练、artifact manifest 和状态投影。
- 从模型注册表和训练分派提取模型 ID、可用性、架构版本、二分类输出、评估策略和解释性方法矩阵。
- 从前端请求构造和状态渲染核对字段默认值、兼容路由、下载路径和展示优先级。
- 用测试确认并记录兼容约束、路径安全、并发状态转换和历史模型行为。

**验收标准：** 每份当前文档中的关键事实都能定位到至少一个代码或测试依据；冲突项形成明确的“保留、更新或标记历史”结论。

## 任务 2：更新当前有效 Markdown

**预计修改：**

- `README.md`：面向使用者的功能、安装、启动、目录、模型能力和验证方式。
- `CONTEXT.md`：面向维护者的当前架构、核心不变量和已知限制，去除重复或互相矛盾的旧规则。
- `AGENTS.md`：仅同步会影响后续代理判断的现行事实与验证规则，不塞入实现细节。
- `docs/frontend_backend_handoff.md`：同步实际请求/响应、状态、artifact 与前端行为契约。
- `deploy/server_deploy.md`：同步当前 Web/worker 启动、环境变量、目录默认值和验收步骤。
- `CLAUDE.md`、`docs/github_publish_policy.md`、`docs/adr/0001-stable-run-architecture.md`：仅在代码事实或项目规则已变化时做最小修订。

**历史文件策略：**

- `AutoAI_开发计划.md` 保持历史内容，必要时只加强顶部历史状态说明和当前权威文档指向。
- `docs/hplc-pipeline-review-prompt.md` 保留为评审快照；若已过时，只添加状态/日期/权威实现指向，不将其扩写为现行契约。
- `docs/plans/` 中已存在的计划不因当前实现变化被追溯改写。

**验收标准：** 当前文档不再对模型数量、解释性方法、拉曼处理顺序、Run 执行方式或 artifact 下载边界给出互相冲突的结论；内部链接和命令路径有效。

## 任务 3：完善生产代码注释

**预计修改模块：**

- 启动与 HTTP 边界：`run.py`、`backend/app/main.py`、`backend/app/http/`、`backend/app/routers/`。
- 数据处理：`backend/app/parsers.py`、`backend/app/hplc.py`、`backend/app/datasets/repository.py`、`backend/app/feature_selection.py`。
- Run 子系统：`backend/app/runs/`，重点说明命令侧状态权威、事务、claim token、lease、manifest 提交和 legacy projection。
- 训练与解释性：`backend/app/training.py`、`backend/app/training_explainability.py`、`backend/app/dscarnet_mapping.py`、`backend/app/classification_policy.py`。
- 模型层：`backend/app/models/registry.py`、`profiles.py` 和各模型实现，重点说明张量形状、架构版本、二分类输出、可选依赖和兼容类。
- 前端：`static/index.html` 与 `static/js/` 中的页面状态、请求构造、轮询、结果投影和兼容处理。
- 部署脚本：只给环境假设、进程职责和危险默认值加注释，不改脚本行为。

**注释规则：**

- 模块 docstring 说明职责、边界和主要调用关系。
- 公共函数/类 docstring 说明参数语义、返回值、持久化副作用、异常和关键不变量。
- 就地注释解释“为什么这样做”：数据泄漏防护、数值稳定性、并发 CAS、路径安全、向后兼容、张量变换和算法公式。
- 不给显而易见的赋值、循环、导入和测试断言逐行翻译；不写易随实现漂移的冗余注释。
- 测试文件只在 fixture、并发编排或契约意图不直观处补注释，避免扩大维护负担。

**验收标准：** 关键模块在不阅读全部实现的情况下即可理解输入、输出、状态边界和失败方式；`git diff --check` 无空白错误，注释编辑没有引入行为 diff。

## 任务 4：验证与审查

**自动验证：**

```powershell
$env:PYTHONPATH='D:\PythonProject\AutoAI'
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m compileall 'D:\PythonProject\AutoAI\backend\app' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' -m pytest 'D:\PythonProject\AutoAI\backend\tests' -q
& 'C:\Users\lenovo\anaconda3\envs\pytorch\python.exe' 'D:\PythonProject\AutoAI\run.py' --help
```

前端如有任何脚本注释变更：抽取 `static/index.html` 的内联 `<script>` 内容，并运行 `node --check --input-type=commonjs`；两个独立 JS 文件分别运行 `node --check`。同时运行 `git diff --check` 和定向文档术语搜索，确认没有残留冲突。

根目录存在 `data.csv` 时，使用轻量传统模型完成上传、queued Run、worker 执行、评估和 artifact 下载闭环；不存在或环境不满足时不创建/提交替代数据，并在交付说明中记录原因。

**最终审查：** 对照本计划检查修改文件清单、用户原有改动、测试结果、未解决风险和文档时间属性。只有代码注释、文档校准和验证都完成后才结束目标。

## 风险与默认决策

- “给代码加详细注释”默认指生产代码的维护性注释，不追求每一行都有注释；过密注释会掩盖核心不变量并迅速过时。
- 大型文件 `backend/app/training.py`、`backend/app/feature_selection.py` 和 `static/index.html` 采用按逻辑区块补充说明的方式，避免为注释而重排代码造成高风险 diff。
- 当前文档对 v1 十模型与 v2 十五项能力目录存在冲突；最终以注册表、训练分派和契约测试为准，并清楚区分“目录存在”“当前可用”“历史只读兼容”。
- 注释过程中发现的真实行为缺陷不在本轮直接修复；先记录证据和影响，除非该缺陷阻止文档准确化或验证完成。

## 执行结果（2026-07-15）

- 已通读根目录入口、全部 `backend/app` 生产模块、`backend/tests`、正式前端、依赖约束、部署脚本和项目 Markdown。
- 已校准当前文档的模型目录、解释性矩阵、拉曼/HPLC 顺序、Run/worker、Manifest、部署与历史材料边界，并新增 `docs/README.md` 文档导航。
- `backend/app` 每个 Python 模块及所有公开顶层类/函数均已有模块说明或 docstring；训练、解释性、状态机、前端和部署脚本另补充了关键不变量与流程注释。
- Python compileall、启动器帮助、内联/独立 JavaScript、PowerShell、Bash 和 `git diff --check` 均通过。
- 全量自动化测试最终结果：`283 passed, 3 skipped`。
- 本地 `data.csv` 真实验证：Logistic Regression 分层 8:1:1 训练成功；另通过上传 API 创建 queued Run，由独立 worker 执行到 `succeeded`，Manifest 为 `manifest.json`，`metrics.json` 下载重复 5 次均返回 HTTP 200。
- 用户原有未跟踪计划 `docs/plans/2026-07-14-raman-csv-excel-safe.md` 保持未修改。
