# AutoAI GitHub 提交规范

本文档规定以后 Codex 执行“git 到 GitHub”“git一下”“提交并推送”等操作时，哪些内容可以进入 GitHub，哪些内容必须留在本地。默认原则是：GitHub 只保存可协作、可审查、可复现项目结构的内容；数据、模型、运行产物、本地 agent 状态和认证材料不进入普通 Git 仓库。

本次规范只约束未来提交行为，不修改 `.gitignore`，不迁出已跟踪文件，不重写 Git 历史。

## 提交前总原则

1. 先运行 `git status --short`，向用户展示本次变更范围。
2. 不使用 `git add .`、`git add -A` 或宽泛通配符；只用精确路径 stage 本次任务相关文件。
3. 只提交与用户本轮目标直接相关的文件；看到已有用户改动时，不回滚、不顺手整理、不混入提交。
4. 任何数据、模型、二进制、大文件、截图、zip、历史 UI 方案、本地 agent 目录，默认先停下并向用户确认。
5. 提交前检查 `git diff --cached --stat` 和必要的 `git diff --cached -- <path>`，确认 staged 内容与说明一致。
6. 若 GitNexus 可用且本次要 commit，按项目规则先执行 `detect_changes()`，确认影响范围符合预期。
7. 推送前再次确认没有敏感文件、大对象、运行产物、无关历史文件或本地 agent 状态。

## 默认应提交

这些文件通常属于项目源代码或协作契约，可以在与任务相关时提交：

- 后端源码：`backend/app/` 下的 `.py` 源码。
- 测试：`backend/tests/` 下的测试文件。
- 依赖声明：如 `backend/requirements.txt`。
- 项目入口：`run.py`。
- 正式前端：`static/index.html` 以及被正式主界面引用的必要 CSS/JS。
- 项目文档：`README.md`、`CONTEXT.md`、`AGENTS.md`、`docs/` 下的接口、部署、规范文档。
- 部署说明：`deploy/` 下的文本文档和脚本，前提是不含服务器凭据、内网地址敏感信息或密钥。
- 配置样例：只提交脱敏模板，例如 `.env.example`；不提交真实配置。

## 谨慎提交

这些内容不是绝对禁止，但每次都必须说明用途、确认体积、确认无敏感信息：

- 小型脱敏样例数据：只用于测试或 README 示例，优先放在明确的样例目录，并写清来源和脱敏方式。
- 文档配图、UI 对比截图：只在文档确实引用时提交；临时截图默认不提交。
- `static/ui-*.html` 和多套历史/候选 UI 文件：只有在用户明确要求保存某个 UI 方案时提交。
- 临时评审文档、prompt、方案草稿：只有确认会长期复用时提交；否则留在 `work/`。
- 小型二进制附件：必须低于本文档大小阈值，并说明为什么不能用文本或外部链接替代。

## 默认不提交

以下内容默认留在本地，不进入 GitHub。即使它们当前出现在工作区，也不能被 `git add .` 顺手带入：

- 临时工作区：`work/`。
- 最终导出物或演示产物：`outputs/`。
- 运行数据区：`storage/`，包括 `storage/uploads`、`storage/preprocessed`、`storage/runs`。
- 实验原始数据或真实数据目录：`拉曼/`、`色谱/`、`四个模型/`。
- 训练和解释性产物：`metrics.json`、`predictions.csv`、`feature_importance.json/csv`、`sample_feature_importance.json/csv`、`dscarnet_mapping.json` 等，除非是小型脱敏样例并经用户确认。
- 模型权重和序列化文件：`*.joblib`、`*.pkl`、`*.pt`、`*.pth`、`*.onnx`、`*.bin`、`*.h5`。
- 压缩包和归档：`*.zip`、`*.7z`、`*.rar`、`*.tar`、`*.gz`。
- 视频、音频和渲染结果：`*.mp4`、`*.mov`、`*.avi`、`*.mkv`、`*.mp3`、`*.wav`。
- 依赖和 vendor 目录：`node_modules/`、临时 `vendor/`、下载的 ffmpeg、浏览器运行时、Remotion/webpack 缓存。
- Python 和工具缓存：`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.cache/`。
- 日志和本地进程文件：`*.log`、`*.pid`、`*.out`、`*.err`。
- 本地打包产物：`autoai_deploy.zip`、`backend.zip`、`backend/app/models.zip` 等。

## 绝不提交

以下内容禁止提交。发现它们处于 staged 状态时必须取消 stage，并提示用户：

- 真实密钥和认证材料：`.env`、`auth.json`、token、API key、密码、cookie、OAuth 缓存、SSH 私钥。
- 本地敏感缓存：`.sandbox-secrets`、认证缓存目录、浏览器登录态导出。
- Git 内部目录：`.git/`。
- 本地 agent、索引和状态目录：`.codegraph/`、`.gitnexus/`、`.geniux/`、`.claude/`、`.omc/`。
- 含真实客户、实验对象、未公开研究数据或个人信息的数据文件。
- 任何用户明确说“不要上传”“本地留存”“私有数据”的文件。

## 大小阈值

普通 Git 只适合保存小型文本和必要的小资源。AutoAI 采用比 GitHub 更保守的阈值：

- 单文件超过 10 MiB：默认不提交；若确实要提交，必须得到用户明确确认并写明理由。
- 单文件超过 50 MiB：禁止进入普通 Git；需要先改用 Git LFS、GitHub Release、DVC 或外部存储方案。
- 单文件超过 100 MiB：不得进入普通 Git，GitHub 会阻断普通仓库中的此类文件。
- 数据集、模型权重、视频、压缩包、数据库、node_modules、ffmpeg/browser runtime，即使低于 10 MiB，也默认不提交。

可选替代方案：

- 小型、脱敏、稳定的样例数据：可以放入 Git，但必须说明用途。
- 大数据或模型版本：优先 DVC、对象存储、网盘、服务器数据盘或 Git LFS。
- 大型二进制发布包：优先 GitHub Release，不放入源码历史。

## AutoAI 当前风险事实

截至制定本文档时，本仓库存在以下历史和本地风险，后续提交时必须牢记：

- GitHub 远端为 `https://github.com/fingercd/AutoAI.git`。
- 当前仓库已有 57 个 tracked 数据 CSV，包括 `data.csv`、`拉曼/`、`色谱/` 下的文件。
- 本地 `.git/` 约 3.7 GB，说明历史里曾经进入过大对象。
- 历史大对象包含 `work/`、`outputs/`、`node_modules/`、ffmpeg、视频、渲染缓存等内容。
- “历史里已经出现过”不代表“以后可以继续提交”。未来提交必须按本文档规则重新判断。
- 本文档不会自动清理历史；如果后续要瘦身仓库，应另起阶段，先备份，再评估 `.gitignore`、`git rm --cached`、Git LFS/DVC、`git-filter-repo` 或重建干净仓库。

## Codex 执行“git一下”的标准流程

当用户要求提交或推送时，Codex 必须按以下流程执行：

1. 运行 `git status --short`，列出本次候选变更。
2. 将变更分成“应提交”“需确认”“不提交”“禁止提交”四类。
3. 向用户展示计划提交的文件清单，尤其说明不会提交的数据、模型、输出、缓存和本地 agent 目录。
4. 使用精确路径 stage 文件，例如 `git add docs/github_publish_policy.md`。
5. 运行 `git diff --cached --stat`，必要时查看具体 diff。
6. 若发现误 stage，使用精确路径取消 stage，例如 `git restore --staged <path>`。
7. 若要 commit，提交信息必须描述本次真实变更，不夹带无关内容。
8. 若要 push，推送前再次运行 `git status --short` 和 staged/commit 检查，确认没有敏感文件或大对象。

禁止流程：

- 禁止直接 `git add .` 后提交。
- 禁止把未解释来源的数据、模型、zip、视频、缓存、日志加入提交。
- 禁止为了让工作区“干净”而提交用户无关改动。
- 禁止回滚、删除、移动用户已有数据或大文件来迎合提交。
- 禁止把 `.gitignore` 尚未覆盖的本地目录误认为可提交。

## 常见场景判断

- 后端功能或 bug 修复：通常提交 `backend/app/`、`backend/tests/`、相关文档；不提交 `storage/runs` 里的训练结果。
- 前端主界面改动：通常提交 `static/index.html` 和正式引用的资源；候选 UI、截图需用户确认。
- 文档-only 改动：只 stage 对应 md 文件。
- 训练或评估验证后产生文件：默认不提交任何运行产物，只在交付说明中说明验证结果。
- `data.csv`、`拉曼/`、`色谱/` 有变化：先停下确认，不自动提交。
- 出现 `*.zip`、`*.mp4`、`*.joblib`、`*.pt`、`*.pth`、`*.onnx`、`node_modules/`、ffmpeg：普通 Git 提交中拒绝。
- 出现 `.codegraph/`、`.gitnexus/`、`.geniux/`、`.claude/`：本地 agent 状态，不提交。
- 出现 `.env`、`auth.json`、token、API key：绝不提交；若已经暴露，先提示用户轮换密钥，再讨论历史清理。

## 误提交后的处理

- 尚未 commit：用 `git restore --staged <path>` 取消 stage。
- 已 commit 但未 push：按文件风险选择 amend、rebase 或重新提交；大文件可用 `git rm --cached` 从提交中移除但保留本地文件。
- 已 push 的普通数据或大文件：不要继续追加同类文件；另起清理阶段评估 `git-filter-repo`、Git LFS/DVC 或重建仓库。
- 已 push 的密钥或凭据：先轮换/吊销密钥，再决定是否需要历史重写和联系 GitHub 支持。

## 参考依据

- GitHub Docs: Ignoring files - https://docs.github.com/en/get-started/git-basics/ignoring-files
- GitHub Docs: About large files on GitHub - https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub Docs: Removing sensitive data from a repository - https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
- GitHub Docs: Push protection - https://docs.github.com/en/code-security/concepts/secret-security/push-protection
- Git LFS - https://git-lfs.com/
- DVC Get Started - https://doc.dvc.org/start
