# AutoAI-v2 GitHub 发布规范

AutoAI-v2 的 GitHub 仓库只保存可协作、可审查和可复现的源码、配置与长期维护文档。数据、模型、运行产物、本地 agent 状态、凭据和临时方案不得进入仓库。

## 仓库基线

- 正式仓库：`fingercd/AutoAI-v2`。
- 可见性：private。
- 默认分支：`main`。
- 首次发布来自筛选后的干净目录，使用单一初始提交，不复制旧 `.git` 或旧提交对象。
- 旧仓库 `fingercd/AutoAI` 是历史备份，不重命名、不归档、不删除，也不作为 v2 的 remote 替换目标。

## 可以提交

- `backend/app/` 后端源码。
- `backend/tests/` 自动化测试。
- `backend/requirements*.txt` 和验证约束。
- `run.py`。
- 正式前端 `static/index.html` 与 `static/js/`。
- `deploy/` 中不含凭据的脚本与说明。
- `README.md`、`CONTEXT.md`、`AGENTS.md`、`CLAUDE.md`、ADR 和长期维护文档。

## 必须排除

- 真实数据与上传：`data.csv`、其他 CSV、拉曼/色谱数据目录、`storage/`。
- 模型与运行结果：PT/PTH/PKL/JOBLIB/ONNX、Run artifacts、SQLite/DB、日志和 PID。
- 压缩包、音视频、二进制工具、缓存、虚拟环境和构建目录。
- `.env`、`auth.json`、`.sandbox-secrets/`、证书、私钥和认证缓存。
- `.claude/`、`.codex/`、`.agents/`、`.codegraph/`、`.gitnexus/` 等本地工具状态。
- Office 附件、截图、评审 prompt、`docs/superpowers/`、当前内部实施计划和临时空文件。明确标记的 `AutoAI_开发计划.md` 仅作为历史路线档案保留。
- 未进入正式产品的 UI 候选页面、样式和脚本。

忽略规则只是最后一道保护；发布快照仍必须按允许清单复制和 stage，不能依赖 `git add -A` 后再排查。

## 首次干净发布流程

1. 从当前正式源码筛选允许文件到新的隔离目录，不复制 `.git`。
2. 检查文件名、内容、大小和 Git 状态；单文件不得超过 10 MiB。
3. 在隔离目录执行 `git init -b main`。
4. 只按审核后的路径 stage，检查 `git diff --cached --stat` 和 staged 文件清单。
5. 创建单一提交：`chore: publish clean AutoAI-v2 baseline`。
6. 创建 private 远端；同名仓库已存在时停止，不覆盖、不删除、不强推。
7. 推送 `main`，设置默认分支并复核私密性。
8. 邀请协作者时只授予需要的权限；write/push 不等于 admin。

## 日常提交规则

1. 先执行 `git status --short`，区分本次修改和已有工作。
2. 禁止在仓库根目录盲用 `git add .` 或 `git add -A`。
3. 使用精确路径 stage，并检查 staged diff。
4. 测试、编译和真实数据验证按 `AGENTS.md` 执行。
5. 推送前复查敏感文件名、对象大小和远端 URL。

## 发布前检查示例

PowerShell：

```powershell
git ls-files
git status --short
git diff --cached --stat
git ls-files | ForEach-Object {
  $item = Get-Item -LiteralPath $_
  if ($item.Length -gt 10MB) { $item.FullName }
}
```

敏感命名检查至少覆盖：`.env`、`auth`、`token`、`secret`、`credential`、私钥扩展和本地 agent 目录。不得读取或打印疑似凭据内容；命中后只报告路径并从快照移除。

## 泄漏处理

一旦敏感信息进入提交或远端：立即停止继续推送，轮换/吊销凭据，再评估历史重写。仅删除工作区文件或追加 `.gitignore` 不能清除已有 Git 历史。
