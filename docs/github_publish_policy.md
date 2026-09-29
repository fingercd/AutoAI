# Pan Git 与发布规范

> 最近核对：2026-09-29。长期规则以 [AGENTS](../AGENTS.md) 为准；当前状态见 [CONTEXT](../CONTEXT.md)。旧的 AutoAI-v2/main 首次发布流程已停止作为操作指南。

## 仓库和权限边界

- 正式远端：`https://github.com/fingercd/AutoAI.git`。
- 唯一开发目录：服务器 `/users/fotile/AutoAI/Pan`；开发及推送分支：`pan/agent`。
- 不创建分支、worktree、fork 或可继续开发的仓库副本，不切换其他分支开发。GitHub 的 master 和其他现有分支保留，不能据此推送或删除它们。
- 所有人工/智能体开发 Git 命令只从上述物理根目录执行，禁止通过 `-C`、`--git-dir` 或 Git 环境变量绕到别的仓库。
- 当前任务必须明确授权提交或推送。文档更新、检查或目录清理本身不构成提交/推送授权。

## 命令前核对

先进入已知服务器目录，用 shell 核对物理路径，再运行 Git 核对：

```bash
cd /users/fotile/AutoAI/Pan
test "$(pwd -P)" = /users/fotile/AutoAI/Pan || exit 1
test -z "${GIT_DIR-}${GIT_WORK_TREE-}${GIT_COMMON_DIR-}" || exit 1
test "$(git rev-parse --show-toplevel)" = /users/fotile/AutoAI/Pan || exit 1
test "$(git branch --show-current)" = pan/agent || exit 1
git status --short
git remote get-url origin
```

origin 必须对应 fingercd/AutoAI。若目录、分支、远端或已有改动不符合当前任务，不切换到别处继续操作。

## 经授权的提交和推送

只对本次授权文件使用精确路径暂存，检查 staged 文件清单和完整 diff；禁止 `git add .`/`git add -A`。数据、凭据、数据库、模型、日志、缓存及归档都不进入提交。

推送前在同一个 Pan 根目录读取远端 pan/agent 并核对非快进风险。网络/认证失败或远端状态不明时停止，不在本地副本补推、不强推。确有推送授权且核对通过时，唯一推送目标为：

```bash
git push origin HEAD:refs/heads/pan/agent
```

禁止 `--force`、`--force-with-lease`、`--mirror` 和 `--all`。推送后在同目录读取远端 SHA 核对结果；不更改远端默认分支或仓库可见性，不删除 GitHub 其他分支。

## 仓库内容与发布快照

Git 保存源码、配置、测试和维护文档。真实数据、storage、权重、日志、数据库、凭据、压缩归档、环境与本地工具状态均排除。忽略规则不能代替 staged diff 检查；敏感文件检查只报告文件名，不输出内容。

服务使用已验收的不可变 release 和持久 storage。发布快照不是开发副本，不在其中编辑、建分支、提交或推送。服务发布与 Git push 是两项不同操作，按 [部署说明](../deploy/server_deploy.md) 和 [服务控制契约](service_control.md) 单独授权执行。

清理旧开发副本前归档独有历史、检查进程和 Git 对象库依赖；不删除当前 release、控制器来源或持久数据。已存在的可恢复归档不作为开发入口。
