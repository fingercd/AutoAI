# Pan 服务器部署说明

> 最近核对：2026-09-29。开发只在 `/users/fotile/AutoAI/Pan` 的 pan/agent 进行；Git 规则见 [AGENTS](../AGENTS.md)。本文记录当前部署，不授权自动重启、迁移、安装依赖或发布。

## 当前目录和服务

| 用途 | 已核对位置或值 |
|---|---|
| 开发目录 | `/users/fotile/AutoAI/Pan` |
| 远端/分支 | `fingercd/AutoAI` / `pan/agent` |
| 服务节点与端口 | node3 / 18085，server 模式以现场健康检查为准 |
| 发布入口 | `/users/fotile/AutoAI/step78/current` |
| 当前 release | `019f1cc673f9f289419e4ec1e68204659d94e34b` |
| 持久存储 | `/users/fotile/AutoAI/step78/storage` |
| 控制记录 | `/users/fotile/AutoAI/step78/runtime` |
| 现场适配器 | `/users/fotile/AutoAI/step78/service_control.py` |
| 控制器来源 | 固定 fcea2b1555a8d3f5d4f815494e46771839fe8336 release |

step78 是保留的发布/数据目录，不是允许继续开发的分支。不得整目录清理，也不在 release 中进行开发 Git 操作。`current`、控制器来源 release 和存储用途不同，不能仅凭“只保留最新开发分支”删除回退或运行资源。

当前运行环境位于既有服务器验收目录，维护时复用实际进程使用的解释器；不要照旧 AutoAI-v2 示例重新建环境或复制仓库。历史 install/run/qsub 脚本保留为环境模板，不代表当前现场控制入口；调用前须核对其路径和行为。

## 访问与身份

从已授权的 SSH 线路将本地端口转发到 node3 的 18085，浏览器访问转发端口。转发不改变服务现有 server 模式，也不取消 Bearer 认证。

服务端安全注入令牌和 Principal；不读取或打印认证文件，不把 token 放到命令行参数、URL、日志或仓库。浏览器令牌只保存在当前标签页 sessionStorage。server 训练使用 dataset_id/test_dataset_id，不接受浏览器传入服务器 data_path。CORS 只接受明确来源，不用通配符。

`/health` 为匿名健康探针；业务 API 的授权方式按当前接口契约执行。结果页显示既有 Train/Valid/Test 或 pooled OOF；Agent 仍只使用工具契约规定的输入，这不是部署时要变更的 Test 访问机制。

## 启停和后续发布

使用现场适配器调用正式 [Linux release 控制器](../docs/service_control.md) 的 status/start/stop/restart。控制器核对进程身份、数据库排空条件、端口和 worker 契约；不能绕过它另启第二组生产 Web/worker，不能直接按 PID 强杀。

后续发布只取 Pan 主线已验收版本，不另建开发分支。先形成一致备份和恢复证据，再通过控制器排空、停止和切换 current；启动验收失败时按既有控制器回退代码，保留发布后新增数据，不用旧数据库覆盖当前生产。完整边界以控制器契约为准。

若发布 checkout 使用 Git alternates，共享对象库属于发布依赖。删除旧开发副本前，必须保全所需对象并核验 release 的提交/树/文件，再解除对待删除目录的引用；不能只检查 current 符号链接。

## 验证与维护

文档修改和闲置源码副本清理只检查路径、归档、对象完整性、现行文档和运行身份，不触发训练或服务重启。发布操作另行验证 `/health` 的 deployment_mode、worker 可用性与 `training-worker-guard-v1` 兼容性；有真实训练授权时才执行隔离的小流程。

存储包括上传、Run artifacts、SQLite 数据和 Agent 日志，均不进入 Git。备份开放 SQLite 时须使用既定一致性方法，不能只复制主文件而忽略 WAL。旧所有权迁移先完成备份和 dry-run，再按明确授权处理；不作为启动时的自动动作。

新模型/依赖变化按任务授权和现有环境处理；DSCARNet 已退役，不再提供其安装步骤。GPU 作业先核对实际占用与进程归属，本次文档维护不分配 GPU、不更新依赖。
