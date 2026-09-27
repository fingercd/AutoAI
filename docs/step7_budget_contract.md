# 第七步预算协议与运行边界

第八步增量见 [训练检查协议](step8_guard_contract.md)。新 Guard Run 使用 guard-v1 worker；
旧预算 Run 的 SQL 领取闸门同时接受 budget-v1 和 guard-v1，下文 v5/v7 为历史协议描述。

## 新任务与旧任务

配方协议 `agent-recipes-revision-v5` 使用 `agent-state-v7`、
`budget-policy-v1` 和 `agent-context-budget-v1`。CLI 新建预算任务时显式传
`--budget-awareness on` 或 `--budget-awareness off`；两组均执行相同的硬预算。
`on` 的选择上下文含剩余额度、未知占用及各合法配方的 fit/epoch 上界，并可建议
`stop_ml_session`；`off` 沿原单次选择流程，不向 LLM 显示预算卡。无可行配方时
两组都由编排器终止，不发起训练。

新任务的策略和 digest 同时冻结到 Session、State、后端训练账本及本机
CallJournal。恢复只接受原 canonical Journal 路径及原策略绑定；复制存储或修改
策略不会创建新的额度。旧 checkpoint/Session 按其原协议读取，不自动升级。

## 执行与收尾

Run、fit、已进入的 epoch 由后端账本扣账；LLM、API、Token、修复与重试由
CallJournal 扣账。预约上界和已发生用量分开记录。调度到训练时，预算 Run 仅由
`training-worker-budget-v1` worker 领取，并在受监督子进程运行。数据库领取闸门
拒绝旧 worker 合同领取预算 Run。worker 退出证据不明确时保留未知 hold，不能将
Run 发布为成功。

### Linux 进程监督

Windows 继续使用带 kill-on-close 的 Job Object。Linux 通过同一 `supervise`
接口启动独立、单线程的 `PR_SET_CHILD_SUBREAPER` 监督进程；训练仍进入现有
`execute_claimed_run`，没有另一套训练核心。监督关系成立且父侧启动检查通过后才
释放训练输入。worker 持有的专用管道不传给训练子进程，worker 被 SIGKILL 后
管道 EOF 会触发清理；监督进程也独立检查冻结截止时间。

清理只对内核列出的直接子进程发 SIGKILL，并反复收养、清理、waitpid 回收后代，
因此 `setsid`、父进程先退出不会逃离监督。只在 `waitpid` 返回 ECHILD 后发出
整树退出证明，父侧还要求监督进程正常退出。结果用临时文件传输，避免大 JSON
填满 stdout 管道阻塞退出。截止、取消及 claim 失效仍由原预算和发布事务收口。

该机制不要求 root/cgroup 写权限，要求 Linux subreaper 与 `/proc/.../children`。
它面向受信任训练代码，不是抵抗恶意同 UID 程序的安全沙箱。监督进程本身被杀、
内核不可中断等待或缺失退出证明时保留 unknown hold，绝不按主子进程已退出释放
额度；超时的监督进程继续清理。worker 崩溃后的恢复沿用原账本，不自动重复 fit。

同一 operation 的再次发送按此前最后一次已发送记录分类：LLM 无效/过大输出后的
再次请求计入 output_repairs，其余再次请求（包括未知发送后的重试）计入
network_retries。两者都同时消耗对应的 LLM/API call 额度，不是额外免费调用。
分类和预约在同一事务持久化；发送前失败释放预约，发送后中断保留未知占用，
恢复结算不重复扣减。旧记录不事后回填未采集的修复/重试消耗。

普通工作截止后不开始新的模型选择或提交；任务总截止后不发新网络请求。受保护的
收尾额度供反馈、对账、Finalize/terminate 及确认使用。Session terminate 请求包含
稳定 `client_request_id` 和枚举 reason，仅当无活动 Run 且后端预算已结算时成功。
Finalize 与 terminate 在同一 Session 上互斥。成功 Run 可以选择不锁定，明确终止
后 `selected_run_id=null`。

State v7/v8 中，已落盘且契约有效的 LLM 提案若超过冻结训练额度，Graph 将
decision 标记为 `invalid` / `recipe_training_budget_exceeded`，choose operation
保持 `confirmed`。Journal 保留原提案、响应引用与实际 token；不另选配方，
不提交 Run，转入 `budget_exhausted` 的 terminate + 后端确认流程。收尾前核对
Session，未解决映射先对账，活动 Run、收尾 API 额度不足或结果未知时保持
恢复/待处理状态，不能宣称 Session 已终止。对账释放后继续原收尾，不回到选型。
崩溃后重放同一已落盘提案不新增 LLM 调用；已确认终止的任务重放不发网络请求。
预算提示 on 的输出协议仍先限制可负担配方，越过该协议的响应按原输出修复
规则处理，不把协议错误当作上述已确认提案。

## 成本报告

`scripts/task_cost_report.py` 只读导出 `task-cost-report-v1` 的七个文件：
`cost_summary.json`、`llm_requests.csv`、`operation_attempts.csv`、
`training_usage.csv`、`timeline_events.csv`、`coverage.json`、`cost_report.md`。
导出时须指定 Journal、Agent 数据库、thread/task/session 身份；有 Run 时还须指定
Runs 数据库、run_id 和由该数据库定位的 run_dir。报告校验 scope、冻结计划、
Manifest 和事件关联；缺失或冲突来源不写出看似有效的已结算报告。旧记录只读兼容，
未采集量保持 unknown，父子重叠耗时不能直接相加。
