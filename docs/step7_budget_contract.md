# 第七步预算协议与运行边界

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

## 成本报告

`scripts/task_cost_report.py` 只读导出 `task-cost-report-v1` 的七个文件：
`cost_summary.json`、`llm_requests.csv`、`operation_attempts.csv`、
`training_usage.csv`、`timeline_events.csv`、`coverage.json`、`cost_report.md`。
导出时须指定 Journal、Agent 数据库、thread/task/session 身份；有 Run 时还须指定
Runs 数据库、run_id 和由该数据库定位的 run_dir。报告校验 scope、冻结计划、
Manifest 和事件关联；缺失或冲突来源不写出看似有效的已结算报告。旧记录只读兼容，
未采集量保持 unknown，父子重叠耗时不能直接相加。
