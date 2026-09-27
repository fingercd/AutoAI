# 第九步：反馈诊断契约

新 recipe CLI 默认启用 `--feedback-diagnosis on`，可显式关闭。新任务使用
`agent-recipes-revision-v7` / `agent-state-v9`；旧 revision 与 direct_action 不启用诊断。
选择与 Finalize 使用原 Prompt，诊断意见不参与调度，不产生第二个 Run。

## 可信输入

训练核心在既有 Train-only 选优对象上采集 Train/Valid 评价；传统模型采集后才按原算法
refit Train+Valid，RF 仍以 OOB 选参；深度模型使用已恢复的最佳权重。
私有、可选、不可下载的 `training_validation_audit.json`（training-validation-audit-v1）
记录 fold/snapshot、分区摘要、处理 fit 来源、trial/epoch、匿名类别支持和指标。
采集不额外训练或重算。人工 CV 按折比较，external_test 保持原语义；Agent 仍仅 holdout。
审计不存在或语义无效只令同源比较 unavailable，不以旧审计猜测或补零。

`agent-observation-v2.extensions.feedback_evidence` 是闭合的 agent-feedback-evidence-v1。
服务先验证 Principal/Session/Run 绑定和 Guard/Manifest；仅消费同次 assessment 验证过的
文档对象，坏产物不提供性能数值。训练成本独立读取同 scope 后端账本；budget-awareness
关闭时只给 withheld，诊断 Prompt 中不显示任何成本数量。

代码计算 Train−Valid 的描述性差距和支持量，不设置低分/过拟合阈值，不从 Test 取值。
失败使用受控原因；任意异常字符串、路径、标签、样品标识、堆栈不能变成原因推断。
准入时的 DatasetIntegrityError 在原 Guard preflight 内转换为 guard_dataset_changed，
所以已确认的数据改变拒绝不创建 Run，并能按原流程关闭 Session。

## 解释与建议

输入/上下文/Prompt/报告版本依次为 agent-diagnosis-input-v1、agent-context-diagnosis-v1、
agent-diagnosis-prompt-v1、agent-diagnosis-report-v1。
代码拥有事实、指标和问题分类；LLM 只能提交 input_digest、assessment、hypotheses、
suggestions。解释必须引用实际展示的事实/知识 ID，带 tentative/unknown 和限制；
overfitting_risk 还必须引用同源正差距。高分、低分、零分均可返回空问题列表。
自由文本的科学正确性仍需人工审查，结构通过不代表因果已证实。

建议 action_id 从冻结 recipe/config 编译，参数是 const。解释只选择 ID，不提交参数。
所有建议 executable_now=false，不存在 Dispatcher 注册，不开放 Test 调参、换 seed、
安装依赖、执行代码、取消 Guard 或新模型。memory/replanning 仍 disabled。

## 路由和预算

成功 eligible：observe → 冻结输入 → diagnose → 有计量的 observe 刷新 → 原 Finalize。
失败/取消/无资格/confirmed 无 Run 拒绝：先 inspect/reconcile/terminate/confirm，再可选诊断。
后端关闭但诊断未完成使用 post_termination_diagnosis 可恢复阶段；确认结果保留，不重复终止。
活动 Run、未知 mapping/退出继续原安全恢复，不能用诊断覆盖 unknown。

复用同一 LLMAdapter 和 CallJournal；诊断属于 work，不消费受保护 tail。
每任务最多 2 次诊断物理发送，最多一次修复或明确网络失败的重试，共享上限。
未知响应/用量不自动重发；work 不足给 deterministic_only/unavailable，合法收尾仍继续。
默认 tail 仍为 3 LLM / 9 API / 对应 token / 60 秒。总 deadline 后禁止新增请求。

## 持久化与历史

canonical Journal 增加 diagnosis_inputs_v1 / diagnosis_reports_v1，不回填旧行。
首发前冻结规范输入和实际显示上下文；输入按 thread/generation 唯一且不可变。
用量和已校验 proposal 沿原 finish 同事务确认。checkpoint 落后可重建报告，不能重费。
报告和输入均校验完整性/绑定，报告事实必须等于输入。最多两代，共享发送上限。
证据改变保留旧报告并标 stale；当前资格仍由后端复验决定。来源模块纳入冻结摘要。

终态 status/export/resume 只读，零网络，不因规则变化补诊断；导出 freshness=snapshot_only，
current_eligibility=unverified。活动任务拒绝混规则、模型、开关或生产来源恢复。

```text
python scripts/export_diagnosis.py --storage <canonical-dir> --thread-id <id> --output <new-dir>
```

只输出 diagnosis.json、diagnosis.md，拒绝覆盖已有目录。原成本七文件契约保持不变；
指定 Run 产物损坏时仍拒绝完整成本导出。诊断导出另附 diagnosis-cost-evidence-v1，
以独立账本和冻结训练成本表明 known/held/unknown，artifact_status=unverified；
它不是产物完整性证明，也不等于成本七文件。off 输出 disabled，不生成诊断报告/调用。

## 验证与研究边界

专项位于 test_step9_evidence、test_step9_search_audit、test_step9_contracts、
test_step9_graph、test_step9_recovery。脚本化 provider、真实训练、真实 Qwen 分开登记。
最终工程证据放 work/step9_acceptance；原始数据库/数据/模型不进 Git。
建议本步不执行，不能据此宣称训练或 Test 质量改善；独立验收、研究消融及部署另行进行。
