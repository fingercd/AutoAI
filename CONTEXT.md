# Pan 当前实现与发布状态

## 核对基准

最近核对：2026-09-29。本次为文档统一及闲置开发副本清理，业务源码和现有服务不变。

| 项目 | 当前状态 |
|---|---|
| 唯一开发目录 | `/users/fotile/AutoAI/Pan` |
| 开发主线 | `pan/agent`；不创建或切换到其他开发分支 |
| 源码基准 | `019f1cc673f9f289419e4ec1e68204659d94e34b` |
| 发布目录 | `/users/fotile/AutoAI/step78/current` 指向同一 SHA 的 release |
| 服务控制器来源 | 固定 release `fcea2b1555a8d3f5d4f815494e46771839fe8336`，不随 current 改变 |
| 新配方任务 | `agent-recipes-revision-v7` / `agent-state-v9`，反馈诊断默认启用 |
| 执行范围 | 同一 recipe Session 为 `max_runs=1`；诊断建议不执行 |
| worker 合同 | `training-worker-guard-v1` |
| 候选历史 | `2699a55` 的第5/6步研究补充、`56108ee` 的第十步候选未合入；归档供参考，不作为当前功能 |

GitHub 的 `master` 和其他现有分支保留，但日常开发只使用上述服务器主线。本次已删除未被工作树使用的 py/step4-old-head 引用；跨节点进程核验未完成的10个工作树及 development 链接暂保留，禁止继续作为开发入口。临时分支/副本清理清单和归档记录保存在 `work/maintenance-20260929/`，其文件不进入 Git。发布快照、历史 Agent 参考库和运行资源不作为新开发入口。

## 项目与阶段

Pan 以既有光谱/HPLC 分类训练后端为基础开发小样本训练 Agent。用户确认的开发顺序为 State 闭环、全模型接口、证据与有限配方、领域先验、动态处理、有界搜索、统一预算、训练检查、反馈诊断、有限重规划、不确定性与停止、案例记忆、固定评测与消融。

当前主线包含第1至第9步的实现及收尾修复。第十步候选未合入，第11至第13步不在此宣称完成；实现存在不等于已获得论文效果证据。早期 `AutoAI_开发计划.md` 和 `docs/plans/` 只作历史材料，不替代当前顺序和操作规则。

## 当前 Agent 行为

- CLI 使用 `python -m agent_poc.orchestration start/resume/status`，通过 Agent v2 的冻结能力和合法配方调用现有后端；旧 v1、旧 revision 和 `direct_action` 按各自协议读取/恢复。
- 新配方任务默认 `processing-mode=fixed`、`search-mode=fixed`、`knowledge=off`、`fail-fast-guard=on`；反馈诊断默认启用。当前默认预算感知为 on，可显式选择 off，硬预算与给 LLM 展示预算是不同概念。完整参数以 [CLI/State 契约](docs/langgraph_state_contract.md) 为准。
- Session 冻结数据指纹、按 Sample_ID 的评估计划、Train-only 证据、模型配置、配方、知识及预算。新配方请求使用当前规定的 grouped holdout；不要将人工训练的其他评估入口当成 Agent 已开放能力。
- 人工训练与 Agent 共用 Run 提交和 worker 执行；HTTP 只创建 queued Run，BackgroundTasks 不执行训练。Agent 账本与 Runs 数据库通过持久映射和显式 reconciliation 恢复崩溃窗口。
- Agent 只能使用规定工具与后端生成的输入。Observation、Session、诊断和选择不提供 Test；通用结果页/API 仍按既定权限显示 Train/Valid/Test。本次不增加 Test 计算或开放时间门禁。
- 第九步用同源 Train/Valid 审计和 Guard 证据生成解释及只读建议；不产生第二个 Run。失败先确认后端关闭再解释，成功诊断后沿原 Finalize 流程收尾，终态历史读取不新增网络调用。
- 预算预约、实际消耗、失败/修复/重试和 unknown/held 持久记录；Linux 与 Windows 的既有进程监督实现保留。

## 数据、模型与结果

现有业务数据合同保持不变：新预处理输出 `wide-feature-v2`，前四列 `Index,Label,Sample_ID,Name`，其后为真实、有限、唯一、严格递增的数值坐标；读取兼容没有 Name 的 v1。曲线共享轴、独立测试同轴、同组标签一致和当前重复次数约束继续执行。拉曼先截取再基线校正；HPLC 使用批次公共点数、0–50 分钟目标轴和既有插值规则。详细内容见 [README](README.md#建模-csv) 与 [前后端契约](docs/frontend_backend_handoff.md)。

核对时能力目录注册14个目标，其中13个有训练实现；实际可用性还取决于环境依赖。DSCARNet 已退役，新请求和别名被拒绝，历史产物按权限和完整性规则只读；Mamba 当前不可用，不做近似替代。模型能力以代码目录和 `GET /api/models` 为准。

人工分类支持 grouped holdout、按 Sample_ID 留一 CV、外部测试 holdout；Agent 请求范围单独按其协议执行。传统搜索通常按 Valid balanced accuracy、RF 按 OOB 选优；传统最终 Train+Valid 重训与选参阶段区别记录。深度模型保留既有 Valid loss 最佳权重规则。这里描述现状，不新增搜索或评估整改。

结果接口为 `run-result-v1`。CV 主测试指标取 pooled OOF，fold mean/std 作为审计。新训练生成单样品解释，历史全局解释按原 Manifest 只读。模型权重和内部 joblib 不在公开下载白名单；当前没有正式 ROC/AUC/PR 产物，不绘制虚假图表。

## 运行与维护入口

FastAPI 托管经典前端和 v2 工作台；`run.py` 为公共启动器，另有 `run_classic.py` 和 `run_v2.py`。当前服务器实例使用独立 release 与持久 storage，现场启动/停止须按 [部署说明](deploy/server_deploy.md) 和 [服务控制契约](docs/service_control.md) 操作，不直接在开发目录另启生产进程。

server 模式使用服务端 Principal、Bearer 认证和明确 CORS；浏览器 token 只进标签页 sessionStorage。身份、数据完整性、Manifest 和 worker 兼容性检查继续有效。

知识开启时使用冻结的正文向量 Top-3 检索，目前发布材料为5张有来源的知识卡；knowledge-off 不加载 embedding。生产知识调用仍须配置实际 tokenizer/context window，整卡裁剪、来源绑定和恢复规则见 [Agent 配方契约](docs/agent_step2_contract.md)。本次没有运行新的知识效果实验。

所有开发 Git 操作先遵守 [AGENTS](AGENTS.md) 的目录与分支核对；提交/推送规则见 [发布规范](docs/github_publish_policy.md)。修改文档检查差异、事实和链接，不自动运行训练或重启服务。其它验证按实际改动选择，不能将某台开发机的绝对路径当作服务器命令。
