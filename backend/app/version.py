"""Web、结果投影与训练 Worker 之间共享的公开契约版本。

【模块级说明】
- 职责：集中定义三类对外数据契约的版本号常量，相当于 Web 结果页、Run
  摘要投影与训练 Worker 产物之间的“协议版本登记表”，本文件不含任何逻辑。
- 系统位置：被结果序列化（runs 路由 / result 投影）、Run 产物 Manifest
  校验以及前端契约字符串测试共同引用；任何一端升级契约格式都必须同步
  这里的版本号并评估前后兼容。
- 关键设计约束：Worker 进程与 Web 进程相互独立、可能版本不一致，因此用
  显式版本字符串（而非隐式约定）判定一份训练产物能否被当前 Web 端完整
  解释；契约名本身是前后端共享的字符串协议，改动会影响前端与测试。
"""

from __future__ import annotations


# 训练结果接口 `GET /api/training/runs/{run_id}/result` 的响应契约版本；
# 前端结果页（`#/results?run_id=...`）按 'run-result-v1' 定义的结构渲染指标、
# 混淆矩阵与可解释性内容。
RUN_RESULT_CONTRACT_VERSION = 'run-result-v1'
# Run 产物清单（Manifest）契约版本：v2 要求显式 catalog 及每个 artifact 的
# SHA-256 与大小校验，结果页下载白名单和完整性验证都以它为准。
ARTIFACT_MANIFEST_CONTRACT_VERSION = 'run-artifact-manifest-v2'
# Run 列表摘要投影（`GET /api/training/runs?projection=summary`）的契约版本，
# 该投影可返回可空的 `test_macro_f1` 字段。
RUN_SUMMARY_CONTRACT_VERSION = 'v1'

# Worker must understand frozen processing semantics before it claims new Runs.
WORKER_CONTRACT_VERSION = 'training-worker-guard-v1'

# 汇总字典：一次性暴露全部 Web 端契约版本，供契约/健康类接口原样返回，
# 前端可据此快速判断前后端契约是否匹配；键名同样是公开协议的一部分。
WEB_CONTRACTS: dict[str, str] = {
    'run_result': RUN_RESULT_CONTRACT_VERSION,
    'artifact_manifest': ARTIFACT_MANIFEST_CONTRACT_VERSION,
    'run_summary': RUN_SUMMARY_CONTRACT_VERSION,
}
