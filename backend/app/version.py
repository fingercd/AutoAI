"""Web、结果投影与训练 Worker 之间共享的公开契约版本。"""

from __future__ import annotations


RUN_RESULT_CONTRACT_VERSION = 'run-result-v1'
ARTIFACT_MANIFEST_CONTRACT_VERSION = 'run-artifact-manifest-v2'
RUN_SUMMARY_CONTRACT_VERSION = 'v1'

# Worker 产出是否能被当前 Web 结果页完整解释，以 Manifest 契约为边界。
WORKER_CONTRACT_VERSION = ARTIFACT_MANIFEST_CONTRACT_VERSION

WEB_CONTRACTS: dict[str, str] = {
    'run_result': RUN_RESULT_CONTRACT_VERSION,
    'artifact_manifest': ARTIFACT_MANIFEST_CONTRACT_VERSION,
    'run_summary': RUN_SUMMARY_CONTRACT_VERSION,
}
