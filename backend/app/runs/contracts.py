"""Run 状态机使用的不可变领域对象和兼容状态映射。

本模块是 runs 包的契约核心，被状态机、投影层和路由层共同依赖：

- RunState / LEGACY_STATUS：规范状态机与旧前端 status 字段之间的语义桥；
- public_error_message：对外错误信息的脱敏出口，防止泄露服务器绝对路径；
- Principal：服务端注入的权限范围（owner/tenant），local 模式两个字段均为空；
- RunRecord：SQLite 中一个 Run 的完整规范快照（frozen dataclass，不可变）。

关键约束：server 模式下身份只经 Principal 注入，请求体不接受 owner_id/tenant_id；
所有对外暴露的错误字符串必须经过 public_error_message 过滤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

# 规范状态机词汇表：queued → running → succeeded/failed/cancelled。
# 数据库与内部逻辑只用这套词，旧前端词汇经 LEGACY_STATUS 在读取时映射。
RunState = Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']

# 旧前端 status.json / 列表接口使用的老状态词表。
# 注意 cancelled 历史上展示为 paused——这是对外兼容契约，不能随意改词。
LEGACY_STATUS: dict[RunState, str] = {
    'queued': 'pending',
    'running': 'running',
    'succeeded': 'success',
    'failed': 'failed',
    'cancelled': 'paused',
}

# 匹配 Windows 盘符路径（C:\ 或 C:/ 开头）与 POSIX 绝对路径（/a/b/ 形式），
# 用于拦截异常消息里夹带的服务器文件系统信息。
_ABSOLUTE_PATH_PATTERN = re.compile(
    r'(?:[A-Za-z]:[\\/])|(?:^|\s)/(?:[^/\s]+/)+',
    flags=re.IGNORECASE,
)


def public_error_message(message: object) -> str:
    """阻止旧错误字符串把服务器绝对路径带入 API/状态投影。

    空消息归一为“训练失败”；一旦检测到绝对路径就整段替换为占位文案——
    宁可损失可读性，也不向浏览器泄露部署目录结构。
    返回值永远是可以安全展示给前端用户的字符串。
    """
    # message 可能是 None 或异常对象，先统一转成去空白文本再判断。
    text = str(message or '').strip()
    if not text:
        return '训练失败'
    # 命中路径模式即整段替换，不做局部打码——局部打码容易漏网。
    if _ABSOLUTE_PATH_PATTERN.search(text):
        return '训练失败；详细路径信息仅记录在服务器日志中'
    return text


@dataclass(frozen=True)
class Principal:
    """由服务端注入的所有者/租户范围；本地模式两个字段均为空。

    server 模式下身份只从这里进入系统：请求体不接受 owner_id/tenant_id，
    Run 的读取、取消、删除与下载都必须按 Principal 的 scope 过滤。
    """
    owner_id: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """SQLite 中一个 Run 的完整规范快照。

    frozen=True 保证记录读出后不可被意外修改——状态变更必须走 store 的状态机。
    字段分组：标识与状态（run_id/state/version）、数据与配置（dataset_id/
    legacy_data_path/config）、Worker 租约（claim_token/worker_id/lease_expires_at）、
    结果与错误（manifest_name/error/error_details/dataset_snapshot）、
    权限范围（owner_id/tenant_id）以及各类时间戳。
    """
    run_id: str
    state: RunState
    version: int
    dataset_id: str | None
    legacy_data_path: str | None
    config: dict[str, Any]
    progress: dict[str, Any] = field(default_factory=dict)
    claim_token: str | None = None
    worker_id: str | None = None
    lease_expires_at: str | None = None
    manifest_name: str | None = None
    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    dataset_snapshot: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None
    tenant_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    # legacy_status 是派生只读视图：库中只存规范 state，旧词表在读取时即时映射。
    @property
    def legacy_status(self) -> str:
        return LEGACY_STATUS[self.state]
