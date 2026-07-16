"""请求身份注入边界。

当前是单机模式，因此所有请求都得到空身份。未来接入认证时应只改这个服务端
边界，不能重新允许客户端在 JSON 中声明 owner_id 或 tenant_id。
"""

from fastapi import Request

from ..runs.contracts import Principal


def get_principal(_: Request) -> Principal:
    """返回当前请求的服务端 Principal；本地模式故意不区分租户。"""
    return Principal(owner_id=None, tenant_id=None)
