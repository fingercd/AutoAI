from fastapi import Request

from ..runs.contracts import Principal


def get_principal(_: Request) -> Principal:
    return Principal(owner_id=None, tenant_id=None)
