"""Request identity shared by Run and Batch routes."""
import hashlib
import json

from fastapi import HTTPException

from ..runs.repository import SubmissionConflict


def request_identity(payload, key):
    if key is None:
        return None
    if not 1 <= len(key) <= 128 or not key.isascii() or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise HTTPException(422, detail='Idempotency-Key 必须是 1–128 个可打印非空 ASCII 字符')
    try:
        encoded = json.dumps(payload.model_dump(), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise HTTPException(422, detail='提交请求不得包含非有限数值') from exc
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def existing_submission(repository, *, kind, key, fingerprint, principal):
    try:
        return repository.submission_target(kind=kind, request_key=key, fingerprint=fingerprint, principal=principal)
    except SubmissionConflict as exc:
        raise HTTPException(409, detail={'code': 'idempotency_conflict', 'message': str(exc)}) from exc
