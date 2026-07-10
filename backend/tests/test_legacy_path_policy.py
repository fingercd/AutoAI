from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.app.routers.deps import resolve_legacy_dataset_path


def test_legacy_data_path_must_be_inside_an_allowed_root(tmp_path):
    uploads = tmp_path / 'uploads'
    uploads.mkdir()
    allowed = uploads / 'allowed.csv'
    allowed.write_text('x', encoding='utf-8')
    assert resolve_legacy_dataset_path(allowed, allowed_roots=[uploads]) == str(allowed.resolve())

    outside = tmp_path / 'outside.csv'
    outside.write_text('x', encoding='utf-8')
    with pytest.raises(HTTPException) as exc:
        resolve_legacy_dataset_path(outside, allowed_roots=[uploads])
    assert exc.value.status_code == 403
