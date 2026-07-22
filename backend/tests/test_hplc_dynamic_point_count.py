from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def _write_hplc(path: Path, point_count: int) -> Path:
    x = np.linspace(0.0, 50.0, point_count, dtype=np.float64)
    y = 2.0 * x + 1.0
    pd.DataFrame({0: x, 1: y}).to_csv(path, index=False, header=False)
    return path


def test_hplc_consistent_8000_point_batch_builds_dynamic_grid_and_names(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    files = [_write_hplc(tmp_path / f"sample-{index}.CSV", 8000) for index in range(2)]

    result = preprocess_hplc_files_with_preview(
        files,
        display_names=[item.name for item in files],
    )

    assert result["inspection"]["processable"] is True
    assert result["inspection"]["common_point_count"] == 8000
    assert result["hplc_axis"]["grid_point_count"] == 8000
    assert result["frame"].shape == (2, 8004)
    assert result["frame"].columns[:4].tolist() == [
        "Index",
        "Label",
        "Sample_ID",
        "Name",
    ]
    assert result["frame"]["Name"].tolist() == [item.name for item in files]


def test_hplc_inspection_uses_unique_majority_and_lists_outlier_names(tmp_path):
    from backend.app.hplc import inspect_hplc_files

    files = [
        _write_hplc(tmp_path / "majority-a.csv", 5),
        _write_hplc(tmp_path / "majority-b.csv", 5),
        _write_hplc(tmp_path / "outlier.csv", 6),
    ]

    result = inspect_hplc_files(files, display_names=[item.name for item in files])

    assert result["processable"] is False
    assert result["expected_point_count"] == 5
    assert result["common_point_count"] is None
    assert result["files"][2]["status"] == "point_count_mismatch"
    assert result["files"][2]["point_count"] == 6
    assert "outlier.csv（6 点）" in result["message"]


def test_hplc_inspection_does_not_choose_expected_count_when_modes_tie(tmp_path):
    from backend.app.hplc import inspect_hplc_files

    files = [
        _write_hplc(tmp_path / "five.csv", 5),
        _write_hplc(tmp_path / "six.csv", 6),
    ]

    result = inspect_hplc_files(files, display_names=[item.name for item in files])

    assert result["processable"] is False
    assert result["expected_point_count"] is None
    assert [item["point_count"] for item in result["point_count_groups"]] == [5, 6]
    assert "没有唯一多数点数" in result["message"]
    assert "five.csv" in result["message"]
    assert "six.csv" in result["message"]


def test_hplc_inspect_api_returns_original_names_and_cleans_temp_uploads(
    tmp_path, monkeypatch
):
    from backend.app.main import app
    from backend.app.routers import preprocess as preprocess_router

    source = _write_hplc(tmp_path / "原始文件.CSV", 8)
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(preprocess_router, "UPLOADS_DIR", uploads)

    with source.open("rb") as handle:
        response = TestClient(app).post(
            "/api/preprocess/hplc/inspect",
            files=[("files", (source.name, handle, "text/csv"))],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["processable"] is True
    assert payload["common_point_count"] == 8
    assert payload["files"][0]["name"] == source.name
    assert not uploads.exists() or not list(uploads.iterdir())


def test_hplc_preprocess_api_mismatch_error_uses_original_names(tmp_path, monkeypatch):
    from backend.app.main import app
    from backend.app.routers import preprocess as preprocess_router

    files = [
        _write_hplc(tmp_path / "normal-a.csv", 5),
        _write_hplc(tmp_path / "normal-b.csv", 5),
        _write_hplc(tmp_path / "different.csv", 6),
    ]
    monkeypatch.setattr(preprocess_router, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(preprocess_router, "PREPROCESSED_DIR", tmp_path / "outputs")
    opened = [item.open("rb") for item in files]
    try:
        response = TestClient(app).post(
            "/api/preprocess/hplc",
            files=[
                ("files", (path.name, handle, "text/csv"))
                for path, handle in zip(files, opened)
            ],
        )
    finally:
        for handle in opened:
            handle.close()

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "多数文件为 5" in detail
    assert "different.csv（6 点）" in detail
    assert "uuid" not in detail.lower()
