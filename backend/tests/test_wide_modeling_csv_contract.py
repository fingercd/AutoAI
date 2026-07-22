from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def test_wide_modeling_frame_round_trips_real_axis_scalar_intensity_and_zero_padded_id(
    tmp_path,
):
    from backend.app.parsers import build_wide_modeling_frame, load_modeling_csv

    x_axis = np.asarray([0.0, 0.125, 1.2345678901234567], dtype=np.float64)
    result = build_wide_modeling_frame(
        indices=["curve-1", "curve-2"],
        x_arrays=[x_axis, x_axis.copy()],
        intensity_arrays=[
            np.asarray([1.234567, 2.0, 3.0]),
            np.asarray([4.0, 5.0, 6.0]),
        ],
        source_names=["first.csv", "second.csv"],
    )

    assert list(result.frame.columns) == [
        "Index",
        "Label",
        "Sample_ID",
        "0",
        "0.125",
        "1.2345678901234567",
    ]
    assert "Name" not in result.frame.columns
    assert result.frame.iloc[0, 3:].tolist() == [1.23457, 2.0, 3.0]
    assert result.output_precision == {
        "format": "wide-feature-v1",
        "xxx_encoding": "column_headers",
        "xxx_precision": "float64-roundtrip",
        "intensity_decimal_places": 5,
        "adaptive": False,
        "feature_count": 3,
        "total_column_count": 6,
        "excel_column_limit": 16_384,
        "excel_compatible": True,
    }

    result.frame["Label"] = ["A", "A"]
    result.frame["Sample_ID"] = ["001", "001"]
    path = tmp_path / "wide.csv"
    result.frame.to_csv(path, index=False, encoding="utf-8-sig")

    loaded = load_modeling_csv(path)
    assert loaded.frame.columns.tolist() == ["Index", "Label", "Sample_ID"]
    assert loaded.sample_id == ["001", "001"]
    assert loaded.labels == ["A", "A"]
    assert loaded.x_axis == [x_axis.tolist(), x_axis.tolist()]
    np.testing.assert_allclose(
        loaded.intensity,
        np.asarray([[1.23457, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_load_modeling_csv_strictly_rejects_legacy_six_column_arrays(tmp_path):
    from backend.app.parsers import load_modeling_csv

    path = tmp_path / "legacy.csv"
    path.write_text(
        'Index,Name,XXX,Intensity,Label,Sample_ID\n'
        '1,sample,"[0, 1]","[2, 3]",A,S1\n',
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match="旧六列数组格式.*wide-feature-v1"):
        load_modeling_csv(path)


@pytest.mark.parametrize(
    ("feature_headers", "message"),
    [
        (["0", "0"], "表头包含重复列名"),
        (["0", "0.0"], "坐标数值重复"),
        (["0", "not-an-axis"], "不是有效的真实 XXX 数值"),
        (["0", "NaN"], "必须是有限数值"),
        (["1", "0"], "必须按列严格递增"),
    ],
)
def test_load_modeling_csv_rejects_invalid_real_axis_headers(
    tmp_path,
    feature_headers,
    message,
):
    from backend.app.parsers import load_modeling_csv

    path = _write_csv(
        tmp_path / "invalid-axis.csv",
        ["Index", "Label", "Sample_ID", *feature_headers],
        [[1, "A", "S1", 10, 20]],
    )

    with pytest.raises(ValueError, match=message):
        load_modeling_csv(path)


@pytest.mark.parametrize("invalid_intensity", ["", "not-a-number", "NaN", "inf", "-inf"])
def test_load_modeling_csv_rejects_non_finite_or_non_numeric_intensity(
    tmp_path,
    invalid_intensity,
):
    from backend.app.parsers import load_modeling_csv

    path = _write_csv(
        tmp_path / "invalid-intensity.csv",
        ["Index", "Label", "Sample_ID", "0", "1"],
        [[1, "A", "S1", 10, invalid_intensity]],
    )

    with pytest.raises(ValueError, match="Intensity 必须是有限数值"):
        load_modeling_csv(path)


def test_wide_modeling_frame_rejects_different_axes_in_one_batch():
    from backend.app.parsers import build_wide_modeling_frame

    with pytest.raises(ValueError, match="second.csv 的 XXX 与 first.csv 不一致.*公共真实轴"):
        build_wide_modeling_frame(
            indices=[1, 2],
            x_arrays=[
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([0.0, 1.0000000000000002, 2.0]),
            ],
            intensity_arrays=[
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([4.0, 5.0, 6.0]),
            ],
            source_names=["first.csv", "second.csv"],
        )


def test_local_sample_summary_returns_clear_400_for_legacy_default_data(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    from backend.app.main import app
    from backend.app.routers import catalog

    legacy = tmp_path / "data.csv"
    legacy.write_text(
        'Index,Name,XXX,Intensity,Label,Sample_ID\n'
        '1,sample,"[0,1]","[2,3]",A,S1\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(catalog, "DEFAULT_DATA", legacy)

    response = TestClient(app).get("/api/sample/summary")

    assert response.status_code == 400
    assert "旧六列数组格式" in response.json()["detail"]
    assert "wide-feature-v1" in response.json()["detail"]
