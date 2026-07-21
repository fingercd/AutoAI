import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _write_hplc_source(
    path: Path,
    *,
    x: np.ndarray | None = None,
    point_count: int = 7500,
) -> Path:
    source_x = (
        np.linspace(0.0, 50.0, point_count, dtype=np.float64)
        if x is None
        else np.asarray(x, dtype=np.float64)
    )
    source_y = 7.0 + 1.25 * source_x
    pd.DataFrame({0: source_x, 1: source_y}).to_csv(
        path, index=False, header=False
    )
    return path


def test_hplc_target_row_selection_uses_injected_grid_limit():
    from backend.app.hplc import (
        HplcGridConfig,
        _select_hplc_target_axis,
        build_hplc_target_axis,
    )

    config = HplcGridConfig(1.0, 3.0, 5, 1e-10)
    selection = _select_hplc_target_axis(
        start_row=2, end_row=4, config=config
    )
    full_axis = build_hplc_target_axis(config)

    assert np.array_equal(selection.full_axis, full_axis)
    assert np.array_equal(selection.indices, np.array([1, 2, 3]))
    assert np.array_equal(selection.target_x, full_axis[1:4])
    assert selection.offset == 1
    assert selection.length == 3
    assert selection.selected_start_row == 2
    assert selection.selected_end_row == 4

    through_end = _select_hplc_target_axis(
        start_row=3, end_row=None, config=config
    )
    assert np.array_equal(through_end.target_x, full_axis[2:])
    assert through_end.selected_end_row == config.point_count


@pytest.mark.parametrize(
    ("start_row", "end_row", "message"),
    [
        (0, 4, "当前为 0"),
        (1, 6, "当前为 6"),
        (6, None, "当前为 6"),
        (4, 3, "起始行不能大于终止行"),
        (1.5, 4, "必须是 1 到 5 的整数"),
        (True, 4, "必须是 1 到 5 的整数"),
    ],
)
def test_hplc_target_row_selection_rejects_invalid_bounds(
    start_row, end_row, message
):
    from backend.app.hplc import HplcGridConfig, _select_hplc_target_axis

    config = HplcGridConfig(0.0, 4.0, 5, 1e-10)
    with pytest.raises(ValueError, match=message):
        _select_hplc_target_axis(
            start_row=start_row, end_row=end_row, config=config
        )


def test_hplc_target_time_selection_uses_closed_real_minute_grid():
    from backend.app.hplc import HplcGridConfig, _select_hplc_target_axis

    config = HplcGridConfig(0.0, 4.0, 5, 1e-10)
    selection = _select_hplc_target_axis(
        range_mode="x_value", x_min=1.1, x_max=3.0, config=config
    )
    assert np.array_equal(selection.target_x, np.array([2.0, 3.0]))
    assert selection.offset == 2
    assert selection.length == 2
    assert selection.selected_start_row == 3
    assert selection.selected_end_row == 4

    partly_outside = _select_hplc_target_axis(
        range_mode="x_value", x_min=-1.0, x_max=1.0, config=config
    )
    assert np.array_equal(partly_outside.target_x, np.array([0.0, 1.0]))

    unbounded = _select_hplc_target_axis(
        range_mode="x_value", config=config
    )
    assert np.array_equal(unbounded.target_x, np.linspace(0.0, 4.0, 5))


@pytest.mark.parametrize(
    ("x_min", "x_max", "message"),
    [
        (np.nan, 1.0, "下限必须是有限数值"),
        (0.0, np.inf, "上限必须是有限数值"),
        (3.0, 2.0, "下限不能大于上限"),
        (4.1, 5.0, "没有数据"),
    ],
)
def test_hplc_target_time_selection_rejects_invalid_or_empty_ranges(
    x_min, x_max, message
):
    from backend.app.hplc import HplcGridConfig, _select_hplc_target_axis

    config = HplcGridConfig(0.0, 4.0, 5, 1e-10)
    with pytest.raises(ValueError, match=message):
        _select_hplc_target_axis(
            range_mode="x_value", x_min=x_min, x_max=x_max, config=config
        )


def test_hplc_interpolation_rejects_single_target_point_but_raw_mode_keeps_it(
    tmp_path
):
    from backend.app.hplc import HplcGridConfig, preprocess_hplc_files_with_preview

    config = HplcGridConfig(0.0, 4.0, 5, 1e-10)
    source = _write_hplc_source(
        tmp_path / "five_points.csv", x=np.linspace(0.0, 4.0, 5)
    )

    with pytest.raises(ValueError, match="少于 2 个点"):
        preprocess_hplc_files_with_preview(
            [source], start_row=5, end_row=5, config=config
        )

    raw_result = preprocess_hplc_files_with_preview(
        [source],
        start_row=5,
        end_row=5,
        interpolate=False,
        config=config,
    )
    assert raw_result["hplc_axis"] is None
    assert raw_result["common_time"] == []
    assert raw_result["curves"][0]["x"] == [4]
    assert len(raw_result["curves"][0]["processed_y"]) == 1


def test_hplc_raw_mode_filters_each_original_real_time_axis(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    source_x = np.linspace(60.0, 110.0, 7500, dtype=np.float64)
    source = _write_hplc_source(tmp_path / "late_window.csv", x=source_x)
    result = preprocess_hplc_files_with_preview(
        [source],
        range_mode="x_value",
        x_min=65.0,
        x_max=66.0,
        interpolate=False,
    )

    expected = source_x[(source_x >= 65.0) & (source_x <= 66.0)]
    assert result["hplc_axis"] is None
    assert result["common_time"] == []
    assert len(result["curves"][0]["x"]) == len(expected)
    np.testing.assert_allclose(
        result["curves"][0]["x"], np.round(expected, 5), rtol=0, atol=0
    )


def test_hplc_partial_slice_descriptor_round_trips_exact_full_grid_positions(
    tmp_path,
):
    from backend.app.hplc import (
        DEFAULT_HPLC_GRID,
        build_hplc_target_axis,
        preprocess_hplc_files_with_preview,
    )
    from backend.app.parsers import load_modeling_csv

    source = _write_hplc_source(tmp_path / "partial.csv")
    result = preprocess_hplc_files_with_preview(
        [source], start_row=100, end_row=4000
    )
    expected_axis = build_hplc_target_axis(DEFAULT_HPLC_GRID)[99:4000]
    descriptor = json.loads(result["frame"].iloc[0]["XXX"])

    assert descriptor == {
        "type": "linspace-slice-v1",
        "grid_start": 0.0,
        "grid_stop": 50.0,
        "grid_count": 7500,
        "offset": 99,
        "length": 3901,
        "unit": "minute",
    }
    assert np.array_equal(np.asarray(result["common_time"]), expected_axis)
    assert result["hplc_axis"] == {
        "start": float(expected_axis[0]),
        "stop": float(expected_axis[-1]),
        "unit": "minute",
        "point_count": 3901,
        "step_minutes": DEFAULT_HPLC_GRID.step_minutes,
        "mapping": "piecewise_linear",
        "input_point_count_required": 7500,
        "encoding": "linspace-slice-v1",
        "grid_start": 0.0,
        "grid_stop": 50.0,
        "grid_point_count": 7500,
        "selected_start_row": 100,
        "selected_end_row": 4000,
    }
    assert len(json.loads(result["frame"].iloc[0]["Intensity"])) == 3901

    modeling = result["frame"].copy()
    modeling["Label"] = "A"
    modeling["Sample_ID"] = "S1"
    modeling_path = tmp_path / "modeling.csv"
    modeling.to_csv(modeling_path, index=False, encoding="utf-8-sig")
    loaded = load_modeling_csv(modeling_path)
    assert np.array_equal(np.asarray(loaded.x_axis[0]), expected_axis)


def test_hplc_time_range_pipeline_reports_actual_grid_points(tmp_path):
    from backend.app.hplc import (
        DEFAULT_HPLC_GRID,
        build_hplc_target_axis,
        preprocess_hplc_files_with_preview,
    )

    source = _write_hplc_source(tmp_path / "time_range.csv")
    result = preprocess_hplc_files_with_preview(
        [source], range_mode="x_value", x_min=0.66, x_max=26.67
    )
    full_axis = build_hplc_target_axis(DEFAULT_HPLC_GRID)
    positions = np.flatnonzero((full_axis >= 0.66) & (full_axis <= 26.67))
    expected_axis = full_axis[positions]
    descriptor = json.loads(result["frame"].iloc[0]["XXX"])

    assert np.array_equal(np.asarray(result["common_time"]), expected_axis)
    assert descriptor["offset"] == int(positions[0])
    assert descriptor["length"] == len(positions)
    assert result["hplc_axis"]["selected_start_row"] == int(positions[0]) + 1
    assert result["hplc_axis"]["selected_end_row"] == int(positions[-1]) + 1
    assert result["hplc_axis"]["start"] == float(expected_axis[0])
    assert result["hplc_axis"]["stop"] == float(expected_axis[-1])
    assert result["output_precision"]["xxx_max_characters"] == len(
        result["frame"].iloc[0]["XXX"]
    )


def test_modeling_axis_parser_keeps_legacy_formats_and_reads_slice_exactly():
    from backend.app.parsers import _parse_modeling_axis

    descriptor = {
        "type": "linspace-slice-v1",
        "grid_start": 0.0,
        "grid_stop": 4.0,
        "grid_count": 5,
        "offset": 1,
        "length": 3,
        "unit": "minute",
    }
    assert _parse_modeling_axis(descriptor, 2) == [1.0, 2.0, 3.0]
    assert _parse_modeling_axis("[0, 0.5, 1]", 2) == [0.0, 0.5, 1.0]
    assert _parse_modeling_axis(
        {"type": "linspace-v1", "start": 0, "stop": 1, "count": 3}, 2
    ) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    "changes",
    [
        {"grid_start": "0"},
        {"grid_start": np.nan},
        {"grid_stop": 0.0},
        {"grid_count": True},
        {"grid_count": 5.0},
        {"grid_count": 1},
        {"grid_count": 1_000_001},
        {"offset": True},
        {"offset": -1},
        {"length": True},
        {"length": 0},
        {"offset": 4, "length": 2},
        {"unit": "second"},
        {"unexpected": 1},
    ],
)
def test_modeling_axis_parser_rejects_malformed_slice_descriptors(changes):
    from backend.app.parsers import _parse_modeling_axis

    descriptor = {
        "type": "linspace-slice-v1",
        "grid_start": 0.0,
        "grid_stop": 4.0,
        "grid_count": 5,
        "offset": 0,
        "length": 5,
        "unit": "minute",
    }
    descriptor.update(changes)
    with pytest.raises(ValueError):
        _parse_modeling_axis(descriptor, 2)


@pytest.mark.parametrize("end_row", [7501, 9000])
@pytest.mark.parametrize("interpolate", [True, False])
def test_hplc_api_rejects_row_end_beyond_grid_without_output(
    tmp_path, monkeypatch, end_row, interpolate
):
    from backend.app.main import app
    from backend.app.routers import preprocess as preprocess_router

    source = _write_hplc_source(tmp_path / "source.csv")
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(preprocess_router, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(preprocess_router, "PREPROCESSED_DIR", outputs)

    client = TestClient(app)
    with source.open("rb") as handle:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (source.name, handle, "text/csv"))],
            data={
                "end_row": str(end_row),
                "hplc_interpolate": str(interpolate).lower(),
            },
        )

    assert response.status_code == 400
    assert f"1 到 7500" in response.json()["detail"]
    assert f"当前为 {end_row}" in response.json()["detail"]
    assert not outputs.exists() or not list(outputs.iterdir())


def test_hplc_api_writes_only_one_self_contained_modeling_csv(
    tmp_path, monkeypatch
):
    from backend.app.main import app
    from backend.app.parsers import load_modeling_csv
    from backend.app.routers import preprocess as preprocess_router

    source = _write_hplc_source(tmp_path / "source.csv")
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(preprocess_router, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(preprocess_router, "PREPROCESSED_DIR", outputs)

    client = TestClient(app)
    with source.open("rb") as handle:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (source.name, handle, "text/csv"))],
            data={"start_row": "100", "end_row": "4000"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "xxx_download_url" not in payload
    assert "xxx_rows" not in payload
    assert payload["download_url"].startswith("/api/files?path=")
    generated = list(outputs.iterdir())
    assert len(generated) == 1
    assert generated[0].name.startswith("hplc_")
    assert generated[0].suffix == ".csv"
    assert "_xxx" not in generated[0].name

    frame = pd.read_csv(generated[0])
    assert list(frame.columns) == [
        "Index",
        "Name",
        "XXX",
        "Intensity",
        "Label",
        "Sample_ID",
    ]
    frame["Label"] = "A"
    frame["Sample_ID"] = "S1"
    modeling_path = tmp_path / "roundtrip.csv"
    frame.to_csv(modeling_path, index=False, encoding="utf-8-sig")
    loaded = load_modeling_csv(modeling_path)
    assert np.array_equal(
        np.asarray(loaded.x_axis[0]), np.asarray(payload["common_time"])
    )
