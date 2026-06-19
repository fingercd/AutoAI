from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.parsers import load_modeling_csv, summarize_modeling_csv
from backend.app.training import train_model


ROOT = Path(__file__).resolve().parents[2]
USER_RAMAN_CSV = Path.home() / "Desktop" / "课件" / "raman_b2084be144.csv"


def test_data_csv_summary():
    summary = summarize_modeling_csv(ROOT / "data.csv")
    assert summary["samples"] == 90
    assert summary["classes"] == 2
    assert summary["curve_length"] == 160


def test_modeling_csv_accepts_gbk_and_index_alias(tmp_path):
    source = tmp_path / "gbk_modeling.csv"
    frame = (
        "AutoAI 谱学建模平台,Name,XXX,Intensity,Label,Repeat_index\n"
        '1,s1,"[1, 2, 3]","[4, 5, 6]",A,1\n'
        '2,s2,"[1, 2, 3]","[6, 5, 4]",B,2\n'
    )
    source.write_bytes(frame.encode("gbk"))

    summary = summarize_modeling_csv(source)

    assert summary["samples"] == 2
    assert summary["classes"] == 2
    assert summary["curve_length"] == 3
    assert summary["columns"][0] == "Index"


def _write_grouped_modeling_csv(path: Path, group_count: int = 11, repeats: int = 2) -> None:
    rows = ["Index,Name,XXX,Intensity,Label,Repeat_index"]
    index = 1
    for group in range(1, group_count + 1):
        label = "A" if group <= (group_count + 1) // 2 else "B"
        for repeat in range(repeats):
            rows.append(f'{index},s{group}_{repeat},"[1, 2, 3, 4]","[{group}, {group + 1}, {group + 2}, {group + 3}]",{label},{group}')
            index += 1
    path.write_text("\n".join(rows), encoding="utf-8")


def test_repeat_index_summary_and_incomplete_group_error(tmp_path):
    source = tmp_path / "bad_repeat.csv"
    _write_grouped_modeling_csv(source, group_count=4, repeats=2)
    text = source.read_text(encoding="utf-8")
    source.write_text("\n".join(text.splitlines()[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match="Repeat_index 重复测量次数不一致"):
        load_modeling_csv(source)


def test_custom_split_uses_repeat_index_groups(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=11, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        source,
        {
            "model_type": "knn",
            "split_mode": "custom",
            "split_train": 7,
            "split_valid": 1,
            "split_test": 2,
        },
    )
    split = json.loads((Path(result["run_dir"]) / "split.json").read_text(encoding="utf-8"))
    frame = load_modeling_csv(source).frame

    split_group_counts = {
        name: frame.iloc[idxs]["Repeat_index"].nunique()
        for name, idxs in split.items()
    }
    assert split_group_counts == {"train": 8, "valid": 1, "test": 2}
    assert set(frame.iloc[split["train"]]["Repeat_index"]).isdisjoint(set(frame.iloc[split["valid"]]["Repeat_index"]))
    assert set(frame.iloc[split["train"]]["Repeat_index"]).isdisjoint(set(frame.iloc[split["test"]]["Repeat_index"]))


def test_custom_split_ratio_must_sum_to_ten(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=5, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    with pytest.raises(ValueError, match="相加必须等于 10"):
        train_model(source, {"model_type": "knn", "split_train": 7, "split_valid": 1, "split_test": 1})


def test_external_test_dataset_uses_train_valid_split(tmp_path, monkeypatch):
    import backend.app.training as training

    train_source = tmp_path / "train.csv"
    test_source = tmp_path / "test.csv"
    _write_grouped_modeling_csv(train_source, group_count=10, repeats=2)
    _write_grouped_modeling_csv(test_source, group_count=4, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(train_source, {"model_type": "knn", "test_data_path": str(test_source)})
    split = json.loads((Path(result["run_dir"]) / "split.json").read_text(encoding="utf-8"))

    assert len(split["train"]) == 16
    assert len(split["valid"]) == 4
    assert len(split["test"]) == 8
    assert result["sample_count"] == 20
    assert result["test_sample_count"] == 8


def test_external_test_dataset_rejects_unknown_label(tmp_path, monkeypatch):
    import backend.app.training as training

    train_source = tmp_path / "train.csv"
    test_source = tmp_path / "test.csv"
    _write_grouped_modeling_csv(train_source, group_count=6, repeats=2)
    _write_grouped_modeling_csv(test_source, group_count=2, repeats=2)
    text = test_source.read_text(encoding="utf-8").replace(",A,", ",C,").replace(",B,", ",C,")
    test_source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    with pytest.raises(ValueError, match="测试集包含训练集中不存在的 Label"):
        train_model(train_source, {"model_type": "knn", "test_data_path": str(test_source)})


def test_train_smoke(tmp_path, monkeypatch):
    import backend.app.training as training

    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    result = train_model(ROOT / "data.csv", {"epochs": 1, "batch_size": 16})
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()


@pytest.mark.parametrize("model_type", ["cnn1d", "mlp", "transformer", "unet1d", "dscarnet", "knn", "random_forest", "svm", "xgboost"])
def test_all_model_types_train_one_epoch(tmp_path, monkeypatch, model_type):
    import backend.app.training as training

    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    result = train_model(
        ROOT / "data.csv",
        {
            "epochs": 1,
            "batch_size": 32,
            "model_type": model_type,
            "early_stopping_patience": 5,
            "hidden_size": 32,
            "transformer_heads": 4,
        },
    )

    assert result["status"] == "success"
    assert result["model_type"] == model_type
    assert result["actual_epochs"] == 1
    if result.get("model_family") == "traditional_ml":
        assert (tmp_path / result["run_id"] / "model.pkl").exists()
    else:
        assert (tmp_path / result["run_id"] / "model.pt").exists()


@pytest.mark.skipif(not USER_RAMAN_CSV.exists(), reason="local user Raman CSV fixture is not available")
@pytest.mark.parametrize("model_type", ["KNN", "UNet1D", "DSCARNet"])
def test_user_raman_csv_trains_new_model_choices(tmp_path, monkeypatch, model_type):
    import backend.app.training as training

    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    result = train_model(
        USER_RAMAN_CSV,
        {
            "epochs": 1,
            "batch_size": 16,
            "model_type": model_type,
            "early_stopping_patience": 5,
            "hidden_size": 32,
            "unet_depth": 3,
            "dscarnet_inception_blocks": 1,
            "knn_n_neighbors": 5,
        },
    )

    assert result["status"] == "success"
    assert result["actual_epochs"] == 1
    assert result["sample_count"] == 50


def test_raman_baseline_order_changes_processing_scope(tmp_path, monkeypatch):
    import backend.app.parsers as parsers

    source = tmp_path / "raman.csv"
    lines = ["RamanShift,Intensity"]
    lines.extend(f"{idx},{idx * idx + 10}" for idx in range(1, 12))
    source.write_text("\n".join(lines), encoding="utf-8")

    def fake_baseline_correct(x, y, method):
        return y + len(y) * 100

    monkeypatch.setattr(parsers, "_baseline_correct", fake_baseline_correct)
    range_first = parsers.preprocess_raw_files(
        [source],
        kind="raman",
        start_row=2,
        end_row=3,
        baseline_order="range_then_baseline",
    )
    baseline_first = parsers.preprocess_raw_files(
        [source],
        kind="raman",
        start_row=2,
        end_row=3,
        baseline_order="baseline_then_range",
    )

    assert ast.literal_eval(range_first.iloc[0]["Intensity"]) == [214.0, 219.0]
    assert ast.literal_eval(baseline_first.iloc[0]["Intensity"]) == [1114.0, 1119.0]


def test_preprocess_x_value_range_selects_by_axis(tmp_path):
    from backend.app import parsers

    source = tmp_path / "chrom.csv"
    lines = ["Time,Intensity"]
    lines.extend(f"{idx * 0.5},{idx * 10}" for idx in range(1, 9))
    source.write_text("\n".join(lines), encoding="utf-8")

    result = parsers.preprocess_raw_files_with_preview(
        [source],
        kind="chromatography",
        range_mode="x_value",
        x_min=1.0,
        x_max=2.0,
    )

    assert result["curves"][0]["x"] == [1.0, 1.5, 2.0]
    assert result["curves"][0]["raw_y"] == [20.0, 30.0, 40.0]
    assert ast.literal_eval(result["frame"].iloc[0]["XXX"]) == [1.0, 1.5, 2.0]


def test_raman_preprocess_api_returns_curve_preview(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    source = tmp_path / "raman.csv"
    lines = ["RamanShift,Intensity"]
    lines.extend(f"{idx},{idx * idx + 10}" for idx in range(1, 12))
    source.write_text("\n".join(lines), encoding="utf-8")

    client = TestClient(app)
    with source.open("rb") as file:
        response = client.post(
            "/api/preprocess/raman",
            files=[("files", ("raman.csv", file, "text/csv"))],
            data={
                "start_row": "2",
                "end_row": "10",
                "baseline_order": "range_then_baseline",
                "baseline_method": "poly",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_method"] == "poly"
    assert payload["curves"]
    assert set(payload["curves"][0]) == {"name", "x", "raw_y", "corrected_y"}
    assert len(payload["curves"][0]["x"]) == 9
    assert len(payload["curves"][0]["raw_y"]) == 9
    assert len(payload["curves"][0]["corrected_y"]) == 9


def test_preprocess_preserves_original_names_and_returns_all_curves(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = []
    for idx in range(12):
        source = tmp_path / f"sample_{idx:02d}_original.csv"
        lines = ["RamanShift,Intensity"]
        lines.extend(f"{point},{point * point + idx}" for point in range(1, 12))
        source.write_text("\n".join(lines), encoding="utf-8")
        files.append(source)

    client = TestClient(app)
    opened = [path.open("rb") for path in files]
    try:
        response = client.post(
            "/api/preprocess/raman",
            files=[("files", (path.name, handle, "text/csv")) for path, handle in zip(files, opened)],
            data={
                "start_row": "2",
                "end_row": "10",
                "baseline_order": "range_then_baseline",
                "baseline_method": "poly",
            },
        )
    finally:
        for handle in opened:
            handle.close()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["curves"]) == 12
    assert payload["curves"][0]["name"] == "sample_00_original"
    assert payload["curves"][-1]["name"] == "sample_11_original"
    assert payload["preview"][0]["Name"] == "sample_00_original"


def test_chromatography_preprocess_api_returns_curve_preview(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    source = tmp_path / "chrom.csv"
    lines = ["Time,Intensity"]
    lines.extend(f"{idx},{idx * 2 + (10 if idx == 5 else 0)}" for idx in range(1, 12))
    source.write_text("\n".join(lines), encoding="utf-8")

    client = TestClient(app)
    with source.open("rb") as file:
        response = client.post(
            "/api/preprocess/chromatography",
            files=[("files", ("chrom.csv", file, "text/csv"))],
            data={"start_row": "2", "end_row": "10"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_method"] is None
    assert payload["curves"]
    assert set(payload["curves"][0]) == {"name", "x", "raw_y"}
    assert len(payload["curves"][0]["x"]) == 9
    assert len(payload["curves"][0]["raw_y"]) == 9


@pytest.mark.parametrize("path", ["/ui", "/ui/workbench", "/ui/wizard", "/ui/dashboard", "/ui/console"])
def test_ui_variant_routes(path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    response = client.get(path)

    assert response.status_code == 200
    assert "AutoAI" in response.text


def test_ui_variant_assets_are_served():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    assert client.get("/static/autoai-variants.css").status_code == 200
    script_response = client.get("/static/autoai-variants.js")
    assert script_response.status_code == 200
    assert "initAutoAIVariant" in script_response.text


# ---------------------------------------------------------------------------
# HPLC preprocessing tests
# ---------------------------------------------------------------------------


def _make_hplc_fixture(tmp_path, n_files=2, n_points=50, offset_range=0.005, add_negatives=True):
    """Generate XJ-GC-like test data: uniform time grid + constant offset + Gaussian peaks."""
    files = []
    for i in range(n_files):
        offset = np.random.uniform(-offset_range, offset_range)
        x = np.linspace(0 + offset, 10 + offset, n_points, dtype=np.float64)
        y = (
            100 * np.exp(-0.5 * ((x - 3) / 1.0) ** 2)
            + 80 * np.exp(-0.5 * ((x - 7) / 1.5) ** 2)
            + np.random.normal(0, 0.3, n_points).astype(np.float64)
        )
        if add_negatives:
            y[:3] = -0.3  # simulate baseline drift negatives
        path = tmp_path / f"hplc_{i}.csv"
        pd.DataFrame({0: x, 1: y}).to_csv(path, index=False, header=False)
        files.append(path)
    return files


# ---- Unit tests for pure functions ----


def test_compute_common_time_identical():
    from backend.app.hplc import compute_common_time_axis

    x1 = np.linspace(0, 10, 100, dtype=np.float32)
    x2 = np.linspace(0, 10, 100, dtype=np.float32)
    common = compute_common_time_axis([x1, x2])
    assert len(common) == 100
    np.testing.assert_allclose(common[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(common[-1], 10.0, atol=1e-6)


def test_compute_common_time_offset():
    from backend.app.hplc import compute_common_time_axis

    x1 = np.linspace(0.003, 10.003, 200, dtype=np.float32)
    x2 = np.linspace(0.000, 10.000, 200, dtype=np.float32)
    common = compute_common_time_axis([x1, x2])
    # Overlap: start = max(0.000, 0.003) = 0.003, end = min(10.0, 10.003) = 10.0
    np.testing.assert_allclose(common[0], 0.003, atol=1e-5)
    np.testing.assert_allclose(common[-1], 10.0, atol=1e-5)
    assert len(common) == 200  # median length


def test_compute_common_time_single_file():
    from backend.app.hplc import compute_common_time_axis

    x = np.linspace(1, 50, 7500, dtype=np.float32)
    common = compute_common_time_axis([x])
    np.testing.assert_array_equal(common, x)


def test_compute_common_time_no_overlap():
    from backend.app.hplc import compute_common_time_axis

    x1 = np.linspace(0, 5, 100, dtype=np.float32)
    x2 = np.linspace(6, 10, 100, dtype=np.float32)
    with pytest.raises(ValueError, match="无重叠"):
        compute_common_time_axis([x1, x2])


def test_hplc_interpolate_two_files(tmp_path):
    from backend.app.hplc import hplc_interpolate

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50)
    pairs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        pairs.append((df.iloc[:, 0].values.astype(np.float32), df.iloc[:, 1].values.astype(np.float32)))

    matrix, common_x = hplc_interpolate(pairs)
    assert matrix.shape == (2, 50)
    assert common_x.shape == (50,)
    assert not np.any(np.isnan(matrix))


def test_hplc_subtract_min_basic():
    from backend.app.hplc import hplc_subtract_min

    m = np.array([[-1.0, 0.0, 5.0], [-0.5, 2.0, 10.0]], dtype=np.float32)
    result = hplc_subtract_min(m)
    np.testing.assert_allclose(result[0].min(), 0.0, atol=1e-7)
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-7)  # -1 - (-1) = 0
    np.testing.assert_allclose(result[0, 2], 6.0, atol=1e-7)  # 5 - (-1) = 6
    np.testing.assert_allclose(result[1].min(), 0.0, atol=1e-7)


def test_hplc_subtract_min_all_positive():
    from backend.app.hplc import hplc_subtract_min

    m = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]], dtype=np.float32)
    result = hplc_subtract_min(m)
    assert np.all(result >= 0)
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-7)  # 1 - 1 = 0


def test_hplc_normalize_area_basic():
    from backend.app.hplc import hplc_normalize_area

    m = np.array([[0.0, 1.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    result = hplc_normalize_area(m)
    area = float(np.trapezoid(result[0]))
    np.testing.assert_allclose(area, 1.0, atol=1e-5)


def test_hplc_normalize_area_zero_protection():
    from backend.app.hplc import hplc_normalize_area

    m = np.zeros((2, 10), dtype=np.float32)
    result = hplc_normalize_area(m)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_hplc_pipeline_default(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    files = _make_hplc_fixture(tmp_path, n_files=3, n_points=50)
    result = preprocess_hplc_files_with_preview(
        files, subtract_min=True, normalize_area=True, interpolate=True
    )
    assert result["frame"].shape[0] == 3
    assert len(result["curves"]) == 3
    assert "common_time" in result
    # After full pipeline, all values >= 0
    for curve in result["curves"]:
        assert np.min(curve["processed_y"]) >= 0.0
    # Area normalization: each curve area ≈ 1.0
    for curve in result["curves"]:
        area = float(np.trapezoid(curve["processed_y"]))
        assert abs(area - 1.0) < 0.1  # loose tolerance for random noise


def test_hplc_pipeline_interpolate_off(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50, offset_range=0)
    result = preprocess_hplc_files_with_preview(files, interpolate=False)
    # With same offsets, X should be unchanged
    np.testing.assert_allclose(result["curves"][0]["x"], result["curves"][1]["x"], atol=1e-4)


def test_hplc_pipeline_subtract_min_off(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50, add_negatives=True)
    result = preprocess_hplc_files_with_preview(files, subtract_min=False, normalize_area=False)
    # Negatives may persist
    has_negative = any(np.min(curve["processed_y"]) < 0 for curve in result["curves"])
    assert has_negative  # our fixture has negatives


def test_hplc_pipeline_normalize_off(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50)
    result = preprocess_hplc_files_with_preview(files, normalize_area=False)
    area = float(np.trapezoid(result["curves"][0]["processed_y"]))
    # Area should NOT be 1.0 (normalization disabled)
    assert abs(area - 1.0) > 0.01


# ---- API integration tests ----


def test_hplc_preprocess_api_200(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
            data={"hplc_interpolate": "true", "hplc_subtract_min": "true", "hplc_normalize_area": "true"},
        )
    finally:
        for h in opened:
            h.close()
    assert response.status_code == 200


def test_hplc_preprocess_api_curve_keys(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
        )
    finally:
        for h in opened:
            h.close()
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["curves"][0]) == {"name", "x", "raw_y", "processed_y"}


def test_hplc_preprocess_api_response_fields(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
        )
    finally:
        for h in opened:
            h.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["hplc_interpolate"] is True
    assert payload["hplc_subtract_min"] is True
    assert payload["hplc_normalize_area"] is True
    assert "common_time" in payload
    assert len(payload["common_time"]) >= 48  # overlap may trim 1–2 points with offset
    assert "common_time_path" in payload
    assert payload["baseline_method"] is None


def test_hplc_preprocess_api_toggles_off(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2, n_points=50, offset_range=0)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
            data={"hplc_interpolate": "false", "hplc_subtract_min": "false", "hplc_normalize_area": "false"},
        )
    finally:
        for h in opened:
            h.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["hplc_interpolate"] is False
    assert payload["hplc_subtract_min"] is False
    assert payload["hplc_normalize_area"] is False


def test_hplc_preprocess_rejects_invalid_kind():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    response = client.post("/api/preprocess/invalid_kind")
    assert response.status_code in {400, 422}


def test_hplc_preprocess_empty_files():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    response = client.post("/api/preprocess/hplc", files=[])
    assert response.status_code in {400, 422}


def test_hplc_csv_downloadable(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=1, n_points=50)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        resp = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
        )
    finally:
        for h in opened:
            h.close()
    assert resp.status_code == 200
    download_url = resp.json()["download_url"]
    dl_resp = client.get(download_url)
    assert dl_resp.status_code == 200
    assert "Index,Name,XXX,Intensity,Label,Repeat_index" in dl_resp.text
