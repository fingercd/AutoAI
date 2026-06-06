from pathlib import Path
import ast

import pytest

from backend.app.parsers import summarize_modeling_csv
from backend.app.training import train_model


ROOT = Path(__file__).resolve().parents[2]


def test_data_csv_summary():
    summary = summarize_modeling_csv(ROOT / "data.csv")
    assert summary["samples"] == 90
    assert summary["classes"] == 2
    assert summary["curve_length"] == 160


def test_train_smoke(tmp_path, monkeypatch):
    import backend.app.training as training

    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    result = train_model(ROOT / "data.csv", {"epochs": 1, "batch_size": 16})
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()


@pytest.mark.parametrize("model_type", ["cnn1d", "mlp", "resnet1d", "transformer", "cnn_transformer"])
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
    assert set(payload["curves"][0]) == {"name", "x", "raw_y", "corrected_y"}
    assert len(payload["curves"][0]["x"]) == 9
    assert len(payload["curves"][0]["raw_y"]) == 9
    assert len(payload["curves"][0]["corrected_y"]) == 9
