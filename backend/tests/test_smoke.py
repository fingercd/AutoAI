from pathlib import Path

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
