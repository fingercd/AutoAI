from pathlib import Path
import ast

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


def test_raman_baseline_order_changes_processing_scope(tmp_path, monkeypatch):
    import backend.app.parsers as parsers

    source = tmp_path / "raman.csv"
    source.write_text(
        "RamanShift,Intensity\n"
        "1,10\n"
        "2,20\n"
        "3,30\n"
        "4,40\n",
        encoding="utf-8",
    )

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

    assert ast.literal_eval(range_first.iloc[0]["Intensity"]) == [220.0, 230.0]
    assert ast.literal_eval(baseline_first.iloc[0]["Intensity"]) == [420.0, 430.0]
