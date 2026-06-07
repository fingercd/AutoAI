from pathlib import Path
import ast
import json

import pytest

from backend.app.parsers import load_modeling_csv, summarize_modeling_csv
from backend.app.training import train_model


ROOT = Path(__file__).resolve().parents[2]


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
    assert set(payload["curves"][0]) == {"name", "x", "raw_y"}
    assert len(payload["curves"][0]["x"]) == 9
    assert len(payload["curves"][0]["raw_y"]) == 9
