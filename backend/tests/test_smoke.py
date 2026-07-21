from pathlib import Path
import ast
import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.parsers import load_modeling_csv, summarize_modeling_csv
from backend.app.training import train_model


ROOT = Path(__file__).resolve().parents[2]
USER_RAMAN_CSV = Path.home() / "Desktop" / "课件" / "raman_b2084be144.csv"


def test_data_csv_summary(tmp_path):
    source = tmp_path / "sample_data.csv"
    _write_grouped_modeling_csv(source, group_count=30, repeats=3, curve_length=160)

    summary = summarize_modeling_csv(source)
    assert summary["samples"] == 90
    assert summary["classes"] == 2
    assert summary["curve_length"] == 160


def test_modeling_csv_accepts_gbk_and_index_alias(tmp_path):
    source = tmp_path / "gbk_modeling.csv"
    frame = (
        "SpecAutoAI 谱学建模平台,Name,XXX,Intensity,Label,Repeat_index\n"
        '1,s1,"[1, 2, 3]","[4, 5, 6]",A,1\n'
        '2,s2,"[1, 2, 3]","[6, 5, 4]",B,2\n'
    )
    source.write_bytes(frame.encode("gbk"))

    summary = summarize_modeling_csv(source)

    assert summary["samples"] == 2
    assert summary["classes"] == 2
    assert summary["curve_length"] == 3
    assert summary["columns"][0] == "Index"
    assert "Sample_ID" in summary["columns"]
    assert "Repeat_index" not in summary["columns"]


def test_sample_id_summary_uses_natural_numeric_order(tmp_path):
    source = tmp_path / "natural_order.csv"
    rows = ["Index,Name,XXX,Intensity,Label,Sample_ID"]
    for index, sample_id in enumerate(("1", "10", "11", "2", "3"), start=1):
        rows.append(f'{index},s{sample_id},"[1,2]","[{index},{index + 1}]",A,{sample_id}')
    source.write_text("\n".join(rows), encoding="utf-8")

    summary = summarize_modeling_csv(source)

    assert [item["sample_id"] for item in summary["sample_id"]["groups"]] == ["1", "2", "3", "10", "11"]


def test_modeling_csv_rejects_conflicting_sample_id_and_legacy_column(tmp_path):
    source = tmp_path / "conflicting_sample_id.csv"
    source.write_text(
        "Index,Name,XXX,Intensity,Label,Sample_ID,Repeat_index\n"
        '1,s1,"[1,2]","[3,4]",A,1,2\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Sample_ID.*不一致"):
        load_modeling_csv(source)


def _write_grouped_modeling_csv(path: Path, group_count: int = 11, repeats: int = 2, curve_length: int = 4) -> None:
    rows = ["Index,Name,XXX,Intensity,Label,Sample_ID"]
    index = 1
    x_axis = list(range(curve_length))
    for group in range(1, group_count + 1):
        label = "A" if group <= (group_count + 1) // 2 else "B"
        for repeat in range(repeats):
            y = [float(group + point + repeat * 0.01) for point in range(curve_length)]
            if label == "B" and curve_length:
                y[curve_length // 2] += 5.0
            rows.append(f'{index},s{group}_{repeat},"{x_axis}","{y}",{label},{group}')
            index += 1
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_feature_signal_csv(path: Path, group_count: int = 12, repeats: int = 2) -> None:
    rows = ["Index,Name,XXX,Intensity,Label,Sample_ID"]
    x_axis = list(range(40))
    index = 1
    for group in range(1, group_count + 1):
        label = "A" if group <= group_count // 2 else "B"
        for repeat in range(repeats):
            y = np.zeros(40, dtype=float)
            y += group * 0.01 + repeat * 0.001
            if label == "B":
                y[25] += 3.0
            rows.append(
                f'{index},signal_{group}_{repeat},"{x_axis}",'
                f'"{y.round(6).tolist()}",{label},{group}'
            )
            index += 1
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_inconsistent_axis_csv(path: Path, group_count: int = 12, repeats: int = 2) -> None:
    rows = ["Index,Name,XXX,Intensity,Label,Sample_ID"]
    index = 1
    for group in range(1, group_count + 1):
        label = "A" if group <= group_count // 2 else "B"
        for repeat in range(repeats):
            x_axis = [0.0, 1.0, 2.0, 3.0, 10.0 + index]
            y = [0.0, 0.1 * group, 0.2 * group, 0.3 * group, 0.4 * group]
            if label == "B":
                y[2] += 4.0
            rows.append(
                f'{index},axis_{group}_{repeat},"{x_axis}",'
                f'"{y}",{label},{group}'
            )
            index += 1
    path.write_text("\n".join(rows), encoding="utf-8")


class _FakeAggMap:
    instances: list["_FakeAggMap"] = []

    def __init__(self, dfx, metric="correlation"):
        self.metric = metric
        self.alist = list(dfx.columns)
        self.fit_input_shape = dfx.shape
        self.fit_kwargs = {}
        self.isfit = False
        _FakeAggMap.instances.append(self)

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        self.isfit = True
        feature_count = len(self.alist)
        side = max(5, int(np.ceil(np.sqrt(max(1, feature_count)))))
        rows = []
        for idx, name in enumerate(self.alist):
            rows.append({"x": idx % side, "y": idx // side, "v": name})
        self.fmap_shape = (side, side)
        self.feature_names_reshape = [row["v"] for row in rows]
        self.df_grid = pd.DataFrame(rows)
        return self

    def batch_transform(self, array_2d, scale=True, scale_method="minmax", n_jobs=4, fillnan=0):
        values = np.asarray(array_2d, dtype=np.float32)
        side_h, side_w = self.fmap_shape
        output = np.zeros((values.shape[0], side_h, side_w, 1), dtype=np.float32)
        for feature_idx in range(min(values.shape[1], len(self.alist))):
            output[:, feature_idx // side_w, feature_idx % side_w, 0] = values[:, feature_idx]
        return output


def test_build_feature_windows_uses_nearest_equal_width_divisor():
    from backend.app.feature_selection import (
        build_feature_windows,
        resolve_equal_width_window_count,
    )
    from backend.app.training import TrainConfig

    windows = build_feature_windows(10, window_count=4)

    assert windows == [
        {"window_index": 0, "start_index": 0, "end_index": 1},
        {"window_index": 1, "start_index": 2, "end_index": 3},
        {"window_index": 2, "start_index": 4, "end_index": 5},
        {"window_index": 3, "start_index": 6, "end_index": 7},
        {"window_index": 4, "start_index": 8, "end_index": 9},
    ]
    assert resolve_equal_width_window_count(160, 100) == 80
    assert {window["end_index"] - window["start_index"] + 1 for window in build_feature_windows(160, 100)} == {2}
    assert resolve_equal_width_window_count(157, 100) == 157
    assert build_feature_windows(5, window_count=100) == [
        {"window_index": 0, "start_index": 0, "end_index": 0},
        {"window_index": 1, "start_index": 1, "end_index": 1},
        {"window_index": 2, "start_index": 2, "end_index": 2},
        {"window_index": 3, "start_index": 3, "end_index": 3},
        {"window_index": 4, "start_index": 4, "end_index": 4},
    ]
    assert build_feature_windows(0, window_count=50) == []
    assert TrainConfig().feature_window_count == 100


def test_sample_occlusion_importance_finds_true_label_signal_window():
    from backend.app.feature_selection import sample_occlusion_importance

    x = np.zeros((2, 40), dtype=np.float32)
    x[1, 20:30] = 2.0
    y = np.asarray([0, 1], dtype=np.int64)

    def score(values):
        logits = np.column_stack([np.zeros(values.shape[0]), values[:, 20:30].mean(axis=1)])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    result = sample_occlusion_importance(
        x,
        y,
        x_axis=np.arange(40, dtype=np.float32),
        splits={"train": [0], "valid": [0], "test": [1]},
        label_names=["A", "B"],
        score_fn=score,
        mean_indices=[0],
        metadata=[
            {"index": 1, "name": "sample_a", "sample_id": "1"},
            {"index": 2, "name": "sample_b", "sample_id": "2"},
        ],
        window_count=4,
        top_k=2,
    )

    assert result["status"] == "ready"
    assert [sample["dataset"] for sample in result["samples"]] == ["test"]
    sample_b = next(sample for sample in result["samples"] if sample["true_label"] == "B")
    assert sample_b["correct"] is True
    assert len(sample_b["windows"]) == 4
    assert sample_b["top_segments"]
    top_window = min(sample_b["windows"], key=lambda item: item["rank"])
    assert "original_loss" in top_window
    assert "masked_loss" in top_window
    assert "masked_true_probability" in top_window
    assert "true_probability_drop" in top_window
    assert "normalized_importance" in top_window
    assert top_window["importance"] == pytest.approx(top_window["masked_loss"] - top_window["original_loss"])
    normalized_values = [window["normalized_importance"] for window in sample_b["windows"]]
    assert min(normalized_values) == pytest.approx(0.0)
    assert max(normalized_values) == pytest.approx(1.0)
    assert top_window["normalized_importance"] == pytest.approx(1.0)
    assert sample_b["top_segments"][0]["normalized_importance"] == pytest.approx(1.0)
    assert sample_b["primary_segment"] == sample_b["top_segments"][0]
    assert any(segment["start_index"] <= 29 and segment["end_index"] >= 20 for segment in sample_b["top_segments"])


def test_sample_occlusion_importance_allows_no_positive_segments():
    from backend.app.feature_selection import sample_occlusion_importance

    x = np.ones((1, 12), dtype=np.float32)
    y = np.asarray([0], dtype=np.int64)

    result = sample_occlusion_importance(
        x,
        y,
        x_axis=np.arange(12, dtype=np.float32),
        splits={"test": [0]},
        label_names=["A"],
        score_fn=lambda values: np.ones((values.shape[0], 1), dtype=np.float32),
        mean_indices=[0],
        metadata=[{"index": 1, "name": "flat", "sample_id": "1"}],
        window_count=3,
        top_k=2,
    )

    assert result["status"] == "ready"
    assert result["method"] == "sample_occlusion_log_loss"
    assert result["importance_metric"] == "masked_true_class_log_loss_minus_original_true_class_log_loss"
    assert result["samples"][0]["top_segments"] == []
    assert result["samples"][0]["primary_segment"] is None
    assert all("original_loss" in window and "masked_loss" in window for window in result["samples"][0]["windows"])
    assert all(window["importance"] == 0 for window in result["samples"][0]["windows"])
    assert all(window["normalized_importance"] == 0 for window in result["samples"][0]["windows"])


def test_log_loss_importance_detects_probability_changes_without_label_flip():
    from backend.app.feature_selection import sample_occlusion_importance

    x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int64)

    def score(values):
        positive = np.clip(0.79 + 0.10 * values[:, 0] + 0.01 * values[:, 1], 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - positive, positive])

    result = sample_occlusion_importance(
        x,
        y,
        x_axis=np.arange(2, dtype=np.float32),
        splits={"train": [0], "test": [1]},
        label_names=["A", "B"],
        score_fn=score,
        mean_indices=[0],
        window_count=2,
    )

    windows = result["samples"][0]["windows"]
    assert result["samples"][0]["pred_label"] == "B"
    assert windows[0]["masked_true_probability"] == pytest.approx(0.80)
    assert windows[1]["masked_true_probability"] == pytest.approx(0.89)
    assert windows[0]["importance"] > windows[1]["importance"] > 0


def test_log_loss_occlusion_caps_perturbed_inference_batches():
    from backend.app.feature_selection import sample_occlusion_importance

    x = np.zeros((302, 2), dtype=np.float32)
    y = np.ones(302, dtype=np.int64)
    calls = []

    def score(values):
        calls.append(len(values))
        return np.tile(np.asarray([[0.1, 0.9]], dtype=np.float32), (len(values), 1))

    sample_occlusion_importance(
        x,
        y,
        x_axis=np.arange(2, dtype=np.float32),
        splits={"train": [0], "test": list(range(1, 302))},
        label_names=["A", "B"],
        score_fn=score,
        mean_indices=[0],
        window_count=2,
        max_perturbed_rows=256,
    )

    assert calls[0] == 301
    assert calls[1:]
    assert max(calls[1:]) <= 256


def test_sample_feature_csv_columns_follow_importance_method(tmp_path):
    import torch
    from torch import nn
    from backend.app.feature_selection import (
        sample_deep_attribution_importance,
        sample_occlusion_importance,
        write_sample_feature_importance_artifacts,
    )

    x = np.asarray([[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int64)
    metadata = [
        {"index": 1, "name": "train", "sample_id": "1"},
        {"index": 2, "name": "test", "sample_id": "2"},
    ]

    occlusion = sample_occlusion_importance(
        x,
        y,
        x_axis=np.arange(4, dtype=np.float32),
        splits={"train": [0], "test": [1]},
        label_names=["A", "B"],
        score_fn=lambda values: np.tile(np.asarray([[0.2, 0.8]], dtype=np.float32), (values.shape[0], 1)),
        mean_indices=[0],
        metadata=metadata,
        window_count=2,
    )
    occlusion_dir = tmp_path / "occlusion"
    occlusion_dir.mkdir()
    write_sample_feature_importance_artifacts(occlusion_dir, occlusion)
    occlusion_csv = pd.read_csv(tmp_path / "occlusion" / "sample_feature_importance.csv")

    class TinyGradientClassifier(nn.Module):
        def forward(self, inputs):
            signal = inputs.mean(dim=2)
            return torch.cat([-signal, signal], dim=1)

    deep = sample_deep_attribution_importance(
        TinyGradientClassifier(),
        x,
        y,
        x_axis=np.arange(4, dtype=np.float32),
        splits={"train": [0], "test": [1]},
        label_names=["A", "B"],
        metadata=metadata,
        model_type="cnn1d",
        top_k=2,
    )
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    write_sample_feature_importance_artifacts(deep_dir, deep)
    deep_csv = pd.read_csv(tmp_path / "deep" / "sample_feature_importance.csv")

    assert "importance_metric" in occlusion_csv.columns
    assert "original_loss" in occlusion_csv.columns
    assert "masked_loss" in occlusion_csv.columns
    assert "masked_true_probability" in occlusion_csv.columns
    assert "true_probability_drop" in occlusion_csv.columns
    assert "importance_metric" in deep_csv.columns
    assert "original_loss" not in deep_csv.columns
    assert "masked_loss" not in deep_csv.columns


def test_deep_gradcam_records_sample_axis_and_auxiliary_sanity():
    import torch
    from torch import nn
    from backend.app.feature_selection import sample_deep_attribution_importance

    class TinyConvClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 2, kernel_size=1, bias=False)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(torch.tensor([[[1.0]], [[-1.0]]]))
                self.classifier.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
            return self.classifier(x)

    x = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.0, -0.1, -0.2],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0], dtype=np.int64)
    metadata = [
        {"index": 1, "name": "train", "sample_id": "1", "sample_x_axis": [0, 1, 2, 3, 4]},
        {"index": 2, "name": "sample_a", "sample_id": "2", "sample_x_axis": [10, 11, 12, 13, 99]},
        {"index": 3, "name": "sample_b", "sample_id": "3", "sample_x_axis": [20, 21, 22, 23, 199]},
    ]

    result = sample_deep_attribution_importance(
        TinyConvClassifier(),
        x,
        y,
        x_axis=[0, 1, 2, 3, 4],
        splits={"train": [0], "test": [1, 2]},
        label_names=["A", "B"],
        metadata=metadata,
        model_type="cnn1d",
        top_k=2,
    )

    assert result["status"] == "ready"
    assert result["method"] == "gradcam_1d"
    assert result["sanity_checks"]["auxiliary_method"] == "input_gradient_attribution"
    assert result["sanity_checks"]["sample_count"] == 2
    assert "disagreement_sample_count" in result["sanity_checks"]
    sample = next(item for item in result["samples"] if item["name"] == "sample_a")
    assert sample["sample_x_axis"] == [10, 11, 12, 13, 99]
    assert sample["windows"][0]["start_x"] == pytest.approx(10)
    assert sample["windows"][-1]["end_x"] == pytest.approx(99)
    assert sample["sanity_checks"]["auxiliary_method"] == "input_gradient_attribution"
    assert sample["auxiliary_top_segments"]
    assert sample["primary_segment"] == sample["top_segments"][0]


def test_dscarnet_registry_uses_dual_2d_builder():
    import torch
    from backend.app.models.dscarnet import DSCARNet1D, DualDSCARNet2D
    from backend.app.models.registry import build_deep_model, build_dscarnet_model
    from backend.app.training import TrainConfig

    config = TrainConfig(model_type="dscarnet", dscarnet_inception_blocks=1)

    with pytest.raises(ValueError, match="2D"):
        build_deep_model(config, input_length=40, class_count=2, sample_count=8)

    model = build_dscarnet_model(
        config,
        input_shape1=(5, 5, 1),
        input_shape2=(5, 5, 1),
        class_count=2,
    )

    assert isinstance(model, DualDSCARNet2D)
    assert not isinstance(model, DSCARNet1D)
    assert model.last_avf is None
    logits = model(torch.ones(2, 1, 5, 5), torch.ones(2, 1, 5, 5))
    assert logits.shape == (2, 1)


def test_registry_allows_only_current_active_classification_models():
    from backend.app.models.registry import DEEP_MODEL_TYPES, TRADITIONAL_MODEL_TYPES, canonical_model_type

    assert TRADITIONAL_MODEL_TYPES == {
        "pls_da",
        "pca_lda",
        "logistic_regression",
        "svm",
        "random_forest",
        "xgboost",
    }
    assert DEEP_MODEL_TYPES == {
        "pca_mlp",
        "cnn1d",
        "cnn1d_se",
        "resnet1d",
        "inception1d",
        "tcn1d",
        "cnn_transformer1d",
        "dscarnet",
    }
    assert canonical_model_type("PLS-DA") == "pls_da"
    assert canonical_model_type("1D-Transformer") == "cnn_transformer1d"
    assert canonical_model_type("1D-ResNet") == "resnet1d"
    assert canonical_model_type("1D-Inception") == "inception1d"
    assert canonical_model_type("1D-TCN") == "tcn1d"
    for retired in ["knn", "mlp", "unet1d", "plsr", "svr"]:
        with pytest.raises(ValueError, match="当前仅支持分类任务的 10 类模型"):
            canonical_model_type(retired)


@pytest.mark.parametrize("input_length", [1500, 5000, 9000])
@pytest.mark.parametrize("model_type", ["cnn1d", "transformer1d", "resnet1d", "inception1d", "tcn1d"])
def test_deep_model_architectures_accept_dimension_bands(input_length, model_type):
    import torch
    from backend.app.models.registry import build_deep_model
    from backend.app.training import TrainConfig

    model = build_deep_model(
        TrainConfig(model_type=model_type, hidden_size=32, transformer_heads=4),
        input_length=input_length,
        class_count=3,
        sample_count=12,
    )

    logits = model(torch.randn(2, 1, input_length))

    assert logits.shape == (2, 3)


def test_dscarnet_aggmap_mapping_fits_train_only_and_saves(tmp_path):
    from backend.app.dscarnet_mapping import fit_dscarnet_2d_mapping, save_dscarnet_mapping_artifacts

    _FakeAggMap.instances = []
    x = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)

    mapped = fit_dscarnet_2d_mapping(
        x,
        train_indices=[0, 1, 2, 3],
        pca_components=3,
        cluster_channels=9,
        seed=7,
        aggmap_factory=_FakeAggMap,
    )

    assert len(_FakeAggMap.instances) == 2
    assert _FakeAggMap.instances[0].fit_input_shape == (4, 8)
    assert _FakeAggMap.instances[1].fit_input_shape == (4, 3)
    assert _FakeAggMap.instances[0].fit_kwargs["cluster_channels"] == 9
    assert mapped.x_sar.shape[0] == 6
    assert mapped.x_car.shape[0] == 6
    assert mapped.x_sar.shape[1:] == (1, 5, 5)
    assert mapped.x_car.shape[1:] == (1, 5, 5)
    assert mapped.metadata["fit_scope"] == "train"
    assert mapped.metadata["pca_components"] == 3
    assert "github.com/songlinlu/DSCAR" in mapped.metadata["source_url"]

    save_dscarnet_mapping_artifacts(tmp_path, mapped)

    payload = json.loads((tmp_path / "dscarnet_mapping.json").read_text(encoding="utf-8"))
    assert payload["fit_scope"] == "train"
    assert payload["input_shape_sar"] == [1, 5, 5]
    assert payload["input_shape_car"] == [1, 5, 5]
    assert (tmp_path / "dscarnet_pca.joblib").exists()
    assert (tmp_path / "dscarnet_sar_aggmap.joblib").exists()
    assert (tmp_path / "dscarnet_car_aggmap.joblib").exists()


def test_dscarnet_lapjv_compat_uses_linear_assignment():
    from backend.app.dscarnet_mapping import scipy_lapjv_compat

    row_assign, col_assign, total_cost = scipy_lapjv_compat(
        np.asarray(
            [
                [4.0, 1.0, 3.0],
                [2.0, 0.0, 5.0],
                [3.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    assert row_assign.tolist() == [1, 0, 2]
    assert col_assign.tolist() == [1, 0, 2]
    assert total_cost == pytest.approx(5.0)


def test_dscarnet_training_uses_dual_2d_mapping_and_gradcam_artifacts(tmp_path, monkeypatch):
    import backend.app.dscarnet_mapping as dscarnet_mapping
    import backend.app.training as training

    source = tmp_path / "feature_signal.csv"
    _write_feature_signal_csv(source, group_count=12, repeats=2)
    _FakeAggMap.instances = []
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(dscarnet_mapping, "_load_aggmap_class", lambda: _FakeAggMap)

    result = train_model(
        source,
        {
            "model_type": "dscarnet",
            "normalization": "none",
            "epochs": 1,
            "batch_size": 8,
            "split_train": 6,
            "split_valid": 2,
            "split_test": 2,
            "feature_top_k": 2,
            "dscarnet_inception_blocks": 1,
        },
    )

    run_dir = Path(result["run_dir"])
    sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))
    mapping_payload = json.loads((run_dir / "dscarnet_mapping.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["sample_feature_importance"]["method"] == "dscarnet_dual_2d_gradcam"
    assert result["sample_feature_importance"]["importance_metric"] == "sar_gradcam_plus_car_pca_backprojection"
    assert "feature_importance" not in result
    assert not (run_dir / "feature_importance.json").exists()
    assert not (run_dir / "feature_importance.csv").exists()
    assert sample_payload["method"] == "dscarnet_dual_2d_gradcam"
    assert sample_payload["importance_metric"] == "sar_gradcam_plus_car_pca_backprojection"
    assert sample_payload["window_count"] == 40
    assert sample_payload["samples"]
    assert all(sample["primary_segment"] for sample in sample_payload["samples"])
    assert sample_payload["dscarnet_mapping"]["source_url"] == mapping_payload["source_url"]
    assert all(len(sample["windows"]) == 40 for sample in sample_payload["samples"])
    assert all("sar_top_segments" in sample and "car_top_segments" in sample for sample in sample_payload["samples"])
    assert all(
        0.0 <= window["normalized_importance"] <= 1.0
        for sample in sample_payload["samples"]
        for window in sample["windows"]
    )
    assert (run_dir / "dscarnet_pca.joblib").exists()
    assert (run_dir / "dscarnet_sar_aggmap.joblib").exists()
    assert (run_dir / "dscarnet_car_aggmap.joblib").exists()


def test_sample_id_summary_and_incomplete_group_error(tmp_path):
    source = tmp_path / "bad_repeat.csv"
    _write_grouped_modeling_csv(source, group_count=4, repeats=2)
    text = source.read_text(encoding="utf-8")
    source.write_text("\n".join(text.splitlines()[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match="Sample_ID 重复测量次数不一致"):
        load_modeling_csv(source)


def test_outer_leave_one_cv_uses_each_sample_id_once(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=6, repeats=2, curve_length=12)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        source,
        {
            "model_type": "pls_da",
            "split_mode": "leave_one_sample_id_cv",
            "normalization": "zscore",
            "feature_selection_enabled": False,
        },
    )
    cv_payload = json.loads((Path(result["run_dir"]) / "cv_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(Path(result["run_dir"]) / "cv_predictions.csv")
    frame = load_modeling_csv(source).frame

    assert result["evaluation_strategy"] == "leave_one_sample_id_cv"
    assert cv_payload["fold_count"] == frame["Sample_ID"].nunique()
    assert result["total_target_epochs"] == result["fold_count"] * result["target_epochs"]
    assert result["current_fold"] == result["fold_count"]
    assert result["completed_folds"] == result["fold_count"]
    assert result["fold_progress_text"] == f'{result["fold_count"]}/{result["fold_count"]}'
    assert result["current_fold_sample_id"]
    assert result["metrics"]["test"]["accuracy"] == pytest.approx(
        result["cv_summary"]["pooled_test"]["accuracy"]
    )
    assert result["metrics"]["test"]["macro_f1"] == pytest.approx(
        result["cv_summary"]["pooled_test"]["macro_f1"]
    )
    assert result["metrics"]["test"]["aggregation"] == "pooled_out_of_fold"
    assert result["cv_summary"]["primary_test_aggregation"] == "pooled_out_of_fold"
    assert result["cv_summary"]["fold_mean"]["test"]["accuracy"] == pytest.approx(
        float(pd.read_csv(Path(result["run_dir"]) / "fold_metrics.csv")["accuracy"].mean())
    )
    assert "accuracy" in result["metrics"]
    assert sorted((item["test_sample_id"] for item in cv_payload["folds"]), key=int) == sorted(
        frame["Sample_ID"].astype(str).unique(), key=int
    )
    assert predictions["Sample_ID"].astype(str).nunique() == frame["Sample_ID"].nunique()
    assert set(predictions["dataset"]) == {"test"}
    for fold in cv_payload["folds"]:
        assert fold["test_sample_id"] not in fold["train_sample_ids"]
        assert fold["test_sample_id"] not in fold["valid_sample_ids"]
        assert set(fold["train_sample_ids"]).isdisjoint(fold["valid_sample_ids"])
        assert set(fold["test_sample_ids"]).isdisjoint(fold["train_sample_ids"])
        assert set(fold["test_sample_ids"]).isdisjoint(fold["valid_sample_ids"])
        assert len(fold["splits"]["train"]) == 8
        assert len(fold["splits"]["valid"]) == 2
        assert len(fold["splits"]["test"]) == 2


def test_cv_primary_test_macro_metrics_use_pooled_out_of_fold_predictions():
    import backend.app.training as training

    perfect_binary_split = {"true": [0, 1], "pred": [0, 1]}
    fold_split_evals = [
        {
            "train": perfect_binary_split,
            "valid": perfect_binary_split,
            "test": {"true": [0, 0], "pred": [0, 0]},
        },
        {
            "train": perfect_binary_split,
            "valid": perfect_binary_split,
            "test": {"true": [1, 1], "pred": [1, 1]},
        },
    ]

    metrics, cv_summary = training._build_metrics_payload(
        fold_split_evals,
        ["A", "B"],
        "leave_one_sample_id_cv",
    )

    assert cv_summary["fold_mean"]["test"]["macro_f1"] == pytest.approx(0.5)
    assert metrics["test"]["macro_precision"] == pytest.approx(1.0)
    assert metrics["test"]["macro_recall"] == pytest.approx(1.0)
    assert metrics["test"]["macro_f1"] == pytest.approx(1.0)
    assert metrics["test"]["aggregation"] == "pooled_out_of_fold"


def test_stratified_holdout_uses_single_8_1_1_split(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=10, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        source,
        {
            "model_type": "pls_da",
            "split_mode": "stratified_holdout",
            "split_train": 8,
            "split_valid": 1,
            "split_test": 1,
            "feature_selection_enabled": False,
        },
    )
    split_payload = json.loads((Path(result["run_dir"]) / "split.json").read_text(encoding="utf-8"))

    assert result["evaluation_strategy"] == "stratified_holdout"
    assert result["fold_count"] == 1
    assert result["total_target_epochs"] == result["target_epochs"]
    assert set(result["metrics"]).issuperset({"accuracy", "train", "valid", "test"})
    assert result["metrics"]["test"]["accuracy"] == result["metrics"]["accuracy"]
    assert all("accuracy" in result["metrics"][split] for split in ("train", "valid", "test"))
    assert len(split_payload[0]["splits"]["train"]) == 16
    assert len(split_payload[0]["splits"]["valid"]) == 2
    assert len(split_payload[0]["splits"]["test"]) == 2


def test_external_test_dataset_uses_train_valid_holdout(tmp_path, monkeypatch):
    import backend.app.training as training

    train_source = tmp_path / "train.csv"
    test_source = tmp_path / "test.csv"
    _write_grouped_modeling_csv(train_source, group_count=10, repeats=2)
    _write_grouped_modeling_csv(test_source, group_count=4, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        train_source,
        {
            "model_type": "pls_da",
            "test_data_path": str(test_source),
            "split_train": 8,
            "split_valid": 2,
            "split_test": 0,
            "feature_selection_enabled": False,
        },
    )
    split_payload = json.loads((Path(result["run_dir"]) / "split.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(Path(result["run_dir"]) / "predictions.csv")

    assert result["evaluation_strategy"] == "external_test_holdout"
    assert result["test_sample_count"] == 8
    assert result["fold_count"] == 1
    assert not split_payload[0]["splits"]["test"]
    assert len(split_payload[0]["splits"]["train"]) == 16
    assert len(split_payload[0]["splits"]["valid"]) == 4
    assert len(split_payload[0]["external_test_indices"]) == 8
    train_groups = set(split_payload[0]["train_sample_ids"])
    valid_groups = set(split_payload[0]["valid_sample_ids"])
    assert train_groups.isdisjoint(valid_groups)
    assert set(split_payload[0]["test_sample_ids"]) == {"1", "2", "3", "4"}
    assert set(predictions["dataset"]) == {"external_test"}


@pytest.mark.parametrize(
    "strategy, with_external_test",
    [
        ("stratified_holdout", False),
        ("leave_one_sample_id_cv", False),
        ("external_test_holdout", True),
    ],
)
def test_training_rejects_split_ratios_that_do_not_sum_to_ten_for_every_strategy(
    tmp_path,
    monkeypatch,
    strategy,
    with_external_test,
):
    import backend.app.training as training

    train_source = tmp_path / "train.csv"
    _write_grouped_modeling_csv(train_source, group_count=10, repeats=2)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")
    config = {
        "model_type": "pls_da",
        "split_mode": strategy,
        "split_train": 8,
        "split_valid": 6,
        "split_test": 1,
        "feature_selection_enabled": False,
    }
    if with_external_test:
        test_source = tmp_path / "test.csv"
        _write_grouped_modeling_csv(test_source, group_count=4, repeats=2)
        config["test_data_path"] = str(test_source)

    with pytest.raises(ValueError, match="训练、验证、测试.*10|相加必须等于 10"):
        train_model(train_source, config)


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
        train_model(train_source, {"model_type": "pls_da", "test_data_path": str(test_source)})


def test_train_smoke(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "sample_data.csv"
    _write_grouped_modeling_csv(source, group_count=30, repeats=3, curve_length=160)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    result = train_model(source, {"epochs": 1, "batch_size": 16})
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "cv_metrics.json").exists()
    assert (run_dir / "fold_metrics.csv").exists()
    assert (run_dir / "cv_predictions.csv").exists()
    assert not (run_dir / "feature_importance.json").exists()
    assert not (run_dir / "feature_importance.csv").exists()
    assert (run_dir / "sample_feature_importance.json").exists()
    assert (run_dir / "sample_feature_importance.csv").exists()
    assert "feature_importance" not in result
    assert result["sample_feature_importance"]["artifact"] == "sample_feature_importance.json"
    assert result["evaluation_strategy"] == "stratified_holdout"
    assert "classification_report" in result["metrics"]
    assert result["metrics"]["test"]["accuracy"] == result["metrics"]["accuracy"]
    assert result["total_target_epochs"] == result["target_epochs"]


def test_training_writes_only_sample_importance_artifacts_and_downloads(tmp_path, monkeypatch):
    import backend.app.main as main
    import backend.app.routers.deps as router_deps
    import backend.app.training as training
    from fastapi.testclient import TestClient

    source = tmp_path / "feature_signal.csv"
    _write_feature_signal_csv(source)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(router_deps, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        source,
            {
                "model_type": "svm",
                "normalization": "none",
                "split_mode": "leave_one_sample_id_cv",
                "split_train": 8,
                "split_valid": 2,
                "split_test": 0,
            "feature_window_count": 4,
            "feature_top_k": 2,
            "feature_n_repeats": 2,
        },
    )
    run_dir = Path(result["run_dir"])
    sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))

    assert "feature_importance" not in result
    assert result["sample_feature_importance"]["status"] == "ready"
    assert not (run_dir / "feature_importance.json").exists()
    assert not (run_dir / "feature_importance.csv").exists()
    assert any(
        segment["start_index"] <= 29 and segment["end_index"] >= 20
        for sample in sample_payload["samples"]
        for segment in sample["top_segments"]
    )
    assert sample_payload["window_count"] == 4
    assert sample_payload["method"] == "sample_occlusion_log_loss"
    assert sample_payload["importance_metric"] == "masked_true_class_log_loss_minus_original_true_class_log_loss"
    assert (run_dir / "sample_feature_importance.json").exists()
    assert (run_dir / "sample_feature_importance.csv").exists()

    client = TestClient(main.app)
    assert client.get(f"/api/training/runs/{result['run_id']}/artifact/feature_importance.json").status_code == 404
    assert client.get(f"/api/training/runs/{result['run_id']}/artifact/feature_importance.csv").status_code == 404
    assert client.get(f"/api/training/runs/{result['run_id']}/artifact/sample_feature_importance.json").status_code == 200


def test_training_warns_when_sample_x_axes_are_inconsistent(tmp_path, monkeypatch):
    import backend.app.training as training

    source = tmp_path / "inconsistent_axis.csv"
    _write_inconsistent_axis_csv(source)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")

    result = train_model(
        source,
        {
            "model_type": "cnn1d",
            "normalization": "none",
            "epochs": 1,
            "batch_size": 8,
            "split_train": 6,
            "split_valid": 2,
            "split_test": 2,
            "feature_window_count": 3,
            "feature_top_k": 2,
            "feature_n_repeats": 1,
        },
    )
    run_dir = Path(result["run_dir"])
    sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))

    assert result["x_axis_warning"]["status"] == "inconsistent"
    assert not (run_dir / "feature_importance.json").exists()
    assert sample_payload["x_axis_warning"]["status"] == "inconsistent"
    assert sample_payload["samples"]
    assert all("sample_x_axis" in sample for sample in sample_payload["samples"])
    assert any(
        sample["sample_x_axis"][-1] != sample_payload["x_axis"][-1]
        for sample in sample_payload["samples"]
    )


@pytest.mark.parametrize("model_type", ["pls_da", "svm", "random_forest", "xgboost", "cnn1d", "transformer1d", "resnet1d", "inception1d", "tcn1d", "dscarnet"])
def test_all_model_types_train_one_epoch(tmp_path, monkeypatch, model_type):
    import backend.app.dscarnet_mapping as dscarnet_mapping
    import backend.app.training as training

    source = tmp_path / f"{model_type}_grouped.csv"
    _write_grouped_modeling_csv(source, group_count=6, repeats=2, curve_length=40)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    if model_type == "dscarnet":
        _FakeAggMap.instances = []
        monkeypatch.setattr(dscarnet_mapping, "_load_aggmap_class", lambda: _FakeAggMap)
    result = train_model(
        source,
        {
            "epochs": 1,
            "batch_size": 8,
            "model_type": model_type,
            "split_mode": "leave_one_sample_id_cv",
            "early_stopping_patience": 5,
            "hidden_size": 32,
            "transformer_heads": 4,
            "feature_window_count": 4,
            "feature_top_k": 2,
            "feature_n_repeats": 1,
        },
    )

    assert result["status"] == "success"
    expected_model_type = "cnn_transformer1d" if model_type == "transformer1d" else model_type
    assert result["model_type"] == expected_model_type
    assert result["evaluation_strategy"] == "leave_one_sample_id_cv"
    assert result["fold_count"] == 6
    run_dir = tmp_path / result["run_id"]
    if result.get("model_family") == "traditional_ml":
        assert (run_dir / "model.pkl").exists()
        assert result["sample_feature_importance"]["status"] == "ready"
        assert (run_dir / "sample_feature_importance.json").exists()
        assert (run_dir / "sample_feature_importance.csv").exists()
        sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))
        assert "feature_importance" not in result
        assert not (run_dir / "feature_importance.json").exists()
        assert not (run_dir / "feature_importance.csv").exists()
        assert sample_payload["importance_metric"] == "masked_true_class_log_loss_minus_original_true_class_log_loss"
        assert sample_payload["method"] == "sample_occlusion_log_loss"
        assert sample_payload["window_count"] == 4
    else:
        assert (run_dir / "model.pt").exists()
        assert result["sample_feature_importance"]["status"] == "ready"
        assert result["sample_feature_importance"]["artifact"] == "sample_feature_importance.json"
        sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))
        assert sample_payload["status"] == "ready"
        assert sample_payload["method"] in {"gradcam_1d", "sample_occlusion_log_loss", "dscarnet_dual_2d_gradcam"}
        expected_window_count = 4 if sample_payload["method"] == "sample_occlusion_log_loss" else 40
        assert sample_payload["window_count"] == expected_window_count
        assert sample_payload["x_axis_warning"]["status"] in {"consistent", "inconsistent"}
        assert sample_payload["samples"]
        assert all(sample["primary_segment"] is None or sample["primary_segment"]["importance"] > 0 for sample in sample_payload["samples"])
        assert all(
            len(sample["windows"]) == sample_payload["window_count"]
            for sample in sample_payload["samples"]
        )
        assert all("sample_x_axis" in sample for sample in sample_payload["samples"])
        if sample_payload["method"] == "gradcam_1d":
            assert sample_payload["sanity_checks"]["auxiliary_method"] == "input_gradient_attribution"
            assert all(
                "sanity_checks" in sample and "auxiliary_top_segments" in sample
                for sample in sample_payload["samples"]
            )
        if sample_payload["method"] == "dscarnet_dual_2d_gradcam":
            assert sample_payload["importance_metric"] == "sar_gradcam_plus_car_pca_backprojection"
            assert "dscarnet_mapping" in sample_payload
            assert (run_dir / "dscarnet_mapping.json").exists()
        assert all(
            0.0 <= window["normalized_importance"] <= 1.0
            for sample in sample_payload["samples"]
            for window in sample["windows"]
        )


def test_cv_deep_sample_feature_importance_accumulates_all_fold_test_samples(tmp_path, monkeypatch):
    import torch
    from torch import nn
    import backend.app.training as training

    class FastGradientModel(nn.Module):
        def forward(self, inputs):
            signal = inputs.mean(dim=2)
            return torch.cat([-signal, signal], dim=1)

    def fake_fit_deep_fold(**kwargs):
        return (
            FastGradientModel(),
            [
                {
                    "epoch": 1,
                    "train_loss": 0.0,
                    "valid_accuracy": 1.0,
                    "valid_macro_f1": 1.0,
                    "best_valid_macro_f1": 1.0,
                    "bad_epochs": 0,
                }
            ],
            None,
            None,
        )

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=30, repeats=3, curve_length=160)
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(training, "_fit_deep_fold", fake_fit_deep_fold)

    result = train_model(
        source,
        {
            "model_type": "transformer1d",
            "split_mode": "leave_one_sample_id_cv",
            "epochs": 1,
            "batch_size": 16,
            "feature_top_k": 3,
        },
    )

    run_dir = Path(result["run_dir"])
    sample_payload = json.loads((run_dir / "sample_feature_importance.json").read_text(encoding="utf-8"))
    sample_csv = pd.read_csv(run_dir / "sample_feature_importance.csv")

    assert result["test_sample_count"] == 90
    assert result["sample_feature_importance"]["sample_count"] == result["test_sample_count"]
    assert sample_payload["sample_count"] == result["test_sample_count"]
    assert len(sample_payload["samples"]) == result["test_sample_count"]
    assert all("fold_index" in sample for sample in sample_payload["samples"])
    assert sample_payload["requested_window_count"] == 100
    assert sample_payload["window_count"] == 80
    assert sample_payload["window_width"] == 2
    assert sample_payload["window_policy"] == "nearest_divisor_equal_width"
    assert len(sample_csv) == 90 * sample_payload["window_count"]
    assert "fold_index" in sample_csv.columns
    assert sample_csv["fold_index"].notna().all()
    assert "original_loss" in sample_csv.columns
    assert "masked_loss" in sample_csv.columns


def test_create_run_persists_independent_queued_runs(tmp_path):
    import backend.app.main as main
    from fastapi.testclient import TestClient

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=6, repeats=2)

    client = TestClient(main.app)
    uploaded = client.post(
        "/api/datasets/upload",
        files={"file": (source.name, source.read_bytes(), "text/csv")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["dataset_name"] == source.name
    dataset_id = uploaded.json()["dataset_id"]

    first_response = client.post(
        "/api/training/runs",
        json={"dataset_id": dataset_id, "config": {"model_type": "pls_da"}},
    )
    second_response = client.post(
        "/api/training/runs",
        json={"dataset_id": dataset_id, "config": {"model_type": "svm"}},
    )

    assert first_response.status_code == 202
    assert second_response.status_code == 202
    assert first_response.json()["status"] == "pending"
    assert first_response.json()["state"] == "queued"
    assert second_response.json()["status"] == "pending"
    assert second_response.json()["state"] == "queued"
    first_run = client.get(f'/api/training/runs/{first_response.json()["run_id"]}')
    assert first_run.status_code == 200
    assert first_run.json()["dataset_name"] == source.name
    assert first_run.json()["config"]["dataset_name"] == source.name


def test_train_model_stops_when_status_is_replaced_mid_loop(tmp_path, monkeypatch):
    import backend.app.training as training

    class FastTraditionalModel:
        def predict(self, values):
            return (np.asarray(values)[:, 0] > 0).astype(np.int64)

        def predict_proba(self, values):
            pred = self.predict(values)
            probs = np.full((len(pred), 2), 0.1, dtype=float)
            probs[np.arange(len(pred)), pred] = 0.9
            return probs

    def fake_fit_traditional_fold(*args, **kwargs):
        status_file = runs_dir / run_id / "status.json"
        status_file.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "paused",
                    "replaced_by": "newer_run",
                    "pause_reason": "replaced_by_new_run",
                    "paused_at": "2026-07-07T12:00:00",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return FastTraditionalModel(), training.TrainConfig(model_type="pls_da"), {"macro_f1": 1.0}, []

    source = tmp_path / "grouped.csv"
    _write_grouped_modeling_csv(source, group_count=6, repeats=2)
    runs_dir = tmp_path / "runs"
    run_id = "replace_me"
    monkeypatch.setattr(training, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(training, "_fit_traditional_fold", fake_fit_traditional_fold)

    with pytest.raises(RuntimeError, match="替换|replaced"):
        train_model(
            source,
            {
                "model_type": "pls_da",
                "split_mode": "leave_one_sample_id_cv",
                "feature_selection_enabled": False,
            },
            run_id=run_id,
        )

    status = json.loads((runs_dir / run_id / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "paused"
    assert status["replaced_by"] == "newer_run"



@pytest.mark.skipif(not USER_RAMAN_CSV.exists(), reason="local user Raman CSV fixture is not available")
@pytest.mark.parametrize("model_type", ["PLS-DA", "1D-ResNet", "DSCARNet"])
def test_user_raman_csv_trains_new_model_choices(tmp_path, monkeypatch, model_type):
    import backend.app.dscarnet_mapping as dscarnet_mapping
    import backend.app.training as training

    monkeypatch.setattr(training, "RUNS_DIR", tmp_path)
    if str(model_type).lower() == "dscarnet":
        _FakeAggMap.instances = []
        monkeypatch.setattr(dscarnet_mapping, "_load_aggmap_class", lambda: _FakeAggMap)
    result = train_model(
        USER_RAMAN_CSV,
        {
            "epochs": 1,
            "batch_size": 16,
            "model_type": model_type,
            "early_stopping_patience": 5,
            "hidden_size": 32,
            "dscarnet_inception_blocks": 1,
            "feature_selection_enabled": False,
        },
    )

    assert result["status"] == "success"
    assert result["evaluation_strategy"] == "stratified_holdout"
    assert result["sample_count"] == 50


def test_raman_baseline_is_applied_after_range_selection(tmp_path, monkeypatch):
    import backend.app.parsers as parsers

    source = tmp_path / "raman.csv"
    lines = ["RamanShift,Intensity"]
    lines.extend(f"{idx},{idx * idx + 10}" for idx in range(1, 12))
    source.write_text("\n".join(lines), encoding="utf-8")

    received = {}

    def fake_baseline_correct(x, y, method):
        received["x"] = np.asarray(x).tolist()
        received["y"] = np.asarray(y).tolist()
        return y + len(y) * 100

    monkeypatch.setattr(parsers, "_baseline_correct", fake_baseline_correct)
    result = parsers.preprocess_raw_files_with_preview(
        [source],
        kind="raman",
        start_row=2,
        end_row=3,
    )

    assert received == {"x": [2.0, 3.0], "y": [14.0, 19.0]}
    assert ast.literal_eval(result["frame"].iloc[0]["Intensity"]) == [214.0, 219.0]
    assert result["curves"][0]["raw_y"] == [14.0, 19.0]
    assert result["curves"][0]["corrected_y"] == [214.0, 219.0]


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


def test_preprocess_csv_arrays_are_excel_safe_and_rounded_to_five_decimals(tmp_path, monkeypatch):
    from backend.app import parsers

    source = tmp_path / "raman.csv"
    x_values = np.round(180.91 + np.arange(2048) * 0.73, 2)
    y_values = (np.arange(2048, dtype=np.float32) * np.float32(0.1234567)) - 100
    source.write_text(
        "RamanShift,Intensity\n"
        + "\n".join(f"{x:.2f},{float(y):.9g}" for x, y in zip(x_values, y_values)),
        encoding="utf-8",
    )
    monkeypatch.setattr(parsers, "_baseline_correct", lambda x, y, method: y)

    result = parsers.preprocess_raw_files_with_preview(
        [source],
        kind="raman",
        start_row=100,
        end_row=2000,
    )
    row = result["frame"].iloc[0]
    serialized_x = json.loads(row["XXX"])
    serialized_y = json.loads(row["Intensity"])
    expected_x = x_values[99:2000]
    expected_y = y_values[99:2000]

    assert len(row["XXX"]) <= parsers.EXCEL_CELL_CHARACTER_LIMIT
    assert len(row["Intensity"]) <= parsers.EXCEL_CELL_CHARACTER_LIMIT
    assert "450.82000732421875" not in row["XXX"]
    np.testing.assert_allclose(serialized_x, expected_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(serialized_y, np.round(expected_y.astype(np.float64), 5), rtol=0, atol=1e-12)
    assert result["curves"][0]["x"] == serialized_x
    assert result["curves"][0]["corrected_y"] == serialized_y

    output = tmp_path / "result.csv"
    result["frame"].to_csv(output, index=False, encoding="utf-8-sig")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2
    assert all(len(item) == 6 for item in rows)


def test_preprocess_rejects_9000_point_arrays_over_excel_cell_limit(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.routers import preprocess as preprocess_router

    source = tmp_path / "chromatography_9000_points.csv"
    source.write_text(
        "Time,Intensity\n"
        + "\n".join(f"{idx},{idx}" for idx in range(9000)),
        encoding="utf-8",
    )
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "preprocessed"
    monkeypatch.setattr(preprocess_router, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(preprocess_router, "PREPROCESSED_DIR", outputs)

    client = TestClient(app)
    with source.open("rb") as handle:
        response = client.post(
            "/api/preprocess/chromatography",
            files=[("files", (source.name, handle, "text/csv"))],
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "超过 Excel 单元格上限 32767" in detail
    assert "请缩小行号范围或 X 轴数值范围" in detail
    assert not outputs.exists() or not list(outputs.iterdir())


def test_adaptive_serialization_selects_highest_fitting_precision():
    from backend.app.parsers import _serialize_modeling_arrays

    values = np.array([1.23456, 2.34567, 3.45678], dtype=np.float64)

    result = _serialize_modeling_arrays(
        [values],
        "XXX",
        ["sample"],
        character_limit=17,
    )

    assert result.decimal_places == 2
    assert result.max_characters == 16
    assert result.arrays[0].serialized == "[1.23,2.35,3.46]"
    assert result.arrays[0].values == json.loads(result.arrays[0].serialized)


def test_adaptive_serialization_uses_one_precision_for_the_whole_batch():
    from backend.app.parsers import _serialize_modeling_arrays

    short_values = np.array([1.2, 2.3, 3.4], dtype=np.float64)
    long_values = np.array([1.23456, 2.34567, 3.45678], dtype=np.float64)

    result = _serialize_modeling_arrays(
        [short_values, long_values],
        "Intensity",
        ["short", "long"],
        character_limit=17,
    )

    assert result.decimal_places == 2
    assert all(item.decimal_places == 2 for item in result.arrays)
    assert result.max_source_name == "long"


def test_adaptive_serialization_compacts_integral_values_and_negative_zero():
    from backend.app.parsers import _serialize_modeling_arrays

    values = np.array([-0.0, 1.49, 2.49], dtype=np.float64)

    result = _serialize_modeling_arrays(
        [values],
        "XXX",
        ["sample"],
        character_limit=7,
    )

    assert result.decimal_places == 0
    assert result.arrays[0].serialized == "[0,1,2]"
    assert ".0" not in result.arrays[0].serialized
    assert "-0" not in result.arrays[0].serialized


def test_adaptive_serialization_rejects_precision_that_erases_variation():
    from backend.app.parsers import _serialize_modeling_arrays

    values = np.array([0.01, 0.02], dtype=np.float64)

    with pytest.raises(ValueError, match="失去全部有效变化"):
        _serialize_modeling_arrays(
            [values],
            "Intensity",
            ["sample"],
            character_limit=9,
        )


def test_adaptive_serialization_rejects_when_zero_decimals_are_still_too_long():
    from backend.app.parsers import _serialize_modeling_arrays

    values = np.array([100, 200], dtype=np.float64)

    with pytest.raises(ValueError, match="即使保留 0 位小数.*超过 Excel 单元格上限 8"):
        _serialize_modeling_arrays(
            [values],
            "XXX",
            ["sample"],
            character_limit=8,
        )


def test_7500_point_chromatography_adapts_precision_and_round_trips(tmp_path):
    from backend.app.parsers import preprocess_raw_files_with_preview

    x = np.linspace(0, 75, 7500, dtype=np.float64)
    files = []
    for index in range(2):
        y = 10 + np.sin(x + index * 0.1)
        source = tmp_path / f"chrom_{index}.csv"
        pd.DataFrame({"Time": x, "Intensity": y}).to_csv(source, index=False)
        files.append(source)

    result = preprocess_raw_files_with_preview(files, kind="chromatography")
    precision = result["output_precision"]

    assert precision["xxx_decimal_places"] < 5
    assert precision["intensity_decimal_places"] < 5
    assert precision["xxx_max_characters"] <= 32767
    assert precision["intensity_max_characters"] <= 32767
    for row_index, curve in enumerate(result["curves"]):
        row = result["frame"].iloc[row_index]
        assert len(curve["x"]) == 7500
        assert len(curve["raw_y"]) == 7500
        assert json.loads(row["XXX"]) == curve["x"]
        assert json.loads(row["Intensity"]) == curve["raw_y"]

    frame = result["frame"].copy()
    frame["Label"] = ["A", "B"]
    frame["Sample_ID"] = ["1", "2"]
    output = tmp_path / "adaptive_modeling.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    loaded = load_modeling_csv(output)
    assert loaded.intensity.shape == (2, 7500)
    assert len(loaded.x_axis[0]) == 7500


def test_read_raw_spectrum_no_header_preserves_first_row(tmp_path):
    from backend.app.parsers import read_raw_spectrum

    source = tmp_path / "no_header.csv"
    source.write_text("\n".join(f"{idx},{idx * 10}" for idx in range(5)), encoding="utf-8")

    x, y = read_raw_spectrum(source, kind="hplc")

    np.testing.assert_allclose(x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(y, [0, 10, 20, 30, 40])


def test_read_raw_spectrum_header_csv_still_reads_values(tmp_path):
    from backend.app.parsers import read_raw_spectrum

    source = tmp_path / "with_header.csv"
    source.write_text(
        "Time,Intensity\n" + "\n".join(f"{idx},{idx * 10}" for idx in range(5)),
        encoding="utf-8",
    )

    x, y = read_raw_spectrum(source, kind="hplc")

    np.testing.assert_allclose(x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(y, [0, 10, 20, 30, 40])


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
                "baseline_method": "poly",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert "baseline_order" not in payload
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
def test_obsolete_ui_variant_routes_are_not_exposed(path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    response = client.get(path)

    assert response.status_code == 404


def test_obsolete_ui_variant_assets_are_not_served():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    assert client.get("/static/autoai-variants.css").status_code == 404
    assert client.get("/static/autoai-variants.js").status_code == 404


def test_main_ui_prefers_sample_feature_importance_panel():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")

    assert 'id="resultExplainability"' in content
    assert "sample_feature_importance" in results
    assert "function renderSampleImportance" in results
    assert "resultSampleExplanationSelect" in results
    assert "加载单样品解释" in results
    assert "sample.top_segments" in results
    assert "normalized_importance" in results
    assert "renderGlobalFeatureImportance" not in content
    assert "run.feature_importance" not in content
    assert 'id="featureWindowOptions"' in content
    assert "featureWindowCount" in content
    assert 'value="100"' in content
    assert 'max="5000"' in content
    assert "function usesFeatureWindowCount" in content
    assert '$("featureWindowOptions").classList.toggle("hidden", !usesFeatureWindowCount(modelType));' in content
    assert 'if (usesFeatureWindowCount(payload.model_type)) {' in content
    assert 'payload.feature_window_count = Number($("featureWindowCount").value);' in content
    assert 'feature_window_count: Number($("featureWindowCount").value)' not in content


def test_main_ui_enforces_cv_split_sum_and_prevents_duplicate_train_requests():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'if (!hasExternalTest && splitTrain + splitValid + splitTest !== 10)' in content
    assert 'if (!hasExternalTest && !cvEnabled && splitTrain + splitValid + splitTest !== 10)' not in content
    assert "训练、验证、测试比例相加必须等于 10" in content
    assert "trainingRequestInFlight" in content
    assert "trainingRequestInFlight = true" in content
    assert "trainingRequestInFlight = false" in content
    assert "startButton.disabled = true" in content
    assert "startButton.disabled = false" in content


def test_main_ui_external_dataset_forces_eight_two_and_hides_cv():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "function applySplitPreset(mode)" in content
    assert 'id="cvOptions"' in content
    assert 'id="splitTestField"' in content
    assert '$("cvEnabled").checked = false' in content
    assert '$("cvOptions").classList.toggle("hidden", hasExternalTest)' in content
    assert 'applySplitPreset("external")' in content


def test_main_ui_uses_documented_deep_training_defaults():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'option value="normal"' not in content
    assert 'option value="deep"' not in content
    assert 'id="epochs" type="number" value="200" min="1" max="200"' in content
    assert 'id="batchSize" type="number" value="8"' in content
    assert 'id="learningRate" type="number" value="0.001"' in content
    assert 'id="earlyStoppingPatience" type="number" value="20"' in content
    assert 'value = $("trainTime").value === "deep" ? "100" : "50"' not in content


def test_main_ui_handles_cancelled_runs_and_bounds_explainability_lists():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")

    assert '["cancelled", "paused"].includes(canonicalState)' in content
    assert "训练已取消" in content
    assert "run.pause_reason" in content
    assert "run.replaced_by" in content
    assert "sample?.fold_index" in results
    assert "第 ${sample.fold_index} 折" in results
    assert ".filter(Boolean).slice(0, 8)" in results
    assert "sample.top_segments" in results
    assert "不同模型使用其实际生成的解释方法" in results
    assert "primaryFeatureSegment" not in content
    assert "renderIntensitySummary" not in content
    assert 'id="intensitySummary"' not in content


def test_main_ui_exposes_custom_split_and_cv_epoch_summary():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")

    assert '["数据量", summary.samples]' in content
    assert '["样本数", sampleIds.group_count]' in content
    assert '["样品种类",' not in content
    assert 'const splitTrain = hasExternalTest ? 8 : readSplitNumber("splitTrain", 8);' in content
    assert 'const splitValid = hasExternalTest ? 2 : readSplitNumber("splitValid", 1);' in content
    assert 'const splitTest = hasExternalTest ? 0 : readSplitNumber("splitTest", 1);' in content
    assert 'const cvEnabled = $("cvEnabled").checked;' in content
    assert 'split_mode: hasExternalTest ? "external_test_holdout" : (cvEnabled ? "leave_one_sample_id_cv" : splitMode),' in content
    assert '$("customSplitOptions").classList.toggle("hidden"' not in content
    assert 'id="splitMode"' not in content
    assert 'id="cvEnabled"' in content
    assert "开启交叉验证" in content
    assert "几个 Sample_ID 就跑几折" in content
    assert content.index('id="advancedOptions"') < content.index('id="deepOptions"')
    assert 'id="trainTimeBlock"' not in content
    assert "function historyRows(result)" in results
    assert "function renderHistory(result)" in results
    assert "resultHistoryFoldSelect" in results
    assert "drawHistoryChart" in results
    assert "fold_progress_text" in content
    assert "completed_folds" in content
    assert "total_target_epochs" in content
    assert "result.metrics?.primary || result.metrics?.splits?.test" in results
    assert "坐标提示" not in content
    assert "归因提示" not in content
    assert "红色标记为" not in content
    assert "含义：" not in content


def test_training_records_use_summary_projection_and_result_page_owns_split_metrics():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")

    assert '["训练时间", "耗时", "Run ID", "上传的数据集名", "模型", "测试集 Macro F1", "状态", ""]' in content
    assert "run.test_macro_f1" in content
    assert 'new URLSearchParams({ projection: "summary", limit: "50" })' in content
    assert 'const datasetName = String(run.dataset_name || run.config?.dataset_name || "-");' in content
    assert "const splits = result.metrics?.splits || {};" in results
    assert "const rows = ['train', 'valid', 'test']" in results
    assert "'Macro F1'" in results
    assert "result.evaluation?.primary_aggregation" in results


def test_product_brand_is_specautoai_in_ui_and_openapi():
    from backend.app.main import app

    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "<title>SpecAutoAI 谱学建模平台</title>" in content
    assert "<h1>SpecAutoAI 谱学数据预处理及 AI 建模平台</h1>" in content
    assert app.title == "SpecAutoAI"


def test_main_ui_manual_explains_sample_id_group_split():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "AI 建模操作说明" in content
    assert "数据划分比例" in content
    assert "开启交叉验证" in content
    assert "外层留一交叉验证" not in content
    assert "独立测试集" in content
    assert "external_test_holdout" in content
    assert "加载独立测试集" in content
    assert "已停用" not in content
    assert "相同 Sample_ID 会整组进入同一数据分区" in content
    assert "相同 Sample_ID 会整组划分，避免重复测量泄漏到不同集合" in content
    assert "几个 Sample_ID 就跑几折" in content


def test_main_ui_preserves_hplc_range_and_interpolation_controls():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "色谱预处理操作说明" in content
    assert "服务端配置的固定时间轴" in content
    assert 'id="chromHplcInterp" checked' in content
    assert 'id="chromRangeMode"' in content
    assert 'id="chromStart"' in content
    assert 'id="chromEnd"' in content
    assert 'id="chromHplcSubMin"' not in content
    assert 'id="chromHplcNormArea"' not in content
    assert "不执行消负或面积归一化" in content


def test_chromatography_ui_uses_hplc_by_default():
    content = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="chromHplcMode"' not in content
    assert "启用 HPLC 高级预处理" not in content
    assert 'const effectiveKind = isChrom ? "hplc" : kind;' in content
    assert 'form.append("hplc_interpolate"' in content
    assert "result.hplc_axis" in content


# ---------------------------------------------------------------------------
# HPLC preprocessing tests
# ---------------------------------------------------------------------------


def _make_hplc_fixture(tmp_path, n_files=2, n_points=7500, add_negatives=True):
    """Generate a full-range HPLC fixture with configurable point count."""
    files = []
    for i in range(n_files):
        x = np.linspace(0, 50, n_points, dtype=np.float64)
        if n_points > 2 and i:
            # Keep endpoints fixed while making internal source coordinates slightly irregular.
            x[1:-1] += np.sin(np.linspace(0, np.pi, n_points - 2)) * (i * 1e-4)
        y = (
            100 * np.exp(-0.5 * ((x - 15) / 2.0) ** 2)
            + 80 * np.exp(-0.5 * ((x - 35) / 3.0) ** 2)
            + i
        )
        if add_negatives:
            y[:3] = -0.3  # simulate baseline drift negatives
        path = tmp_path / f"hplc_{i}.csv"
        pd.DataFrame({0: x, 1: y}).to_csv(path, index=False, header=False)
        files.append(path)
    return files


# ---- Unit tests for pure functions ----


def test_hplc_grid_is_driven_by_injected_config():
    from backend.app.hplc import HplcGridConfig, build_hplc_target_axis

    config = HplcGridConfig(1.0, 3.0, 5, 1e-10)
    np.testing.assert_allclose(build_hplc_target_axis(config), [1, 1.5, 2, 2.5, 3])
    assert config.step_minutes == 0.5


def test_hplc_linear_mapping_uses_bracketing_source_points():
    from backend.app.hplc import HplcGridConfig, map_hplc_intensity

    config = HplcGridConfig(0.0, 0.03, 4, 1e-10)
    source_x = np.array([0.0, 0.008, 0.021, 0.03])
    source_y = 100 + 3000 * source_x
    mapped = map_hplc_intensity(source_x, source_y, "manual.csv", config)
    np.testing.assert_allclose(mapped, [100, 130, 160, 190], rtol=0, atol=1e-10)


def test_hplc_fixed_axis_default_contract():
    from backend.app.hplc import DEFAULT_HPLC_GRID, build_hplc_target_axis

    axis = build_hplc_target_axis()
    assert len(axis) == DEFAULT_HPLC_GRID.point_count
    assert axis[0] == DEFAULT_HPLC_GRID.start_minutes
    assert axis[-1] == DEFAULT_HPLC_GRID.stop_minutes
    assert np.all(np.diff(axis) > 0)
    np.testing.assert_allclose(np.diff(axis), DEFAULT_HPLC_GRID.step_minutes, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("point_count", [7499, 7501])
def test_hplc_rejects_non_configured_point_count(tmp_path, point_count):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    source = _make_hplc_fixture(tmp_path, n_files=1, n_points=point_count)[0]
    with pytest.raises(ValueError, match=rf"解析到 {point_count} 个有效色谱点.*恰好 7500"):
        preprocess_hplc_files_with_preview([source])


def test_hplc_rejects_non_increasing_source_axis(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    source = _make_hplc_fixture(tmp_path, n_files=1)[0]
    frame = pd.read_csv(source, header=None)
    frame.iloc[101, 0] = frame.iloc[100, 0]
    frame.to_csv(source, index=False, header=False)
    with pytest.raises(ValueError, match="不严格递增"):
        preprocess_hplc_files_with_preview([source])


def test_hplc_rejects_source_that_cannot_cover_configured_range(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview

    source = tmp_path / "short_range.csv"
    x = np.linspace(0.1, 49.9, 7500)
    pd.DataFrame({0: x, 1: np.sin(x)}).to_csv(source, index=False, header=False)
    with pytest.raises(ValueError, match="相差超过一个采样间隔.*无法安全线性映射"):
        preprocess_hplc_files_with_preview([source])


def test_hplc_row_range_maps_to_same_fixed_axis_slice_with_small_edge_phase(tmp_path):
    from backend.app.hplc import DEFAULT_HPLC_GRID, build_hplc_target_axis, preprocess_hplc_files_with_preview

    source = tmp_path / "instrument_phase.csv"
    source_x = 0.0020833333333333 + np.arange(7500, dtype=np.float64) * (50 / 7500)
    source_y = 10 + 2 * source_x
    pd.DataFrame({0: source_x, 1: source_y}).to_csv(source, index=False, header=False)

    result = preprocess_hplc_files_with_preview([source], start_row=1, end_row=4000)

    expected_x = build_hplc_target_axis(DEFAULT_HPLC_GRID)[:4000]
    assert result["hplc_axis"]["point_count"] == 4000
    assert len(result["common_time"]) == 4000
    np.testing.assert_allclose(result["common_time"], expected_x, rtol=0, atol=0)
    digits = result["output_precision"]["intensity_decimal_places"]
    np.testing.assert_allclose(
        result["curves"][0]["processed_y"],
        10 + 2 * expected_x,
        rtol=0,
        atol=0.5 * 10 ** (-digits) + 1e-12,
    )
    descriptor = json.loads(result["frame"].iloc[0]["XXX"])
    assert descriptor["grid_count"] == 7500
    assert descriptor["offset"] == 0
    assert descriptor["length"] == 4000


def test_hplc_pipeline_uses_fixed_axis_descriptor_and_preserves_scale(tmp_path):
    from backend.app.hplc import preprocess_hplc_files_with_preview
    from backend.app.parsers import load_modeling_csv

    files = _make_hplc_fixture(tmp_path, n_files=2)
    result = preprocess_hplc_files_with_preview(files)
    assert result["frame"].shape[0] == 2
    assert result["hplc_axis"]["point_count"] == 7500
    assert result["hplc_axis"]["mapping"] == "piecewise_linear"
    assert len(result["common_time"]) == 7500
    descriptor = json.loads(result["frame"].iloc[0]["XXX"])
    assert descriptor == {
        "type": "linspace-slice-v1",
        "grid_start": 0.0,
        "grid_stop": 50.0,
        "grid_count": 7500,
        "offset": 0,
        "length": 7500,
        "unit": "minute",
    }
    assert len(result["frame"].iloc[0]["XXX"]) < 32767
    for curve in result["curves"]:
        assert len(curve["x"]) == len(curve["processed_y"]) == 7500
        assert curve["raw_y"] == curve["processed_y"]
    assert min(result["curves"][0]["processed_y"]) < 0

    modeling = result["frame"].copy()
    modeling["Label"] = ["A", "B"]
    modeling["Sample_ID"] = ["S1", "S2"]
    output = tmp_path / "hplc_modeling.csv"
    modeling.to_csv(output, index=False, encoding="utf-8-sig")
    loaded = load_modeling_csv(output)
    assert len(loaded.x_axis[0]) == 7500
    np.testing.assert_allclose(loaded.x_axis[0], result["common_time"], rtol=0, atol=0)


# ---- API integration tests ----


def test_hplc_preprocess_openapi_preserves_interpolation_switch():
    from backend.app.main import app

    schema = app.openapi()
    body = schema["paths"]["/api/preprocess/{kind}"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    body_name = body["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][body_name]["properties"]

    assert "hplc_interpolate" in properties
    assert "hplc_subtract_min" not in properties
    assert "hplc_normalize_area" not in properties


def test_hplc_preprocess_api_200(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2)
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


def test_hplc_preprocess_api_curve_keys(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2)
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

    files = _make_hplc_fixture(tmp_path, n_files=2)
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
    assert payload["warnings"] == []
    assert "hplc_subtract_min" not in payload
    assert "hplc_normalize_area" not in payload
    assert "common_time" in payload
    assert len(payload["common_time"]) == 7500
    assert payload["hplc_axis"]["start"] == 0.0
    assert payload["hplc_axis"]["stop"] == 50.0
    assert payload["hplc_axis"]["point_count"] == 7500
    assert payload["hplc_axis"]["mapping"] == "piecewise_linear"
    assert "xxx_download_url" not in payload
    assert "xxx_rows" not in payload
    assert "common_time_path" not in payload
    assert payload["intensity_summary"]
    assert payload["intensity_summary"][0]["point_count"] == 7500
    assert payload["intensity_summary"][0]["all_zero"] is False
    assert payload["intensity_summary"][0]["max"] > payload["intensity_summary"][0]["min"]
    assert payload["baseline_method"] is None
    precision = payload["output_precision"]
    assert precision["adaptive"] is True
    assert precision["max_decimal_places"] == 5
    assert precision["xxx_encoding"] == "linspace-slice-v1"
    assert 0 <= precision["intensity_decimal_places"] <= 5
    assert precision["xxx_max_characters"] <= 32767
    assert precision["intensity_max_characters"] <= 32767
    assert precision["excel_cell_character_limit"] == 32767


def test_hplc_preprocess_api_row_range_selects_matching_target_axis_slice(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=1)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
            data={"end_row": "100"},
        )
    finally:
        for h in opened:
            h.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["hplc_axis"]["point_count"] == 100
    assert len(payload["common_time"]) == 100
    assert len(payload["curves"][0]["processed_y"]) == 100


def test_hplc_preprocess_api_interpolation_off_exports_selected_original_axes(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app

    files = _make_hplc_fixture(tmp_path, n_files=2)
    shifted = pd.read_csv(files[1], header=None)
    shifted.iloc[:, 0] += 0.01
    shifted.to_csv(files[1], index=False, header=False)
    client = TestClient(app)
    opened = [f.open("rb") for f in files]
    try:
        response = client.post(
            "/api/preprocess/hplc",
            files=[("files", (f.name, h, "text/csv")) for f, h in zip(files, opened)],
            data={"end_row": "7500", "hplc_interpolate": "false"},
        )
    finally:
        for handle in opened:
            handle.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["hplc_interpolate"] is False
    assert payload["x_axis_consistent"] is False
    assert payload["common_time"] == []
    assert payload["hplc_axis"] is None
    assert payload["warnings"]
    assert "所选原始 X 轴正常生成" in payload["warnings"][0]
    assert all(len(curve["x"]) == 7500 for curve in payload["curves"])


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

    files = _make_hplc_fixture(tmp_path, n_files=1)
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
    assert "Index,Name,XXX,Intensity,Label,Sample_ID" in dl_resp.text
    downloaded = pd.read_csv(io.StringIO(dl_resp.text))
    x_descriptor = ast.literal_eval(downloaded.iloc[0]["XXX"])
    intensity = ast.literal_eval(downloaded.iloc[0]["Intensity"])
    assert x_descriptor["type"] == "linspace-slice-v1"
    assert x_descriptor["grid_count"] == 7500
    assert x_descriptor["offset"] == 0
    assert x_descriptor["length"] == 7500
    assert len(intensity) == 7500
    assert not any(pd.isna(value) for value in intensity)
    assert any(abs(float(value)) > 1e-12 for value in intensity)

    assert "xxx_download_url" not in resp.json()
    assert "xxx_rows" not in resp.json()
