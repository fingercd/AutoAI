from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .parsers import load_modeling_csv
from .paths import RUNS_DIR


class CNN1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, sample_count: int, dropout: float | None = None, hidden_size: int = 64) -> None:
        super().__init__()
        if input_length < 500:
            channels = [1, 16, 32]
        elif input_length <= 3000:
            channels = [1, 24, 48, 64]
        else:
            channels = [1, 32, 64, 96, 128]
        dropout = dropout if dropout is not None else (0.45 if sample_count < 100 else 0.25)
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels[-1], class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class MLPBaseline(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 128) -> None:
        super().__init__()
        dropout = 0.35 if dropout is None else dropout
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 16)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 16), class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ResNet1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 48) -> None:
        super().__init__()
        dropout = 0.3 if dropout is None else dropout
        self.stem = nn.Sequential(nn.Conv1d(1, hidden_size, kernel_size=7, padding=3), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock1D(hidden_size, dropout), nn.MaxPool1d(2), ResidualBlock1D(hidden_size, dropout))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_size, class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class Transformer1D(nn.Module):
    def __init__(
        self,
        input_length: int,
        class_count: int,
        dropout: float | None = None,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else dropout
        heads = max(1, heads)
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.proj = nn.Linear(1, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, input_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_size, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        seq = self.proj(seq) + self.pos[:, : seq.shape[1], :]
        seq = self.encoder(seq)
        return self.head(seq.mean(dim=1))


class CNNTransformer1D(nn.Module):
    def __init__(
        self,
        input_length: int,
        class_count: int,
        dropout: float | None = None,
        hidden_size: int = 64,
        heads: int = 4,
    ) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else dropout
        heads = max(1, heads)
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(hidden_size, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.conv(x).transpose(1, 2)
        seq = self.encoder(seq)
        return self.head(seq.mean(dim=1))


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    normalization: str = "zscore"
    split_mode: str = "stratified"
    class_balance: str = "none"
    model_type: str = "cnn1d"
    early_stopping_patience: int = 20
    dropout: float | None = None
    hidden_size: int = 64
    transformer_heads: int = 4


def _build_model(config: TrainConfig, input_length: int, class_count: int, sample_count: int) -> nn.Module:
    model_type = config.model_type.lower()
    if model_type in {"cnn1d", "1d-cnn", "1dcnn"}:
        return CNN1D(input_length, class_count, sample_count, config.dropout, config.hidden_size)
    if model_type in {"mlp", "mlp_baseline"}:
        return MLPBaseline(input_length, class_count, config.dropout, max(config.hidden_size, 32))
    if model_type in {"resnet1d", "resnet"}:
        return ResNet1D(input_length, class_count, config.dropout, max(config.hidden_size, 16))
    if model_type in {"transformer", "transformer_encoder"}:
        return Transformer1D(input_length, class_count, config.dropout, max(config.hidden_size, 16), config.transformer_heads)
    if model_type in {"cnn_transformer", "cnn+transformer"}:
        return CNNTransformer1D(input_length, class_count, config.dropout, max(config.hidden_size, 16), config.transformer_heads)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def _normalize(x: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return x.astype(np.float32), {"mode": mode}
    if mode == "minmax":
        mins = x.min(axis=1, keepdims=True)
        maxs = x.max(axis=1, keepdims=True)
        return ((x - mins) / np.maximum(maxs - mins, 1e-8)).astype(np.float32), {"mode": mode}
    if mode == "area":
        area = np.trapz(np.abs(x), axis=1, keepdims=True)
        return (x / np.maximum(area, 1e-8)).astype(np.float32), {"mode": mode}
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    return ((x - means) / np.maximum(stds, 1e-8)).astype(np.float32), {"mode": "zscore"}


def _split_indices(labels: np.ndarray, repeat_index: np.ndarray, config: TrainConfig) -> dict[str, list[int]]:
    indices = np.arange(len(labels))
    if config.split_mode == "repeat_leave_one":
        values, counts = np.unique(repeat_index, return_counts=True)
        test_repeat = values[np.argmax(counts)]
        test_idx = indices[repeat_index == test_repeat]
        rest_idx = indices[repeat_index != test_repeat]
        train_idx, valid_idx = train_test_split(
            rest_idx,
            test_size=0.2,
            random_state=config.seed,
            stratify=labels[rest_idx] if len(np.unique(labels[rest_idx])) > 1 else None,
        )
    else:
        train_valid_idx, test_idx = train_test_split(
            indices,
            test_size=0.1,
            random_state=config.seed,
            stratify=labels,
        )
        valid_ratio = 0.1 / 0.9
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_ratio,
            random_state=config.seed,
            stratify=labels[train_valid_idx],
        )
    return {"train": train_idx.tolist(), "valid": valid_idx.tolist(), "test": test_idx.tolist()}


def _loader(x: np.ndarray, y: np.ndarray, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    tx = torch.tensor(x[indices], dtype=torch.float32).unsqueeze(1)
    ty = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, indices: list[int], labels: list[str]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[indices], dtype=torch.float32).unsqueeze(1))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = y[indices]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist(),
        "probabilities": probs.tolist(),
        "pred": pred.tolist(),
        "true": true.tolist(),
    }


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def train_model(data_path: str | Path, config_data: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    config = TrainConfig(**{**TrainConfig().__dict__, **(config_data or {})})
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir / "status.json"
    previous_status: dict[str, Any] = {}
    if status_file.exists():
        try:
            previous_status = json.loads(status_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_status = {}

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = load_modeling_csv(data_path)
    sample_count = int(len(dataset.labels))
    x, norm_config = _normalize(dataset.intensity, config.normalization)
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels], dtype=np.int64)
    repeat_index = dataset.frame["Repeat_index"].astype(str).to_numpy()
    splits = _split_indices(y, repeat_index, config)

    class_weights = None
    if config.class_balance == "class_weight":
        counts = np.bincount(y, minlength=len(label_names)).astype(np.float32)
        class_weights = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / len(label_names), dtype=torch.float32)

    model = _build_model(config, input_length=x.shape[1], class_count=len(label_names), sample_count=x.shape[0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []
    best_score = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, by in _loader(x, y, splits["train"], config.batch_size, True):
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        valid_eval = _evaluate(model, x, y, splits["valid"], label_names)
        improved = valid_eval["macro_f1"] > best_score + 1e-8
        if improved:
            best_score = valid_eval["macro_f1"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "valid_accuracy": valid_eval["accuracy"],
                "valid_macro_f1": valid_eval["macro_f1"],
                "best_valid_macro_f1": best_score,
                "bad_epochs": bad_epochs,
            }
        )
        if config.early_stopping_patience > 0 and bad_epochs >= config.early_stopping_patience:
            history[-1]["early_stopped"] = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    evaluations = {name: _evaluate(model, x, y, idx, label_names) for name, idx in splits.items()}
    metrics = {
        split: {key: value for key, value in result.items() if key not in {"probabilities", "pred", "true"}}
        for split, result in evaluations.items()
    }

    prediction_rows = []
    for split, idxs in splits.items():
        result = evaluations[split]
        for local_idx, source_idx in enumerate(idxs):
            row = {
                "dataset": split,
                "index": dataset.frame.iloc[source_idx]["Index"],
                "Repeat_index": dataset.frame.iloc[source_idx]["Repeat_index"],
                "true_label": label_names[result["true"][local_idx]],
                "pred_label": label_names[result["pred"][local_idx]],
            }
            for label, prob in zip(label_names, result["probabilities"][local_idx]):
                row[f"prob_{label}"] = float(prob)
            prediction_rows.append(row)

    config_out = {**config.__dict__, "data_path": str(Path(data_path).resolve()), "preprocess": norm_config}
    (run_dir / "config.json").write_text(json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps({idx: label for idx, label in enumerate(label_names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), run_dir / "model.pt")
    status_payload = {
        **previous_status,
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "history": history,
        "model_type": config.model_type,
        "sample_count": sample_count,
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history),
        "best_valid_macro_f1": best_score,
        "config": config_out,
        "data_path": str(Path(data_path).resolve()),
        "completed_at": _now_iso(),
    }
    (run_dir / "status.json").write_text(
        json.dumps(
            status_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "status": "success",
        "metrics": metrics,
        "history": history,
        "model_type": config.model_type,
        "sample_count": sample_count,
        "label_names": label_names,
        "target_epochs": config.epochs,
        "actual_epochs": len(history),
        "best_valid_macro_f1": best_score,
        "run_dir": str(run_dir.resolve()),
    }


def list_runs() -> list[dict[str, Any]]:
    runs = []
    for path in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        status_file = path / "status.json"
        if status_file.exists():
            try:
                runs.append(json.loads(status_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                runs.append({"run_id": path.name, "status": "unknown"})
    return runs
