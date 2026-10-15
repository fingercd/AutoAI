"""Offline execution-contract tests; CUDA requests are mocked, no GPU job runs."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from backend.app import execution_runtime as runtime
from backend.app import feature_selection, training
from backend.app.models import build_deep_model, build_traditional_model


DEEP_MODELS = ("pca_mlp", "cnn1d", "cnn1d_se", "resnet1d", "inception1d", "tcn1d", "cnn_transformer1d")


def test_execution_defaults_and_invalid_options():
    config = training.TrainConfig()
    assert config.execution_device == "cpu"
    assert config.cpu_threads is None
    for value in (0, -1, 1.5, True, "4"):
        with pytest.raises(ValueError, match="cpu_threads"):
            training.TrainConfig(cpu_threads=value)
    with pytest.raises(ValueError, match="execution_device"):
        training.TrainConfig(execution_device="cuda:1")


def test_runtime_default_does_not_change_process_threads(monkeypatch):
    import threadpoolctl

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary execution must not change process thread limits")

    monkeypatch.setattr(torch, "set_num_threads", unexpected)
    monkeypatch.setattr(torch, "set_num_interop_threads", unexpected)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", unexpected)
    runtime.initialize_execution_runtime()


def test_explicit_runtime_thread_allocation(monkeypatch):
    import os
    import threadpoolctl

    calls = []
    marker = object()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "17")
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("torch", value)))
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 32)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.append(("interop", value)))
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **kwargs: calls.append(("native", kwargs["limits"])) or marker)
    monkeypatch.setattr(runtime, "_threadpool_limiter", None)
    runtime.initialize_execution_runtime(4)
    assert calls == [("torch", 4), ("interop", 4), ("native", 4)]
    assert runtime._threadpool_limiter is marker
    assert all(os.environ[name] == "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))


def test_cuda_is_explicit_and_fp32_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.resolve_execution_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        runtime.resolve_execution_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    assert runtime.resolve_execution_device("cuda") == torch.device("cuda:0")
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_synchronization_only_waits_for_requested_cuda_device(monkeypatch):
    devices = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: devices.append(device))
    runtime.synchronize_device("cpu")
    runtime.synchronize_device("cuda:0")
    assert devices == [torch.device("cuda:0")]


def test_parameterless_model_uses_buffer_device():
    model = nn.Module()
    model.register_buffer("fixed_projection", torch.empty(2, device="meta"))
    assert runtime.model_device(model) == torch.device("meta")


@pytest.mark.parametrize("model_type", DEEP_MODELS)
@pytest.mark.parametrize("class_count", [2, 3])
def test_all_deep_architectures_and_buffers_move_together(model_type, class_count):
    # Meta tensors execute shape rules without real inference or GPU allocation.
    config = training.TrainConfig(model_type=model_type)
    values = np.random.default_rng(7).normal(size=(8, 64)).astype(np.float32)
    model = build_deep_model(config, input_length=64, class_count=class_count,
                             sample_count=8, x_train=values if model_type == "pca_mlp" else None)
    model.to(device="meta", dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.empty((2, 1, 64), device="meta", dtype=torch.float32))
    assert logits.shape == (2, 1 if class_count == 2 else class_count)
    assert all(parameter.device.type == "meta" and parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(buffer.device.type == "meta" for buffer in model.buffers())


def test_checkpoint_restores_parameters_and_fixed_buffers(tmp_path):
    model = nn.Linear(3, 1)
    model.register_buffer("pca_mean", torch.arange(3, dtype=torch.float32))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = nn.Linear(3, 1)
    restored.register_buffer("pca_mean", torch.zeros(3))
    assert runtime.load_deep_checkpoint(restored, path) is restored
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)


def test_checkpoint_cuda_reload_uses_explicit_map_location(monkeypatch):
    calls = []
    model = SimpleNamespace(
        to=lambda **kwargs: calls.append(("model", kwargs)),
        load_state_dict=lambda state: calls.append(("state", state)),
    )
    monkeypatch.setattr(runtime, "resolve_execution_device", lambda value: torch.device("cuda:0"))
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: calls.append(("load", kwargs)) or {"weight": "mock"})
    runtime.load_deep_checkpoint(model, "model.pt", execution_device="cuda")
    assert calls == [
        ("model", {"device": torch.device("cuda:0"), "dtype": torch.float32}),
        ("load", {"map_location": torch.device("cuda:0"), "weights_only": True}),
        ("state", {"weight": "mock"}),
    ]


@pytest.mark.parametrize("model_type,ordinary_jobs", [("random_forest", -1), ("xgboost", 1)])
def test_traditional_constructor_honors_worker_allocation_without_changing_defaults(monkeypatch, model_type, ordinary_jobs):
    if model_type == "xgboost":
        import xgboost
        monkeypatch.setattr(xgboost, "XGBClassifier", lambda **kwargs: SimpleNamespace(**kwargs))
    labels = np.asarray([0, 1])
    config = training.TrainConfig(model_type=model_type)
    ordinary = build_traditional_model(config, labels, 2)
    assert ordinary.n_jobs == ordinary_jobs
    config.cpu_threads = 4
    allocated = build_traditional_model(config, labels, 2)
    assert allocated.n_jobs == 4


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, values):
        return self.weight.expand(values.shape[0], 1)


def mock_cuda_tensors(monkeypatch):
    devices = []
    tensor = torch.tensor
    move = torch.Tensor.to

    def cpu_tensor(*args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            devices.append(("create", torch.device(device)))
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    def cpu_move(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "cuda":
            devices.append(("move", torch.device(device)))
            return self
        return move(self, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "to", cpu_move)
    return devices


def test_prediction_validation_and_occlusion_inputs_follow_cuda_model(monkeypatch):
    devices = mock_cuda_tensors(monkeypatch)
    monkeypatch.setattr(training, "model_device", lambda model: torch.device("cuda:0"))
    model = ToyModel()
    x = np.zeros((2, 4), dtype=np.float32)
    y = np.asarray([0, 1])
    training._evaluate(model, x, y, [0, 1], ["A", "B"])
    probabilities = training._deep_probabilities(model, x)
    loss = training._evaluate_deep_loss(model, training._loader(x, y, [0, 1], 2, False), nn.BCEWithLogitsLoss())
    assert probabilities.shape == (2, 2)
    assert np.isfinite(loss)
    assert devices == [("create", torch.device("cuda:0"))] * 2 + [("move", torch.device("cuda:0"))] * 2


@pytest.mark.parametrize("class_count", [2, 3])
def test_cuda_fit_moves_model_batches_and_class_weights_with_fp32(monkeypatch, tmp_path, class_count):
    devices = mock_cuda_tensors(monkeypatch)
    model = ToyModel()
    model.to = lambda **kwargs: devices.append(("model", kwargs)) or model
    if class_count == 3:
        model.forward = lambda values: model.weight.expand(values.shape[0], 3)
    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.param_groups = [{"lr": kwargs["lr"]}]
        def zero_grad(self):
            pass
        def step(self):
            pass
    class FakeScheduler:
        def __init__(self, optimizer, **kwargs):
            pass
        def step(self, loss):
            pass
    monkeypatch.setattr(training, "resolve_execution_device", lambda value: torch.device("cuda:0"))
    monkeypatch.setattr(training, "model_device", lambda model: torch.device("cuda:0"))
    monkeypatch.setattr(training, "build_deep_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch.optim, "AdamW", FakeOptimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", FakeScheduler)
    synchronized = []
    monkeypatch.setattr(training, "synchronize_device", lambda device: synchronized.append(device))
    y = np.arange(class_count, dtype=np.int64)
    x = np.zeros((class_count, 4), dtype=np.float32)
    config = training.TrainConfig(execution_device="cuda", epochs=1, class_balance="class_weight", batch_size=8)
    _, history = training._fit_deep_fold(config=config, model_type="cnn1d", x=x, y=y,
        splits={"train": list(range(class_count)), "valid": list(range(class_count))},
        label_names=[str(i) for i in range(class_count)], run_dir=tmp_path, sample_count=class_count)
    assert len(history) == 1
    assert ("model", {"device": torch.device("cuda:0"), "dtype": torch.float32}) in devices
    assert sum(kind == "create" for kind, _ in devices) >= 2
    assert sum(kind == "move" for kind, _ in devices) >= 5
    assert synchronized == [torch.device("cuda:0")] * 3


@pytest.mark.parametrize("method", ["input_gradient", "gradcam"])
def test_cuda_attribution_inputs_and_labels_follow_model_device(monkeypatch, method):
    devices = mock_cuda_tensors(monkeypatch)
    monkeypatch.setattr(feature_selection, "_model_device", lambda model: torch.device("cuda:0"))
    model = nn.Sequential(nn.Conv1d(1, 2, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    x = np.ones((2, 4), dtype=np.float32)
    y = np.asarray([0, 1])
    if method == "gradcam":
        attributions, scores, actual = feature_selection._gradcam_1d_attributions(model, x, y, 4)
        assert actual == "gradcam_1d"
    else:
        attributions, scores = feature_selection._input_gradient_attributions(model, x, y)
    assert attributions.shape == (2, 4)
    assert scores.shape == (2, 2)
    assert devices == [("create", torch.device("cuda:0"))] * 2
