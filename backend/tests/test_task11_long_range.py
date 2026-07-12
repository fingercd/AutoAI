import pytest
import torch
from torch import nn

from backend.app.models.cnn_mamba1d import (
    CNNMamba1D,
    ModelDependencyError,
    build_cnn_mamba1d,
    mamba_available,
)
from backend.app.models.cnn_transformer1d import CNNTransformer1D


def test_cnn_transformer_has_three_cnn_stages_and_batch_first_encoder():
    model = CNNTransformer1D(
        input_length=64,
        class_count=3,
        dropout=0.1,
        hidden_size=32,
        transformer_heads=4,
        transformer_layers=1,
    )

    assert len([layer for layer in model.cnn if isinstance(layer, nn.Conv1d)]) == 3
    assert len([layer for layer in model.cnn if isinstance(layer, nn.BatchNorm1d)]) == 3
    assert len([layer for layer in model.cnn if isinstance(layer, nn.ReLU)]) == 3
    assert len([layer for layer in model.cnn if isinstance(layer, nn.MaxPool1d)]) == 3
    assert model.encoder.layers[0].self_attn.batch_first is True
    assert model.encoder.layers[0].self_attn.embed_dim == model.d_model
    assert model.dropout.p == pytest.approx(0.1)


def test_cnn_transformer_forward_uses_batch_t_d_model_and_has_input_gradient():
    model = CNNTransformer1D(
        input_length=64,
        class_count=2,
        dropout=0.0,
        hidden_size=32,
        transformer_heads=4,
        transformer_layers=1,
    )
    seen_shapes: list[tuple[int, ...]] = []

    def capture_encoder_input(_module, inputs, _output):
        seen_shapes.append(tuple(inputs[0].shape))

    handle = model.encoder.register_forward_hook(capture_encoder_input)
    try:
        values = torch.randn(2, 1, 64, requires_grad=True)
        logits = model(values)
        logits.sum().backward()
    finally:
        handle.remove()

    assert logits.shape == (2, 2)
    assert seen_shapes and seen_shapes[0][0] == 2
    assert seen_shapes[0][2] == model.d_model
    assert values.grad is not None
    assert values.grad.shape == values.shape


def test_cnn_mamba_dependency_boundary_is_explicit():
    if mamba_available():
        model = CNNMamba1D(input_length=64, class_count=2, hidden_size=32, mamba_layers=1)
        values = torch.randn(2, 1, 64, requires_grad=True)
        logits = model(values)
        logits.sum().backward()
        assert logits.shape == (2, 2)
        assert values.grad is not None
        return

    with pytest.raises(ModelDependencyError, match="mamba-ssm"):
        CNNMamba1D(input_length=64, class_count=2, hidden_size=32)
    with pytest.raises(ModelDependencyError, match="pip install mamba-ssm"):
        build_cnn_mamba1d(input_length=64, class_count=2, hidden_size=32)
