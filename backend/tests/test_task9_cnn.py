import pytest
import torch
from torch import nn

from backend.app.models.cnn1d import CNN1DDocumentV2
from backend.app.models.cnn_se1d import CNNSE1DDocumentV2, SEBlock1D


@pytest.mark.parametrize(
    ("sample_count", "expected_channels"),
    [
        (100, [8, 16, 32]),
        (101, [16, 32, 64]),
        (299, [16, 32, 64]),
        (300, [32, 64, 128]),
    ],
)
def test_document_cnn_uses_n_profile(sample_count, expected_channels):
    model = CNN1DDocumentV2(input_length=1000, class_count=3, sample_count=sample_count)

    convolutions = [module for module in model.features if isinstance(module, nn.Conv1d)]

    assert [module.out_channels for module in convolutions] == expected_channels


@pytest.mark.parametrize(
    ("input_length", "expected_kernels", "expected_pools"),
    [
        (1000, [7, 5, 3], [2, 2, 2]),
        (1001, [9, 5, 3], [4, 2, 2]),
        (2999, [9, 5, 3], [4, 2, 2]),
        (3000, [9, 7, 5], [4, 4, 2]),
    ],
)
def test_document_cnn_uses_l_profile(input_length, expected_kernels, expected_pools):
    model = CNN1DDocumentV2(input_length=input_length, class_count=3, sample_count=100)

    convolutions = [module for module in model.features if isinstance(module, nn.Conv1d)]
    pools = [module for module in model.features if isinstance(module, nn.MaxPool1d)]

    assert [module.kernel_size[0] for module in convolutions] == expected_kernels
    assert [module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0] for module in pools] == expected_pools


def test_document_cnn_keeps_dropout_in_the_pooled_head_only():
    model = CNN1DDocumentV2(input_length=1000, class_count=3, sample_count=100)

    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    assert dropouts == [model.dropout]
    assert all(not isinstance(module, nn.Dropout) for module in model.features)


def test_se_block_uses_document_reduction_and_preserves_gate_shape():
    block = SEBlock1D(channels=16, reduction=8)
    values = torch.randn(2, 16, 31)

    result = block(values)

    assert block.hidden_channels == max(16 // 8, 4)
    assert block.fc[0].in_features == 16
    assert block.fc[0].out_features == 4
    assert block.fc[-1].out_features == 16
    assert result.shape == values.shape


def test_cnn_se_has_one_se_block_after_each_convolution_block():
    model = CNNSE1DDocumentV2(input_length=1000, class_count=3, sample_count=100)

    convolutions = [module for module in model.features if isinstance(module, nn.Conv1d)]
    se_blocks = [module for module in model.features if isinstance(module, SEBlock1D)]

    assert len(convolutions) == len(se_blocks) == 3
    assert model.features[2].__class__ is nn.ReLU
    assert isinstance(model.features[3], SEBlock1D)


@pytest.mark.parametrize("model_class", [CNN1DDocumentV2, CNNSE1DDocumentV2])
def test_document_cnn_forward_outputs_batch_by_classes(model_class):
    model = model_class(input_length=64, class_count=5, sample_count=150)
    values = torch.randn(4, 1, 64)

    logits = model(values)

    assert logits.shape == (4, 5)


@pytest.mark.parametrize("model_class", [CNN1DDocumentV2, CNNSE1DDocumentV2])
def test_document_cnn_exposes_last_conv_for_gradcam(model_class):
    model = model_class(input_length=64, class_count=3, sample_count=150)
    convolutions = [module for module in model.features if isinstance(module, nn.Conv1d)]

    assert model.gradcam_target_layer() is convolutions[-1]
    assert isinstance(model.gradcam_target_layer(), nn.Conv1d)


def test_document_cnn_accepts_resolved_profile_values():
    profile = {
        "train_sample_count": 301,
        "feature_count": 3001,
        "dropout": 0.2,
        "values": {
            "channels": [4, 8, 12],
            "kernels": [5, 3, 3],
            "pools": [2, 2, 2],
        },
    }

    model = CNN1DDocumentV2(profile=profile, class_count=2)

    assert model.channels == (4, 8, 12)
    assert model.kernels == (5, 3, 3)
    assert model.pools == (2, 2, 2)
    assert model.dropout.p == pytest.approx(0.2)
