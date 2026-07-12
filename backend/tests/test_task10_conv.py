import pytest
import torch
from torch import nn

from backend.app.models.inception1d_v2 import Inception1DDocumentV2
from backend.app.models.resnet1d_v2 import ResNet1DDocumentV2
from backend.app.models.tcn1d_v2 import TCN1DDocumentV2


@pytest.mark.parametrize("model_class", [ResNet1DDocumentV2, Inception1DDocumentV2, TCN1DDocumentV2])
def test_document_conv_models_forward_and_have_head(model_class):
    model = model_class(input_length=64, class_count=3, sample_count=150)
    logits = model(torch.randn(2, 1, 64))

    assert logits.shape == (2, 3)
    assert isinstance(model.head[0], nn.AdaptiveAvgPool1d)
    assert isinstance(model.head[1], nn.Flatten)
    assert isinstance(model.head[2], nn.Dropout)
    assert isinstance(model.head[3], nn.Linear)


def test_resnet_has_three_blocks_and_explicit_target_layer():
    model = ResNet1DDocumentV2(input_length=1000, class_count=2, sample_count=100)

    assert len(model.blocks) == 3
    assert model.blocks[0].shortcut.__class__ is nn.Identity
    assert isinstance(model.blocks[1].shortcut, nn.Sequential)
    assert model.gradcam_target_layer() is model.blocks[-1].conv2

    seen: list[tuple[int, ...]] = []
    handle = model.gradcam_target_layer().register_forward_hook(lambda _m, _i, output: seen.append(tuple(output.shape)))
    try:
        model(torch.randn(2, 1, 64))
    finally:
        handle.remove()
    assert seen and len(seen[0]) == 3


def test_inception_has_four_equal_width_conv_branches_and_target_hook():
    model = Inception1DDocumentV2(input_length=1000, class_count=2, sample_count=100)
    block = model.gradcam_target_layer()

    assert len(block.branches) == 4
    assert len({branch.out_channels for branch in block.branches}) == 1
    assert all(isinstance(branch, nn.Conv1d) for branch in block.branches)

    seen: list[tuple[int, ...]] = []
    handle = block.register_forward_hook(lambda _m, _i, output: seen.append(tuple(output.shape)))
    try:
        model(torch.randn(2, 1, 64))
    finally:
        handle.remove()
    assert seen and len(seen[0]) == 3


@pytest.mark.parametrize(
    ("input_length", "expected_dilations"),
    [(1000, (1, 2, 4)), (1001, (1, 2, 4, 8)), (3001, (1, 2, 4, 8))],
)
def test_tcn_uses_only_document_dilations_and_targets_last_second_conv(input_length, expected_dilations):
    model = TCN1DDocumentV2(input_length=input_length, class_count=2, sample_count=100)

    assert model.dilations == expected_dilations
    assert [block.dilation for block in model.blocks] == list(expected_dilations)
    assert model.gradcam_target_layer() is model.blocks[-1].conv2
    assert model.gradcam_target_layer().dilation == (expected_dilations[-1],)
