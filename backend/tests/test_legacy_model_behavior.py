from backend.app.models.cnn1d import CNN1D
from backend.app.models.transformer import Transformer1D


def test_legacy_cnn1d_keeps_master_two_class_output_and_width():
    model = CNN1D(input_length=800, class_count=2, sample_count=50)
    assert model.classifier.out_features == 2
    assert model.features[0].out_channels == 16


def test_legacy_transformer_keeps_master_patch_configuration():
    model = Transformer1D(input_length=800, class_count=3)
    assert model.patch.kernel_size == (32,)
    assert model.patch.stride == (16,)
