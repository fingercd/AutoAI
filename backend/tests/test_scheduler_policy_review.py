"""Strict scheduler boundary, before API/client integration."""
import pytest
from backend.app.model_config import resolve_model_params


def test_real_scheduler_and_legacy_training_validation():
    import torch
    from backend.app.contracts import TrainingSpec, TrainingConfigValidationError
    for factor in (0, 1):
        with pytest.raises(ValueError): resolve_model_params('cnn1d', {'scheduler_factor':factor})
        with pytest.raises(TrainingConfigValidationError):
            TrainingSpec.from_legacy({'model_type':'cnn1d','scheduler_factor':factor}).validated(has_external_test=False)
    params = resolve_model_params('cnn1d', {'scheduler_factor':0.5})
    optimizer = torch.optim.AdamW(torch.nn.Linear(2,1).parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params['scheduler_factor'])
    assert scheduler.factor == 0.5
