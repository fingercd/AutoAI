"""Policies describe executable controls without importing a trainer."""
import subprocess
import sys
import pytest
from backend.app.model_catalog import MODEL_DECLARATIONS, MODEL_ALIASES, model_availability
from backend.app.model_config import model_capability_snapshot, model_policy, resolve_model_params


def test_lightweight_import_and_alias_identity():
    child = subprocess.run([sys.executable, '-B', '-c',
        'import sys; import backend.app.model_catalog; import backend.app.model_config; '
        'assert not any(n in sys.modules for n in ("torch","sklearn","xgboost","aggmap","backend.app.models"))'],
        capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    from backend.app.models import registry
    from backend.app import contracts
    assert registry.MODEL_ALIASES is MODEL_ALIASES
    assert contracts._MODEL_ALIASES is MODEL_ALIASES


def test_missing_dependency_and_unimplemented_version(monkeypatch):
    import backend.app.model_catalog as catalog
    original = catalog.importlib.util.find_spec
    monkeypatch.setattr(catalog.importlib.util, 'find_spec', lambda name: None if name == 'xgboost' else original(name))
    assert model_availability('xgboost') == (False, 'dependency_missing_xgboost')
    assert model_availability('logistic_regression') == (True, None)
    monkeypatch.setattr(catalog.importlib.util, 'find_spec', lambda name: object())
    assert model_availability('cnn_mamba1d') == (False, 'not_implemented_for_version')


def test_semantic_digest_separates_availability(monkeypatch):
    import backend.app.model_config as config
    first = model_capability_snapshot()
    monkeypatch.setattr(config, 'model_availability', lambda _: (False, 'implementation_missing'))
    second = model_capability_snapshot()
    assert first['catalog_digest'] == second['catalog_digest']
    assert first['availability_digest'] != second['availability_digest']
    assert not any(m['available'] for m in second['models'])


@pytest.mark.parametrize('model', [m.id for m in MODEL_DECLARATIONS if m.implemented])
def test_defaults_are_executable_and_overrides_closed(model):
    from backend.app.training import TrainConfig
    policy = model_policy(model)
    defaults = resolve_model_params(model)
    assert defaults == {key: getattr(TrainConfig(), key) for key in defaults}
    assert defaults == resolve_model_params(model, defaults)
    with pytest.raises(ValueError):
        resolve_model_params(model, {'device':'cuda'})
    for field in policy['parameters']:
        name = field['name']
        for invalid in (None, True, False, {}, [], float('nan'), float('inf')):
            with pytest.raises(ValueError):
                resolve_model_params(model, {name: invalid})
        if field['value_type'] != 'string':
            with pytest.raises(ValueError):
                resolve_model_params(model, {name: str(field['default'])})
            if field['minimum'] is not None:
                with pytest.raises(ValueError):
                    resolve_model_params(model, {name: field['minimum'] - 1})
            if field['maximum'] is not None:
                with pytest.raises(ValueError):
                    resolve_model_params(model, {name: field['maximum'] + 1})


def test_xgboost_gamma_survives_candidates_and_rf_controls():
    import numpy as np
    from backend.app.training import TrainConfig, _traditional_candidate_configs
    config = TrainConfig(xgboost_gamma=0.75, random_forest_n_estimators=50, random_forest_search_iterations=2)
    candidates = _traditional_candidate_configs(config, 'xgboost', 128, np.array([0,1]*10))
    assert len(candidates) == 12
    assert all(c.xgboost_gamma == 0.75 for c in candidates)
    candidates = _traditional_candidate_configs(config, 'random_forest', 128, np.array([0,1]*10))
    assert len(candidates) == 2
    assert all(c.random_forest_n_estimators == 50 and c.random_forest_oob_score is True for c in candidates)
