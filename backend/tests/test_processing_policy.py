import numpy as np
import pytest
from backend.tests.test_agent_model_sessions import api

from backend.app.processing_policy import (
    EXECUTABLE_MODELS,
    freeze_fixed_processing,
    legal_processing,
    validate_processing,
)
from backend.app.training import _fit_x_normalizer, _transform_x_with_normalizer


def test_finite_support_matrix():
    assert len(EXECUTABLE_MODELS) == 13
    assert sum(len(legal_processing(model, class_count=2)) for model in EXECUTABLE_MODELS) == 96
    assert sum(len(legal_processing(model, class_count=3)) for model in EXECUTABLE_MODELS) == 92
    with pytest.raises(ValueError, match="multiclass_xgboost"):
        validate_processing("xgboost", "area", "class_weight", class_count=3)
    with pytest.raises(ValueError, match="unsupported_for_model"):
        validate_processing("pca_lda", "zscore", "class_weight", class_count=2)


def test_fixed_processing_requires_exact_mapping():
    assert freeze_fixed_processing(["svm"])["svm"] == {
        "normalization": "zscore", "class_balance": "none"
    }
    with pytest.raises(ValueError, match="model_set_mismatch"):
        freeze_fixed_processing(["svm", "cnn1d"], {"svm": {"normalization": "none", "class_balance": "none"}})


def test_area_uses_unit_feature_index_and_finite_float32():
    matrix = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    normalizer = _fit_x_normalizer(matrix, "area")
    assert normalizer == {"mode": "area"}
    actual = _transform_x_with_normalizer(matrix, normalizer)
    np.testing.assert_allclose(actual[:2], matrix[:2] / 4.0)
    np.testing.assert_array_equal(actual[2], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(
        _transform_x_with_normalizer([[3.0]], _fit_x_normalizer(np.array([[3.0]]), "area")),
        np.array([[3e8]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="area must be finite"):
        _transform_x_with_normalizer([[1e308, 1e308, 1e308]], normalizer)
    with pytest.raises(ValueError, match="float32 output must be finite"):
        _transform_x_with_normalizer([[1e100, 1.0]], {"mode": "none"})


def test_normalizer_statistics_use_train_only():
    train = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    valid = np.array([[20.0, 30.0]], dtype=np.float32)
    normalizer = _fit_x_normalizer(train, "minmax")
    np.testing.assert_allclose(_transform_x_with_normalizer(valid, normalizer), [[10.0, 14.5]])
    with pytest.raises(ValueError, match="invalid_normalization"):
        _fit_x_normalizer(train, "bad")


def test_default_zscore_matches_pre_step5_featurewise_calculation():
    train = np.array([[1.0, 4.0, 6.0], [3.0, 8.0, 12.0],
                      [5.0, 6.0, 18.0]], dtype=np.float32)
    held_out = np.array([[7.0, 10.0, 24.0]], dtype=np.float32)
    normalizer = _fit_x_normalizer(train, "zscore")
    old_mean = train.mean(axis=0)
    old_scale = np.maximum(train.std(axis=0), 1e-8)
    assert normalizer["mode"] == "zscore"
    np.testing.assert_array_equal(normalizer["mean"], old_mean)
    np.testing.assert_array_equal(normalizer["scale"], old_scale)
    np.testing.assert_array_equal(
        _transform_x_with_normalizer(held_out, normalizer),
        ((held_out - old_mean) / old_scale).astype(np.float32),
    )


def test_training_spec_rejects_unsupported_weight_before_queuing():
    from backend.app.contracts import TrainingConfigValidationError, TrainingSpec

    with pytest.raises(TrainingConfigValidationError, match="class_weight_unsupported_for_model"):
        TrainingSpec.from_legacy({
            "model_type": "pca_lda", "class_balance": "class_weight",
        }).validated(has_external_test=False)
    with pytest.raises(TrainingConfigValidationError, match="invalid_normalization"):
        TrainingSpec.from_legacy({
            "model_type": "svm", "normalization": "not_a_mode",
        }).validated(has_external_test=False)


def test_old_source_bindings_only_match_verified_default_processing():
    from backend.app.model_config import (
        _DEFAULT_PROCESSING_BINDING_COMPAT, compatible_search_strategy_binding,
        search_strategy_binding,
    )

    assert len(_DEFAULT_PROCESSING_BINDING_COMPAT) == len(EXECUTABLE_MODELS)
    for model_id, (old_digest, new_digest) in _DEFAULT_PROCESSING_BINDING_COMPAT.items():
        assert search_strategy_binding(model_id)["digest"] == new_digest
        assert compatible_search_strategy_binding(model_id, old_digest, "zscore", "none")
        assert not compatible_search_strategy_binding(model_id, old_digest, "area", "none")
        assert not compatible_search_strategy_binding(model_id, old_digest, "zscore", "class_weight")
        assert not compatible_search_strategy_binding(model_id, "0" * 64, "zscore", "none")


def test_multiclass_xgboost_weight_is_rejected_before_run_creation(api, tmp_path):
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    from backend.app.runs.repository import RunRepository

    client, storage, _ = api
    source = write_grouped_classification_csv(tmp_path / 'three-classes.csv',
        groups_per_class=4,repeats=2,feature_count=16,labels=('A','B','C'))
    with source.open('rb') as handle:
        uploaded = client.post('/api/datasets/upload', files={
            'file': ('three-classes.csv',handle,'text/csv')})
    assert uploaded.status_code == 200
    response = client.post('/api/training/runs',json={
        'dataset_id': uploaded.json()['dataset_id'],
        'config': {'model_type':'xgboost','class_balance':'class_weight'},
    })
    assert response.status_code == 422
    assert RunRepository(storage / 'runs.sqlite3').list() == []


@pytest.mark.parametrize('model_type',['logistic_regression','svm','random_forest','xgboost'])
def test_imbalanced_traditional_fit_consumes_reported_weight(model_type):
    from backend.app.models import build_traditional_model
    from backend.app.training import TrainConfig, _processing_stage_audit

    x=np.random.default_rng(14).normal(size=(9,16)).astype(np.float32)
    y=np.asarray([0]*6+[1]*3,dtype=np.int64)
    config=TrainConfig(model_type=model_type,class_balance='class_weight',
        random_forest_n_estimators=5,xgboost_n_estimators=3)
    model=build_traditional_model(config,y,2)
    model.fit(x,y)
    audit=_processing_stage_audit(stage='selection_train',indices=list(range(len(y))),
        normalizer={'mode':'none'},y=y,label_names=['A','B'],
        model_type=model_type,class_balance='class_weight')
    if model_type=='xgboost':
        assert model.get_params()['scale_pos_weight']==audit['weighting']['value']==2.0
    else:
        assert model.get_params()['class_weight']=='balanced'
        assert audit['weighting']['kind']=='balanced'
        np.testing.assert_allclose(audit['weighting']['values'],[.75,1.5])
        if model_type=='svm':
            np.testing.assert_allclose(model.class_weight_,audit['weighting']['values'])
        if model_type=='logistic_regression':
            from sklearn.utils.class_weight import compute_class_weight
            np.testing.assert_allclose(compute_class_weight('balanced',classes=np.unique(y),y=y),
                audit['weighting']['values'])
            unweighted=build_traditional_model(TrainConfig(model_type=model_type),y,2)
            unweighted.fit(x,y)
            assert not np.allclose(model.coef_,unweighted.coef_)


@pytest.mark.parametrize('class_count',[2,3])
def test_imbalanced_deep_loss_consumes_reported_weight(monkeypatch,tmp_path,class_count):
    from backend.app import training
    from backend.app.training import TrainConfig, _processing_stage_audit

    counts=[6,3] if class_count==2 else [6,3,3]
    y=np.asarray([label for label,count in enumerate(counts) for _ in range(count)],dtype=np.int64)
    x=np.random.default_rng(18).normal(size=(len(y),32)).astype(np.float32)
    splits={'train':list(range(len(y))),'valid':list(range(len(y)-3,len(y)))}
    observed=[]
    constructor=training.nn.BCEWithLogitsLoss if class_count==2 else training.nn.CrossEntropyLoss
    method='BCEWithLogitsLoss' if class_count==2 else 'CrossEntropyLoss'
    def capture(*args,**kwargs):
        loss=constructor(*args,**kwargs)
        observed.append(loss)
        return loss
    monkeypatch.setattr(training.nn,method,capture)
    config=TrainConfig(model_type='cnn1d',class_balance='class_weight',epochs=1,batch_size=4)
    training._fit_deep_fold(config=config,model_type='cnn1d',x=x,y=y,splits=splits,
        label_names=[str(i) for i in range(class_count)],run_dir=tmp_path,sample_count=len(y))
    assert observed
    audit=_processing_stage_audit(stage='train',indices=splits['train'],normalizer={'mode':'none'},
        y=y,label_names=[str(i) for i in range(class_count)],model_type='cnn1d',class_balance='class_weight')
    if class_count==2:
        assert audit['weighting']['kind']=='bce_pos_weight'
        np.testing.assert_allclose(observed[0].pos_weight.detach().numpy(),[audit['weighting']['value']])
    else:
        assert audit['weighting']['kind']=='ce_class_weight'
        np.testing.assert_allclose(observed[0].weight.detach().numpy(),audit['weighting']['values'])
