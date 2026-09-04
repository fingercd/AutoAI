from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.app.models.pca_lda import build_pca_lda
from backend.app.models.pls_da import build_pls_da
from backend.app.training_visualization import (
    FEATURE_VISUALIZATION_SCHEMA,
    build_model_feature_visualization,
    build_pls_permutation_plot,
    classical_mds_from_proximity,
    random_forest_proximity,
)


def _fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(2), 15)
    x = rng.normal(size=(30, 8))
    x[y == 1, :3] += 2.5
    splits = {
        "train": list(range(0, 10)) + list(range(15, 25)),
        "valid": list(range(10, 12)) + list(range(25, 27)),
        "test": list(range(12, 15)) + list(range(27, 30)),
    }
    metadata = [
        {"name": f"S{index + 1}", "sample_id": str(index // 3 + 1)}
        for index in range(len(y))
    ]
    return x, y, splits, metadata


def test_pls_visualization_is_deterministic_finite_and_contains_native_plots():
    x, y, splits, metadata = _fixture()
    model = build_pls_da(2).fit(x[sorted(splits["train"] + splits["valid"])], y[sorted(splits["train"] + splits["valid"])])
    kwargs = dict(
        model_type="pls_da",
        model=model,
        x=x,
        selection_x=x,
        y=y,
        label_names=["A", "B"],
        splits=splits,
        metadata=metadata,
        x_axis=[float(index) for index in range(x.shape[1])],
        evaluation_strategy="stratified_holdout",
        seed=11,
        permutation_count=5,
    )

    first = build_model_feature_visualization(**kwargs)
    second = build_model_feature_visualization(**kwargs)

    assert first == second
    assert first["schema_version"] == FEATURE_VISUALIZATION_SCHEMA
    assert first["status"] == "ready"
    # Binary PLS-DA has one identifiable response direction, so a native
    # two-dimensional score chart is correctly unavailable; retained plots
    # remain deterministic and use the fitted PLS representation.
    assert {plot["id"] for plot in first["plots"]} == {"pls_permutation", "pls_vip"}
    assert len(next(plot for plot in first["plots"] if plot["id"] == "pls_vip")["items"]) == x.shape[1]
    json.dumps(first, ensure_ascii=False, allow_nan=False)


def test_pls_permutation_ignores_test_labels():
    x, y, splits, _ = _fixture()
    first = build_pls_permutation_plot(
        x=x,
        y=y,
        splits=splits,
        class_count=2,
        n_components=2,
        seed=4,
        permutation_count=4,
    )
    changed = y.copy()
    changed[splits["test"]] = 1 - changed[splits["test"]]
    second = build_pls_permutation_plot(
        x=x,
        y=changed,
        splits=splits,
        class_count=2,
        n_components=2,
        seed=4,
        permutation_count=4,
    )
    assert first == second


def test_pca_lda_visualization_uses_fitted_pipeline_scores():
    x, y, splits, metadata = _fixture()
    fit_indices = sorted(splits["train"] + splits["valid"])
    model = build_pca_lda(3).fit(x[fit_indices], y[fit_indices])

    result = build_model_feature_visualization(
        model_type="pca_lda",
        model=model,
        x=x,
        y=y,
        label_names=["A", "B"],
        splits=splits,
        metadata=metadata,
        x_axis=list(range(x.shape[1])),
        evaluation_strategy="external_test_holdout",
        seed=1,
    )

    expected = model.named_steps["pca"].transform(x)[:, :2]
    points = next(plot for plot in result["plots"] if plot["id"] == "pca_scores")["points"]
    actual = np.asarray([[point["x"], point["y"]] for point in points])
    np.testing.assert_allclose(actual, expected)


def test_random_forest_proximity_is_symmetric_and_drives_two_plots():
    x, y, splits, metadata = _fixture()
    model = RandomForestClassifier(n_estimators=24, max_depth=3, random_state=3).fit(x, y)
    proximity = random_forest_proximity(model, x)
    coordinates, ratios = classical_mds_from_proximity(proximity)

    np.testing.assert_allclose(proximity, proximity.T)
    np.testing.assert_allclose(np.diag(proximity), 1.0)
    assert coordinates.shape == (len(x), 2)
    assert all(value >= 0 for value in ratios)

    result = build_model_feature_visualization(
        model_type="random_forest",
        model=model,
        x=x,
        y=y,
        label_names=["A", "B"],
        splits=splits,
        metadata=metadata,
        x_axis=list(range(x.shape[1])),
        evaluation_strategy="stratified_holdout",
        seed=1,
    )
    assert result["status"] == "ready"
    assert {plot["type"] for plot in result["plots"]} == {"scatter", "dendrogram"}


def test_cv_and_non_native_models_return_explicit_unsupported_state():
    x, y, splits, metadata = _fixture()
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    common = dict(
        model=model,
        x=x,
        y=y,
        label_names=["A", "B"],
        splits=splits,
        metadata=metadata,
        x_axis=list(range(x.shape[1])),
        seed=1,
    )
    cv = build_model_feature_visualization(
        model_type="random_forest",
        evaluation_strategy="leave_one_sample_id_cv",
        **common,
    )
    unsupported = build_model_feature_visualization(
        model_type="svm",
        evaluation_strategy="stratified_holdout",
        **common,
    )

    assert cv["status"] == "unsupported"
    assert "交叉验证" in cv["reason"]
    assert unsupported["status"] == "unsupported"
    assert "PCA/t-SNE" in unsupported["reason"]
