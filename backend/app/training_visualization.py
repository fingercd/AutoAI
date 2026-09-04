"""Generate model-native, auditable feature visualizations after training.

Only fitted representations that belong to the selected model are exposed.  The
module deliberately returns an explicit unsupported state for models without a
native two-dimensional representation instead of applying a generic PCA/t-SNE
and presenting it as model explainability.

TEMPORARILY_HIDDEN: retained for a future, explicitly approved restoration;
new training paths do not call this module or publish its artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression


FEATURE_VISUALIZATION_SCHEMA = "model-feature-visualization-v1"
PLS_PERMUTATION_COUNT = 50
MAX_PROXIMITY_SAMPLES = 120


def _finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("模型特征可视化包含非有限数值")
    return result


def _split_lookup(splits: dict[str, list[int]]) -> dict[int, str]:
    return {
        int(index): split_name
        for split_name, indices in splits.items()
        for index in indices
    }


def _points(
    coordinates: np.ndarray,
    *,
    source_indices: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    splits: dict[str, list[int]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_by_index = _split_lookup(splits)
    rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(source_indices.tolist()):
        meta = metadata[source_index] if source_index < len(metadata) else {}
        rows.append(
            {
                "index": int(source_index),
                "name": str(meta.get("name") or meta.get("sample_id") or f"S{source_index + 1}"),
                "sample_id": str(meta.get("sample_id") or ""),
                "label": str(label_names[int(y[source_index])]),
                "split": split_by_index.get(int(source_index), "unknown"),
                "x": _finite_float(coordinates[local_index, 0]),
                "y": _finite_float(coordinates[local_index, 1]),
            }
        )
    return rows


def _confidence_ellipse(coordinates: np.ndarray, fit_indices: np.ndarray) -> dict[str, Any] | None:
    """Return a descriptive 95% covariance ellipse from the fitted sample pool."""
    values = np.asarray(coordinates[fit_indices, :2], dtype=np.float64)
    if values.shape[0] < 3 or not np.isfinite(values).all():
        return None
    covariance = np.cov(values, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 1e-12:
        return None
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # sqrt(chi2.ppf(.95, df=2)); fixed constant avoids a scipy.stats import.
    radii = np.sqrt(eigenvalues) * np.sqrt(5.991464547107979)
    angles = np.linspace(0.0, 2.0 * np.pi, 97)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    boundary = circle @ np.diag(radii) @ eigenvectors.T + values.mean(axis=0)
    return {
        "level": 0.95,
        "kind": "training_pool_covariance",
        "points": [
            {"x": _finite_float(point[0]), "y": _finite_float(point[1])}
            for point in boundary
        ],
    }


def _scatter(
    *,
    plot_id: str,
    title: str,
    coordinates: np.ndarray,
    x_label: str,
    y_label: str,
    y: np.ndarray,
    label_names: list[str],
    splits: dict[str, list[int]],
    metadata: list[dict[str, Any]],
    description: str,
    source_indices: np.ndarray | None = None,
    ellipse_fit_indices: np.ndarray | None = None,
) -> dict[str, Any] | None:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None
    values = values[:, :2]
    if not np.isfinite(values).all():
        return None
    indices = np.arange(values.shape[0], dtype=np.int64) if source_indices is None else source_indices
    plot: dict[str, Any] = {
        "id": plot_id,
        "type": "scatter",
        "title": title,
        "description": description,
        "x_label": x_label,
        "y_label": y_label,
        "points": _points(
            values,
            source_indices=np.asarray(indices, dtype=np.int64),
            y=y,
            label_names=label_names,
            splits=splits,
            metadata=metadata,
        ),
    }
    if ellipse_fit_indices is not None and source_indices is None:
        ellipse = _confidence_ellipse(values, np.asarray(ellipse_fit_indices, dtype=np.int64))
        if ellipse:
            plot["ellipse"] = ellipse
    return plot


def _one_hot(values: np.ndarray, class_count: int) -> np.ndarray:
    output = np.zeros((len(values), class_count), dtype=np.float64)
    output[np.arange(len(values)), np.asarray(values, dtype=np.int64)] = 1.0
    return output


def _pls_r2_q2(
    model: PLSRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    class_count: int,
) -> tuple[float, float]:
    train_target = _one_hot(y_train, class_count)
    valid_target = _one_hot(y_valid, class_count)
    train_mean = train_target.mean(axis=0, keepdims=True)
    train_prediction = np.asarray(model.predict(x_train), dtype=np.float64).reshape(train_target.shape)
    valid_prediction = np.asarray(model.predict(x_valid), dtype=np.float64).reshape(valid_target.shape)
    r2_denominator = max(float(np.sum((train_target - train_mean) ** 2)), 1e-12)
    q2_denominator = max(float(np.sum((valid_target - train_mean) ** 2)), 1e-12)
    r2 = 1.0 - float(np.sum((train_target - train_prediction) ** 2)) / r2_denominator
    q2 = 1.0 - float(np.sum((valid_target - valid_prediction) ** 2)) / q2_denominator
    return _finite_float(r2), _finite_float(q2)


def build_pls_permutation_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, list[int]],
    class_count: int,
    n_components: int,
    seed: int,
    permutation_count: int = PLS_PERMUTATION_COUNT,
) -> dict[str, Any] | None:
    """Build a fixed Train/Valid label-permutation validation plot; Test is unused."""
    train_indices = np.asarray(splits.get("train", []), dtype=np.int64)
    valid_indices = np.asarray(splits.get("valid", []), dtype=np.int64)
    if train_indices.size < 2 or valid_indices.size < 1:
        return None
    pool_indices = np.concatenate([train_indices, valid_indices])
    original_y = np.asarray(y[pool_indices], dtype=np.int64)
    component_count = max(1, min(int(n_components), len(train_indices) - 1, x.shape[1]))

    def fit(permuted_y: np.ndarray) -> tuple[float, float]:
        by_index = dict(zip(pool_indices.tolist(), permuted_y.tolist()))
        train_y = np.asarray([by_index[int(index)] for index in train_indices], dtype=np.int64)
        valid_y = np.asarray([by_index[int(index)] for index in valid_indices], dtype=np.int64)
        fitted = PLSRegression(n_components=component_count, scale=False, max_iter=500, tol=1e-6)
        fitted.fit(x[train_indices], _one_hot(train_y, class_count))
        return _pls_r2_q2(
            fitted,
            x[train_indices],
            train_y,
            x[valid_indices],
            valid_y,
            class_count,
        )

    original_r2, original_q2 = fit(original_y)
    original_hot = _one_hot(original_y, class_count).reshape(-1)
    rng = np.random.default_rng(seed)
    points: list[dict[str, Any]] = []
    permuted_q2: list[float] = []
    for _ in range(max(1, int(permutation_count))):
        permuted_y = rng.permutation(original_y)
        r2, q2 = fit(permuted_y)
        permuted_hot = _one_hot(permuted_y, class_count).reshape(-1)
        correlation = float(np.corrcoef(original_hot, permuted_hot)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        points.append(
            {
                "correlation": _finite_float(correlation),
                "r2": r2,
                "q2": q2,
                "original": False,
            }
        )
        permuted_q2.append(q2)
    points.append(
        {
            "correlation": 1.0,
            "r2": original_r2,
            "q2": original_q2,
            "original": True,
        }
    )
    p_value = (1 + sum(value >= original_q2 for value in permuted_q2)) / (len(permuted_q2) + 1)
    return {
        "id": "pls_permutation",
        "type": "permutation",
        "title": "PLS-DA 标签置换检验",
        "description": "固定 Train/Valid 划分并置换训练池标签；Q² 来自未参与拟合的 Valid，Test 未参与。",
        "x_label": "置换标签与原标签相关性",
        "y_label": "R²Y / Q²",
        "permutation_count": len(permuted_q2),
        "q2_p_value": _finite_float(p_value),
        "points": points,
    }


def _pls_vip(model: Any, x_axis: list[float]) -> dict[str, Any] | None:
    pls = getattr(model, "model", None)
    weights = np.asarray(getattr(pls, "x_weights_", []), dtype=np.float64)
    scores = np.asarray(getattr(pls, "x_scores_", []), dtype=np.float64)
    loadings = np.asarray(getattr(pls, "y_loadings_", []), dtype=np.float64)
    if weights.ndim != 2 or scores.ndim != 2 or loadings.ndim != 2 or weights.size == 0:
        return None
    strength = np.sum(scores**2, axis=0) * np.sum(loadings**2, axis=0)
    denominator = float(np.sum(strength))
    if denominator <= 1e-12:
        return None
    normalized_weights = weights / np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)
    vip = np.sqrt(weights.shape[0] * ((normalized_weights**2) @ strength) / denominator)
    coordinates = np.asarray(x_axis, dtype=np.float64)
    if coordinates.size != vip.size or not np.isfinite(coordinates).all():
        coordinates = np.arange(vip.size, dtype=np.float64)
    items = [
        {
            "feature_index": int(index),
            "label": f"{coordinates[index]:.12g}",
            "x": _finite_float(coordinates[index]),
            "value": _finite_float(vip[index]),
        }
        for index in np.argsort(vip)[::-1].tolist()
    ]
    return {
        "id": "pls_vip",
        "type": "bar",
        "title": "PLS-DA VIP 变量重要性",
        "description": "展示 VIP 最高的 24 个特征；VIP>1 是常用启发式，不等于统计显著性。",
        "x_label": "特征坐标",
        "y_label": "VIP",
        "threshold": 1.0,
        "display_limit": 24,
        "items": items,
    }


def balanced_sample_indices(y: np.ndarray, splits: dict[str, list[int]], limit: int) -> np.ndarray:
    all_indices = np.arange(len(y), dtype=np.int64)
    if len(all_indices) <= limit:
        return all_indices
    split_by_index = _split_lookup(splits)
    strata: dict[tuple[int, str], list[int]] = {}
    for index, class_id in enumerate(np.asarray(y, dtype=np.int64).tolist()):
        strata.setdefault((int(class_id), split_by_index.get(index, "unknown")), []).append(index)
    quota = max(1, limit // max(1, len(strata)))
    selected: list[int] = []
    for key in sorted(strata):
        values = strata[key]
        positions = np.linspace(0, len(values) - 1, min(quota, len(values)), dtype=int)
        selected.extend(values[position] for position in positions.tolist())
    if len(selected) < limit:
        selected_set = set(selected)
        selected.extend(index for index in all_indices.tolist() if index not in selected_set)
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def random_forest_proximity(model: Any, x: np.ndarray) -> np.ndarray:
    leaves = np.asarray(model.apply(x), dtype=np.int64)
    if leaves.ndim != 2 or leaves.shape[1] == 0:
        raise ValueError("随机森林没有可用叶节点索引")
    proximity = np.mean(leaves[:, None, :] == leaves[None, :, :], axis=2, dtype=np.float64)
    return np.asarray(proximity, dtype=np.float64)


def classical_mds_from_proximity(proximity: np.ndarray) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(proximity, dtype=np.float64)
    squared_distance = np.clip(1.0 - values, 0.0, 1.0)
    count = squared_distance.shape[0]
    centering = np.eye(count) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * centering @ squared_distance @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = [(int(index), float(eigenvalues[index])) for index in order if eigenvalues[index] > 1e-12]
    coordinates = np.zeros((count, 2), dtype=np.float64)
    for axis, (index, value) in enumerate(positive[:2]):
        coordinates[:, axis] = eigenvectors[:, index] * np.sqrt(value)
    total = sum(value for _, value in positive) or 1.0
    ratios = [value / total for _, value in positive[:2]]
    return coordinates, ratios + [0.0] * (2 - len(ratios))


def _random_forest_plots(
    *,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    splits: dict[str, list[int]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = balanced_sample_indices(y, splits, MAX_PROXIMITY_SAMPLES)
    proximity = random_forest_proximity(model, x[selected])
    coordinates, ratios = classical_mds_from_proximity(proximity)
    sampled = len(selected) < len(y)
    scatter = _scatter(
        plot_id="random_forest_proximity_mds",
        title="随机森林 proximity-MDS",
        coordinates=coordinates,
        x_label=f"MDS 1（{ratios[0] * 100:.1f}%）",
        y_label=f"MDS 2（{ratios[1] * 100:.1f}%）",
        y=y,
        label_names=label_names,
        splits=splits,
        metadata=metadata,
        source_indices=selected,
        description="两样品落入同一叶节点的树占比定义为 proximity，再做经典多维尺度分析。",
    )
    plots: list[dict[str, Any]] = []
    if scatter:
        scatter.update({"sampled": sampled, "source_sample_count": int(len(y))})
        plots.append(scatter)
    if len(selected) < 2:
        return plots
    distances = np.sqrt(np.clip(1.0 - proximity, 0.0, 1.0))
    hierarchy = linkage(squareform(distances, checks=False), method="average")
    display_labels = [
        str(metadata[index].get("name") or metadata[index].get("sample_id") or f"S{index + 1}")
        for index in selected.tolist()
    ]
    rendered = dendrogram(hierarchy, labels=display_labels, no_plot=True)
    leaf_source_indices = [int(selected[position]) for position in rendered["leaves"]]
    plots.append(
        {
            "id": "random_forest_proximity_dendrogram",
            "type": "dendrogram",
            "title": "随机森林 proximity 层次聚类",
            "description": "基于 √(1-proximity) 的平均连接聚类；这是森林相似性树，不是单棵决策树。",
            "y_label": "√(1 − proximity)",
            "segments": [
                {
                    "x": [_finite_float(value) for value in xs],
                    "y": [_finite_float(value) for value in ys],
                }
                for xs, ys in zip(rendered["icoord"], rendered["dcoord"])
            ],
            "leaves": [
                {
                    "x": float(5 + position * 10),
                    "name": str(rendered["ivl"][position]),
                    "label": str(label_names[int(y[source_index])]),
                    "index": source_index,
                }
                for position, source_index in enumerate(leaf_source_indices)
            ],
            "sampled": sampled,
            "source_sample_count": int(len(y)),
        }
    )
    return plots


def _unsupported(model_type: str, evaluation_strategy: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_VISUALIZATION_SCHEMA,
        "status": "unsupported",
        "model_type": model_type,
        "evaluation_strategy": evaluation_strategy,
        "reason": reason,
        "plots": [],
    }


def build_model_feature_visualization(
    *,
    model_type: str,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    splits: dict[str, list[int]],
    metadata: list[dict[str, Any]],
    x_axis: list[float],
    evaluation_strategy: str,
    seed: int,
    selection_x: np.ndarray | None = None,
    permutation_count: int = PLS_PERMUTATION_COUNT,
) -> dict[str, Any]:
    """Build the optional model-feature-visualization artifact payload."""
    if evaluation_strategy == "leave_one_sample_id_cv":
        return _unsupported(
            model_type,
            evaluation_strategy,
            "交叉验证各折的低维坐标存在旋转、符号和尺度不唯一，首版不直接拼接，以免产生误导。",
        )
    values = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.int64)
    fit_indices = np.asarray(sorted({*splits.get("train", []), *splits.get("valid", [])}), dtype=np.int64)
    plots: list[dict[str, Any]] = []
    if model_type == "pls_da":
        pls = getattr(model, "model", None)
        scores = np.asarray(pls.transform(values), dtype=np.float64) if pls is not None else np.empty((0, 0))
        score_plot = _scatter(
            plot_id="pls_scores",
            title="PLS-DA 潜变量得分图",
            coordinates=scores,
            x_label="PLS latent variable 1",
            y_label="PLS latent variable 2",
            y=target,
            label_names=label_names,
            splits=splits,
            metadata=metadata,
            ellipse_fit_indices=fit_indices,
            description="坐标来自已拟合 PLS-DA 的前两个潜变量；颜色表示真实类别，点形表示数据分区。",
        )
        if score_plot:
            plots.append(score_plot)
        vip = _pls_vip(model, x_axis)
        if vip:
            plots.append(vip)
        permutation = build_pls_permutation_plot(
            x=np.asarray(selection_x if selection_x is not None else values, dtype=np.float64),
            y=target,
            splits=splits,
            class_count=len(label_names),
            n_components=int(getattr(model, "n_components", 2)),
            seed=seed,
            permutation_count=permutation_count,
        )
        if permutation:
            plots.append(permutation)
    elif model_type == "pca_lda":
        pca = getattr(model, "named_steps", {}).get("pca")
        scores = np.asarray(pca.transform(values), dtype=np.float64) if pca is not None else np.empty((0, 0))
        ratios = np.asarray(getattr(pca, "explained_variance_ratio_", []), dtype=np.float64)
        score_plot = _scatter(
            plot_id="pca_scores",
            title="PCA-LDA 的 PCA 得分图",
            coordinates=scores,
            x_label=f"PC1（{ratios[0] * 100:.1f}%）" if ratios.size > 0 else "PC1",
            y_label=f"PC2（{ratios[1] * 100:.1f}%）" if ratios.size > 1 else "PC2",
            y=target,
            label_names=label_names,
            splits=splits,
            metadata=metadata,
            ellipse_fit_indices=fit_indices,
            description="坐标来自分类 Pipeline 中实际拟合的 PCA 步骤；LDA 仍使用配置的全部 PCA 成分。",
        )
        if score_plot:
            plots.append(score_plot)
    elif model_type == "pca_mlp":
        mean = getattr(model, "pca_mean", None)
        components = getattr(model, "pca_components", None)
        if mean is not None and components is not None:
            mean_values = mean.detach().cpu().numpy()
            component_values = components.detach().cpu().numpy()
            scores = (values - mean_values) @ component_values.T
        else:
            scores = np.empty((0, 0))
        score_plot = _scatter(
            plot_id="pca_mlp_scores",
            title="PCA-MLP 输入得分图",
            coordinates=scores,
            x_label="PC1",
            y_label="PC2",
            y=target,
            label_names=label_names,
            splits=splits,
            metadata=metadata,
            ellipse_fit_indices=np.asarray(splits.get("train", []), dtype=np.int64),
            description="坐标来自 PCA-MLP 固化的训练折 PCA 输入层，不是 MLP 隐层的事后降维。",
        )
        if score_plot:
            plots.append(score_plot)
    elif model_type == "random_forest":
        plots.extend(
            _random_forest_plots(
                model=model,
                x=values,
                y=target,
                label_names=label_names,
                splits=splits,
                metadata=metadata,
            )
        )
    else:
        return _unsupported(
            model_type,
            evaluation_strategy,
            "该模型没有与示例图等价、可直接审计的二维潜变量或随机森林 proximity；未用通用 PCA/t-SNE 冒充模型原生解释。",
        )
    if not plots:
        return {
            "schema_version": FEATURE_VISUALIZATION_SCHEMA,
            "status": "unavailable",
            "model_type": model_type,
            "evaluation_strategy": evaluation_strategy,
            "reason": "当前拟合模型没有足够的有效成分或样品生成二维可视化。",
            "plots": [],
        }
    result = {
        "schema_version": FEATURE_VISUALIZATION_SCHEMA,
        "status": "ready",
        "model_type": model_type,
        "evaluation_strategy": evaluation_strategy,
        "scope": "holdout_fitted_model",
        "warning": "得分图用于描述模型表征，不单独证明类别差异或泛化能力；请结合 Test 指标与置换检验解读。",
        "plots": plots,
    }
    # Enforce strict JSON compatibility at the producer boundary.
    import json

    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
