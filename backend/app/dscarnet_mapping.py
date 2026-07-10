from __future__ import annotations

import io
import sys
import types
import collections
import collections.abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from .runs.artifacts import RunArtifactWriter


DSCAR_SOURCE_URL = "https://github.com/songlinlu/DSCAR"
DSCAR_DUAL_SOURCE_URL = "https://raw.githubusercontent.com/songlinlu/DSCAR/main/utils/dscarnet.py"
AGGMAP_DOCS_URL = "https://bidd-aggmap.readthedocs.io/en/latest/api.html"
_LAPJV_BACKEND = "lapjv"


@dataclass
class DSCARNetMappedInputs:
    x_sar: np.ndarray
    x_car: np.ndarray
    pca: PCA
    sar_mapper: Any
    car_mapper: Any
    metadata: dict[str, Any]

    @property
    def model_input_shape_sar(self) -> tuple[int, int, int]:
        channels, height, width = self.x_sar.shape[1:]
        return int(height), int(width), int(channels)

    @property
    def model_input_shape_car(self) -> tuple[int, int, int]:
        channels, height, width = self.x_car.shape[1:]
        return int(height), int(width), int(channels)


def _load_aggmap_class() -> Any:
    global _LAPJV_BACKEND
    _install_legacy_collections_aliases()
    try:
        from aggmap import AggMap
    except ModuleNotFoundError as exc:
        if exc.name == "lapjv":
            _install_lapjv_compat_module()
            _LAPJV_BACKEND = "scipy_linear_sum_assignment"
            for module_name in ["aggmap", "aggmap.map", "aggmap.utils.matrixopt"]:
                sys.modules.pop(module_name, None)
            from aggmap import AggMap
            _patch_aggmap_pairwise_distance()
            return AggMap
        raise ImportError(
            "DSCARNet 需要安装并能导入 aggmap。当前环境导入失败；"
            "可先尝试 python -m pip install aggmap==1.2.1 --no-deps colorlog colored "
            "umap-learn python-highcharts future pynndescent"
        ) from exc
    except Exception as exc:  # pragma: no cover - exact dependency failure varies by environment
        raise ImportError(
            "DSCARNet 需要安装并能导入 aggmap。当前环境导入失败；"
            "可先尝试 python -m pip install aggmap==1.2.1 --no-deps colorlog colored "
            "umap-learn python-highcharts future pynndescent"
        ) from exc
    _patch_aggmap_pairwise_distance()
    return AggMap


def _install_legacy_collections_aliases() -> None:
    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def scipy_lapjv_compat(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2D")
    row_indices, col_indices = linear_sum_assignment(cost)
    row_assign = np.full(cost.shape[0], -1, dtype=np.int32)
    col_assign = np.full(cost.shape[1], -1, dtype=np.int32)
    row_assign[row_indices] = col_indices
    col_assign[col_indices] = row_indices
    total_cost = float(cost[row_indices, col_indices].sum())
    return row_assign, col_assign, total_cost


def _install_lapjv_compat_module() -> None:
    if "lapjv" in sys.modules:
        return
    module = types.ModuleType("lapjv")
    module.lapjv = scipy_lapjv_compat
    sys.modules["lapjv"] = module


def _serial_pairwise_distance(npydata: np.ndarray, n_cpus: int = 1, method: str = "correlation") -> np.ndarray:
    values = np.asarray(npydata, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("npydata must be 2D")
    feature_count = int(values.shape[1])
    if feature_count <= 1:
        return np.zeros((feature_count, feature_count), dtype=np.float64)
    distances = squareform(pdist(values.T, metric=method))
    return np.nan_to_num(distances, copy=False).clip(0, np.inf)


def _patch_aggmap_pairwise_distance() -> None:
    try:
        import aggmap.utils.calculator as calculator
    except Exception:
        return
    calculator.pairwise_distance = _serial_pairwise_distance


def _feature_columns(n_features: int) -> list[str]:
    return [f"f_{idx}" for idx in range(int(n_features))]


def _component_columns(n_components: int) -> list[str]:
    return [f"pc_{idx}" for idx in range(int(n_components))]


def _fit_mapper(
    values: np.ndarray,
    columns: list[str],
    *,
    cluster_channels: int,
    aggmap_factory: Any,
) -> Any:
    frame = pd.DataFrame(values, columns=columns)
    try:
        mapper = aggmap_factory(frame, metric="euclidean", n_cpus=1)
    except TypeError:
        mapper = aggmap_factory(frame, metric="euclidean")
    return mapper.fit(cluster_channels=int(cluster_channels), verbose=0)


def _ensure_min_hw_nhwc(values: np.ndarray, min_size: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"AggMap batch_transform 应返回 NHWC 3D/4D 数组，实际 shape={arr.shape}")
    n, height, width, channels = arr.shape
    target_h = max(int(height), int(min_size))
    target_w = max(int(width), int(min_size))
    if target_h == height and target_w == width:
        return arr
    padded = np.zeros((n, target_h, target_w, channels), dtype=np.float32)
    padded[:, :height, :width, :] = arr
    return padded


def _nhwc_to_nchw(values: np.ndarray) -> np.ndarray:
    arr = _ensure_min_hw_nhwc(values)
    return np.moveaxis(arr, -1, 1).astype(np.float32, copy=False)


def fit_dscarnet_2d_mapping(
    x: np.ndarray,
    train_indices: list[int],
    *,
    pca_components: int = 30,
    cluster_channels: int = 9,
    seed: int = 42,
    aggmap_factory: Any | None = None,
) -> DSCARNetMappedInputs:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("DSCARNet 需要非空二维特征矩阵")
    train_indices = [int(idx) for idx in train_indices if 0 <= int(idx) < x.shape[0]]
    if len(train_indices) < 2:
        raise ValueError("DSCARNet AggMap/PCA 至少需要 2 个训练样品")

    aggmap_factory = aggmap_factory or _load_aggmap_class()
    n_components = max(1, min(int(pca_components), int(x.shape[1]), int(len(train_indices) - 1)))

    feature_columns = _feature_columns(x.shape[1])
    sar_mapper = _fit_mapper(
        x[train_indices],
        feature_columns,
        cluster_channels=cluster_channels,
        aggmap_factory=aggmap_factory,
    )
    x_sar = _nhwc_to_nchw(sar_mapper.batch_transform(x, scale_method="minmax", n_jobs=1))

    pca = PCA(n_components=n_components, random_state=int(seed))
    train_pca = pca.fit_transform(x[train_indices])
    all_pca = pca.transform(x).astype(np.float32)
    component_columns = _component_columns(n_components)
    car_mapper = _fit_mapper(
        train_pca.astype(np.float32),
        component_columns,
        cluster_channels=cluster_channels,
        aggmap_factory=aggmap_factory,
    )
    x_car = _nhwc_to_nchw(car_mapper.batch_transform(all_pca, scale_method="minmax", n_jobs=1))

    metadata = {
        "method": "dscarnet_aggmap_sar_car",
        "fit_scope": "train",
        "source_url": DSCAR_SOURCE_URL,
        "dual_dscarnet_source_url": DSCAR_DUAL_SOURCE_URL,
        "aggmap_docs_url": AGGMAP_DOCS_URL,
        "metric": "euclidean",
        "distance_backend": "scipy_pdist_serial",
        "scale_method": "minmax",
        "lapjv_backend": _LAPJV_BACKEND,
        "cluster_channels": int(cluster_channels),
        "pca_components": int(n_components),
        "feature_count": int(x.shape[1]),
        "sample_count": int(x.shape[0]),
        "train_sample_count": int(len(train_indices)),
        "feature_columns": feature_columns,
        "component_columns": component_columns,
        "input_shape_sar": [int(item) for item in x_sar.shape[1:]],
        "input_shape_car": [int(item) for item in x_car.shape[1:]],
        "model_input_shape_sar": [int(item) for item in (x_sar.shape[2], x_sar.shape[3], x_sar.shape[1])],
        "model_input_shape_car": [int(item) for item in (x_car.shape[2], x_car.shape[3], x_car.shape[1])],
    }
    return DSCARNetMappedInputs(
        x_sar=x_sar,
        x_car=x_car,
        pca=pca,
        sar_mapper=sar_mapper,
        car_mapper=car_mapper,
        metadata=metadata,
    )


def save_dscarnet_mapping_artifacts(run_dir: str | Path, mapped: DSCARNetMappedInputs) -> dict[str, Any]:
    run_path = Path(run_dir)
    writer = RunArtifactWriter(run_path)
    metadata = dict(mapped.metadata)
    metadata.update(
        {
            "pca_artifact": "dscarnet_pca.joblib",
            "sar_mapper_artifact": "dscarnet_sar_aggmap.joblib",
            "car_mapper_artifact": "dscarnet_car_aggmap.joblib",
        }
    )
    writer.write_json("dscarnet_mapping.json", metadata)
    for name, value in (
        ("dscarnet_pca.joblib", mapped.pca),
        ("dscarnet_sar_aggmap.joblib", mapped.sar_mapper),
        ("dscarnet_car_aggmap.joblib", mapped.car_mapper),
    ):
        buffer = io.BytesIO()
        joblib.dump(value, buffer)
        writer.write_private_bytes(name, buffer.getvalue())
    return metadata
