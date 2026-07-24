"""将一维光谱拟合并转换为 DSCARNet 的 SAR/CAR 二维输入。

SAR 使用 AggMap 对原始特征建立二维布局；CAR 先在当前训练折拟合 PCA，再对主成分
建立 AggMap。所有映射对象只由 train 数据拟合并复用于 valid/test，避免信息泄漏。
模块同时隔离 AggMap 1.2.1 的旧依赖兼容问题，并把 joblib 映射对象作为私有产物。
"""

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
    """一折 train/valid/test 的映射张量、拟合对象与可审计元数据。"""
    x_sar: np.ndarray | None
    x_car: np.ndarray | None
    pca: PCA | None
    sar_mapper: Any | None
    car_mapper: Any | None
    metadata: dict[str, Any]

    @property
    def model_input_shape_sar(self) -> tuple[int, int, int]:
        if self.x_sar is None:
            raise ValueError("当前 DSCARNet 模式不包含 SAR 输入")
        channels, height, width = self.x_sar.shape[1:]
        return int(height), int(width), int(channels)

    @property
    def model_input_shape_car(self) -> tuple[int, int, int]:
        if self.x_car is None:
            raise ValueError("当前 DSCARNet 模式不包含 CAR 输入")
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
    """用 SciPy 线性分配模拟 AggMap 所需的 lapjv 返回格式。"""
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
    mode: str = "dual",
    cancel_check: Any | None = None,
    progress_callback: Any | None = None,
) -> DSCARNetMappedInputs:
    """仅在 train 上拟合 SAR/CAR 映射，再转换三组输入为 NCHW 张量。"""
    def checkpoint(stage: str, label: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, label)
        if cancel_check is not None:
            cancel_check()

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("DSCARNet 需要非空二维特征矩阵")
    train_indices = [int(idx) for idx in train_indices if 0 <= int(idx) < x.shape[0]]
    if len(train_indices) < 2:
        raise ValueError("DSCARNet AggMap/PCA 至少需要 2 个训练样品")

    mode = str(mode or "dual").strip().lower()
    if mode not in {"sar", "car", "dual"}:
        raise ValueError("dscarnet mode 必须是 sar、car 或 dual")
    aggmap_factory = aggmap_factory or _load_aggmap_class()
    n_components = max(1, min(int(pca_components), int(x.shape[1]), int(len(train_indices) - 1)))

    feature_columns = _feature_columns(x.shape[1])
    sar_mapper = None
    x_sar = None
    if mode in {"sar", "dual"}:
        checkpoint("dscarnet_sar_layout", "正在构建 DSCARNet SAR 二维布局")
        sar_mapper = _fit_mapper(x[train_indices], feature_columns, cluster_channels=cluster_channels, aggmap_factory=aggmap_factory)
        checkpoint("dscarnet_sar_transform", "正在转换 DSCARNet SAR 输入")
        x_sar = _nhwc_to_nchw(sar_mapper.batch_transform(x, scale_method="minmax", n_jobs=1))
        checkpoint("dscarnet_sar_ready", "DSCARNet SAR 映射已完成")

    pca = None
    car_mapper = None
    x_car = None
    component_columns: list[str] = []
    if mode in {"car", "dual"}:
        checkpoint("dscarnet_pca", "正在拟合 DSCARNet CAR 主成分")
        pca = PCA(n_components=n_components, random_state=int(seed))
        train_pca = pca.fit_transform(x[train_indices])
        all_pca = pca.transform(x).astype(np.float32)
        component_columns = _component_columns(n_components)
        checkpoint("dscarnet_car_layout", "正在构建 DSCARNet CAR 二维布局")
        car_mapper = _fit_mapper(train_pca.astype(np.float32), component_columns, cluster_channels=cluster_channels, aggmap_factory=aggmap_factory)
        checkpoint("dscarnet_car_transform", "正在转换 DSCARNet CAR 输入")
        x_car = _nhwc_to_nchw(car_mapper.batch_transform(all_pca, scale_method="minmax", n_jobs=1))
        checkpoint("dscarnet_car_ready", "DSCARNet CAR 映射已完成")

    metadata = {
        "method": f"dscarnet_aggmap_{mode}",
        "mode": mode,
        "schema_version": 2,
        "active_branches": [item for item in ("sar", "car") if mode == "dual" or mode == item],
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
        "input_shape_sar": None if x_sar is None else [int(item) for item in x_sar.shape[1:]],
        "input_shape_car": None if x_car is None else [int(item) for item in x_car.shape[1:]],
        "model_input_shape_sar": None if x_sar is None else [int(item) for item in (x_sar.shape[2], x_sar.shape[3], x_sar.shape[1])],
        "model_input_shape_car": None if x_car is None else [int(item) for item in (x_car.shape[2], x_car.shape[3], x_car.shape[1])],
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
    """保存公开映射元数据和私有 PCA/AggMap joblib 对象。"""
    run_path = Path(run_dir)
    writer = RunArtifactWriter(run_path)
    metadata = dict(mapped.metadata)
    artifact_values = (
        ("dscarnet_pca.joblib", mapped.pca, "pca_artifact"),
        ("dscarnet_sar_aggmap.joblib", mapped.sar_mapper, "sar_mapper_artifact"),
        ("dscarnet_car_aggmap.joblib", mapped.car_mapper, "car_mapper_artifact"),
    )
    for name, value, metadata_key in artifact_values:
        if value is not None:
            metadata[metadata_key] = name
    writer.write_json("dscarnet_mapping.json", metadata)
    for name, value, _metadata_key in artifact_values:
        if value is None:
            continue
        buffer = io.BytesIO()
        joblib.dump(value, buffer)
        writer.write_private_bytes(name, buffer.getvalue())
    return metadata
