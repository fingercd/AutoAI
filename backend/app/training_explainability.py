"""模型 ID 到解释性算法的单一映射表。

训练器、模型目录和 artifact 元数据都应引用这里，避免同一模型在文档、前端和
后端被分配不同解释方法。DSCARNet 的方法名还编码实际启用的 SAR/CAR 分支。
"""

from __future__ import annotations


EXPLAINABILITY_METHOD_BY_MODEL = {
    "pls_da": "window_occlusion_log_loss",
    "pca_lda": "window_occlusion_log_loss",
    "logistic_regression": "window_occlusion_log_loss",
    "svm": "window_occlusion_log_loss",
    "random_forest": "window_occlusion_log_loss",
    "xgboost": "window_occlusion_log_loss",
    "pca_mlp": "window_occlusion_log_loss",
    "cnn1d": "gradcam_1d",
    "cnn1d_se": "gradcam_1d",
    "transformer1d": "window_occlusion_log_loss",
    "resnet1d": "gradcam_1d",
    "inception1d": "gradcam_1d",
    "tcn1d": "gradcam_1d",
    "cnn_transformer1d": "window_occlusion_log_loss",
    "cnn_mamba1d": "window_occlusion_log_loss",
}


def explainability_method(model_type: str, *, dscarnet_mode: str = "dual") -> str:
    """返回规范模型的解释方法；未知模型显式标记 unsupported。"""
    model_key = str(model_type or "").strip().lower()
    model_key = {"transformer": "cnn_transformer1d", "transformer1d": "cnn_transformer1d"}.get(model_key, model_key)
    if model_key == "dscarnet":
        mode = str(dscarnet_mode or "dual").strip().lower()
        methods = {
            "sar": "dscarnet_sar_2d_gradcam",
            "car": "dscarnet_car_2d_gradcam",
            "dual": "dscarnet_dual_2d_gradcam",
        }
        if mode not in methods:
            raise ValueError(f"未知 DSCARNet 解释模式: {dscarnet_mode}")
        return methods[mode]
    try:
        return EXPLAINABILITY_METHOD_BY_MODEL[model_key]
    except KeyError as exc:
        raise ValueError(f"未知模型，无法选择解释方法: {model_type}") from exc
