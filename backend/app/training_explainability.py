from __future__ import annotations


EXPLAINABILITY_METHOD_BY_MODEL = {
    "pls_da": "window_permutation",
    "pca_lda": "window_permutation",
    "logistic_regression": "window_permutation",
    "svm": "window_permutation",
    "random_forest": "window_permutation",
    "xgboost": "window_permutation",
    "pca_mlp": "input_gradient_attribution",
    "cnn1d": "gradcam_1d",
    "cnn1d_se": "gradcam_1d",
    "transformer1d": "input_gradient_attribution",
    "resnet1d": "gradcam_1d",
    "inception1d": "gradcam_1d",
    "tcn1d": "gradcam_1d",
    "cnn_transformer1d": "input_gradient_attribution",
    "cnn_mamba1d": "input_gradient_attribution",
}


def explainability_method(model_type: str, *, dscarnet_mode: str = "dual") -> str:
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
