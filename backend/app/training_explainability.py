# =============================================================================
# 模块说明：模型可解释性方法的"单一事实来源"（single source of truth）
# =============================================================================
# 本文件只维护一件事：每个模型 ID 对应哪一种可解释性算法。
#
# 在系统中的位置：
#   - 训练流水线在生成 feature_importance / Grad-CAM 等 artifact 之前，会通过
#     explainability_method() 查询应使用哪种解释算法；
#   - 模型能力目录（capabilities）和 artifact 元数据（Manifest）也引用这里的方法名，
#     保证前端展示、后端执行、文档描述三者不会各说各话。
#
# 解释方法分两大类（与 AGENTS.md 的建模/可解释性规则一致）：
#   1. window_occlusion_log_loss：窗口遮挡法。把输入光谱/色谱的一段窗口遮掉，
#      比较遮挡前后真实类别的 log-loss 增量（log(p_before / p_after)），
#      增量越大说明该窗口对真实类别的置信度越重要。用于传统机器学习模型
#      以及没有卷积结构可挂 Grad-CAM 的深度模型（如 pca_mlp、cnn_transformer1d）。
#   2. gradcam_1d：1D Grad-CAM / Grad-CAM-like，利用卷积层的梯度与激活图
#      生成特征重要性热图，用于 cnn1d / resnet1d 等 1D 卷积类模型，
#      并保留输入梯度 sanity check。
#
# 历史只读兼容：DSCARNet 已退役，旧结果的方法名称使用 AggMap/PCA 的 SAR/CAR 双通路 2D 映射，
# 方法名必须编码当前实际启用的分支（sar / car / dual），因此在
# explainability_method() 中按 dscarnet_mode 动态生成。
#
# 设计约束：新增模型时必须在这里登记；未登记的模型显式抛错（fail-fast），
# 禁止训练器"猜"一个解释方法静默兜底。
"""模型 ID 到解释性算法的单一映射表。

训练器、模型目录和 artifact 元数据都应引用这里，避免同一模型在文档、前端和
后端被分配不同解释方法。DSCARNet 的方法名还编码实际启用的 SAR/CAR 分支。
"""

from __future__ import annotations


# 静态映射表：模型 ID（canonical key） -> 解释方法名。
# 六个传统机器学习模型（pls_da / pca_lda / logistic_regression / svm /
# random_forest / xgboost）以及 pca_mlp、cnn_transformer1d、cnn_mamba1d
# 统一走窗口遮挡法；纯 1D 卷积家族（cnn1d、cnn1d_se、resnet1d、inception1d、
# tcn1d）走 gradcam_1d。
# 注意：cnn_mamba1d 当前因 mamba-ssm 依赖不可用而不可训练，但仍在目录中登记，
# 以便其能力条目 available=false 时也能给出稳定的解释方法名。
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
    # "transformer1d" 是 "cnn_transformer1d" 的兼容别名，两者都指向遮挡法
    "transformer1d": "window_occlusion_log_loss",
    "resnet1d": "gradcam_1d",
    "inception1d": "gradcam_1d",
    "tcn1d": "gradcam_1d",
    "cnn_transformer1d": "window_occlusion_log_loss",
    "cnn_mamba1d": "window_occlusion_log_loss",
}


def explainability_method(model_type: str, *, dscarnet_mode: str = "dual") -> str:
    """返回规范模型的解释方法；未知模型显式标记 unsupported。"""
    # 归一化输入：容忍大小写和首尾空白差异，避免 "CNN1D" / " cnn1d " 这类
    # 调用方写法导致查表失败；同时把历史别名 transformer / transformer1d
    # 折叠到规范名 cnn_transformer1d，保证别名与正名行为完全一致。
    model_key = str(model_type or "").strip().lower()
    model_key = {"transformer": "cnn_transformer1d", "transformer1d": "cnn_transformer1d"}.get(model_key, model_key)
    if model_key in ("dscarnet", "dscar_net"):
        # 仅解释旧产物的方法标签，不提供训练或归因执行： AggMap/PCA 的 SAR/CAR 2D 映射，
        # 方法名必须反映实际启用的通路分支，因此不在静态表中，按 mode 动态返回。
        mode = str(dscarnet_mode or "dual").strip().lower()
        methods = {
            "sar": "dscarnet_sar_2d_gradcam",
            "car": "dscarnet_car_2d_gradcam",
            "dual": "dscarnet_dual_2d_gradcam",
        }
        if mode not in methods:
            # 模式拼写非法时同样 fail-fast，避免静默回退到错误的通路。
            raise ValueError(f"未知 DSCARNet 解释模式: {dscarnet_mode}")
        return methods[mode]
    try:
        # 普通模型直接查静态映射表。
        return EXPLAINABILITY_METHOD_BY_MODEL[model_key]
    except KeyError as exc:
        # fail-fast：未登记的模型不允许训练器自行猜测解释方法，
        # 抛 ValueError 并把原始 KeyError 链为 __cause__ 便于排查。
        raise ValueError(f"未知模型，无法选择解释方法: {model_type}") from exc
