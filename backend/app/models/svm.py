"""带概率输出的 SVM 分类器构造器。

模块定位：
    本文件实现传统机器学习模型 ``svm``（支持向量机，SVC）的
    工厂函数，是分类模型 v2 能力目录中 15 个目标模型之一。

系统协作：
    - 由 ``backend/app/models/registry.py`` 按训练配置调用
      ``build_svm`` 构造，超参数来自训练配置（``svm_c``、
      ``svm_gamma``、``svm_kernel``、``seed`` 与类别权重）；
    - 输入特征来自预处理输出的 ``wide-feature-v2`` 宽表，外层
      训练流程会先用仅由训练集拟合的标准化器处理特征，再送入
      本模型训练（SVM 对特征量纲极其敏感，这一步在流水线外层
      统一完成，本文件不涉及）。

关键设计约束：
    - 下游评估依赖 ``predict_proba``（Log-loss、可解释性分析），
      因此必须打开 SVC 的概率校准（``probability=True``）；
    - kernel 只允许 ``linear`` / ``rbf``，非法取值静默回退 rbf，
      防止非法配置直接打断训练任务。
"""

from __future__ import annotations

from sklearn.svm import SVC


def build_svm(
    c: float = 1.0,
    gamma: str | float = 0.03,
    class_weight: str | None = None,
    seed: int = 42,
    kernel: str = "rbf",
) -> SVC:
    """构造带概率校准的 linear/rbf SVC，并防御非法 kernel。

    参数：
        c: 正则化强度参数 C（大写 C 是 SVM 惯例）。C 越大对错分
           惩罚越重，间隔越窄、越趋向过拟合；越小则间隔越宽、
           模型越平滑。来自训练配置 ``svm_c``。
        gamma: RBF 核系数，控制单个样本的影响范围（"scale"/"auto"
           或正浮点数）。默认 0.03 是按本项目光谱特征维度与标准化
           后量级选定的经验值。来自训练配置 ``svm_gamma``。
        class_weight: 类别权重，通常为 None 或 "balanced"；后者按
           类别频率反比加权，用于缓解类别不平衡（每类至少 3 个
           Sample_ID 的划分约束由外层评估流程保证）。
        seed: 随机种子，传入 ``random_state``。SVC 的
           ``probability=True`` 内部使用 Platt scaling（带交叉验证
           的逻辑回归校准），该过程有随机性；固定种子保证同一
           配置下训练结果可复现。
        kernel: 核函数，仅允许 "linear"（线性核）或 "rbf"（径向基
           核，默认）。

    返回：
        配置完成的 sklearn ``SVC`` 实例（尚未拟合）。

    防御逻辑（为什么这么做）：
        - ``gamma`` 允许以字符串形式从配置传入；字符串若不是
          sklearn 内置的 "scale"/"auto"，则尝试转成浮点核系数，
          兼容配置层把 0.03 序列化成文本的情况；
        - kernel 白名单之外的取值（拼写错误、旧配置遗留等）静默
          回退 "rbf"，让训练任务以默认核继续，而不是在 SVC 内部
          抛出 ValueError 中断整个 Run。

    注意：
        ``probability=True`` 会显著增加训练开销（内部多跑一次
        交叉验证校准），但这是评估/可解释性流程对概率输出的硬
        要求，不能关闭。
    """
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        gamma = float(gamma)
    safe_kernel = kernel if kernel in {"linear", "rbf"} else "rbf"
    return SVC(C=c, gamma=gamma, kernel=safe_kernel, probability=True, class_weight=class_weight, random_state=seed)
