"""用 one-hot PLSRegression 响应实现 PLS-DA 分类概率接口。

模块定位：
    本文件实现传统机器学习模型 ``pls_da``（Partial Least Squares
    Discriminant Analysis，偏最小二乘判别分析）的分类器封装，
    是分类模型 v2 能力目录中 15 个目标模型之一。

系统协作：
    - 由 ``backend/app/models/registry.py`` 中的模型注册表通过
      ``build_pls_da`` 工厂函数构造（潜变量数取自训练配置的
      ``pls_components``，缺省为 2）；
    - 训练数据来自预处理阶段输出的 ``wide-feature-v2`` 宽表：
      特征为真实递增坐标对应的强度向量，标签即使为数字也按
      类别名编码（当前仅支持分类，不支持回归）；
    - 下游训练/评估流程统一以 sklearn 风格的 ``fit`` /
      ``predict`` / ``predict_proba`` 接口驱动模型，因此本类必须
      伪装出与 sklearn 分类器一致的调用契约。

关键设计约束：
    - sklearn 的 ``PLSRegression`` 本质上是回归器，只能输出连续
      响应；PLS-DA 的惯用做法是把类别标签 one-hot 化后做多输出
      回归，再把连续响应“翻译”回类别与概率，本类正是这个翻译层；
    - 概率输出没有严格的统计校准（不是真实后验），只是把响应
      做平移 + softmax 归一化，用于满足下游评估对
      ``predict_proba`` 的统一需求（如 Log-loss、阈值无关指标）；
    - 所有标准化参数只由训练集拟合，这一约束由外层训练流程
      保证，本类内部不做任何特征缩放（``scale=False``）。
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


class PLSDAClassifier:
    """把 PLS 连续响应归一化成 sklearn 风格分类概率。

    原理说明：
        PLS-DA = PLS 回归 + 判别后处理。训练时把整数类别标签
        展开成 one-hot 矩阵 Y（形状为 [n_samples, n_classes]），
        用 ``PLSRegression`` 在 X 与 Y 之间提取少量潜变量
        （latent components）做多元回归；预测时得到每个样本在
        各个类别上的连续响应值，响应最大的类别即为预测类别，
        响应经 softmax 归一化后作为“概率”输出。

    为什么不用 sklearn 现成的分类器：
        sklearn 没有内置 PLS-DA，只有 ``PLSRegression``；社区惯例
        就是自行包一层 one-hot + argmax/softmax，本类保持这一惯例，
        同时补齐 sklearn 分类器接口（``classes_``、``predict_proba``），
        使训练主流程可以无差别地把它当作普通分类器使用。
    """

    def __init__(self, n_components: int = 2) -> None:
        """初始化 PLS-DA 分类器。

        参数：
            n_components: PLS 潜变量（主成分）数量。PLS 的潜变量
                同时考虑 X 的方差与 X-Y 协方差，数量越大拟合能力
                越强但越容易过拟合；实际取值来自训练配置的
                ``pls_components``，由 registry 注入。

        边界处理：
            ``max(1, int(n_components))`` 防御非法取值（0、负数、
            浮点字符串等），保证至少保留 1 个潜变量，避免 sklearn
            内部抛出难以定位的错误。
        """
        self.n_components = max(1, int(n_components))
        # scale=False：特征标准化由外层训练流程统一完成（只用训练集
        # 拟合 scaler 再应用到 valid/test），这里不能再缩放，否则
        # 会破坏“标准化参数只来自训练集”的评估约束并造成数据泄漏。
        # max_iter/tol 收紧收敛条件，换取更稳定的潜变量解。
        self.model = PLSRegression(n_components=self.n_components, scale=False, max_iter=500, tol=1e-6)
        # 拟合前为 None，用于在未 fit 时给出明确报错（见 predict）。
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        """拟合 PLS-DA 模型。

        参数：
            x: 形状 [n_samples, n_features] 的特征矩阵（宽表中的
               强度特征，已按公共轴对齐）。
            y: 形状 [n_samples] 的整数类别编码（由外层把类别名
               编码为整数后传入）。

        返回：
            self，遵循 sklearn 的链式调用惯例（fit 返回自身）。

        逻辑说明：
            1. 把 y 强制为 int64，并用 ``np.unique`` 得到升序类别表
               ``classes_``——one-hot 列与类别的一一对应关系完全由
               这张表决定，predict 时再据此把列下标翻回类别值；
            2. 手工构造 float32 one-hot 矩阵（比引入 LabelBinarizer
               更直接，且类别表顺序可控）；
            3. 用 one-hot 矩阵作为多输出回归目标拟合 PLS。
        """
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("PLS-DA 至少需要 2 个样本和 1 个特征")
        if not np.isfinite(values).all():
            raise ValueError("PLS-DA 输入特征必须全部为有限数值")
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1 or y.shape[0] != values.shape[0]:
            raise ValueError("PLS-DA 的标签长度必须与样本数一致")
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("PLS-DA 至少需要两个类别")
        y_one_hot = np.zeros((len(y), len(self.classes_)), dtype=np.float32)
        for col, class_id in enumerate(self.classes_):
            # 第 col 列在“真实类别为 class_id”的样本行上置 1，其余为 0
            y_one_hot[y == class_id, col] = 1.0
        # one-hot 响应经过中心化后最多只有 C-1 个独立方向。若仍把候选
        # 分量数直接交给 PLSRegression，二分类的小训练折会在已耗尽的 Y
        # 残差上继续迭代，SciPy SVD 可能出现 NaN。将有效分量裁剪到
        # min(N-1, P, C-1) 是 PLS-DA 的可辨识秩，不是模型替代或降级。
        effective_components = min(
            self.n_components,
            values.shape[0] - 1,
            values.shape[1],
            self.classes_.size - 1,
        )
        if effective_components < 1:
            raise ValueError("PLS-DA 当前训练折没有可辨识的潜变量")
        self.effective_n_components_ = int(effective_components)
        self.model = PLSRegression(
            n_components=self.effective_n_components_,
            scale=False,
            max_iter=500,
            tol=1e-6,
        )
        # 特征转 float32 与 one-hot 目标 dtype 对齐，避免 sklearn 内部
        # 重复拷贝和类型提升带来的额外内存开销。
        self.model.fit(values, y_one_hot)
        return self

    def _responses(self, x: np.ndarray) -> np.ndarray:
        """计算每个样本在各类别上的 PLS 连续响应（内部辅助）。

        返回形状固定为 [n_samples, n_classes] 的 float64 矩阵：
        - 转 float64 是为了后续 softmax 数值计算更稳定；
        - 单列（二分类时 sklearn 可能返回 1 维或形状 [n, 1]）时
          显式 reshape 成二维，保证 ``argmax(axis=1)`` 等下游
          操作不会因维度问题出错。
        """
        values = np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测类别标签（返回类别值本身，不是列下标）。

        异常：
            未调用 ``fit`` 时 ``classes_`` 为 None，抛出 ValueError，
            防止用未训练模型静默产出无意义预测。

        逻辑：
            对连续响应按行取 argmax 得到获胜类别所在的列下标，
            再用 ``classes_`` 映射回原始类别编码值。
        """
        if self.classes_ is None:
            raise ValueError("PLS-DA model is not fitted")
        responses = self._responses(x)
        return self.classes_[np.argmax(responses, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """输出各类别的归一化“概率”（softmax 后的 PLS 响应）。

        注意（设计意图）：
            这不是统计意义上校准过的后验概率，只是为了让 PLS-DA
            能接入依赖 ``predict_proba`` 的统一评估流程（例如
            Log-loss 与可解释性中的遮挡 Log-loss 增量分析）。

        数值稳定性处理：
            1. 每行减去该行最大值再取 exp——softmax 的标准防溢出
               技巧，不改变归一化结果；
            2. 分母用 ``np.maximum(..., 1e-12)`` 兜底，防止极端输入
               下行和为 0 导致除零产生 NaN。
        """
        responses = self._responses(x)
        responses = responses - responses.max(axis=1, keepdims=True)
        exp = np.exp(responses)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def build_pls_da(n_components: int = 2) -> PLSDAClassifier:
    """构造指定潜变量数量的 PLS-DA 分类器。

    工厂函数存在的意义：模型注册表（registry）按模型名统一调用
    ``build_*`` 系列构造函数，本函数保持该命名契约，让上层无需
    关心具体类名与构造细节。

    参数：
        n_components: PLS 潜变量数，来自训练配置 ``pls_components``。
    """
    return PLSDAClassifier(n_components=n_components)
