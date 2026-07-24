"""PCA 降维后接 LDA 的传统分类 Pipeline 构造器。

模块定位：
    本文件实现传统机器学习模型 ``pca_lda``（主成分分析 + 线性
    判别分析）的工厂函数，是分类模型 v2 能力目录中 15 个目标
    模型之一。

系统协作：
    - 由 ``backend/app/models/registry.py`` 按训练配置调用
      ``build_pca_lda`` 构造，PCA 维度来自配置 ``pca_components``
      （缺省为 2）；
    - 输入特征来自预处理输出的 ``wide-feature-v2`` 宽表（公共轴
      对齐后的强度向量），外层训练流程负责训练集-only 的标准化
      与 train/valid/test 划分。

算法原理：
    PCA 先把高维光谱/色谱特征投影到方差最大的少数主成分方向
    （无监督降维，去共线性、压缩噪声），LDA 再在低维空间中寻找
    类别可分性最强的线性判别方向并给出分类决策。两者串联是
    高维小样本光谱分类的经典组合：PCA 缓解“特征数 >> 样本数”
    导致 LDA 协方差矩阵奇异/过拟合的问题。

关键设计约束：
    - 用 sklearn ``Pipeline`` 串联两步，使交叉验证/网格搜索时
      PCA 只在每折训练子集上拟合，避免数据泄漏；
    - Pipeline 自带 ``predict`` / ``predict_proba``（委托给末级
      LDA），满足下游评估对概率输出的统一要求。
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def build_pca_lda(n_components: int = 2) -> Pipeline:
    """构造 PCA 与 LDA 串联的 sklearn Pipeline。

    参数：
        n_components: PCA 保留的主成分数，来自训练配置
            ``pca_components``。``max(1, int(...))`` 防御 0、负数
            或浮点字符串等非法取值，保证至少保留 1 个主成分。
            注意主成分数同时还受 ``min(n_samples, n_features)``
            限制，超出时由 sklearn 在 fit 阶段处理。

    返回：
        两步 Pipeline：``("pca", PCA(...))`` 负责无监督降维，
        ``("lda", LinearDiscriminantAnalysis())`` 用默认参数
        （solver="svd"，对奇异协方差稳健）负责线性分类。
        实例尚未拟合，由外层训练流程在训练集上 fit。
    """
    return Pipeline(
        [
            ("pca", PCA(n_components=max(1, int(n_components)))),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
