"""稳定数据集 ID 与 SQLite 数据集仓库的公共导出。

【模块职责】
这是 datasets 子包的包入口：只负责把 repository.py 中的两个公共符号
（DatasetRecord 数据类、DatasetRepository 仓库类）重新导出，
让上层可以用 ``from ..datasets import DatasetRepository`` 的简短形式，
而不必关心内部文件布局。实现细节全部在 repository.py。
"""

from .repository import DatasetRecord, DatasetRepository

# __all__ 显式声明包的公共 API 边界：只有这两个类对外，
# 内部辅助函数（如 _sha256、_is_within）不被星号导入泄露。
__all__ = ['DatasetRecord', 'DatasetRepository']
