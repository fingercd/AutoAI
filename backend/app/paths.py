"""项目路径常量与本地存储目录初始化。

所有路径都从当前文件位置推导，避免把开发机盘符写入业务逻辑。这里仅创建目录，
不会创建数据库表；数据库 schema 分别由 DatasetRepository 和 RunRepository 管理。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage"
RUNS_DATABASE = STORAGE_DIR / "runs.sqlite3"
DATASETS_DATABASE = STORAGE_DIR / "datasets.sqlite3"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RUNS_DIR = STORAGE_DIR / "runs"
PREPROCESSED_DIR = STORAGE_DIR / "preprocessed"
STATIC_DIR = PROJECT_ROOT / "static"
DEFAULT_DATA = PROJECT_ROOT / "data.csv"


def ensure_storage() -> None:
    """幂等创建 Web、预处理和训练流程需要的本地目录。"""
    for path in (STORAGE_DIR, UPLOADS_DIR, RUNS_DIR, PREPROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)
