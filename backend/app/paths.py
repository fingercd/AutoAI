from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RUNS_DIR = STORAGE_DIR / "runs"
PREPROCESSED_DIR = STORAGE_DIR / "preprocessed"
STATIC_DIR = PROJECT_ROOT / "static"
DEFAULT_DATA = PROJECT_ROOT / "data.csv"


def ensure_storage() -> None:
    for path in (STORAGE_DIR, UPLOADS_DIR, RUNS_DIR, PREPROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)
