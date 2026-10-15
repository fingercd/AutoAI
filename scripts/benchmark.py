"""CLI entry; worker thread limits are set before numerical imports."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.benchmark_driver import main
if __name__ == '__main__':
    main()
