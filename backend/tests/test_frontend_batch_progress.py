"""Run classic training-panel lifecycle tests against the real inline functions."""
from pathlib import Path
import subprocess


def test_frontend_batch_progress_lifecycle():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ['node', '--test', str(Path(__file__).with_name('frontend_batch_progress.mjs'))],
        cwd=root, capture_output=True, text=True, encoding='utf-8',
    )
    assert result.returncode == 0, result.stdout + result.stderr
