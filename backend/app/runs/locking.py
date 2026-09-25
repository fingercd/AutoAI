"""Per-Run cross-process boundary for Finalize and application deletion."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import time


@contextmanager
def run_lock(database_path: Path, run_id: str):
    folder = Path(database_path).parent / '.run-locks'
    folder.mkdir(exist_ok=True)
    path = folder / (hashlib.sha256(run_id.encode()).hexdigest() + '.lock')
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            until = time.monotonic() + 65
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= until:
                        raise TimeoutError('Run integrity lock unavailable') from None
                    time.sleep(.025)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
