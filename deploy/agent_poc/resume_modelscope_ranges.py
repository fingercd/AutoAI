#!/usr/bin/env python3
"""Resume large ModelScope blobs with HTTP Range after CLI worker failures.

This is an escalation path for public ModelScope LFS files.  It reuses the
same frozen manifest and SHA-256 checks as the CLI path; it never fabricates or
silently accepts a short blob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "qwen35_9b": ("Qwen/Qwen3.5-9B", "qwen3.5-9b"),
    "qwen35_27b": ("Qwen/Qwen3.5-27B", "qwen3.5-27b"),
    "qwen38_27b": ("Qwen/Qwen3.8-27B", "qwen3.8-27b"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODELS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    model_id, _ = MODELS[args.model_key]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest["files"]
    temp_root = args.root / "._____temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    bytes_done = 0

    for item in files:
        relative = Path(item["path"])
        target = args.root / relative
        expected_size = int(item["size"])
        expected_sha = item["sha256"]
        if target.is_file() and target.stat().st_size == expected_size and sha256(target) == expected_sha:
            completed += 1
            bytes_done += expected_size
            print(f"skip_verified path={relative}", flush=True)
            continue

        if relative.as_posix() == ".gitattributes":
            url = f"https://modelscope.cn/models/{model_id}/resolve/master/.gitattributes"
        else:
            encoded = urllib.parse.quote_plus(relative.as_posix())
            url = (
                f"https://www.modelscope.cn/api/v1/models/{model_id}/repo"
                f"?Revision=master&FilePath={encoded}"
            )

        # ModelScope's downloader keeps large incomplete blobs in this flat
        # directory.  Reuse that file so curl sends Range: bytes=<size>-.
        temp = temp_root / relative.name
        if target.is_file() and not temp.exists():
            temp = target
        temp.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"range_download path={relative} current_bytes={temp.stat().st_size if temp.exists() else 0}"
            f" expected_bytes={expected_size}",
            flush=True,
        )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "30",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                "7200",
                "-C",
                "-",
                "-o",
                str(temp),
                url,
            ],
            check=True,
        )
        if not temp.is_file() or temp.stat().st_size != expected_size:
            raise RuntimeError(
                f"size_mismatch path={relative} actual={temp.stat().st_size if temp.exists() else 0}"
                f" expected={expected_size}"
            )
        actual_sha = sha256(temp)
        if actual_sha != expected_sha:
            raise RuntimeError(f"sha256_mismatch path={relative} actual={actual_sha} expected={expected_sha}")
        if temp != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp), str(target))
        completed += 1
        bytes_done += expected_size
        write_status(
            args.status,
            {
                "schema_version": "autoai-model-range-download-v1",
                "model_key": args.model_key,
                "model_id": model_id,
                "phase": "downloading",
                "completed_files": completed,
                "expected_files": len(files),
                "bytes_downloaded": bytes_done,
                "expected_bytes": sum(int(entry["size"]) for entry in files),
                "last_completed_path": relative.as_posix(),
                "updated_at_utc": utc_now(),
            },
        )

    # The verifier treats the presence of the downloader's temp directory as
    # incomplete state, even when it is empty.  Remove only this known empty
    # directory; any unexpected leftover file remains visible and fails the
    # subsequent snapshot verification.
    if temp_root.is_dir() and not any(temp_root.iterdir()):
        temp_root.rmdir()

    write_status(
        args.status,
        {
            "schema_version": "autoai-model-range-download-v1",
            "model_key": args.model_key,
            "model_id": model_id,
            "phase": "complete",
            "completed_files": completed,
            "expected_files": len(files),
            "bytes_downloaded": bytes_done,
            "expected_bytes": sum(int(entry["size"]) for entry in files),
            "updated_at_utc": utc_now(),
        },
    )
    print(f"range_download_complete model_key={args.model_key} files={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
