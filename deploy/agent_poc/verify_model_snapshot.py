#!/usr/bin/env python3
"""Stream-verify a downloaded ModelScope snapshot against a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_SIZE = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    temporary = (
        [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and (path.name == "._____temp" or path.name.endswith(".incomplete"))
        ]
        if root.exists()
        else []
    )
    if temporary:
        failures.append(f"incomplete files present: {temporary[:20]}")

    checked = 0
    expected_bytes = 0
    actual_bytes = 0
    for item in manifest.get("files", []):
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe manifest path: {relative}")
            continue
        path = root / relative
        expected_size = int(item["size"])
        expected_bytes += expected_size
        if not path.is_file():
            failures.append(f"missing: {relative.as_posix()}")
            continue
        actual_size = path.stat().st_size
        actual_bytes += actual_size
        if actual_size != expected_size:
            failures.append(f"size mismatch: {relative.as_posix()} expected={expected_size} actual={actual_size}")
            continue
        actual_sha256 = sha256_file(path)
        checked += 1
        if actual_sha256 != str(item["sha256"]).lower():
            failures.append(f"sha256 mismatch: {relative.as_posix()}")

    report = {
        "schema_version": "autoai-model-verification-v1",
        "model_id": manifest.get("model_id"),
        "model_key": manifest.get("model_key"),
        "snapshot_root": str(root),
        "manifest_fetched_at_utc": manifest.get("fetched_at_utc"),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "checked_files": checked,
        "expected_files": len(manifest.get("files", [])),
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "temporary_files": temporary,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify_snapshot(args.root, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
