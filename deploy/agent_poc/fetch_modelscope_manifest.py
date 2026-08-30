#!/usr/bin/env python3
"""Freeze the ModelScope API file manifest used by the Agent V1 POC."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS: dict[str, str] = {
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_27b": "Qwen/Qwen3.5-27B",
    "qwen38_27b": "Qwen/Qwen3.8-27B",
}


def _api_url(model_id: str, revision: str) -> str:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in model_id.split("/"))
    query = urllib.parse.urlencode({"Recursive": "true", "Revision": revision})
    return f"https://modelscope.cn/api/v1/models/{quoted}/repo/files?{query}"


def _cli_version() -> str | None:
    try:
        result = subprocess.run(
            ['modelscope', '--version'],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[-1][:200] if lines else None


def fetch_manifest(model_id: str, output: Path, *, revision: str = "master") -> dict[str, Any]:
    url = _api_url(model_id, revision)
    request = urllib.request.Request(url, headers={"User-Agent": "AutoAI-Agent-V1/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)

    if str(payload.get("Code")) != "200":
        raise RuntimeError(f"ModelScope manifest failed: {payload.get('Message', 'unknown error')}")
    files = payload.get("Data", {}).get("Files", [])
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"ModelScope returned no files for {model_id}")

    frozen_files = []
    for item in files:
        if item.get("Type") != "blob":
            continue
        path = str(item.get("Path", ""))
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise RuntimeError(f"unsafe ModelScope path in manifest: {path!r}")
        frozen_files.append(
            {
                "path": path,
                "size": int(item["Size"]),
                "sha256": str(item["Sha256"]).lower(),
                "source_revision": str(item.get("Revision", revision)),
                "is_lfs": bool(item.get("IsLFS", False)),
            }
        )
    if not frozen_files:
        raise RuntimeError(f"ModelScope returned no blob files for {model_id}")

    frozen = {
        "schema_version": "autoai-model-manifest-v1",
        "model_key": next((key for key, value in MODELS.items() if value == model_id), None),
        "model_id": model_id,
        "source": "modelscope",
        "source_api_url": url,
        "requested_revision": revision,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "modelscope_cli_version": _cli_version(),
        "files": frozen_files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODELS))
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()

    if args.model_key:
        model_id = MODELS[args.model_key]
        output = args.output or Path(f"{args.model_key}.json")
    elif args.model_id and args.output:
        model_id = args.model_id
        output = args.output
    else:
        parser.error("use --model-key [--output] or --model-id with --output")

    if args.output_dir:
        output = args.output_dir / output.name
    manifest = fetch_manifest(model_id, output, revision=args.revision)
    total = sum(item["size"] for item in manifest["files"])
    print(json.dumps({"model_id": model_id, "output": str(output), "files": len(manifest["files"]), "bytes": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
