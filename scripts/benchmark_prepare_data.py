"""Rebuild the benchmark data manifest after acquiring public source assets."""
from pathlib import Path
import argparse
import json
import time
import uuid
from backend.app.benchmark_data import prepare_assets, prepare_full_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmark-root', type=Path, help='新全量轮次目录；省略时保留旧入口行为')
    parser.add_argument('--source-root', type=Path, help='只读旧来源轮次目录')
    args = parser.parse_args()
    started_at, started = time.time(), time.monotonic()
    status = "failed"
    try:
        result = (prepare_full_assets(args.root, args.benchmark_root, args.source_root)
                  if args.benchmark_root else prepare_assets(args.root))
        status = result["status"]
    finally:
        if args.benchmark_root:
            trace_dir = args.benchmark_root / "audit" / "preparation-times"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace = dict(stage="data_preparation", started_at=started_at,
                         finished_at=time.time(), wall_seconds=time.monotonic() - started,
                         status=status, model_fits=0, lm_requests=0, lm_tokens=0)
            (trace_dir / (uuid.uuid4().hex + ".json")).write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
