#!/usr/bin/env python3
"""Create the deterministic 256-line text calibration set shared by all models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEMPLATES = (
    'Explain how to validate a scientific experiment without leaking the test set.',
    'Summarize a classification result using only validation metrics.',
    'Return a concise plan for comparing logistic regression and support vector machines.',
    'Describe why reproducible random seeds matter in a data pipeline.',
    'Translate this short technical instruction into clear Chinese.',
    'Translate this short technical instruction into clear English.',
    'Identify an unsafe request for a server path and refuse it.',
    'Give a structured answer with a decision, a reason, and no extra fields.',
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=256)
    args = parser.parse_args()
    if args.count < 128 or args.count > 256:
        parser.error('count must be between 128 and 256')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as handle:
        for index in range(args.count):
            text = f'{TEMPLATES[index % len(TEMPLATES)]} Calibration example {index + 1}.'
            handle.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({'output': str(args.output), 'count': args.count, 'sha256': digest}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
