#!/usr/bin/env python3
"""Generate a deterministic wide-feature-v2 acceptance CSV outside Git."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_dataset(path: Path, *, groups_per_class: int = 12, repeats: int = 2, feature_count: int = 32) -> None:
    x_axis = np.linspace(100.0, 100.0 + feature_count - 1, feature_count, dtype=np.float64)
    headers = ['0' if value == 0 else format(float(value), '.17g') for value in x_axis]
    rows: list[dict[str, object]] = []
    index = 1
    group_id = 0
    for class_id, label in enumerate(('A', 'B')):
        for group in range(groups_per_class):
            group_id += 1
            for repeat in range(repeats):
                signal = (
                    class_id * 2.0
                    + group_id * 0.01
                    + repeat * 0.001
                    + np.sin(x_axis * 0.05 + class_id) * 0.02
                )
                row: dict[str, object] = {
                    'Index': index,
                    'Label': label,
                    'Sample_ID': str(group_id),
                    'Name': f'acceptance_{label}_{group_id}_{repeat}.csv',
                }
                row.update(zip(headers, signal.tolist()))
                rows.append(row)
                index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['Index', 'Label', 'Sample_ID', 'Name', *headers]).to_csv(
        path, index=False, encoding='utf-8-sig'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups-per-class', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--feature-count', type=int, default=32)
    args = parser.parse_args()
    make_dataset(
        args.output,
        groups_per_class=args.groups_per_class,
        repeats=args.repeats,
        feature_count=args.feature_count,
    )
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
