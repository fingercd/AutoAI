from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_grouped_classification_csv(
    path: Path,
    *,
    groups_per_class: int,
    repeats: int,
    feature_count: int,
    labels: tuple[str, ...] = ("A", "B"),
) -> Path:
    """Write grouped curves using the repository's one-curve-per-row CSV contract."""

    rows: list[dict[str, object]] = []
    x_axis = np.arange(feature_count, dtype=float)
    group_id = 0
    row_index = 1
    for class_id, label in enumerate(labels):
        for _ in range(groups_per_class):
            group_id += 1
            for repeat in range(repeats):
                signal = class_id * 2.0 + group_id * 0.01 + repeat * 0.001 + x_axis * 0.0001
                rows.append(
                    {
                        "Index": row_index,
                        "Name": f"sample_{group_id}",
                        "XXX": json.dumps(x_axis.tolist()),
                        "Intensity": json.dumps(signal.tolist()),
                        "Label": label,
                        "Sample_ID": str(group_id),
                    }
                )
                row_index += 1
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path
