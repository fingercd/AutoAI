from __future__ import annotations

from typing import Any

from .feature_selection import aggregate_attribution_sanity, aggregate_dscarnet_branch_sanity


def aggregate_attribution_sanity_result(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_attribution_sanity(samples)


def aggregate_dscarnet_branch_sanity_result(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_dscarnet_branch_sanity(samples)


__all__ = [
    'aggregate_attribution_sanity',
    'aggregate_dscarnet_branch_sanity',
    'aggregate_attribution_sanity_result',
    'aggregate_dscarnet_branch_sanity_result',
]
