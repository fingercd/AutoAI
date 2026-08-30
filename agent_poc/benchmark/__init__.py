"""Small-sample benchmark bundle verification and deterministic adapters."""

from .adapter import AdaptedDataset, adapt_pmlb_dataset
from .manifest import (
    BenchmarkValidationError,
    VerifiedBenchmark,
    VerifiedDataset,
    verify_benchmark,
)

__all__ = [
    "AdaptedDataset",
    "BenchmarkValidationError",
    "VerifiedBenchmark",
    "VerifiedDataset",
    "adapt_pmlb_dataset",
    "verify_benchmark",
]
