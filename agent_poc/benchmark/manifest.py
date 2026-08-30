"""Verify the frozen PMLB/TabMini benchmark bundle against a Git policy.

The server-side ``benchmark-manifest.json`` and ``SHA256SUMS`` live beside the
downloaded data, so neither file is an independent trust anchor.  The reviewed
Git policy pins both files' SHA-256 digests.  Only paths reconstructed from that
policy are opened; the inventory's informational ``server_path`` is ignored.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_DATASET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,127}$")
_CHECKSUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\x00\r\n]+)$")
_POLICY_VERSION = "small-sample-benchmark-v1"
_INVENTORY_VERSION = "small-sample-benchmark-candidate-v1"
_ALLOWED_SPLIT = "stratified_holdout"
# This is SHA-256(canonical JSON) of docs/benchmarks/small-sample-benchmark-v1.json.
# It makes the reviewed Git document a real trust anchor even when a CLI caller
# supplies an arbitrary path with the expected filename.
_EXPECTED_POLICY_CANONICAL_SHA256 = (
    "dbb5536eeb05e86574a0f6570f18522f035fe299af3a1ecf8e48c26cc0273aff"
)
_MAX_JSON_BYTES = 1_000_000
_MAX_FEATURES = 16_380
_MAX_ROWS = 2_000_000
_MAX_COMPRESSED_DATA_BYTES = 256 * 1024 * 1024


class BenchmarkValidationError(ValueError):
    """The frozen benchmark bundle does not match the reviewed policy."""


@dataclass(frozen=True)
class VerifiedDataset:
    """A dataset whose metadata, contents, and location have been verified."""

    name: str
    role: str
    task: str
    n_samples: int
    n_features: int
    class_counts: tuple[tuple[str, int], ...]
    data_sha256: str
    data_path: Path = field(repr=False)
    metadata_path: Path = field(repr=False)
    source_feature_names: tuple[str, ...]
    allowed_split: str = _ALLOWED_SPLIT


@dataclass(frozen=True)
class VerifiedBenchmark:
    """Trusted projection of a verified benchmark bundle."""

    manifest_version: str
    suite_name: str
    tabmini_revision: str
    pmlb_revision: str
    native_split: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    checksum_count: int
    benchmark_root: Path = field(repr=False)
    datasets: tuple[VerifiedDataset, ...]

    def dataset(self, name: str) -> VerifiedDataset:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(name)


@dataclass(frozen=True)
class _PolicyDataset:
    name: str
    role: str
    n_samples: int
    n_features: int
    class_counts: tuple[tuple[str, int], ...]
    data_sha256: str


@dataclass(frozen=True)
class _PmlbTable:
    feature_names: tuple[str, ...]
    rows: tuple[tuple[tuple[str, ...], str], ...]
    class_counts: tuple[tuple[str, int], ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkValidationError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "JSON file")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise BenchmarkValidationError(f"JSON file is too large: {path.name}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except BenchmarkValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkValidationError(f"Invalid JSON file: {path.name}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkValidationError(f"JSON root must be an object: {path.name}")
    return payload


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError("Policy JSON is not canonically serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise BenchmarkValidationError(f"Missing {label}: {path.name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise BenchmarkValidationError(f"{label} must be a regular non-symlink file: {path.name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BenchmarkValidationError(f"Unable to hash file: {path.name}") from exc
    return digest.hexdigest()


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    return value


def _as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkValidationError(f"{label} must be a list")
    return value


def _as_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise BenchmarkValidationError(f"{label} must be a non-empty string")
    return value


def _as_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkValidationError(f"{label} must be a positive integer")
    return value


def _as_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkValidationError(f"{label} must be a non-negative integer")
    return value


def _as_sha256(value: Any, label: str) -> str:
    digest = _as_string(value, label)
    if not _SHA256_RE.fullmatch(digest):
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _as_git_revision(value: Any, label: str) -> str:
    revision = _as_string(value, label)
    if not _GIT_REVISION_RE.fullmatch(revision):
        raise BenchmarkValidationError(f"{label} must be a lowercase 40-character Git revision")
    return revision


def _class_counts(value: Any, label: str, n_samples: int) -> tuple[tuple[str, int], ...]:
    raw = _as_object(value, label)
    if len(raw) < 2:
        raise BenchmarkValidationError(f"{label} must contain at least two classes")
    counts: list[tuple[str, int]] = []
    for class_name, count in raw.items():
        if not isinstance(class_name, str) or not class_name:
            raise BenchmarkValidationError(f"{label} contains an invalid class name")
        counts.append((class_name, _as_positive_int(count, f"{label}.{class_name}")))
    counts.sort(key=lambda item: item[0])
    if sum(count for _, count in counts) != n_samples:
        raise BenchmarkValidationError(f"{label} does not sum to n_samples")
    return tuple(counts)


def _parse_policy(payload: dict[str, Any]) -> tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[_PolicyDataset, ...],
    dict[str, Any],
]:
    version = _as_string(payload.get("manifest_version"), "manifest_version")
    if version != _POLICY_VERSION:
        raise BenchmarkValidationError(f"Unsupported policy manifest version: {version}")

    suite = _as_object(payload.get("suite"), "suite")
    suite_name = _as_string(suite.get("name"), "suite.name")
    tabmini_revision = _as_git_revision(
        suite.get("tabmini_revision"), "suite.tabmini_revision"
    )
    pmlb_revision = _as_git_revision(suite.get("pmlb_revision"), "suite.pmlb_revision")

    evaluation = _as_object(payload.get("evaluation"), "evaluation")
    native_split = _as_string(evaluation.get("native_split"), "evaluation.native_split")
    if native_split != _ALLOWED_SPLIT:
        raise BenchmarkValidationError("PMLB benchmark policy only permits stratified_holdout")
    primary_metric = _as_string(
        evaluation.get("native_primary_metric"), "evaluation.native_primary_metric"
    )
    secondary_values = _as_list(
        evaluation.get("native_secondary_metrics"), "evaluation.native_secondary_metrics"
    )
    secondary_metrics = tuple(
        _as_string(item, f"evaluation.native_secondary_metrics[{index}]")
        for index, item in enumerate(secondary_values)
    )
    if len(set(secondary_metrics)) != len(secondary_metrics):
        raise BenchmarkValidationError("evaluation.native_secondary_metrics contains duplicates")

    policy_datasets: list[_PolicyDataset] = []
    names: set[str] = set()
    for index, item in enumerate(_as_list(payload.get("datasets"), "datasets")):
        dataset = _as_object(item, f"datasets[{index}]")
        name = _as_string(dataset.get("name"), f"datasets[{index}].name")
        if not _DATASET_NAME_RE.fullmatch(name):
            raise BenchmarkValidationError(f"Unsafe dataset name: {name}")
        if name in names:
            raise BenchmarkValidationError(f"Duplicate dataset: {name}")
        names.add(name)
        n_samples = _as_positive_int(dataset.get("n_samples"), f"datasets[{index}].n_samples")
        n_features = _as_positive_int(dataset.get("n_features"), f"datasets[{index}].n_features")
        if n_features > _MAX_FEATURES:
            raise BenchmarkValidationError(f"Dataset has too many features: {name}")
        policy_datasets.append(
            _PolicyDataset(
                name=name,
                role=_as_string(dataset.get("role"), f"datasets[{index}].role"),
                n_samples=n_samples,
                n_features=n_features,
                class_counts=_class_counts(
                    dataset.get("class_counts"), f"datasets[{index}].class_counts", n_samples
                ),
                data_sha256=_as_sha256(dataset.get("sha256"), f"datasets[{index}].sha256"),
            )
        )
    if not policy_datasets:
        raise BenchmarkValidationError("Policy contains no datasets")

    integrity = _as_object(payload.get("integrity"), "integrity")
    _as_sha256(integrity.get("benchmark_manifest_sha256"), "integrity.benchmark_manifest_sha256")
    _as_sha256(integrity.get("sha256sums_sha256"), "integrity.sha256sums_sha256")
    _as_positive_int(integrity.get("sha256sums_entries"), "integrity.sha256sums_entries")
    return (
        version,
        suite_name,
        tabmini_revision,
        pmlb_revision,
        native_split,
        primary_metric,
        secondary_metrics,
        tuple(policy_datasets),
        integrity,
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    if "\\" in value or ":" in value:
        raise BenchmarkValidationError(f"Unsafe checksum path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BenchmarkValidationError(f"Unsafe checksum path: {value}")
    if path.as_posix() != value:
        raise BenchmarkValidationError(f"Non-canonical checksum path: {value}")
    return path


def _child_path(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkValidationError(f"Missing frozen file: {relative.as_posix()}") from exc
    root_resolved = root.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise BenchmarkValidationError(f"Frozen path escapes benchmark root: {relative.as_posix()}")
    _require_regular_file(candidate, "frozen file")
    return resolved


def _reject_bundle_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise BenchmarkValidationError("benchmark_root must be a non-symlink directory")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            if (base / name).is_symlink():
                raise BenchmarkValidationError("Benchmark bundle must not contain symlinks")


def _expected_critical_paths(
    datasets: Iterable[_PolicyDataset], tabmini_revision: str
) -> set[str]:
    result = {
        "sources/PMLB-LICENSE",
        f"sources/TabMini-{tabmini_revision}.tar.gz",
        "sources/TabMini-LICENSE",
        "sources/TabMini-README.md",
        "sources/TabMini-data_info.py",
    }
    for dataset in datasets:
        prefix = f"data/{dataset.name}"
        result.update(
            {
                f"{prefix}/README.md",
                f"{prefix}/metadata.yaml",
                f"{prefix}/summary_stats.tsv",
                f"{prefix}/{dataset.name}.tsv.gz",
            }
        )
    return result


def _actual_critical_paths(root: Path) -> set[str]:
    result: set[str] = set()
    for top_name in ("data", "sources"):
        top = root / top_name
        if top.is_symlink() or not top.is_dir():
            raise BenchmarkValidationError(f"Missing benchmark directory: {top_name}")
        for path in top.rglob("*"):
            if path.is_symlink():
                raise BenchmarkValidationError("Benchmark bundle must not contain symlinks")
            if path.is_file():
                result.add(path.relative_to(root).as_posix())
    return result


def _parse_checksums(path: Path) -> dict[str, str]:
    _require_regular_file(path, "SHA256SUMS")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkValidationError("Invalid SHA256SUMS encoding") from exc
    if not lines or any(not line for line in lines):
        raise BenchmarkValidationError("SHA256SUMS must contain only non-empty entries")
    checksums: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise BenchmarkValidationError("Malformed SHA256SUMS entry")
        digest, raw_path = match.groups()
        relative = _safe_relative_path(raw_path).as_posix()
        if relative in checksums:
            raise BenchmarkValidationError(f"Duplicate checksum path: {relative}")
        checksums[relative] = digest
    return checksums


def _yaml_scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_metadata(path: Path) -> tuple[str, str, tuple[str, ...]]:
    _require_regular_file(path, "dataset metadata")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkValidationError(f"Invalid metadata encoding: {path.name}") from exc
    dataset_name: str | None = None
    task: str | None = None
    feature_names: list[str] = []
    in_features = False
    for line in lines:
        if line.startswith("dataset:"):
            dataset_name = _yaml_scalar(line.split(":", 1)[1])
        elif line.startswith("task:"):
            task = _yaml_scalar(line.split(":", 1)[1])
        elif line.startswith("features:"):
            in_features = True
        elif in_features:
            if line and not line[0].isspace() and not line.startswith("#"):
                in_features = False
                continue
            match = re.match(r"^\s{2}-\s+name:\s*(.+?)\s*$", line)
            if match:
                feature_names.append(_yaml_scalar(match.group(1)))
    if not dataset_name or not task or not feature_names:
        raise BenchmarkValidationError(f"Incomplete dataset metadata: {path.name}")
    if any(not name for name in feature_names) or len(set(feature_names)) != len(feature_names):
        raise BenchmarkValidationError(f"Invalid or duplicate feature names in metadata: {path.name}")
    return dataset_name, task, tuple(feature_names)


def _read_pmlb_bytes(
    payload: bytes, *, display_name: str, max_rows: int = _MAX_ROWS
) -> _PmlbTable:
    """Validate a PMLB gzip payload already captured from a trusted descriptor."""

    if not isinstance(payload, bytes) or not payload:
        raise BenchmarkValidationError(f"PMLB data is empty: {display_name}")
    if len(payload) > _MAX_COMPRESSED_DATA_BYTES:
        raise BenchmarkValidationError(f"Compressed PMLB data is too large: {display_name}")
    rows: list[tuple[tuple[str, ...], str]] = []
    counts: Counter[str] = Counter()
    try:
        compressed = gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb")
        with compressed, io.TextIOWrapper(
            compressed, encoding="utf-8", newline=""
        ) as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header is None or len(header) < 2 or header[-1] != "target":
                raise BenchmarkValidationError(f"Invalid PMLB header: {display_name}")
            feature_names = tuple(header[:-1])
            if (
                any(not name for name in feature_names)
                or len(set(feature_names)) != len(feature_names)
                or len(feature_names) > _MAX_FEATURES
            ):
                raise BenchmarkValidationError(f"Invalid PMLB feature columns: {display_name}")
            for row_number, row in enumerate(reader, start=2):
                if row_number - 1 > max_rows:
                    raise BenchmarkValidationError(f"PMLB row limit exceeded: {display_name}")
                if len(row) != len(header):
                    raise BenchmarkValidationError(
                        f"PMLB row {row_number} has the wrong column count: {display_name}"
                    )
                features = tuple(value.strip() for value in row[:-1])
                target = row[-1].strip()
                if not target:
                    raise BenchmarkValidationError(f"PMLB row {row_number} has an empty target")
                for value in features:
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise BenchmarkValidationError(
                            f"PMLB row {row_number} contains a non-numeric feature"
                        ) from exc
                    if not math.isfinite(number):
                        raise BenchmarkValidationError(
                            f"PMLB row {row_number} contains a non-finite feature"
                        )
                rows.append((features, target))
                counts[target] += 1
    except BenchmarkValidationError:
        raise
    except (OSError, EOFError, UnicodeError, csv.Error) as exc:
        raise BenchmarkValidationError(f"Unable to read PMLB data: {display_name}") from exc
    if not rows or len(counts) < 2:
        raise BenchmarkValidationError(f"PMLB data must contain at least two classes: {display_name}")
    return _PmlbTable(feature_names, tuple(rows), tuple(sorted(counts.items())))


def _read_pmlb_table(path: Path, *, max_rows: int = _MAX_ROWS) -> _PmlbTable:
    """Read and validate a small PMLB TSV.GZ using only the standard library."""

    _require_regular_file(path, "PMLB data file")
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_COMPRESSED_DATA_BYTES:
            raise BenchmarkValidationError(f"Compressed PMLB data has an invalid size: {path.name}")
        payload = path.read_bytes()
    except BenchmarkValidationError:
        raise
    except OSError as exc:
        raise BenchmarkValidationError(f"Unable to read PMLB data: {path.name}") from exc
    return _read_pmlb_bytes(payload, display_name=path.name, max_rows=max_rows)


def _inventory_datasets(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    version = _as_string(payload.get("manifest_version"), "inventory.manifest_version")
    if version != _INVENTORY_VERSION:
        raise BenchmarkValidationError(f"Unsupported benchmark inventory version: {version}")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(_as_list(payload.get("datasets"), "inventory.datasets")):
        item = _as_object(value, f"inventory.datasets[{index}]")
        name = _as_string(item.get("name"), f"inventory.datasets[{index}].name")
        if not _DATASET_NAME_RE.fullmatch(name) or name in result:
            raise BenchmarkValidationError(f"Invalid or duplicate inventory dataset: {name}")
        result[name] = item
    return result


def verify_benchmark(benchmark_root: Path, policy_manifest_path: Path) -> VerifiedBenchmark:
    """Verify all 17 frozen files and return a trusted, path-safe projection."""

    root = Path(benchmark_root)
    policy_path = Path(policy_manifest_path)
    _reject_bundle_symlinks(root)
    policy = _load_json(policy_path)
    if _canonical_json_sha256(policy) != _EXPECTED_POLICY_CANONICAL_SHA256:
        raise BenchmarkValidationError("Policy manifest is not the Git-pinned benchmark policy")
    (
        policy_version,
        suite_name,
        tabmini_revision,
        pmlb_revision,
        native_split,
        primary_metric,
        secondary_metrics,
        policy_datasets,
        integrity,
    ) = _parse_policy(policy)

    checksum_path = root / "SHA256SUMS"
    inventory_path = root / "benchmark-manifest.json"
    if _sha256(checksum_path) != integrity["sha256sums_sha256"]:
        raise BenchmarkValidationError("SHA256SUMS does not match the Git policy")
    if _sha256(inventory_path) != integrity["benchmark_manifest_sha256"]:
        raise BenchmarkValidationError("benchmark-manifest.json does not match the Git policy")

    expected_paths = _expected_critical_paths(policy_datasets, tabmini_revision)
    expected_count = _as_positive_int(
        integrity["sha256sums_entries"], "integrity.sha256sums_entries"
    )
    if expected_count != len(expected_paths):
        raise BenchmarkValidationError("Git policy checksum count does not match its dataset inventory")
    checksums = _parse_checksums(checksum_path)
    if _sha256(checksum_path) != integrity["sha256sums_sha256"]:
        raise BenchmarkValidationError("SHA256SUMS changed during verification")
    if set(checksums) != expected_paths or len(checksums) != expected_count:
        raise BenchmarkValidationError("SHA256SUMS has missing or extra critical files")
    if _actual_critical_paths(root) != expected_paths:
        raise BenchmarkValidationError("Benchmark bundle has missing or extra critical files")
    for relative, expected_digest in checksums.items():
        path = _child_path(root, _safe_relative_path(relative))
        if _sha256(path) != expected_digest:
            raise BenchmarkValidationError(f"Frozen file checksum mismatch: {relative}")

    inventory = _load_json(inventory_path)
    if _sha256(inventory_path) != integrity["benchmark_manifest_sha256"]:
        raise BenchmarkValidationError("benchmark-manifest.json changed during verification")
    inventory_suite = _as_object(inventory.get("suite"), "inventory.suite")
    if _as_git_revision(
        inventory_suite.get("tabmini_revision"), "inventory.suite.tabmini_revision"
    ) != tabmini_revision:
        raise BenchmarkValidationError("TabMini revision differs from the Git policy")
    if _as_git_revision(
        inventory_suite.get("pmlb_revision"), "inventory.suite.pmlb_revision"
    ) != pmlb_revision:
        raise BenchmarkValidationError("PMLB revision differs from the Git policy")
    inventory_by_name = _inventory_datasets(inventory)
    if set(inventory_by_name) != {dataset.name for dataset in policy_datasets}:
        raise BenchmarkValidationError("Inventory datasets differ from the Git policy")

    verified: list[VerifiedDataset] = []
    for expected in policy_datasets:
        item = inventory_by_name[expected.name]
        if _as_string(item.get("role"), f"inventory.{expected.name}.role") != expected.role:
            raise BenchmarkValidationError(f"Dataset role mismatch: {expected.name}")
        task = _as_string(item.get("task"), f"inventory.{expected.name}.task")
        if task != "binary_classification":
            raise BenchmarkValidationError(f"Unsupported benchmark task: {expected.name}")
        if _as_positive_int(
            item.get("n_samples"), f"inventory.{expected.name}.n_samples"
        ) != expected.n_samples:
            raise BenchmarkValidationError(f"Dataset row count mismatch: {expected.name}")
        if _as_positive_int(
            item.get("n_features"), f"inventory.{expected.name}.n_features"
        ) != expected.n_features:
            raise BenchmarkValidationError(f"Dataset feature count mismatch: {expected.name}")
        if _class_counts(
            item.get("class_counts"),
            f"inventory.{expected.name}.class_counts",
            expected.n_samples,
        ) != expected.class_counts:
            raise BenchmarkValidationError(f"Dataset class counts mismatch: {expected.name}")
        if _as_nonnegative_int(
            item.get("missing_values"), f"inventory.{expected.name}.missing_values"
        ) != 0:
            raise BenchmarkValidationError(f"Dataset contains missing values: {expected.name}")
        if _as_sha256(
            item.get("data_sha256"), f"inventory.{expected.name}.data_sha256"
        ) != expected.data_sha256:
            raise BenchmarkValidationError(f"Dataset digest mismatch: {expected.name}")

        prefix = PurePosixPath("data", expected.name)
        data_relative = prefix / f"{expected.name}.tsv.gz"
        metadata_relative = prefix / "metadata.yaml"
        data_path = _child_path(root, data_relative)
        metadata_path = _child_path(root, metadata_relative)
        if checksums[data_relative.as_posix()] != expected.data_sha256:
            raise BenchmarkValidationError(f"Dataset checksum policy mismatch: {expected.name}")

        metadata_name, metadata_task, metadata_features = _parse_metadata(metadata_path)
        if _sha256(metadata_path) != checksums[metadata_relative.as_posix()]:
            raise BenchmarkValidationError(f"Dataset metadata changed during verification: {expected.name}")
        if metadata_name != expected.name or metadata_task != "classification":
            raise BenchmarkValidationError(f"Dataset metadata mismatch: {expected.name}")
        if len(metadata_features) != expected.n_features:
            raise BenchmarkValidationError(f"Metadata feature count mismatch: {expected.name}")
        table = _read_pmlb_table(data_path)
        if _sha256(data_path) != expected.data_sha256:
            raise BenchmarkValidationError(f"Dataset changed during verification: {expected.name}")
        if table.feature_names != metadata_features:
            raise BenchmarkValidationError(f"Data columns differ from metadata: {expected.name}")
        if len(table.rows) != expected.n_samples or table.class_counts != expected.class_counts:
            raise BenchmarkValidationError(f"Dataset contents differ from inventory: {expected.name}")

        verified.append(
            VerifiedDataset(
                name=expected.name,
                role=expected.role,
                task=task,
                n_samples=expected.n_samples,
                n_features=expected.n_features,
                class_counts=expected.class_counts,
                data_sha256=expected.data_sha256,
                data_path=data_path,
                metadata_path=metadata_path,
                source_feature_names=metadata_features,
            )
        )

    # Recheck every critical input at the return boundary.  This keeps a
    # concurrent replacement from turning an earlier successful check into a
    # stale capability handed to the runner.
    for relative, expected_digest in checksums.items():
        path = _child_path(root, _safe_relative_path(relative))
        if _sha256(path) != expected_digest:
            raise BenchmarkValidationError(f"Frozen file changed during verification: {relative}")

    return VerifiedBenchmark(
        manifest_version=policy_version,
        suite_name=suite_name,
        tabmini_revision=tabmini_revision,
        pmlb_revision=pmlb_revision,
        native_split=native_split,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        checksum_count=len(checksums),
        benchmark_root=root.resolve(strict=True),
        datasets=tuple(verified),
    )


__all__ = [
    "BenchmarkValidationError",
    "VerifiedBenchmark",
    "VerifiedDataset",
    "verify_benchmark",
]
