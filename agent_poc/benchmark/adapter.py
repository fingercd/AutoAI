"""Deterministic PMLB to AutoAI ``wide-feature-v2`` adapter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import (
    BenchmarkValidationError,
    VerifiedDataset,
    _DATASET_NAME_RE,
    _MAX_COMPRESSED_DATA_BYTES,
    _read_pmlb_bytes,
)


_MAX_ADAPTED_OUTPUT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class AdaptedDataset:
    name: str
    role: str
    format_version: str
    split_strategy: str
    n_samples: int
    n_features: int
    class_counts: tuple[tuple[str, int], ...]
    output_path: Path = field(repr=False)
    receipt_path: Path = field(repr=False)
    input_sha256: str
    output_sha256: str
    source_feature_names: tuple[str, ...]


def _safe_output_directory(output_dir: Path) -> Path:
    path = Path(os.path.abspath(output_dir))
    try:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise BenchmarkValidationError(
                    "Adapter output directory ancestors must not be symlinks"
                )
            if current != path and not stat.S_ISDIR(mode):
                raise BenchmarkValidationError(
                    "Adapter output path has a non-directory ancestor"
                )
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = os.lstat(path)
    except BenchmarkValidationError:
        raise
    except OSError as exc:
        raise BenchmarkValidationError("Unable to create adapter output directory") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BenchmarkValidationError("Adapter output directory must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise BenchmarkValidationError("Adapter output directory is not a canonical path")
    if os.name == "posix":
        get_effective_uid = getattr(os, "geteuid", None)
        if callable(get_effective_uid) and info.st_uid != get_effective_uid():
            raise BenchmarkValidationError("Adapter output directory has an unsafe owner")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise BenchmarkValidationError("Adapter output directory is group/world writable")
    return resolved


def _reject_symlink_target(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise BenchmarkValidationError(f"Unable to inspect output target: {path.name}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise BenchmarkValidationError(f"Output target must be a regular file: {path.name}")


def _capture_verified_source(path: Path) -> bytes:
    """Capture one immutable-by-descriptor snapshot without reopening ``path``."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise BenchmarkValidationError("PMLB source must not be a symlink")
        except BenchmarkValidationError:
            raise
        except OSError as exc:
            raise BenchmarkValidationError("Unable to inspect PMLB source") from exc
    flags = os.O_RDONLY | nofollow | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BenchmarkValidationError(
            "Unable to open PMLB source without following symlinks"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise BenchmarkValidationError("PMLB source descriptor is not a regular file")
        if info.st_size <= 0 or info.st_size > _MAX_COMPRESSED_DATA_BYTES:
            raise BenchmarkValidationError("PMLB source has an invalid compressed size")
        captured = bytearray()
        while True:
            read_size = min(
                1024 * 1024,
                _MAX_COMPRESSED_DATA_BYTES + 1 - len(captured),
            )
            block = os.read(descriptor, read_size)
            if not block:
                break
            captured.extend(block)
            if len(captured) > _MAX_COMPRESSED_DATA_BYTES:
                raise BenchmarkValidationError("PMLB source exceeds the compressed size limit")
        if len(captured) != info.st_size:
            raise BenchmarkValidationError("PMLB source changed size while being captured")
        return bytes(captured)
    except BenchmarkValidationError:
        raise
    except OSError as exc:
        raise BenchmarkValidationError("Unable to capture PMLB source descriptor") from exc
    finally:
        os.close(descriptor)


def _render_csv_bytes(
    rows: tuple[tuple[tuple[str, ...], str], ...], dataset: VerifiedDataset
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["Index", "Label", "Sample_ID", "Name"]
        + [str(index) for index in range(dataset.n_features)]
    )
    source_name = dataset.data_path.name
    for row_index, (features, target) in enumerate(rows, start=1):
        writer.writerow(
            [
                row_index,
                target,
                f"{dataset.name}-row-{row_index:06d}",
                source_name,
                *features,
            ]
        )
    payload = buffer.getvalue().encode("utf-8")
    if len(payload) > _MAX_ADAPTED_OUTPUT_BYTES:
        raise BenchmarkValidationError("Adapted CSV exceeds the output size limit")
    return payload


def _render_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    """Publish pre-hashed bytes through a private, fsynced temporary file."""

    _reject_symlink_target(path)
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, mode="wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as exc:
        raise BenchmarkValidationError(f"Unable to publish adapter output: {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _column_summary(feature_names: tuple[str, ...]) -> dict[str, object]:
    canonical = json.dumps(
        {"features": feature_names, "target": "target"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "feature_count": len(feature_names),
        "target": "target",
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def adapt_pmlb_dataset(verified_dataset: VerifiedDataset, output_dir: Path) -> AdaptedDataset:
    """Create a deterministic, leakage-safe wide table and a path-free receipt."""

    if not isinstance(verified_dataset, VerifiedDataset):
        raise TypeError("verified_dataset must be a VerifiedDataset")
    if not _DATASET_NAME_RE.fullmatch(verified_dataset.name):
        raise BenchmarkValidationError("Verified dataset has an unsafe name")
    if verified_dataset.data_path.name != f"{verified_dataset.name}.tsv.gz":
        raise BenchmarkValidationError("Verified dataset source name is inconsistent")
    if verified_dataset.task != "binary_classification":
        raise BenchmarkValidationError("Adapter only supports binary classification")
    if verified_dataset.allowed_split != "stratified_holdout":
        raise BenchmarkValidationError("PMLB row IDs only permit stratified_holdout")

    source_bytes = _capture_verified_source(verified_dataset.data_path)
    input_digest = hashlib.sha256(source_bytes).hexdigest()
    if input_digest != verified_dataset.data_sha256:
        raise BenchmarkValidationError("Verified PMLB data changed before adaptation")
    table = _read_pmlb_bytes(source_bytes, display_name=verified_dataset.data_path.name)
    if table.feature_names != verified_dataset.source_feature_names:
        raise BenchmarkValidationError("PMLB columns changed after bundle verification")
    if len(table.feature_names) != verified_dataset.n_features:
        raise BenchmarkValidationError("PMLB feature count changed after bundle verification")
    if len(table.rows) != verified_dataset.n_samples:
        raise BenchmarkValidationError("PMLB row count changed after bundle verification")
    if table.class_counts != verified_dataset.class_counts:
        raise BenchmarkValidationError("PMLB class counts changed after bundle verification")

    directory = _safe_output_directory(Path(output_dir))
    output_path = directory / f"{verified_dataset.name}.wide-feature-v2.csv"
    receipt_path = directory / f"{verified_dataset.name}.adapter-receipt.json"
    output_bytes = _render_csv_bytes(table.rows, verified_dataset)
    output_digest = hashlib.sha256(output_bytes).hexdigest()
    receipt = {
        "dataset": verified_dataset.name,
        "input_sha256": input_digest,
        "output_sha256": output_digest,
        "source_columns": _column_summary(table.feature_names),
    }
    _atomic_publish_bytes(output_path, output_bytes)
    _atomic_publish_bytes(receipt_path, _render_json_bytes(receipt))

    return AdaptedDataset(
        name=verified_dataset.name,
        role=verified_dataset.role,
        format_version="wide-feature-v2",
        split_strategy="stratified_holdout",
        n_samples=verified_dataset.n_samples,
        n_features=verified_dataset.n_features,
        class_counts=verified_dataset.class_counts,
        output_path=output_path,
        receipt_path=receipt_path,
        input_sha256=input_digest,
        output_sha256=output_digest,
        source_feature_names=table.feature_names,
    )


__all__ = ["AdaptedDataset", "adapt_pmlb_dataset"]
