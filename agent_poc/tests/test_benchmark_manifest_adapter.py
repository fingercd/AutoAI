from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import agent_poc.benchmark.adapter as adapter_module
import agent_poc.benchmark.manifest as manifest_module
from agent_poc.benchmark.adapter import adapt_pmlb_dataset
from agent_poc.benchmark.manifest import BenchmarkValidationError, verify_benchmark


TABMINI_REVISION = "a" * 40
PMLB_REVISION = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gzip_tsv(path: Path, feature_names: tuple[str, ...]) -> None:
    lines = ["\t".join((*feature_names, "target"))]
    for index in range(6):
        values = [str((index + offset) % 4) for offset in range(len(feature_names))]
        lines.append("\t".join((*values, str(index % 2))))
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(("\n".join(lines) + "\n").encode("utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _refresh_policy_integrity(root: Path, policy_path: Path) -> None:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["integrity"]["benchmark_manifest_sha256"] = _sha256(
        root / "benchmark-manifest.json"
    )
    policy["integrity"]["sha256sums_sha256"] = _sha256(root / "SHA256SUMS")
    _write_json(policy_path, policy)
    canonical = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest_module._EXPECTED_POLICY_CANONICAL_SHA256 = hashlib.sha256(canonical).hexdigest()


def _replace_checksum(root: Path, relative: str) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="ascii").splitlines()
    replacement = f"{_sha256(root / relative)}  {relative}"
    rewritten = [replacement if line.endswith(f"  {relative}") else line for line in lines]
    checksum_path.write_text("\n".join(rewritten) + "\n", encoding="ascii")


@pytest.fixture()
def frozen_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "frozen"
    (root / "data").mkdir(parents=True)
    (root / "sources").mkdir()
    (root / "reports").mkdir()
    specs = {
        "molecular_biology_promoters": ("formal_primary", ("p-1", "p1")),
        "haberman": ("formal_auxiliary", ("age",)),
        "parity5": ("pipeline_smoke_only", ("bit1", "bit2", "bit3")),
    }

    datasets: list[dict[str, object]] = []
    policy_datasets: list[dict[str, object]] = []
    for name, (role, features) in specs.items():
        directory = root / "data" / name
        directory.mkdir()
        data_path = directory / f"{name}.tsv.gz"
        _write_gzip_tsv(data_path, features)
        (directory / "metadata.yaml").write_text(
            "\n".join(
                [
                    f"dataset: {name}",
                    "task: classification",
                    "features:",
                    *[f"  - name: {feature}" for feature in features],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (directory / "README.md").write_text(f"# {name}\n", encoding="utf-8")
        (directory / "summary_stats.tsv").write_text("name\tvalue\nrows\t6\n", encoding="utf-8")
        data_digest = _sha256(data_path)
        shared = {
            "name": name,
            "role": role,
            "n_samples": 6,
            "n_features": len(features),
            "class_counts": {"0": 3, "1": 3},
        }
        datasets.append(
            {
                **shared,
                "task": "binary_classification",
                "missing_values": 0,
                "data_sha256": data_digest,
            }
        )
        policy_datasets.append({**shared, "sha256": data_digest})

    source_files = {
        "PMLB-LICENSE": b"MIT\n",
        f"TabMini-{TABMINI_REVISION}.tar.gz": b"frozen archive\n",
        "TabMini-LICENSE": b"MIT\n",
        "TabMini-README.md": b"# TabMini\n",
        "TabMini-data_info.py": b"DATASETS = ()\n",
    }
    for name, content in source_files.items():
        (root / "sources" / name).write_bytes(content)
    (root / "reports" / "note.md").write_text("not a critical input\n", encoding="utf-8")

    critical = sorted(
        path
        for top in (root / "data", root / "sources")
        for path in top.rglob("*")
        if path.is_file()
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in critical
        ),
        encoding="ascii",
    )
    inventory = {
        "manifest_version": "small-sample-benchmark-candidate-v1",
        "server_path": "/untrusted/absolute/path",
        "suite": {
            "name": "untrusted display name",
            "tabmini_revision": TABMINI_REVISION,
            "pmlb_revision": PMLB_REVISION,
        },
        "datasets": datasets,
    }
    _write_json(root / "benchmark-manifest.json", inventory)
    policy = {
        "manifest_version": "small-sample-benchmark-v1",
        "suite": {
            "name": "reviewed suite name",
            "tabmini_revision": TABMINI_REVISION,
            "pmlb_revision": PMLB_REVISION,
        },
        "datasets": policy_datasets,
        "evaluation": {
            "native_primary_metric": "macro_f1",
            "native_secondary_metrics": [
                "balanced_accuracy",
                "failure_rate",
                "wall_clock_seconds",
                "llm_calls",
                "api_calls",
                "retry_attempts",
                "model_fits",
            ],
            "native_split": "stratified_holdout",
        },
        "integrity": {
            "benchmark_manifest_sha256": _sha256(root / "benchmark-manifest.json"),
            "sha256sums_sha256": _sha256(root / "SHA256SUMS"),
            "sha256sums_entries": 17,
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_json(policy_path, policy)
    canonical = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(
        manifest_module,
        "_EXPECTED_POLICY_CANONICAL_SHA256",
        hashlib.sha256(canonical).hexdigest(),
    )
    return root, policy_path


def test_verify_benchmark_checks_all_critical_files_and_ignores_server_path(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle

    verified = verify_benchmark(root, policy_path)

    assert verified.checksum_count == 17
    assert verified.suite_name == "reviewed suite name"
    assert verified.native_split == "stratified_holdout"
    assert verified.dataset("haberman").source_feature_names == ("age",)
    assert "untrusted" not in repr(verified)
    with pytest.raises(KeyError):
        verified.dataset("absent")


def test_verify_benchmark_rejects_changed_frozen_data(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle
    (root / "data" / "haberman" / "haberman.tsv.gz").write_bytes(b"changed")

    with pytest.raises(BenchmarkValidationError, match="checksum mismatch"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_an_unpinned_policy(
    frozen_bundle: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, policy_path = frozen_bundle
    monkeypatch.setattr(manifest_module, "_EXPECTED_POLICY_CANONICAL_SHA256", "0" * 64)

    with pytest.raises(BenchmarkValidationError, match="Git-pinned"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_parent_traversal_checksum_entry(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    lines[0] = f"{'0' * 64}  ../escape"
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    _refresh_policy_integrity(root, policy_path)

    with pytest.raises(BenchmarkValidationError, match="Unsafe checksum path"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_extra_critical_file(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle
    (root / "sources" / "unexpected.bin").write_bytes(b"not reviewed")

    with pytest.raises(BenchmarkValidationError, match="missing or extra critical files"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_inventory_revision_mismatch(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle
    inventory_path = root / "benchmark-manifest.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["suite"]["pmlb_revision"] = "c" * 40
    _write_json(inventory_path, inventory)
    _refresh_policy_integrity(root, policy_path)

    with pytest.raises(BenchmarkValidationError, match="PMLB revision"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_metadata_content_mismatch(
    frozen_bundle: tuple[Path, Path],
) -> None:
    root, policy_path = frozen_bundle
    relative = "data/haberman/metadata.yaml"
    metadata = root / relative
    metadata.write_text(metadata.read_text(encoding="utf-8").replace("dataset: haberman", "dataset: wrong"), encoding="utf-8")
    _replace_checksum(root, relative)
    _refresh_policy_integrity(root, policy_path)

    with pytest.raises(BenchmarkValidationError, match="metadata mismatch"):
        verify_benchmark(root, policy_path)


def test_verify_benchmark_rejects_symlinks(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    link = root / "reports" / "link"
    try:
        os.symlink(tmp_path / "outside", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(BenchmarkValidationError, match="symlink"):
        verify_benchmark(root, policy_path)


def test_adapter_writes_deterministic_wide_feature_v2_and_path_free_receipt(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("haberman")
    output_dir = tmp_path / "work-output"

    first = adapt_pmlb_dataset(verified, output_dir)
    first_bytes = first.output_path.read_bytes()
    second = adapt_pmlb_dataset(verified, output_dir)

    assert first.output_sha256 == second.output_sha256
    assert second.output_path.read_bytes() == first_bytes
    with second.output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Index", "Label", "Sample_ID", "Name", "0"]
    assert rows[1][:4] == ["1", "0", "haberman-row-000001", "haberman.tsv.gz"]
    assert len({row[2] for row in rows[1:]}) == 6
    assert second.split_strategy == "stratified_holdout"

    receipt_text = second.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert set(receipt) == {
        "dataset",
        "input_sha256",
        "output_sha256",
        "source_columns",
    }
    assert receipt["dataset"] == "haberman"
    assert receipt["source_columns"]["feature_count"] == 1
    assert str(tmp_path) not in receipt_text
    assert "server_path" not in receipt_text


def test_adapter_rechecks_data_after_verification(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("parity5")
    verified.data_path.write_bytes(verified.data_path.read_bytes() + b"changed")

    with pytest.raises(BenchmarkValidationError, match="changed before adaptation"):
        adapt_pmlb_dataset(verified, tmp_path / "output")


def test_adapter_rejects_forged_verified_dataset_name(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("parity5")

    with pytest.raises(BenchmarkValidationError, match="unsafe name"):
        adapt_pmlb_dataset(replace(verified, name="../escape"), tmp_path / "output")


def test_adapter_rejects_output_directory_with_symlink_ancestor(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("parity5")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "output-link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(BenchmarkValidationError, match="ancestors must not be symlinks"):
        adapt_pmlb_dataset(verified, link / "nested")


def test_adapter_uses_the_open_descriptor_when_source_path_is_exchanged(
    frozen_bundle: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("haberman")
    source = verified.data_path
    replacement = source.with_name("replacement.tsv.gz")
    replacement.write_bytes(b"not the verified gzip payload")
    real_open = os.open
    exchanged = False

    def exchange_after_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal exchanged
        descriptor = real_open(path, flags, mode)
        if not exchanged and Path(path) == source:
            try:
                os.replace(replacement, source)
            except OSError:
                os.close(descriptor)
                pytest.skip("this platform cannot exchange an open source path")
            exchanged = True
        return descriptor

    monkeypatch.setattr(adapter_module.os, "open", exchange_after_open)
    adapted = adapt_pmlb_dataset(verified, tmp_path / "descriptor-output")

    assert exchanged is True
    with adapted.output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][:4] == ["1", "0", "haberman-row-000001", "haberman.tsv.gz"]
    assert source.read_bytes() == b"not the verified gzip payload"


def test_adapter_hashes_canonical_output_before_publish_without_path_reread(
    frozen_bundle: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("parity5")
    original_read_bytes = Path.read_bytes

    def reject_output_reread(path: Path) -> bytes:
        if path.name.endswith(".wide-feature-v2.csv"):
            raise AssertionError("published CSV must not be reopened to calculate its digest")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_output_reread)
    adapted = adapt_pmlb_dataset(verified, tmp_path / "prehashed-output")

    with adapted.output_path.open("rb") as handle:
        published = handle.read()
    assert hashlib.sha256(published).hexdigest() == adapted.output_sha256
    receipt = json.loads(adapted.receipt_path.read_text(encoding="utf-8"))
    assert receipt["output_sha256"] == adapted.output_sha256


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode contract")
def test_adapter_rejects_group_or_world_writable_output_directory(
    frozen_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, policy_path = frozen_bundle
    verified = verify_benchmark(root, policy_path).dataset("parity5")
    unsafe = tmp_path / "unsafe-output"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)

    with pytest.raises(BenchmarkValidationError, match="group/world writable"):
        adapt_pmlb_dataset(verified, unsafe)
