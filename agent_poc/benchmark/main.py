"""Command line interface for benchmark verification and one-trial execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from ..budget import BudgetController
from ..clients.autoai_client import AutoAIClient
from ..clients.llm_client import LLMClient
from ..state import load_agent_config, load_model_registry
from ..trace import TraceRecorder
from .evaluator import ResultEvaluator
from .manifest import verify_benchmark
from .report import write_trial_report
from .runner import BenchmarkRunner, DatasetUploadClient


_GIT_REVISION = re.compile(r'^[0-9a-f]{40}$')


class CodeRevisionError(RuntimeError):
    """The running source checkout does not expose a trustworthy Git HEAD."""


def resolve_code_revision(
    repo_root: Path | None = None,
    *,
    run_fn: Callable[..., object] = subprocess.run,
) -> str:
    """Resolve a clean checkout HEAD without a shell or caller-supplied SHA."""
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        head_result = run_fn(
            ['git', 'rev-parse', 'HEAD'],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='strict',
            timeout=5.0,
            shell=False,
        )
        revision = str(getattr(head_result, 'stdout')).strip()
    except (OSError, UnicodeError, subprocess.SubprocessError, AttributeError) as exc:
        raise CodeRevisionError('unable to resolve Git HEAD') from exc
    if not _GIT_REVISION.fullmatch(revision):
        raise CodeRevisionError('Git HEAD is not a lowercase 40-character SHA')
    try:
        status_result = run_fn(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='strict',
            timeout=5.0,
            shell=False,
        )
        worktree_status = str(getattr(status_result, 'stdout')).strip()
    except (OSError, UnicodeError, subprocess.SubprocessError, AttributeError) as exc:
        raise CodeRevisionError('unable to verify clean Git checkout') from exc
    if worktree_status:
        raise CodeRevisionError('benchmark requires a clean Git checkout')
    return revision


@dataclass(frozen=True)
class RuntimeDependencies:
    verify_fn: Callable[..., object] = verify_benchmark
    run_command_fn: Callable[[argparse.Namespace], dict[str, Any]] | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m agent_poc.benchmark.main')
    subparsers = parser.add_subparsers(dest='command', required=True)
    verify = subparsers.add_parser('verify')
    verify.add_argument('--benchmark-root', type=Path, required=True)
    verify.add_argument('--policy-manifest', type=Path, required=True)

    run = subparsers.add_parser('run')
    run.add_argument('--benchmark-root', type=Path, required=True)
    run.add_argument('--policy-manifest', type=Path, required=True)
    run.add_argument('--dataset-name', required=True)
    run.add_argument('--model-key', required=True)
    run.add_argument('--models-config', type=Path, required=True)
    run.add_argument('--agent-config', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--autoai-base-url')
    run.add_argument('--seed', type=int, default=42)
    return parser


def _safe_verification_summary(verified: object) -> dict[str, Any]:
    datasets = getattr(verified, 'datasets', ())
    return {
        'status': 'verified',
        'manifest_version': getattr(verified, 'manifest_version', None),
        'suite_name': getattr(verified, 'suite_name', None),
        'checksum_count': getattr(verified, 'checksum_count', None),
        'datasets': [
            {
                'name': getattr(item, 'name', None),
                'role': getattr(item, 'role', None),
                'sha256': getattr(item, 'data_sha256', None),
            }
            for item in datasets
        ],
    }


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    code_revision = resolve_code_revision()
    registry = load_model_registry(args.models_config)
    if args.model_key not in registry:
        raise ValueError('unknown model key')
    model_cfg = registry[args.model_key]
    trace_path = args.output_dir / f'{args.dataset_name}.trace.jsonl'
    cfg = load_agent_config(
        args.agent_config,
        dataset_id='benchmark-upload-pending',
        trace_path=trace_path,
    )
    cfg = replace(cfg, code_revision=code_revision)
    if args.autoai_base_url:
        cfg = replace(cfg, autoai_base_url=args.autoai_base_url)
    token = os.getenv('AUTOAI_API_TOKEN')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    upload_http = httpx.Client(base_url=cfg.autoai_base_url, headers=headers)
    evaluator_http = httpx.Client(base_url=cfg.autoai_base_url, headers=headers)
    budget_controller = BudgetController(
        cfg.budget if cfg.modules.budget_control else None
    )
    autoai = AutoAIClient(
        cfg.autoai_base_url,
        token=token,
        budget_controller=budget_controller,
    )
    llm = LLMClient(
        model_cfg,
        temperature=cfg.temperature,
        timeout_seconds=cfg.llm_timeout_seconds,
        budget_controller=budget_controller,
    )
    trace = TraceRecorder(
        trace_path, model_cfg=model_cfg, code_revision=cfg.code_revision
    )
    try:
        report = BenchmarkRunner().run_trial(
            benchmark_root=args.benchmark_root,
            policy_manifest_path=args.policy_manifest,
            dataset_name=args.dataset_name,
            adapter_output_dir=args.output_dir / 'adapted',
            cfg=cfg,
            llm=llm,
            autoai=autoai,
            trace=trace,
            uploader=DatasetUploadClient(upload_http),
            evaluator=ResultEvaluator(evaluator_http),
            model_key=model_cfg.key,
            model_revision=model_cfg.revision,
            code_revision=code_revision,
            seed=args.seed,
            budget_controller=budget_controller,
        )
    finally:
        autoai.close()
        upload_http.close()
        evaluator_http.close()
    write_trial_report(report, args.output_dir)
    return report


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> int:
    deps = dependencies or RuntimeDependencies()
    args = _parser().parse_args(argv)
    if args.command == 'verify':
        verified = deps.verify_fn(args.benchmark_root, args.policy_manifest)
        output = _safe_verification_summary(verified)
    else:
        run_command = deps.run_command_fn or _run_command
        output = run_command(args)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    if args.command == 'verify':
        return 0
    return 0 if output.get('outcome', {}).get('status') == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
