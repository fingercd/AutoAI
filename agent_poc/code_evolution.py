"""Deterministic, AST-guarded code generation for experimental model candidates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import stat
import types
from pathlib import Path
from queue import Empty
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator,
)


CODE_EVOLUTION_SCHEMA_VERSION = 'code-evolution-v1'
MAX_SOURCE_LINES = 220
MAX_SPEC_FILE_BYTES = 64 * 1024
MAX_FORWARD_MACS = 200_000_000
MAX_ACTIVATION_ELEMENTS = 5_000_000
VERIFY_TIMEOUT_SECONDS = 15
_FORBIDDEN_CALLS = {
    'eval', 'exec', 'open', 'compile', '__import__', 'getattr', 'setattr',
    'delattr', 'input', 'breakpoint',
}
_ALLOWED_IMPORTS = {'torch', 'torch.nn'}


class EvolutionArchitectureSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    conv_blocks: int = Field(ge=1, le=4, strict=True)
    base_channels: int = Field(ge=8, le=128, strict=True)
    kernel_size: int = Field(ge=3, le=15, strict=True)
    dropout: float = Field(ge=0.0, le=0.5, strict=True)
    residual: bool = Field(strict=True)
    attention: Literal['none', 'se'] = 'none'

    @field_validator('kernel_size')
    @classmethod
    def kernel_must_be_odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError('kernel_size must be odd')
        return value


class CodeEvolutionSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal['code-evolution-v1']
    guard_contract: Literal['agent-guard-v1']
    proposal_id: str = Field(pattern=r'^evo_[0-9a-f]{16}$')
    candidate_name: str = Field(pattern=r'^[a-z][a-z0-9_]{2,40}$')
    base_model: Literal[
        'cnn1d', 'cnn1d_se', 'resnet1d',
    ]
    input_length: int = Field(ge=32, le=16380, strict=True)
    class_count: int = Field(ge=2, le=50, strict=True)
    max_parameters: int = Field(ge=1_000, le=5_000_000, strict=True)
    architecture: EvolutionArchitectureSpec

    @model_validator(mode='after')
    def proposal_must_match_spec(self):
        expected = evolution_proposal_id(self.model_dump(mode='json'))
        if self.proposal_id != expected:
            raise ValueError('proposal_id must be derived from the complete spec')
        if self.base_model == 'cnn1d_se' and self.architecture.attention != 'se':
            raise ValueError('cnn1d_se requires se attention')
        if self.base_model == 'resnet1d' and not self.architecture.residual:
            raise ValueError('resnet1d requires residual blocks')
        return self


def evolution_proposal_id(payload: dict[str, object]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key != 'proposal_id'
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return 'evo_' + hashlib.sha256(encoded).hexdigest()[:16]


def spec_digest(spec: CodeEvolutionSpec) -> str:
    encoded = json.dumps(
        spec.model_dump(mode='json'),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def render_candidate_module(spec: CodeEvolutionSpec) -> str:
    architecture = spec.architecture
    output_dim = 1 if spec.class_count == 2 else spec.class_count
    return f'''"""Generated from {CODE_EVOLUTION_SCHEMA_VERSION}; do not edit by hand."""
import torch
from torch import nn

BASE_MODEL = {spec.base_model!r}


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, {architecture.kernel_size}, padding={architecture.kernel_size // 2}),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout({architecture.dropout}),
        )
        self.attention = SqueezeExcite1d(out_channels) if {architecture.attention == 'se'} else nn.Identity()

    def forward(self, x):
        return self.attention(self.body(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = ConvBlock(in_channels, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class CandidateNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_channels = 1
        for index in range({architecture.conv_blocks}):
            out_channels = min({architecture.base_channels} * (2 ** index), 256)
            block_type = ResidualBlock if {architecture.residual} else ConvBlock
            blocks.extend([block_type(in_channels, out_channels), nn.MaxPool1d(2)])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, {output_dim})

    def forward(self, x):
        features = self.pool(self.features(x)).squeeze(-1)
        return self.head(features)
'''


def validate_generated_source(source: str) -> ast.Module:
    if len(source.splitlines()) > MAX_SOURCE_LINES:
        raise ValueError('generated source exceeds line budget')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in _ALLOWED_IMPORTS for alias in node.names):
                raise ValueError('generated source imports a forbidden module')
        elif isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORTS:
                raise ValueError('generated source imports a forbidden module')
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError('generated source contains forbidden scope mutation')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise ValueError('generated source contains a forbidden call')
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith('__')
            and node.attr != '__init__'
        ):
            raise ValueError('generated source contains a forbidden attribute')
    compile(tree, '<code-evolution-candidate>', 'exec')
    return tree


def _complexity(spec: CodeEvolutionSpec) -> tuple[int, int]:
    length = spec.input_length
    in_channels = 1
    macs = 0
    activations = 0
    for index in range(spec.architecture.conv_blocks):
        out_channels = min(spec.architecture.base_channels * (2 ** index), 256)
        macs += length * in_channels * out_channels * spec.architecture.kernel_size
        if spec.architecture.residual and in_channels != out_channels:
            macs += length * in_channels * out_channels
        activations += length * out_channels
        length = max(1, length // 2)
        in_channels = out_channels
    return int(macs), int(activations)


def _verify_worker(spec_json: str, source: str, result_queue) -> None:
    try:
        import torch

        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        except (ImportError, OSError, ValueError):
            pass
        spec = CodeEvolutionSpec.model_validate_json(spec_json)
        tree = validate_generated_source(source)
        module = types.ModuleType('code_evolution_candidate')
        exec(compile(tree, '<code-evolution-candidate>', 'exec'), module.__dict__)
        candidate = module.CandidateNet()
        parameter_count = sum(
            parameter.numel() for parameter in candidate.parameters()
        )
        if parameter_count > spec.max_parameters:
            raise ValueError('candidate exceeds parameter budget')
        candidate.eval()
        with torch.no_grad():
            output = candidate(torch.zeros(1, 1, spec.input_length))
        expected_width = 1 if spec.class_count == 2 else spec.class_count
        if tuple(output.shape) != (1, expected_width):
            raise ValueError('candidate output contract is invalid')
        result_queue.put({
            'ok': True,
            'parameter_count': int(parameter_count),
            'output_shape': list(output.shape),
        })
    except Exception as exc:
        stable_errors = {
            'candidate exceeds parameter budget': 'parameter_budget',
            'candidate output contract is invalid': 'output_contract',
        }
        result_queue.put({
            'ok': False,
            'error_code': stable_errors.get(str(exc), 'verification_failed'),
        })


def verify_candidate(spec: CodeEvolutionSpec, source: str) -> dict[str, object]:
    if source != render_candidate_module(spec):
        raise ValueError('candidate source is not generator-authored')
    validate_generated_source(source)
    estimated_macs, activation_elements = _complexity(spec)
    if estimated_macs > MAX_FORWARD_MACS:
        raise ValueError('candidate exceeds forward compute budget')
    if activation_elements > MAX_ACTIVATION_ELEMENTS:
        raise ValueError('candidate exceeds activation budget')
    context = multiprocessing.get_context('spawn')
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_verify_worker,
        args=(spec.model_dump_json(), source, result_queue),
    )
    process.start()
    process.join(VERIFY_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise ValueError('candidate verification timed out')
    try:
        worker_result = result_queue.get(timeout=1)
    except Empty as exc:
        raise ValueError('candidate verification failed') from exc
    if not worker_result.get('ok'):
        raise ValueError(
            'candidate verification failed: '
            + str(worker_result.get('error_code') or 'unknown')
        )
    return {
        'schema_version': CODE_EVOLUTION_SCHEMA_VERSION,
        'status': 'verified',
        'spec_digest': spec_digest(spec),
        'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
        'source_line_count': len(source.splitlines()),
        'parameter_count': worker_result['parameter_count'],
        'output_shape': worker_result['output_shape'],
        'estimated_forward_macs': estimated_macs,
        'activation_elements': activation_elements,
    }


def materialize_candidate(
    spec: CodeEvolutionSpec,
    *,
    project_root: Path,
) -> dict[str, object]:
    project = project_root.resolve()
    if not (project / 'agent_poc').is_dir() or not (project / 'backend').is_dir():
        raise ValueError('project_root is not an AutoAI checkout')
    output_root = (project / 'work' / 'code-evolution').resolve()
    if project not in output_root.parents:
        raise ValueError('code evolution work root escaped project')
    candidate_dir = (output_root / spec.proposal_id).resolve()
    if output_root not in candidate_dir.parents:
        raise ValueError('candidate path escaped work root')
    source = render_candidate_module(spec)
    report = verify_candidate(spec, source)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    destination = candidate_dir / f'{spec.candidate_name}.py'
    if destination.is_symlink():
        raise ValueError('candidate destination must not be a symlink')
    if destination.exists():
        if not stat.S_ISREG(destination.stat().st_mode):
            raise ValueError('candidate destination is not a regular file')
        if destination.read_text(encoding='utf-8') != source:
            raise ValueError('candidate destination already contains different source')
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(source)
    for name, content in (
        ('spec.json', json.dumps(
            spec.model_dump(mode='json'),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )),
        ('report.json', json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )),
    ):
        artifact = candidate_dir / name
        if artifact.is_symlink():
            raise ValueError('candidate metadata must not be a symlink')
        if artifact.exists():
            if artifact.read_text(encoding='utf-8') != content:
                raise ValueError('candidate metadata differs from locked identity')
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, 'O_NOFOLLOW', 0)
            descriptor = os.open(artifact, flags, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                handle.write(content)
    report['relative_file'] = (
        Path('work') / 'code-evolution' / spec.proposal_id /
        f'{spec.candidate_name}.py'
    ).as_posix()
    return report


def load_spec_file(path: Path) -> CodeEvolutionSpec:
    if path.stat().st_size > MAX_SPEC_FILE_BYTES:
        raise ValueError('code evolution spec exceeds size limit')
    return CodeEvolutionSpec.model_validate_json(path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-file', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    spec = load_spec_file(args.spec_file)
    report = materialize_candidate(spec, project_root=args.project_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
