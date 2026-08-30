from __future__ import annotations

import json
import os

import pytest
import torch

from agent_poc.code_evolution import (
    CodeEvolutionSpec,
    evolution_proposal_id,
    materialize_candidate,
    load_spec_file,
    render_candidate_module,
    validate_generated_source,
    verify_candidate,
)


def _spec(**overrides):
    payload = {
        'schema_version': 'code-evolution-v1',
        'guard_contract': 'agent-guard-v1',
        'candidate_name': 'candidate_small_cnn',
        'base_model': 'cnn1d',
        'input_length': 128,
        'class_count': 2,
        'max_parameters': 500_000,
        'architecture': {
            'conv_blocks': 2,
            'base_channels': 16,
            'kernel_size': 5,
            'dropout': 0.1,
            'residual': True,
            'attention': 'se',
        },
    }
    payload.update(overrides)
    payload['proposal_id'] = evolution_proposal_id(payload)
    return CodeEvolutionSpec.model_validate(payload)


def test_generated_candidate_passes_ast_parameter_and_shape_guards():
    spec = _spec()
    source = render_candidate_module(spec)
    report = verify_candidate(spec, source)
    assert report['status'] == 'verified'
    assert report['parameter_count'] <= spec.max_parameters
    assert report['output_shape'] == [1, 1]


def test_ast_guard_rejects_file_process_and_dynamic_execution():
    for source in (
        'import os\nos.system("id")\n',
        'open("/tmp/leak", "w")\n',
        'exec("print(1)")\n',
    ):
        with pytest.raises(ValueError):
            validate_generated_source(source)
    with pytest.raises(ValueError, match='generator-authored'):
        verify_candidate(
            _spec(),
            'import torch\ntorch.save({}, "/tmp/leak")\n',
        )


def test_spec_rejects_even_kernel_and_parameter_budget_is_enforced():
    with pytest.raises(ValueError):
        _spec(architecture={
            'conv_blocks': 2, 'base_channels': 16, 'kernel_size': 4,
            'dropout': 0.1, 'residual': True, 'attention': 'none',
        })
    strict = _spec(max_parameters=1_000)
    with pytest.raises(ValueError, match='parameter_budget'):
        verify_candidate(strict, render_candidate_module(strict))


def test_materialization_is_confined_and_idempotent(tmp_path):
    (tmp_path / 'agent_poc').mkdir()
    (tmp_path / 'backend').mkdir()
    spec = _spec()
    first = materialize_candidate(spec, project_root=tmp_path)
    second = materialize_candidate(spec, project_root=tmp_path)
    assert first == second
    destination = tmp_path / first['relative_file']
    assert destination.is_file()
    assert (destination.parent / 'spec.json').is_file()
    assert (destination.parent / 'report.json').is_file()
    assert str(destination.resolve()).startswith(str((tmp_path / 'work').resolve()))


def test_materialization_rejects_symlinked_work_root(tmp_path):
    (tmp_path / 'agent_poc').mkdir()
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'work').mkdir()
    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.mkdir()
    os.symlink(outside, tmp_path / 'work' / 'code-evolution')
    with pytest.raises(ValueError, match='escaped project'):
        materialize_candidate(_spec(), project_root=tmp_path)


def test_spec_file_has_bounded_size(monkeypatch, tmp_path):
    spec_file = tmp_path / 'spec.json'
    spec_file.write_text(json.dumps(_spec().model_dump(mode='json')), encoding='utf-8')
    monkeypatch.setattr(
        'agent_poc.code_evolution.MAX_SPEC_FILE_BYTES',
        10,
    )
    with pytest.raises(ValueError, match='size limit'):
        load_spec_file(spec_file)


def test_identity_binds_base_model_and_full_spec():
    cnn = _spec(base_model='cnn1d')
    se = _spec(
        base_model='cnn1d_se',
        architecture={
            'conv_blocks': 2, 'base_channels': 16, 'kernel_size': 5,
            'dropout': 0.1, 'residual': True, 'attention': 'se',
        },
    )
    assert cnn.proposal_id != se.proposal_id
    assert render_candidate_module(cnn) != render_candidate_module(se)


def test_forward_compute_budget_and_parent_rng_are_protected():
    oversized = _spec(
        input_length=16380,
        max_parameters=5_000_000,
        architecture={
            'conv_blocks': 4, 'base_channels': 128, 'kernel_size': 15,
            'dropout': 0.1, 'residual': True, 'attention': 'se',
        },
    )
    with pytest.raises(ValueError, match='forward compute budget'):
        verify_candidate(oversized, render_candidate_module(oversized))

    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()
    spec = _spec()
    verify_candidate(spec, render_candidate_module(spec))
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
