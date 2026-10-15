"""Offline protocol compatibility and class-complete partition checks."""
import json

import numpy as np
import pytest

from backend.app.benchmark_protocol import (
    FULL_PROTOCOL_ID, PROTOCOL_ID, build_test_rounds, load_approved_protocol,
    validate_context_reference,
)
from backend.app.evaluation_plan import DatasetView, digest


def approved_body(protocol_id):
    return dict(protocol_id=protocol_id, governance=dict(user_approved=True),
                hpo=dict(candidate_budgets=[1, 3]), split_seed=42,
                amendments=dict(retain_all_source_rows=True) if protocol_id == FULL_PROTOCOL_ID else {})


def write_approval(path, body):
    approval = {**body, 'approval_digest': digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(approval, ensure_ascii=False), encoding='utf-8')
    return approval


@pytest.mark.parametrize('protocol_id', [PROTOCOL_ID, FULL_PROTOCOL_ID])
def test_approval_is_loaded_without_migration(tmp_path, protocol_id):
    path = tmp_path / 'approval' / 'protocol.json'
    expected = write_approval(path, approved_body(protocol_id))
    original = path.read_bytes()
    assert load_approved_protocol(path) == expected
    assert path.read_bytes() == original
    assert PROTOCOL_ID == 'autoai-classification-benchmark-v3'


@pytest.mark.parametrize('change', ['unknown_version', 'unapproved', 'seed', 'budget', 'digest'])
def test_v4_rejects_unapproved_or_changed_rules(tmp_path, change):
    body = approved_body(FULL_PROTOCOL_ID)
    if change == 'unknown_version':
        body['protocol_id'] = 'autoai-classification-benchmark-unknown'
    elif change == 'unapproved':
        body['governance']['user_approved'] = False
    elif change == 'seed':
        body['split_seed'] = 43
    elif change == 'budget':
        body['hpo']['candidate_budgets'] = [1, 6]
    path = tmp_path / 'protocol.json'
    write_approval(path, body)
    if change == 'digest':
        changed = json.loads(path.read_text())
        changed['amendments']['retain_all_source_rows'] = False
        path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_approved_protocol(path)


def test_v4_context_binds_to_its_own_approval_root(tmp_path, monkeypatch):
    root = tmp_path / 'benchmark-20261010-full-v4'
    approval = write_approval(root / 'approval' / 'protocol.json', approved_body(FULL_PROTOCOL_ID))
    monkeypatch.setenv('AUTOAI_BENCHMARK_ROOT', str(root))
    context = dict(suite_root=str(root), bundle_id='bundle', task_id='task', round_id='01',
                   protocol_digest=approval['approval_digest'], source_digest='a' * 64)
    assert validate_context_reference(context) == approval
    old_digest = digest(approved_body(PROTOCOL_ID))
    with pytest.raises(ValueError, match='approval binding differs'):
        validate_context_reference({**context, 'protocol_digest': old_digest})


def view_with_groups(groups_per_class, repeats=1):
    labels = tuple(label for label in ('A', 'B')
                   for group in range(groups_per_class) for _ in range(repeats))
    sample_ids = tuple(f'{label}-{group}' for label in ('A', 'B')
                       for group in range(groups_per_class) for _ in range(repeats))
    return DatasetView(np.arange(len(labels) * 2, dtype=np.float32).reshape(-1, 2),
                       labels, sample_ids, (1.0, 2.0), 'b' * 64)


def assert_complete_rounds(view, plans):
    assert sorted(i for plan in plans for i in plan.indices['test']) == list(range(len(view.x)))
    for plan in plans:
        assert plan.seed == 42
        for part in ('train', 'valid', 'test'):
            assert {view.labels[i] for i in plan.indices[part]} == {'A', 'B'}
        groups = [set(plan.groups[part]) for part in ('train', 'valid', 'test')]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_class_complete_five_rounds_preserve_known_physical_groups():
    view = view_with_groups(30, repeats=2)
    first = build_test_rounds(view)
    assert len(first) == 5
    assert_complete_rounds(view, first)
    assert [plan.model_dump() for plan in first] == [plan.model_dump() for plan in build_test_rounds(view)]


def test_three_round_fallback_retains_every_class():
    view = view_with_groups(4)
    plans = build_test_rounds(view)
    assert len(plans) == 3
    assert_complete_rounds(view, plans)


def test_infeasible_class_complete_rounds_are_excluded():
    with pytest.raises(ValueError, match='no class-complete approved test partition is feasible'):
        build_test_rounds(view_with_groups(2))


def test_official_training_singleton_class_is_rejected():
    view = view_with_groups(3)
    # Official Train contains A-0, A-1 and B-0: no class-complete Train/Valid split.
    with pytest.raises(ValueError):
        build_test_rounds(view, official_test_indices=[2, 4, 5])
