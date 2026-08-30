from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_poc.priors import (
    DEFAULT_PRIOR_PATH,
    PRIOR_PROJECTION_SCHEMA_VERSION,
    PRIOR_SCHEMA_VERSION,
    load_prior_catalog,
    project_priors,
)
from agent_poc.prompts import build_messages
from agent_poc.state import AgentConfig, ModelConfig
from agent_poc.trace import TraceRecorder, assert_trace_safe


def _observation(*risks: str) -> dict:
    return {
        'context': {
            'evidence_card': {
                'schema_version': 'small-sample-evidence-v1',
                'status': 'ready',
                'scope': 'train_only',
                'statistics': {'risk_codes': list(risks)},
            },
        },
        'experiments': [],
    }


def _proposal(
    suffix: str,
    model_type: str,
    *,
    class_balance: str = 'none',
) -> dict[str, str]:
    return {
        'proposal_id': f'p_{suffix}',
        'model_type': model_type,
        'normalization': 'zscore',
        'class_balance': class_balance,
    }


def test_default_prior_is_fixed_strict_and_has_stable_digest(tmp_path):
    first = load_prior_catalog()
    raw = json.loads(DEFAULT_PRIOR_PATH.read_text(encoding='utf-8'))
    reformatted = tmp_path / 'reformatted.json'
    reformatted.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8',
    )
    second = load_prior_catalog(reformatted)

    assert first.schema_version == PRIOR_SCHEMA_VERSION
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.recipes == second.recipes

    # The source catalog is data-only: no path, proposal/history identifier,
    # threshold, free-form rationale or executable parameter appears in it.
    for recipe in raw['recipes']:
        assert set(recipe) == {
            'risk_codes', 'model_type', 'normalization',
            'class_balance', 'rationale_code',
        }
        assert not any(key.endswith('_id') for key in recipe)
    rendered = json.dumps(raw, ensure_ascii=False).lower()
    assert '/users/' not in rendered
    assert ':\\' not in rendered
    assert 'run_id' not in rendered
    assert 'proposal_id' not in rendered


@pytest.mark.parametrize(
    'mutate, message',
    [
        (lambda raw: raw.update({'extra': True}), 'only schema_version'),
        (
            lambda raw: raw['recipes'][0].update({'model_type': 'arbitrary'}),
            'allowed enum',
        ),
        (
            lambda raw: raw['recipes'][0].update({
                'risk_codes': ['very_small_train_partition', 'high_dimension'],
            }),
            'sorted, unique',
        ),
        (
            lambda raw: raw['recipes'].append(dict(raw['recipes'][0])),
            'duplicate',
        ),
    ],
)
def test_prior_loader_rejects_schema_drift(tmp_path, mutate, message):
    raw = json.loads(DEFAULT_PRIOR_PATH.read_text(encoding='utf-8'))
    mutate(raw)
    candidate = tmp_path / 'invalid.json'
    candidate.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        load_prior_catalog(candidate)


def test_projection_requires_train_only_risk_and_exact_available_proposal():
    catalog = load_prior_catalog()
    logistic = _proposal(
        '1111111111111111',
        'logistic_regression',
    )
    weighted = _proposal(
        '2222222222222222',
        'logistic_regression',
        class_balance='class_weight',
    )
    svm = _proposal('3333333333333333', 'svm')
    forest = _proposal('4444444444444444', 'random_forest')

    projected = project_priors(
        catalog,
        observation=_observation(
            'class_imbalance',
            'high_dimension',
            'very_small_train_partition',
        ),
        allowed_models=('logistic_regression',),
        proposal_recipes=(logistic, weighted, svm, forest),
    )
    assert projected is not None
    assert projected['schema_version'] == PRIOR_PROJECTION_SCHEMA_VERSION
    assert projected['source_digest'] == catalog.digest
    recommendations = projected['recommendations']
    assert {item['proposal_id'] for item in recommendations} == {
        logistic['proposal_id'], weighted['proposal_id'],
    }
    assert {item['model_type'] for item in recommendations} == {
        'logistic_regression'
    }
    assert all('run_id' not in item for item in recommendations)

    assert project_priors(
        catalog,
        observation=_observation('class_imbalance'),
        allowed_models=('logistic_regression',),
        proposal_recipes=(svm,),
    ) is None
    assert project_priors(
        catalog,
        observation={'context': {}, 'experiments': []},
        allowed_models=('logistic_regression',),
        proposal_recipes=(logistic,),
    ) is None


def test_prompt_injects_only_safe_mapped_prior_and_old_call_stays_compatible():
    cfg = AgentConfig(
        autoai_base_url='http://autoai',
        dataset_id='dataset',
        allowed_models=('logistic_regression',),
    )
    proposal = _proposal('1111111111111111', 'logistic_regression')
    projected = project_priors(
        load_prior_catalog(),
        observation=_observation('very_small_train_partition'),
        allowed_models=cfg.allowed_models,
        proposal_recipes=(proposal,),
    )
    messages = build_messages(
        cfg,
        _observation('very_small_train_partition'),
        proposal_recipes=(proposal,),
        prior_guidance=projected,
    )
    payload = json.loads(messages[1]['content'])
    assert payload['static_prior'] == projected
    assert payload['static_prior']['recommendations'][0]['proposal_id'] == (
        proposal['proposal_id']
    )
    prior_text = json.dumps(payload['static_prior']).lower()
    assert 'run_id' not in prior_text
    assert '/users/' not in prior_text

    legacy_payload = json.loads(
        build_messages(cfg, {'experiments': []})[1]['content']
    )
    assert 'static_prior' not in legacy_payload


def test_trace_records_only_prior_schema_and_digest(tmp_path):
    catalog = load_prior_catalog()
    trace_path = tmp_path / 'trace.jsonl'
    recorder = TraceRecorder(
        trace_path,
        model_cfg=ModelConfig(
            key='model',
            base_url='http://127.0.0.1:1/v1',
            served_model_name='model',
            model_path='/private/model',
        ),
        code_revision='revision',
    )
    recorder.record_prior_metadata(
        schema_version=catalog.schema_version,
        digest=catalog.digest,
    )
    assert_trace_safe(trace_path)
    event = json.loads(trace_path.read_text(encoding='utf-8'))
    assert event['event'] == 'static_prior_loaded'
    assert event['prior_schema_version'] == PRIOR_SCHEMA_VERSION
    assert event['prior_digest'] == catalog.digest
    serialized = json.dumps(event).lower()
    for forbidden in (
        'risk_codes', 'model_type', 'normalization', 'class_balance',
        'rationale_code', 'proposal_id', 'run_id', '/users/', 'priors-v1.json',
    ):
        assert forbidden not in serialized

    with pytest.raises(ValueError, match='digest'):
        recorder.record_prior_metadata(
            schema_version=PRIOR_SCHEMA_VERSION,
            digest='not-a-digest',
        )
