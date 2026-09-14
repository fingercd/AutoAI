import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app import knowledge as k
from backend.app.evaluation_plan import digest, build_evaluation_plan, select_train
from backend.app.train_evidence import compute_train_evidence
from backend.tests.test_train_evidence_recipes import view, EVAL


def publication():
    return k.configured_knowledge()


def facts(models=('pls_da','svm','logistic_regression','pca_lda','cnn1d'), *, features=100, groups=20, repeats=2):
    return k.KnowledgeInput(task_type='classification', statistics_version='train-statistics-v1',
        risk_version='train-risk-rules-v1', feature_count=features, sample_group_count=groups,
        repeated_measurement_group_count=repeats, class_count=3, risks=(),
        recipes=tuple(k.RecipeReference(recipe_id='recipe_'+digest(m),model_id=m) for m in models))


def test_five_rules_and_real_nonmatches():
    result=k.match_knowledge(facts(),publication())
    assert len(result.matches)==5
    assert result.matches[0].entry.entry_id=='km_grouped_repeats'
    assert not k.match_knowledge(facts(('cnn1d',),features=2,repeats=0),publication()).matches
    low=k.match_knowledge(facts(features=2,repeats=0),publication())
    assert [m.entry.entry_id for m in low.matches]==['km_lr_regularized_baseline']
    for match in result.matches:
        entry=next(e for e in publication().entries if e.entry_id==match.entry.entry_id)
        without=tuple(r.model_id for r in facts().recipes if r.model_id not in entry.related_models)
        assert match.entry.entry_id not in {m.entry.entry_id for m in k.match_knowledge(facts(without),publication()).matches}


@pytest.mark.parametrize('change',[
    lambda b:b.update(extra=1),
    lambda b:b.update(schema_version='unknown'),
    lambda b:b['entries'].append(b['entries'][0]),
    lambda b:b['entries'][0].update(sources=[]),
    lambda b:b['entries'][0].update(advice='x'*181),
    lambda b:b['entries'][0].update(related_models=['dscarnet']),
    lambda b:b['entries'][0].update(presentation_priority=True),
    lambda b:b['entries'][0].update(presentation_priority=float('nan')),
    lambda b:b['entries'][0]['conditions'][0].update(op='eval'),
    lambda b:b['entries'][0]['conditions'].append(b['entries'][0]['conditions'][0]),
    lambda b:b['entries'][0].update(conflict_group='alone'),
])
def test_publication_boundary(change):
    body=publication().model_dump(mode='json');change(body)
    with pytest.raises(ValidationError):k.KnowledgeSet.model_validate_json(json.dumps(body))


def test_order_digest_and_deep_immutability():
    base=publication();body=base.model_dump(mode='json');body['entries'].reverse()
    for entry in body['entries']:
        entry['related_models'].reverse()
    reordered=k.KnowledgeSet.model_validate_json(json.dumps(body))
    assert base==reordered
    assert k.match_knowledge(facts(),base)==k.match_knowledge(facts(tuple(reversed(('pls_da','svm','logistic_regression','pca_lda','cnn1d')))),reordered)
    with pytest.raises(ValidationError):base.entries[0].advice='changed'
    body['entries'][0]['entry_version']='2'
    assert digest(body)!=digest(base.model_dump(mode='json'))


def test_projection_preserves_limits_and_context_policy():
    result=k.match_knowledge(facts(),publication())
    projected=k.project_knowledge(result,evidence=False,risks=False)
    assert projected['matched_count']==projected['provided_count']==5
    for entry in projected['entries']:
        assert entry['advice'] and entry['basis'] and entry['limitations']
        for reason in entry['evidence']:
            if reason['field'] not in ('task_type','legal_models'):assert reason['actual'] is None
    assert 'https://' not in json.dumps(projected)
    k.KnowledgeProjection.model_validate_json(json.dumps(projected))


def test_train_only_matching_separates_provenance():
    data=view();plan=build_evaluation_plan(data,EVAL,42)
    first=compute_train_evidence(select_train(data,plan))
    changed=data.x.copy();changed[plan.indices['valid']+plan.indices['test']]=1e20
    other=replace(data,x=changed,dataset_sha256='b'*64)
    second=compute_train_evidence(select_train(other,build_evaluation_plan(other,EVAL,42)))
    def input_for(e):
        s=e.statistics
        return facts(features=s.feature_count,groups=s.sample_group_count,repeats=s.repeated_measurement_group_count)
    assert k.match_knowledge(input_for(first),publication())==k.match_knowledge(input_for(second),publication())
    assert first.evidence_digest!=second.evidence_digest
    assert 'imbalance' not in [r.code for r in first.risks]


def test_conflict_group_name_does_not_merge_unrelated_entry():
    template=publication().model_dump(mode='json')['entries'][0]
    entries=[]
    for index in range(7):
        item=json.loads(json.dumps(template))
        item.update(entry_id='collision' if index==6 else f'entry_{index}',conflict_group=None if index==6 else 'collision')
        entries.append(item)
    knowledge=k.KnowledgeSet.model_validate_json(json.dumps(dict(schema_version='knowledge-entry-v1',
        knowledge_set_version='test-v1',entries=entries)))
    projected=k.project_knowledge(k.match_knowledge(facts(),knowledge),evidence=True,risks=True)
    assert projected['provided_count']==1
    assert projected['provided_entry_ids']==['collision']
    assert projected['omitted_count']==6
