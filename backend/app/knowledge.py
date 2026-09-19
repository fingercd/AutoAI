"""Frozen, finite modeling advice. Matching consumes only normalized Train facts."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .evaluation_plan import digest
from .model_catalog import MODELS_BY_ID
from .recipes import RecipeCatalog
from .train_evidence import TrainEvidence

REVISION = 'agent-recipes-revision-v2'
SNAPSHOT_VERSION = 'knowledge-snapshot-v1'
MATCHER_VERSION = 'knowledge-match-v1'
PROJECTION_VERSION = 'knowledge-projection-v1'
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')]
Sha256 = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
RiskCode = Literal['small_sample', 'high_dimension', 'imbalance', 'zero_variance',
                   'duplicate_features', 'conflicting_vectors', 'repeated_measurement']
Statistic = Literal['feature_count', 'sample_group_count', 'repeated_measurement_group_count', 'class_count']


class KnowledgeError(RuntimeError):
    """Publication or stored snapshot failure; never an empty-match result."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)


class TaskEq(FrozenModel):
    kind: Literal['task_eq']
    condition_id: Identifier
    field: Literal['task_type']
    op: Literal['eq']
    value: Literal['classification']


class IntCompare(FrozenModel):
    kind: Literal['int_compare']
    condition_id: Identifier
    field: Statistic
    op: Literal['gt', 'ge', 'eq']
    value: Annotated[int, Field(ge=0)]


class FieldCompare(FrozenModel):
    kind: Literal['field_compare']
    condition_id: Identifier
    field: Literal['feature_count']
    op: Literal['gt']
    other_field: Literal['sample_group_count']


class RiskPresent(FrozenModel):
    kind: Literal['risk_present']
    condition_id: Identifier
    op: Literal['present']
    value: RiskCode


class LegalModelAny(FrozenModel):
    kind: Literal['legal_model_any']
    condition_id: Identifier
    op: Literal['any']
    models: Annotated[tuple[Identifier, ...], Field(min_length=1)]

    @model_validator(mode='after')
    def canonical_models(self):
        if set(self.models) - MODELS_BY_ID.keys():
            raise ValueError('unknown canonical model')
        object.__setattr__(self, 'models', tuple(sorted(set(self.models))))
        return self


Condition = Annotated[TaskEq | IntCompare | FieldCompare | RiskPresent | LegalModelAny,
                      Field(discriminator='kind')]


class Source(FrozenModel):
    source_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=180)]
    publisher: Annotated[str, Field(min_length=1, max_length=100)]
    url: Annotated[str, Field(pattern=r'^https://[^\s]+$', max_length=500)]
    locator: Annotated[str, Field(min_length=1, max_length=160)]
    supported_claim: Annotated[str, Field(min_length=1, max_length=240)]
    applicability_limit: Annotated[str, Field(min_length=1, max_length=300)]


class KnowledgeEntry(FrozenModel):
    schema_version: Literal['knowledge-entry-v1']
    entry_id: Identifier
    entry_version: Annotated[str, Field(pattern=r'^[0-9]{1,8}$')]
    conditions: Annotated[tuple[Condition, ...], Field(min_length=1, max_length=8)]
    related_models: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    advice: Annotated[str, Field(min_length=1, max_length=180)]
    basis: Annotated[str, Field(min_length=1, max_length=240)]
    limitations: Annotated[str, Field(min_length=1, max_length=300)]
    sources: Annotated[tuple[Source, ...], Field(min_length=1, max_length=4)]
    stance: Literal['consider', 'caution']
    presentation_priority: Annotated[int, Field(ge=0, le=100)]
    conflict_group: Identifier | None

    @model_validator(mode='after')
    def normalize(self):
        if set(self.related_models) - MODELS_BY_ID.keys():
            raise ValueError('unknown canonical model')
        if len({c.condition_id for c in self.conditions}) != len(self.conditions):
            raise ValueError('duplicate condition ID')
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError('duplicate source ID')
        object.__setattr__(self, 'related_models', tuple(sorted(set(self.related_models))))
        object.__setattr__(self, 'sources', tuple(sorted(self.sources, key=lambda s: s.source_id)))
        return self


class KnowledgeSet(FrozenModel):
    schema_version: Literal['knowledge-entry-v1']
    knowledge_set_version: Identifier
    entries: Annotated[tuple[KnowledgeEntry, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode='after')
    def closed(self):
        if len({e.entry_id for e in self.entries}) != len(self.entries):
            raise ValueError('duplicate entry ID')
        groups: dict[str, int] = {}
        sources: dict[str, Source] = {}
        for entry in self.entries:
            if entry.conflict_group is not None:
                groups[entry.conflict_group] = groups.get(entry.conflict_group, 0) + 1
            for source in entry.sources:
                if source.source_id in sources and sources[source.source_id] != source:
                    raise ValueError('conflicting source definition')
                sources[source.source_id] = source
        if any(count < 2 for count in groups.values()):
            raise ValueError('unclosed conflict group')
        object.__setattr__(self, 'entries', tuple(sorted(self.entries, key=lambda e: e.entry_id)))
        return self


class RecipeReference(FrozenModel):
    recipe_id: Annotated[str, Field(pattern=r'^recipe_[a-f0-9]{64}$')]
    model_id: Identifier


class KnowledgeInput(FrozenModel):
    task_type: Literal['classification']
    statistics_version: Literal['train-statistics-v1']
    risk_version: Literal['train-risk-rules-v1']
    feature_count: Annotated[int, Field(ge=0)]
    sample_group_count: Annotated[int, Field(ge=0)]
    repeated_measurement_group_count: Annotated[int, Field(ge=0)]
    class_count: Annotated[int, Field(ge=0)]
    risks: tuple[RiskCode, ...]
    recipes: tuple[RecipeReference, ...]

    @model_validator(mode='after')
    def canonical(self):
        if len({r.recipe_id for r in self.recipes}) != len(self.recipes):
            raise ValueError('duplicate recipe reference')
        object.__setattr__(self, 'recipes', tuple(sorted(self.recipes, key=lambda r: r.recipe_id)))
        object.__setattr__(self, 'risks', tuple(sorted(set(self.risks))))
        return self

    @classmethod
    def from_train(cls, evidence: TrainEvidence, catalog: RecipeCatalog):
        stats = evidence.statistics
        return cls(task_type='classification', statistics_version=evidence.statistics_version,
                   risk_version=evidence.risk_version, feature_count=stats.feature_count,
                   sample_group_count=stats.sample_group_count,
                   repeated_measurement_group_count=stats.repeated_measurement_group_count,
                   class_count=len(stats.classes), risks=tuple(r.code for r in evidence.risks),
                   recipes=tuple(RecipeReference(recipe_id=r.recipe_id, model_id=r.model_id) for r in catalog.recipes))


class ConditionEvidence(FrozenModel):
    condition_id: Identifier
    field: Literal['task_type', 'feature_count', 'sample_group_count', 'repeated_measurement_group_count',
                   'class_count', 'risks', 'legal_models']
    actual: int | str | tuple[str, ...]
    compared_group_count: int | None = None


class MatchSource(FrozenModel):
    source_id: Identifier
    supported_claim: Annotated[str, Field(min_length=1, max_length=240)]


class MatchedEntry(FrozenModel):
    """Matched advice, without a second copy of publication conditions/metadata."""
    entry_id: Identifier
    entry_version: Annotated[str, Field(pattern=r'^[0-9]{1,8}$')]
    advice: Annotated[str, Field(min_length=1, max_length=180)]
    basis: Annotated[str, Field(min_length=1, max_length=240)]
    limitations: Annotated[str, Field(min_length=1, max_length=300)]
    sources: tuple[MatchSource, ...]
    stance: Literal['consider','caution']
    presentation_priority: Annotated[int, Field(ge=0, le=100)]
    conflict_group: Identifier | None

    @classmethod
    def from_entry(cls, entry: KnowledgeEntry):
        return cls(entry_id=entry.entry_id, entry_version=entry.entry_version, advice=entry.advice,
            basis=entry.basis, limitations=entry.limitations, stance=entry.stance,
            presentation_priority=entry.presentation_priority, conflict_group=entry.conflict_group,
            sources=tuple(MatchSource(source_id=s.source_id, supported_claim=s.supported_claim) for s in entry.sources))


class KnowledgeMatch(FrozenModel):
    entry: MatchedEntry
    related_recipe_ids: tuple[str, ...]
    evidence: tuple[ConditionEvidence, ...]


class KnowledgeMatchResult(FrozenModel):
    semantic_input_digest: Sha256
    match_digest: Sha256
    matches: tuple[KnowledgeMatch, ...]


def match_knowledge(facts: KnowledgeInput, knowledge: KnowledgeSet) -> KnowledgeMatchResult:
    matched = []
    legal_models = {r.model_id for r in facts.recipes}
    for entry in knowledge.entries:
        recipe_ids = tuple(r.recipe_id for r in facts.recipes if r.model_id in entry.related_models)
        if not recipe_ids:
            continue
        evidence = []
        for condition in entry.conditions:
            compared = None
            match condition:
                case TaskEq():
                    field, actual = 'task_type', facts.task_type
                    applies = actual == condition.value
                case IntCompare():
                    field, actual = condition.field, getattr(facts, condition.field)
                    applies = {'gt': actual > condition.value, 'ge': actual >= condition.value,
                               'eq': actual == condition.value}[condition.op]
                case FieldCompare():
                    field, actual, compared = 'feature_count', facts.feature_count, facts.sample_group_count
                    applies = actual > compared
                case RiskPresent():
                    field, actual = 'risks', condition.value
                    applies = condition.value in facts.risks
                case LegalModelAny():
                    field, actual = 'legal_models', tuple(sorted(legal_models.intersection(condition.models)))
                    applies = bool(actual)
            if not applies:
                break
            evidence.append(ConditionEvidence(condition_id=condition.condition_id, field=field,
                                              actual=actual, compared_group_count=compared))
        else:
            matched.append(KnowledgeMatch(entry=MatchedEntry.from_entry(entry), related_recipe_ids=recipe_ids, evidence=tuple(evidence)))
    matches = tuple(sorted(matched, key=lambda m: (m.entry.presentation_priority, m.entry.entry_id, m.entry.entry_version)))
    semantic = digest(dict(input=facts.model_dump(mode='json'), knowledge_set=knowledge.model_dump(mode='json'),
                           matcher_version=MATCHER_VERSION))
    return KnowledgeMatchResult(semantic_input_digest=semantic,
                                match_digest=digest([m.model_dump(mode='json') for m in matches]), matches=matches)


def project_knowledge(result: KnowledgeMatchResult, *, evidence: bool, risks: bool) -> dict:
    groups: dict[tuple[str, str], list[KnowledgeMatch]] = {}
    for match in result.matches:
        key = ('group', match.entry.conflict_group) if match.entry.conflict_group else ('entry', match.entry.entry_id)
        groups.setdefault(key, []).append(match)
    units = sorted(groups.values(), key=lambda unit: (min(m.entry.presentation_priority for m in unit),
                                                     min(m.entry.entry_id for m in unit)))
    def block(entries):
        return dict(status='ready', projection_version=PROJECTION_VERSION, matched_count=len(result.matches),
                    provided_count=len(entries), omitted_count=len(result.matches)-len(entries),
                    provided_entry_ids=[e['entry_id'] for e in entries], entries=entries)
    entries = []
    for unit in units:
        projected = []
        for match in unit:
            entry = match.entry
            reasons = []
            for item in match.evidence:
                if item.field == 'risks' and not risks:
                    continue
                reason = dict(condition_id=item.condition_id, field=item.field, actual=None, compared_group_count=None)
                if evidence or item.field in ('task_type', 'legal_models', 'risks'):
                    reason['actual'] = item.model_dump(mode='json')['actual']
                    if item.compared_group_count is not None:
                        reason['compared_group_count'] = item.compared_group_count
                reasons.append(reason)
            projected.append(dict(entry_id=entry.entry_id, entry_version=entry.entry_version,
                related_recipe_ids=list(match.related_recipe_ids), stance=entry.stance,
                advice=entry.advice, basis=entry.basis, limitations=entry.limitations,
                sources=[dict(source_id=s.source_id, description=s.supported_claim) for s in entry.sources],
                evidence=reasons, conflict_group=entry.conflict_group,
                conflict_notice='存在适用范围不同的建议' if entry.conflict_group else None))
        candidate = entries + projected
        if len(candidate) <= 6 and len(json.dumps(block(candidate), ensure_ascii=False, sort_keys=True,
                                                separators=(',', ':'), allow_nan=False).encode()) <= 16384:
            entries = candidate
    return block(entries)


@lru_cache(maxsize=4)
def load_knowledge(path: Path) -> KnowledgeSet:
    """A deployment reload chooses a new path or explicitly clears this cache."""
    try:
        return KnowledgeSet.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValidationError) as exc:
        raise KnowledgeError('knowledge publication unavailable or invalid') from exc


def configured_knowledge() -> KnowledgeSet:
    return load_knowledge(Path(os.environ.get('AUTOAI_KNOWLEDGE_FILE',
        str(Path(__file__).parent / 'knowledge_data' / 'common_modeling_v1.json'))))


class ProjectedSource(FrozenModel):
    source_id: Identifier
    description: Annotated[str, Field(min_length=1, max_length=240)]


class ProjectedReason(FrozenModel):
    condition_id: Identifier
    field: Literal['task_type', 'feature_count', 'sample_group_count', 'repeated_measurement_group_count',
                   'class_count', 'risks', 'legal_models']
    actual: int | str | tuple[str, ...] | None = None
    compared_group_count: int | None = None


class ProjectedEntry(FrozenModel):
    entry_id: Identifier
    entry_version: Annotated[str, Field(pattern=r'^[0-9]{1,8}$')]
    related_recipe_ids: tuple[str, ...]
    stance: Literal['consider', 'caution']
    advice: Annotated[str, Field(min_length=1, max_length=180)]
    basis: Annotated[str, Field(min_length=1, max_length=240)]
    limitations: Annotated[str, Field(min_length=1, max_length=300)]
    sources: tuple[ProjectedSource, ...]
    evidence: tuple[ProjectedReason, ...]
    conflict_group: Identifier | None
    conflict_notice: Literal['存在适用范围不同的建议'] | None


class RagProjectedEntry(FrozenModel):
    entry_id: Identifier
    entry_version: Annotated[str, Field(pattern=r'^[0-9]{1,8}$')]
    related_recipe_ids: tuple[str, ...]
    title: Annotated[str, Field(min_length=1, max_length=180)]
    body: Annotated[str, Field(min_length=1, max_length=1200)]
    sources: tuple[ProjectedSource, ...]


class KnowledgeProjection(FrozenModel):
    status: Literal['disabled', 'ready']
    projection_version: Literal['knowledge-projection-v1', 'knowledge-rag-projection-v1'] | None
    matched_count: Annotated[int, Field(ge=0, le=256)]
    provided_count: Annotated[int, Field(ge=0, le=6)]
    omitted_count: Annotated[int, Field(ge=0, le=256)]
    provided_entry_ids: tuple[Identifier, ...]
    entries: tuple[ProjectedEntry | RagProjectedEntry, ...]

    @model_validator(mode='after')
    def counts(self):
        ids = tuple(e.entry_id for e in self.entries)
        if (ids != self.provided_entry_ids or len(set(ids)) != len(ids) or
            len(ids) != self.provided_count or self.matched_count != self.provided_count + self.omitted_count):
            raise ValueError('projection counts mismatch')
        if self.status == 'disabled' and (self.matched_count or self.projection_version is not None):
            raise ValueError('disabled projection has knowledge')
        if self.status == 'ready' and self.projection_version not in (PROJECTION_VERSION, 'knowledge-rag-projection-v1'):
            raise ValueError('ready projection requires version')
        if self.status == 'ready' and any(isinstance(e, RagProjectedEntry) != (self.projection_version == 'knowledge-rag-projection-v1') for e in self.entries):
            raise ValueError('projection method mismatch')
        if len(self.model_dump_json(exclude_unset=True).encode()) > 16384:
            raise ValueError('projection too large')
        return self


class Provenance(FrozenModel):
    dataset_sha256: Sha256
    plan_digest: Sha256
    evidence_digest: Sha256
    catalog_digest: Sha256


class ProjectionPolicy(FrozenModel):
    evidence: bool
    risks: bool


class KnowledgeSnapshot(FrozenModel):
    schema_version: Literal['knowledge-snapshot-v1']
    status: Literal['disabled', 'ready']
    knowledge_set: KnowledgeSet | None
    knowledge_set_digest: Sha256 | None
    matcher_version: Literal['knowledge-match-v1'] | None
    projection_version: Literal['knowledge-projection-v1'] | None
    semantic_input: KnowledgeInput | None
    semantic_input_digest: Sha256 | None
    match_digest: Sha256 | None
    matches: tuple[KnowledgeMatch, ...]
    projection: KnowledgeProjection
    projection_digest: Sha256
    provenance: Provenance | None
    projection_policy: ProjectionPolicy | None

    @model_validator(mode='after')
    def integrity(self):
        if self.projection_digest != digest(self.projection.model_dump(mode='json', exclude_unset=True)):
            raise ValueError('projection digest mismatch')
        if self.projection.status != self.status:
            raise ValueError('snapshot status mismatch')
        bindings = (self.knowledge_set, self.knowledge_set_digest, self.matcher_version, self.projection_version,
                    self.semantic_input, self.semantic_input_digest, self.match_digest, self.provenance, self.projection_policy)
        if self.status == 'disabled':
            if any(value is not None for value in bindings) or self.matches:
                raise ValueError('disabled snapshot has knowledge bindings')
        else:
            if any(value is None for value in bindings):
                raise ValueError('missing knowledge binding')
            if self.knowledge_set_digest != digest(self.knowledge_set.model_dump(mode='json')):
                raise ValueError('knowledge set digest mismatch')
            if self.semantic_input_digest != digest(dict(input=self.semantic_input.model_dump(mode='json'),
                knowledge_set=self.knowledge_set.model_dump(mode='json'), matcher_version=self.matcher_version)):
                raise ValueError('semantic input digest mismatch')
            if self.match_digest != digest([m.model_dump(mode='json') for m in self.matches]):
                raise ValueError('match digest mismatch')
            entries = {e.entry_id: MatchedEntry.from_entry(e) for e in self.knowledge_set.entries}
            matched = {m.entry.entry_id: m for m in self.matches}
            legal = {r.recipe_id for r in self.semantic_input.recipes}
            if len(matched) != len(self.matches) or self.projection.matched_count != len(self.matches):
                raise ValueError('match count mismatch')
            for match in self.matches:
                if entries.get(match.entry.entry_id) != match.entry or set(match.related_recipe_ids) - legal:
                    raise ValueError('match outside frozen knowledge or recipes')
            for entry in self.projection.entries:
                if entry.entry_id not in matched or entry.entry_version != matched[entry.entry_id].entry.entry_version:
                    raise ValueError('projection outside frozen matches')
            expected = project_knowledge(KnowledgeMatchResult(semantic_input_digest=self.semantic_input_digest,
                match_digest=self.match_digest, matches=self.matches), evidence=self.projection_policy.evidence,
                risks=self.projection_policy.risks)
            if expected != self.projection.model_dump(mode='json', exclude_unset=True):
                raise ValueError('projection differs from frozen matches and policy')
        return self

    def wire(self) -> dict:
        body = self.model_dump(mode='json', exclude_unset=True,
            exclude={'knowledge_set', 'semantic_input', 'matches', 'projection_policy'})
        body['schema_version'] = self.schema_version
        body['knowledge_set_version'] = self.knowledge_set.knowledge_set_version if self.knowledge_set else None
        body['matches'] = [dict(entry_id=m.entry.entry_id, entry_version=m.entry.entry_version,
                              related_recipe_ids=list(m.related_recipe_ids)) for m in self.matches]
        return body


def _freeze_legacy_knowledge(*, enabled: bool, evidence: TrainEvidence, catalog: RecipeCatalog,
                     evidence_context: bool, risk_context: bool) -> KnowledgeSnapshot:
    if enabled:
        knowledge = configured_knowledge()
        facts = KnowledgeInput.from_train(evidence, catalog)
        result = match_knowledge(facts, knowledge)
        projection = project_knowledge(result, evidence=evidence_context, risks=risk_context)
        body = dict(status='ready', knowledge_set=knowledge, knowledge_set_digest=digest(knowledge.model_dump(mode='json')),
                    matcher_version=MATCHER_VERSION, projection_version=PROJECTION_VERSION, semantic_input=facts,
                    semantic_input_digest=result.semantic_input_digest, match_digest=result.match_digest, matches=result.matches,
                    projection_policy=ProjectionPolicy(evidence=evidence_context, risks=risk_context),
                    provenance=Provenance(dataset_sha256=evidence.dataset_sha256, plan_digest=evidence.plan_digest,
                                          evidence_digest=evidence.evidence_digest, catalog_digest=catalog.catalog_digest))
    else:
        projection = dict(status='disabled', projection_version=None, matched_count=0, provided_count=0,
                          omitted_count=0, provided_entry_ids=[], entries=[])
        body = dict(status='disabled', knowledge_set=None, knowledge_set_digest=None, matcher_version=None,
                    projection_version=None, semantic_input=None, semantic_input_digest=None, match_digest=None,
                    matches=(), provenance=None, projection_policy=None)
    return KnowledgeSnapshot(**body, schema_version=SNAPSHOT_VERSION, projection=KnowledgeProjection.model_validate_json(json.dumps(projection)),
                             projection_digest=digest(projection))


def decode_snapshot(raw: dict) -> KnowledgeSnapshot:
    try:
        model = RagSnapshot if raw.get('schema_version') == 'knowledge-snapshot-rag-v1' else KnowledgeSnapshot
        return model.model_validate_json(json.dumps(raw, allow_nan=False))
    except (ValidationError, ValueError) as exc:
        raise KnowledgeError('stored knowledge snapshot invalid') from exc


class KnowledgeReference(FrozenModel):
    entry_id: Identifier
    entry_version: Annotated[str, Field(pattern=r'^[0-9]{1,8}$')]


class KnowledgeDecision(FrozenModel):
    schema_version: Literal['knowledge-decision-v1']
    references: Annotated[tuple[KnowledgeReference, ...], Field(max_length=6)]
    projection_digest: Sha256

    @model_validator(mode='after')
    def unique(self):
        if len({r.entry_id for r in self.references}) != len(self.references):
            raise ValueError('duplicate reference')
        return self


def resolve_decision(snapshot: KnowledgeSnapshot, references: list[str]) -> KnowledgeDecision:
    entries = {e.entry_id: e for e in snapshot.projection.entries}
    if len(references) > 6 or len(references) != len(set(references)) or set(references) - entries.keys():
        raise ValueError('knowledge references not provided')
    return KnowledgeDecision(schema_version='knowledge-decision-v1',references=tuple(KnowledgeReference(entry_id=key, entry_version=entries[key].entry_version)
                                             for key in sorted(references)), projection_digest=snapshot.projection_digest)


# The original snapshot class above is a read/verification adapter for persisted
# rule-based sessions. New enabled sessions use only this retrieval path.
from .knowledge_retrieval import Card, CardLibrary, ENCODING, RetrievalError, sha256


class RagQuery(FrozenModel):
    mode: Literal['train_template', 'user_text']
    template_version: Literal['train-query-v1', 'user-text-v1']
    text: Annotated[str, Field(min_length=1, max_length=4096)]
    text_sha256: Sha256

    @model_validator(mode='after')
    def integrity(self):
        expected = 'train-query-v1' if self.mode == 'train_template' else 'user-text-v1'
        if self.template_version != expected or self.text_sha256 != sha256(self.text.encode()):
            raise ValueError('query binding mismatch')
        return self


class RagMatch(FrozenModel):
    entry: Card
    related_recipe_ids: tuple[str, ...]
    score: Annotated[float, Field(ge=-1.00001, le=1.00001, allow_inf_nan=False)]


def project_rag(matches: tuple[RagMatch, ...], *, max_bytes=16384) -> dict:
    entries = []
    def block(items):
        return dict(status='ready', projection_version='knowledge-rag-projection-v1',
                    matched_count=len(matches), provided_count=len(items), omitted_count=len(matches)-len(items),
                    provided_entry_ids=[e['entry_id'] for e in items], entries=items)
    for match in matches:
        card = match.entry
        entry = dict(entry_id=card.entry_id, entry_version=card.entry_version,
                     related_recipe_ids=list(match.related_recipe_ids), title=card.title, body=card.body,
                     sources=[dict(source_id=s.source_id, description=s.locator) for s in card.sources])
        candidate = block(entries + [entry])
        if len(json.dumps(candidate, ensure_ascii=False, separators=(',', ':')).encode()) <= max_bytes:
            entries.append(entry)
    return block(entries)


class RagSnapshot(FrozenModel):
    schema_version: Literal['knowledge-snapshot-rag-v1'] = 'knowledge-snapshot-rag-v1'
    status: Literal['ready'] = 'ready'
    knowledge_set: CardLibrary
    knowledge_set_digest: Sha256
    matcher_version: Literal['body-cosine-topk-v1'] = 'body-cosine-topk-v1'
    projection_version: Literal['knowledge-rag-projection-v1'] = 'knowledge-rag-projection-v1'
    query: RagQuery
    index_manifest: dict
    index_manifest_digest: Sha256
    semantic_input_digest: Sha256
    retrieval_policy: dict
    projection_policy: ProjectionPolicy
    match_digest: Sha256
    matches: Annotated[tuple[RagMatch, ...], Field(max_length=6)]
    projection: KnowledgeProjection
    projection_digest: Sha256
    provenance: Provenance

    @model_validator(mode='after')
    def integrity(self):
        from .knowledge_retrieval import row_map
        if self.knowledge_set_digest != digest(self.knowledge_set.model_dump(mode='json')):
            raise ValueError('frozen cards mismatch')
        if self.index_manifest_digest != digest(self.index_manifest):
            raise ValueError('frozen manifest mismatch')
        if self.index_manifest.get('encoding') != ENCODING or self.index_manifest.get('rows') != row_map(self.knowledge_set):
            raise ValueError('frozen encoder or rows mismatch')
        if self.index_manifest.get('cards_sha256') != sha256(self.knowledge_set.model_dump_json().encode()):
            raise ValueError('frozen publication mismatch')
        policy = self.retrieval_policy
        if set(policy) != {'k', 'tau', 'scope', 'domain', 'recipes', 'projection_max_bytes'}:
            raise ValueError('invalid retrieval policy')
        if type(policy['k']) is not int or not 1 <= policy['k'] <= 6 or len(self.matches) > policy['k']:
            raise ValueError('invalid top-k')
        if type(policy['projection_max_bytes']) is not int or not 256 <= policy['projection_max_bytes'] <= 16384:
            raise ValueError('invalid projection budget')
        if self.semantic_input_digest != digest(dict(query=self.query.model_dump(mode='json'), policy=policy,
                                                       projection_policy=self.projection_policy.model_dump(mode='json'), index_manifest_digest=self.index_manifest_digest)):
            raise ValueError('frozen retrieval inputs mismatch')
        if self.match_digest != digest([m.model_dump(mode='json') for m in self.matches]):
            raise ValueError('frozen retrieval result mismatch')
        cards = {c.entry_id: c for c in self.knowledge_set.entries}
        ids = [m.entry.entry_id for m in self.matches]
        if len(ids) != len(set(ids)) or list(self.matches) != sorted(self.matches, key=lambda m: (-m.score, m.entry.entry_id)):
            raise ValueError('duplicate or unordered retrieval')
        for match in self.matches:
            c = match.entry
            recipes = tuple(r['recipe_id'] for r in policy['recipes'] if not c.related_models or r['model_id'] in c.related_models)
            if (cards.get(c.entry_id) != c or c.status != 'published' or recipes != match.related_recipe_ids
                    or not recipes or (c.scopes and policy['scope'] not in c.scopes)
                    or ('general' not in c.domains and policy['domain'] not in c.domains)
                    or (policy['tau'] is not None and match.score < policy['tau'])):
                raise ValueError('retrieval outside frozen scope or catalog')
        expected = project_rag(self.matches, max_bytes=policy['projection_max_bytes'])
        if expected != self.projection.model_dump(mode='json') or digest(expected) != self.projection_digest:
            raise ValueError('frozen projection mismatch')
        return self

    def wire(self):
        # Query, numeric score, scopes and full sources never reach the LLM/client.
        return dict(schema_version=self.schema_version, status=self.status,
                    knowledge_set_version=self.knowledge_set.knowledge_set_version,
                    knowledge_set_digest=self.knowledge_set_digest, matcher_version=self.matcher_version,
                    projection_version=self.projection_version, semantic_input_digest=self.semantic_input_digest,
                    match_digest=self.match_digest, matches=[dict(entry_id=m.entry.entry_id,
                    entry_version=m.entry.entry_version, related_recipe_ids=list(m.related_recipe_ids)) for m in self.matches],
                    projection=self.projection.model_dump(mode='json'), projection_digest=self.projection_digest,
                    provenance=self.provenance.model_dump(mode='json'))


@lru_cache(maxsize=2)
def _rag_encoder(path: str):
    from .knowledge_retrieval import BodyEncoder
    return BodyEncoder(Path(path))


def freeze_knowledge(*, enabled: bool, evidence: TrainEvidence, catalog: RecipeCatalog,
                     evidence_context: bool, risk_context: bool, query_mode='train_template',
                     user_text=None, domain=None, scope='local'):
    if not enabled:
        return _freeze_legacy_knowledge(enabled=False, evidence=evidence, catalog=catalog,
                                        evidence_context=evidence_context, risk_context=risk_context)
    from .knowledge_retrieval import current_bundle, build_query, retrieve
    try:
        bundle = current_bundle(Path(os.environ.get('AUTOAI_KNOWLEDGE_BUNDLE', 'storage/knowledge')))
        query = RagQuery(**build_query(evidence, mode=query_mode, user_text=user_text, domain=domain))
        encoder = _rag_encoder(os.environ.get('AUTOAI_EMBEDDING_MODEL', 'storage/models/bge-small-zh-v1.5'))
        vector = encoder.encode([query.text], query=True)[0]
        recipes = [dict(recipe_id=r.recipe_id, model_id=r.model_id) for r in catalog.recipes]
        policy = dict(k=3, tau=None, scope=scope, domain=domain, recipes=recipes, projection_max_bytes=16384)
        retrieved = retrieve(bundle, vector, models={r.model_id for r in catalog.recipes}, scope=scope, domain=domain)
        matches = tuple(RagMatch(entry=item['card'], score=item['score'], related_recipe_ids=tuple(
            r.recipe_id for r in catalog.recipes if not item['card'].related_models or r.model_id in item['card'].related_models))
            for item in retrieved)
        projection = project_rag(matches)
        manifest_digest = digest(bundle.manifest)
        return RagSnapshot(schema_version='knowledge-snapshot-rag-v1', status='ready',
            matcher_version='body-cosine-topk-v1', projection_version='knowledge-rag-projection-v1', knowledge_set=bundle.cards, knowledge_set_digest=digest(bundle.cards.model_dump(mode='json')),
            query=query, index_manifest=bundle.manifest, index_manifest_digest=manifest_digest,
            semantic_input_digest=digest(dict(query=query.model_dump(mode='json'), policy=policy, projection_policy=dict(evidence=evidence_context, risks=risk_context), index_manifest_digest=manifest_digest)),
            projection_policy=ProjectionPolicy(evidence=evidence_context, risks=risk_context),
            retrieval_policy=policy, matches=matches, match_digest=digest([m.model_dump(mode='json') for m in matches]),
            projection=KnowledgeProjection.model_validate_json(json.dumps(projection)), projection_digest=digest(projection),
            provenance=Provenance(dataset_sha256=evidence.dataset_sha256, plan_digest=evidence.plan_digest,
                                  evidence_digest=evidence.evidence_digest, catalog_digest=catalog.catalog_digest))
    except (RetrievalError, ValueError, OSError) as exc:
        raise KnowledgeError('knowledge retrieval unavailable or invalid') from exc
