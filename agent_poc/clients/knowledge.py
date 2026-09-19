"""Frozen knowledge projections and negotiated response codecs."""
from __future__ import annotations
from typing import Annotated, Literal
import math
from pydantic import ConfigDict, Field, field_validator, model_validator
from . import contracts as v1
from .contracts import ClosedModel, Identifier, DecisionModeBinding
from .capabilities import digest
from .execution_contracts import CreatedSessionResponse, SessionResponse, HealthResponse, ExperimentResponse
from .preparation import Preparation, RecipeLockedConfig, RECIPE_RESPONSE_MODELS

class KnowledgeSource(ClosedModel):
    source_id: Identifier
    description: str = Field(min_length=1,max_length=240)


class KnowledgeReason(ClosedModel):
    condition_id: Identifier
    field: Literal['task_type','feature_count','sample_group_count','repeated_measurement_group_count','class_count','risks','legal_models']
    actual: int | str | list[str] | None = None
    compared_group_count: int | None = None


class KnowledgeEntryProjection(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    related_recipe_ids: list[str]
    stance: Literal['consider','caution']
    advice: str = Field(min_length=1,max_length=180)
    basis: str = Field(min_length=1,max_length=240)
    limitations: str = Field(min_length=1,max_length=300)
    sources: list[KnowledgeSource] = Field(min_length=1,max_length=4)
    evidence: list[KnowledgeReason] = Field(min_length=1,max_length=8)
    conflict_group: Identifier | None
    conflict_notice: Literal['存在适用范围不同的建议'] | None


class RagEntryProjection(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    related_recipe_ids: list[str]
    title: str = Field(min_length=1, max_length=180)
    body: str = Field(min_length=1, max_length=1200)
    sources: list[KnowledgeSource] = Field(min_length=1, max_length=4)


class KnowledgeProjection(ClosedModel):
    status: Literal['disabled','ready']
    projection_version: Literal['knowledge-projection-v1', 'knowledge-rag-projection-v1'] | None
    matched_count: int = Field(ge=0,le=256)
    provided_count: int = Field(ge=0,le=6)
    omitted_count: int = Field(ge=0,le=256)
    provided_entry_ids: list[Identifier]
    entries: list[KnowledgeEntryProjection | RagEntryProjection] = Field(max_length=6)

    @model_validator(mode='after')
    def coherent(self):
        ids=[e.entry_id for e in self.entries]
        if (ids!=self.provided_entry_ids or len(set(ids))!=len(ids) or len(ids)!=self.provided_count
                or self.matched_count!=self.provided_count+self.omitted_count):
            raise ValueError('knowledge projection count mismatch')
        if self.status=='disabled' and (self.matched_count or self.projection_version is not None):
            raise ValueError('disabled knowledge has outputs')
        if self.status=='ready' and self.projection_version is None:
            raise ValueError('ready knowledge lacks version')
        if len(self.model_dump_json(exclude_unset=True).encode())>16384:
            raise ValueError('knowledge projection too large')
        return self


class KnowledgeMatchSummary(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    related_recipe_ids: list[str]


class KnowledgeProvenance(ClosedModel):
    dataset_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    plan_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    evidence_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    catalog_digest: str = Field(pattern=r'^[a-f0-9]{64}$')


class KnowledgeWire(ClosedModel):
    schema_version: Literal['knowledge-snapshot-v1', 'knowledge-snapshot-rag-v1']
    status: Literal['disabled','ready']
    knowledge_set_version: Identifier | None
    knowledge_set_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    matcher_version: Literal['knowledge-match-v1', 'body-cosine-topk-v1'] | None
    projection_version: Literal['knowledge-projection-v1', 'knowledge-rag-projection-v1'] | None
    semantic_input_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    match_digest: str | None = Field(pattern=r'^[a-f0-9]{64}$')
    matches: list[KnowledgeMatchSummary] = Field(max_length=256)
    projection: KnowledgeProjection
    projection_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    provenance: KnowledgeProvenance | None

    @model_validator(mode='after')
    def integrity(self):
        if self.projection_digest!=digest(self.projection.model_dump(mode='json',exclude_unset=True)):
            raise ValueError('knowledge projection digest mismatch')
        values=(self.knowledge_set_version,self.knowledge_set_digest,self.matcher_version,self.projection_version,
                self.semantic_input_digest,self.match_digest,self.provenance)
        if self.status=='disabled' and (any(v is not None for v in values) or self.matches):
            raise ValueError('disabled knowledge has bindings')
        if self.status=='ready' and any(v is None for v in values):
            raise ValueError('ready knowledge lacks bindings')
        if self.status == 'ready':
            rag = self.schema_version == 'knowledge-snapshot-rag-v1'
            if (self.matcher_version == 'body-cosine-topk-v1') != rag or (self.projection_version == 'knowledge-rag-projection-v1') != rag:
                raise ValueError('knowledge method mismatch')
            if any(isinstance(e, RagEntryProjection) != rag for e in self.projection.entries):
                raise ValueError('knowledge entry method mismatch')
        matches={m.entry_id:m for m in self.matches}
        if len(matches)!=len(self.matches) or len(matches)!=self.projection.matched_count or self.status!=self.projection.status:
            raise ValueError('knowledge matches inconsistent')
        for entry in self.projection.entries:
            if entry.entry_id not in matches:
                raise ValueError('unmatched projection entry')
            match=matches[entry.entry_id]
            if entry.entry_version!=match.entry_version or entry.related_recipe_ids!=match.related_recipe_ids:
                raise ValueError('projection match binding mismatch')
        return self

    def validate_refs(self,refs):
        if len(refs)>6 or len(refs)!=len(set(refs)) or set(refs)-set(self.projection.provided_entry_ids):
            raise ValueError('knowledge reference not provided')


class KnowledgePreparation(Preparation):
    knowledge: KnowledgeWire

    @model_validator(mode='after')
    def knowledge_binding(self):
        k=self.knowledge
        if k.status=='ready':
            p=k.provenance
            if (p.dataset_sha256!=self.catalog.dataset_sha256 or p.catalog_digest!=self.catalog.catalog_digest
                    or p.plan_digest!=self.evaluation_plan.plan_digest or p.evidence_digest!=self.evidence.evidence_digest):
                raise ValueError('knowledge provenance mismatch')
            legal={r.recipe_id for r in self.catalog.recipes}
            if any(set(m.related_recipe_ids)-legal for m in k.matches):
                raise ValueError('knowledge outside legal recipes')
        return self


class KnowledgeLockedConfig(RecipeLockedConfig, DecisionModeBinding):
    protocol_revision: Literal['agent-recipes-revision-v2']
    preparation: KnowledgePreparation

    @field_validator('modules')
    @classmethod
    def known_modules(cls,value):
        if value not in (['train_evidence','legal_recipes'],['train_evidence','legal_recipes','knowledge']):
            raise ValueError('invalid knowledge modules')
        return value

    @model_validator(mode='after')
    def module_binding(self):
        if ('knowledge' in self.modules)!=(self.preparation.knowledge.status=='ready'):
            raise ValueError('knowledge switch mismatch')
        return self


class KnowledgeCreatedSessionResponse(CreatedSessionResponse):
    locked_config: KnowledgeLockedConfig


class KnowledgeSessionResponse(SessionResponse):
    locked_config: KnowledgeLockedConfig


class KnowledgeCapability(ClosedModel):
    available: Literal[True]
    status: Literal['ready']
    schema_version: Literal['knowledge-snapshot-v1', 'knowledge-snapshot-rag-v1']
    reason: str | None


class KnowledgeHealthResponse(HealthResponse):
    protocol_revision: Literal['agent-recipes-revision-v2']
    execution_profiles: list[Literal['train-evidence-recipes-v1']]
    modules: dict[str, v1.ModuleCapability | KnowledgeCapability]

    @field_validator('modules')
    @classmethod
    def known_modules(cls,value):
        from agent_poc.tools import MODULES
        if set(value)-set(MODULES)-{'knowledge'} or 'knowledge' not in value:
            raise ValueError('unknown module')
        if not isinstance(value['knowledge'],KnowledgeCapability):
            raise ValueError('knowledge capability missing')
        return value


class KnowledgeReference(ClosedModel):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')


class KnowledgeDecision(ClosedModel):
    schema_version: Literal['knowledge-decision-v1']
    references: list[KnowledgeReference] = Field(max_length=6)
    projection_digest: str = Field(pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def unique_references(self):
        if len({r.entry_id for r in self.references})!=len(self.references):
            raise ValueError('duplicate knowledge decision reference')
        return self


class KnowledgeExperimentResponse(ExperimentResponse):
    decision_metadata: KnowledgeDecision


KNOWLEDGE_RESPONSE_MODELS = {**RECIPE_RESPONSE_MODELS,
    'inspect_ml_capabilities':KnowledgeHealthResponse,'start_ml_session':KnowledgeCreatedSessionResponse,
    'inspect_ml_session':KnowledgeSessionResponse,'submit_ml_experiment':KnowledgeExperimentResponse}
