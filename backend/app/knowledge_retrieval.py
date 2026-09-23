"""Small immutable body-only retrieval bundles; no embedding dependency at import."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation_plan import digest

MODEL_ID = 'BAAI/bge-small-zh-v1.5'
MODEL_REVISION = '7999e1d3359715c523056ef9478215996d62a620'
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'
ENCODING = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                tokenizer_revision=MODEL_REVISION, pooling='cls', normalization='l2',
                dtype='float32', max_tokens=512, dimension=512,
                body_prefix='', query_prefix=QUERY_PREFIX, truncation=False)
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')]


class RetrievalError(RuntimeError):
    """An invalid publication or encoder must never become an empty retrieval."""


class Closed(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)


class CardSource(Closed):
    source_id: Identifier
    title: str = Field(min_length=1, max_length=180)
    locator: str = Field(min_length=1, max_length=160)
    publisher: str = Field(min_length=1, max_length=100)
    url: str | None = Field(default=None, pattern=r'^https://[^\s]+$', max_length=500)
    supported_claim: str = Field(min_length=1, max_length=500)
    applicability_limit: str = Field(min_length=1, max_length=500)


class Card(Closed):
    entry_id: Identifier
    entry_version: str = Field(pattern=r'^[0-9]{1,8}$')
    title: str = Field(min_length=1, max_length=180)
    body: str = Field(min_length=1, max_length=1200)
    tags: tuple[Identifier, ...] = ()
    task_types: tuple[Literal['classification'], ...] = ('classification',)
    domains: tuple[Identifier, ...] = ('general',)
    related_models: tuple[Identifier, ...] = ()
    sources: tuple[CardSource, ...] = Field(min_length=1, max_length=4)
    status: Literal['draft', 'published', 'retired']
    # Private publications use exact Principal scope strings; general cards are public.
    scopes: tuple[str, ...] = ()

    @model_validator(mode='after')
    def valid(self):
        from .model_catalog import MODELS_BY_ID
        if set(self.related_models) - MODELS_BY_ID.keys():
            raise ValueError('unknown card model')
        if not self.body.strip() or not self.domains or not self.task_types:
            raise ValueError('incomplete card')
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError('duplicate source')
        return self


class CardLibrary(Closed):
    schema_version: Literal['knowledge-cards-v1'] = 'knowledge-cards-v1'
    knowledge_set_version: Identifier
    entries: tuple[Card, ...] = Field(max_length=256)

    @model_validator(mode='after')
    def unique(self):
        if len({e.entry_id for e in self.entries}) != len(self.entries):
            raise ValueError('duplicate card ID')
        return self


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class BodyEncoder:
    """Fixed local weights, CPU correctness baseline, no network or remote code."""
    def __init__(self, directory: Path):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            directory = Path(directory)
            attestation = json.loads((directory / 'model_integrity.json').read_text())
            if attestation['revision'] != MODEL_REVISION or not attestation['files']:
                raise ValueError('model revision mismatch')
            required = {'config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt'}
            if not required.issubset(attestation['files']):
                raise ValueError('incomplete model attestation')
            for name, expected in attestation['files'].items():
                if Path(name).name != name or sha256((directory / name).read_bytes()) != expected:
                    raise ValueError('model file integrity mismatch')
            self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                                   use_safetensors=True).to(device='cpu', dtype=torch.float32).eval()
            self.torch = torch
        except Exception as exc:
            raise RetrievalError('embedding model unavailable or invalid') from exc

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, ENCODING['dimension']), dtype=np.float32)
        try:
            prepared = [(QUERY_PREFIX if query else '') + t for t in texts]
            lengths = [len(self.tokenizer(t, truncation=False)['input_ids']) for t in prepared]
            if max(lengths) > ENCODING['max_tokens']:
                raise RetrievalError('embedding input too long')
            rows = []
            for start in range(0, len(prepared), 16):
                inputs = self.tokenizer(prepared[start:start+16], padding=True, truncation=False, return_tensors='pt')
                with self.torch.no_grad():
                    hidden = self.model(**inputs).last_hidden_state[:, 0]
                    rows.append(self.torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy())
            result = np.concatenate(rows)
            validate_vectors(result, len(texts))
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError('embedding encoding failed') from exc


def validate_vectors(vectors: np.ndarray, count: int) -> None:
    if (vectors.dtype != np.float32 or vectors.shape != (count, ENCODING['dimension'])
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise RetrievalError('invalid embedding shape, dtype, value or norm')


def row_map(cards: CardLibrary) -> list[dict]:
    return [dict(entry_id=c.entry_id, entry_version=c.entry_version, body_sha256=sha256(c.body.encode()))
            for c in cards.entries]


class Bundle:
    def __init__(self, directory: Path, *, expected_digest: str | None = None):
        try:
            raw = (directory / 'index_manifest.json').read_bytes()
            manifest = json.loads(raw)
            if expected_digest is not None and sha256(raw) != expected_digest:
                raise ValueError('publication pointer mismatch')
            if set(manifest) != {'schema_version', 'encoding', 'cards_sha256', 'vectors_sha256', 'rows', 'count', 'dimension'}:
                raise ValueError('unknown manifest fields')
            if manifest['schema_version'] != 'knowledge-index-v1' or manifest['encoding'] != ENCODING:
                raise ValueError('unsupported encoding strategy')
            card_bytes = (directory / 'cards.json').read_bytes()
            vector_bytes = (directory / 'embeddings.npy').read_bytes()
            if sha256(card_bytes) != manifest['cards_sha256'] or sha256(vector_bytes) != manifest['vectors_sha256']:
                raise ValueError('bundle file mismatch')
            cards = CardLibrary.model_validate_json(card_bytes)
            if any(c.status != 'published' for c in cards.entries):
                raise ValueError('unpublished card in bundle')
            if manifest['rows'] != row_map(cards) or manifest['count'] != len(cards.entries) or manifest['dimension'] != 512:
                raise ValueError('row binding mismatch')
            import io
            vectors = np.load(io.BytesIO(vector_bytes), allow_pickle=False)
            validate_vectors(vectors, len(cards.entries))
            vectors.flags.writeable = False
            self.cards, self.vectors, self.manifest = cards, vectors, manifest
            self.manifest_digest = sha256(raw)
        except Exception as exc:
            raise RetrievalError('retrieval bundle unavailable or invalid') from exc


def current_bundle(root: Path) -> Bundle:
    try:
        pointer = json.loads((root / 'current.json').read_text())
        name = pointer['directory']
        if set(pointer) != {'directory', 'manifest_sha256'} or not isinstance(name, str) or not name.startswith('bundle_') or not name[7:].isalnum():
            raise ValueError('invalid publication pointer')
        return Bundle(root / name, expected_digest=pointer['manifest_sha256'])
    except Exception as exc:
        raise RetrievalError('knowledge publication unavailable') from exc


def publish(library: CardLibrary, root: Path, encoder: BodyEncoder, *, previous: Bundle | None = None) -> Bundle:
    """Write a new complete immutable directory, validate, then atomically switch."""
    cards = CardLibrary(knowledge_set_version=library.knowledge_set_version,
                        entries=tuple(sorted((c for c in library.entries if c.status == 'published'), key=lambda c: c.entry_id)))
    cache = {} if previous is None else {c.body: previous.vectors[i] for i, c in enumerate(previous.cards.entries)}
    missing = list(dict.fromkeys(c.body for c in cards.entries if c.body not in cache))
    cache.update(zip(missing, encoder.encode(missing)))
    vectors = np.stack([cache[c.body] for c in cards.entries]) if cards.entries else np.empty((0, 512), dtype=np.float32)
    validate_vectors(vectors, len(cards.entries))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ('bundle_' + uuid4().hex)
    directory.mkdir()
    (directory / 'cards.json').write_text(cards.model_dump_json(), encoding='utf-8')
    np.save(directory / 'embeddings.npy', vectors, allow_pickle=False)
    manifest = dict(schema_version='knowledge-index-v1', encoding=ENCODING,
                    cards_sha256=sha256((directory / 'cards.json').read_bytes()),
                    vectors_sha256=sha256((directory / 'embeddings.npy').read_bytes()),
                    rows=row_map(cards), count=len(cards.entries), dimension=512)
    (directory / 'index_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    bundle = Bundle(directory)
    pointer = root / ('pointer_' + uuid4().hex + '.tmp')
    pointer.write_text(json.dumps(dict(directory=directory.name, manifest_sha256=bundle.manifest_digest)), encoding='utf-8')
    for path in [directory/'cards.json', directory/'embeddings.npy', directory/'index_manifest.json', pointer]:
        with path.open('r+b') as handle:
            os.fsync(handle.fileno())
    if os.name == 'posix':
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    os.replace(pointer, root / 'current.json')
    if os.name == 'posix':
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    return bundle


def build_query(evidence, *, mode='train_template', user_text=None, domain=None) -> dict:
    if mode == 'user_text':
        if not isinstance(user_text, str) or not user_text.strip():
            raise RetrievalError('user_text requires a nonempty explicit description')
        text = user_text.strip()
    elif mode == 'train_template':
        if user_text is not None:
            raise RetrievalError('user_text supplied in template mode')
        stats = evidence.statistics
        ratio = format(stats.feature_count / stats.sample_group_count, '.12g') if stats.sample_group_count else '未知'
        counts = '、'.join(str(c.sample_group_count) for c in sorted(stats.classes, key=lambda c: c.class_id))
        text = (f'初始分类配方选择。领域：{domain or "未知"}。训练独立样品{stats.sample_group_count}个，'
                f'观测{stats.observation_count}条，'
                f'{"存在" if stats.repeated_measurement_group_count else "不存在"}重复测量；'
                f'特征{stats.feature_count}个，特征/独立样品比{ratio}。'
                f'{len(stats.classes)}类，每类训练独立样品{counts}。需要候选模型经验、适用前提与局限。')
    else:
        raise RetrievalError('unknown query mode')
    return dict(mode=mode, template_version='train-query-v1' if mode == 'train_template' else 'user-text-v1',
                text=text, text_sha256=sha256(text.encode()))


def retrieve(bundle: Bundle, query: np.ndarray, *, models: set[str], scope: str,
             domain: str | None = None, k: int = 3, tau: float | None = None) -> list[dict]:
    if type(k) is not int or not 1 <= k <= 6 or (tau is not None and (type(tau) not in (int, float) or not np.isfinite(tau) or not -1 <= tau <= 1)):
        raise RetrievalError('invalid retrieval policy')
    validate_vectors(query.reshape(1, -1), 1)
    eligible = [i for i, c in enumerate(bundle.cards.entries)
                if (not c.scopes or scope in c.scopes) and 'classification' in c.task_types
                and ('general' in c.domains or domain is not None and domain in c.domains)
                and (not c.related_models or set(c.related_models) & models)]
    # One exact matrix product. No second retrieval or reranking stage.
    scores = bundle.vectors[eligible] @ query
    selected = sorted(zip(eligible, scores), key=lambda pair: (-float(pair[1]), bundle.cards.entries[pair[0]].entry_id))
    return [dict(card=bundle.cards.entries[i], score=float(score)) for i, score in selected
            if tau is None or score >= tau][:k]
