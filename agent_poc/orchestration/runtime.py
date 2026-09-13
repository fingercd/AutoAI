"""Single-host persistent runner and a credential-free start/resume/status CLI."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Callable, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith.run_helpers import tracing_context
from pydantic import ValidationError

from agent_poc.clients.autoai_client import AutoAIClient, validate_base_url, validate_identifier
from .llm import LLMAdapter, LLMConfig, LLM_CONFIG_VERSION, PROMPT_VERSION
from .persistence import CallJournal, JSONSerializer, PersistenceError, thread_lock
from .projection import CONTEXT_VERSION
from .state import GraphState, StateModel, apply_patch, fingerprint, new_state


DEFAULT_STORAGE = Path('storage') / 'orchestration'


class RuntimeErrorCode(RuntimeError):
    """Only a locally chosen safe code, never raw dependency exceptions."""


class TaskInterrupted(RuntimeErrorCode):
    def __init__(self, thread_id: str):
        super().__init__('checkpoint_preserved')
        self.thread_id = thread_id


def normalize_endpoint(value: str) -> str:
    parsed = urlsplit(validate_base_url(value))
    host = parsed.hostname.lower()
    if ':' in host:
        host = '[' + host + ']'
    port = parsed.port
    if port is not None and (parsed.scheme, port) not in (('http', 80), ('https', 443)):
        host += ':' + str(port)
    return urlunsplit((parsed.scheme.lower(), host, parsed.path.rstrip('/'), '', ''))


@dataclass(frozen=True, repr=False)
class RuntimeConfig:
    backend_url: str
    principal_scope: str
    llm_config: LLMConfig
    backend_token: str | None = None
    llm_token: str | None = None
    api_timeout: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, 'backend_url', normalize_endpoint(self.backend_url))
        try:
            validate_identifier(self.principal_scope)
            if '..' in self.principal_scope:
                raise ValueError
            if type(self.api_timeout) not in (int, float) or not math.isfinite(self.api_timeout) or self.api_timeout <= 0:
                raise ValueError
            object.__setattr__(self, 'api_timeout', float(self.api_timeout))
            if self.backend_token is not None and (type(self.backend_token) is not str or not self.backend_token):
                raise ValueError
            if self.llm_token is not None and (type(self.llm_token) is not str or not self.llm_token):
                raise ValueError
        except Exception:
            raise RuntimeErrorCode('runtime_configuration_invalid') from None

    def __repr__(self):
        return 'RuntimeConfig(backend=<configured>, principal=<bound>, llm=<configured>)'

    def backend_fingerprint(self) -> str:
        return hashlib.sha256(self.backend_url.encode('utf-8')).hexdigest()

    def principal_fingerprint(self) -> str:
        # The token is the HMAC key, never the message or persisted material.
        # A local unauthenticated service still has an explicit scope label.
        key = (self.backend_token or 'agent-local-scope-v1').encode('utf-8')
        binding = '\0'.join(('agent-principal-binding-v1', self.backend_url, self.principal_scope))
        return hmac.new(key, binding.encode('utf-8'), hashlib.sha256).hexdigest()

    def runtime_config_fingerprint(self) -> str:
        return fingerprint({'api_timeout': self.api_timeout})


@contextmanager
def checkpoint_store(storage: Path | str, *, read_only: bool = False) -> Iterator[SqliteSaver]:
    """Own one SQLite connection; first saver access initializes its schema.

    check_same_thread=False is required by LangGraph's synchronous writer
    worker. SqliteSaver owns its connection mutex and serializes transactions.
    """
    directory = Path(storage)
    if read_only:
        uri = (directory / 'checkpoints.sqlite').resolve().as_uri() + '?mode=ro'
        connection = sqlite3.connect(uri, uri=True, timeout=10, check_same_thread=False)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(directory / 'checkpoints.sqlite', timeout=10, check_same_thread=False)
    try:
        saver = SqliteSaver(connection, serde=JSONSerializer())
        if read_only:
            # Pin all saver SELECTs (checkpoint and pending writes) to one
            # committed snapshot without taking the runner's execution lock.
            connection.execute('BEGIN')
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'checkpoints', 'writes'}.issubset(tables):
                raise RuntimeErrorCode('checkpoint_schema_unavailable')
            # The pinned saver lazily CREATEs tables on first read. The schema
            # was checked above; suppress creation on this read-only handle.
            saver.is_setup = True
        else:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=FULL')
        yield saver
    finally:
        connection.close()


def _graph_config(thread_id: str) -> dict:
    return {'configurable': {'thread_id': thread_id, 'checkpoint_ns': ''},
            'callbacks': [], 'recursion_limit': 12}


def _validate_thread(thread_id: str) -> str:
    try:
        validate_identifier(thread_id)
        if '..' in thread_id:
            raise ValueError
    except Exception:
        raise RuntimeErrorCode('thread_identifier_invalid') from None
    return thread_id


def _state_from_checkpoint(saver: SqliteSaver, thread_id: str) -> GraphState | None:
    checkpoint = saver.get_tuple(_graph_config(thread_id))
    if checkpoint is None:
        return None
    channels = checkpoint.checkpoint['channel_values']
    raw = {key: channels[key] for key in GraphState.__annotations__ if key in channels}
    if set(raw) != set(GraphState.__annotations__):
        # A process may stop after the input checkpoint but before START has
        # copied its input into the graph's declared channels.
        raw = channels.get('__start__', raw)
    return StateModel.model_validate(raw).model_dump(mode='json')


def _verify_binding(state: GraphState, config: RuntimeConfig, thread_id: str) -> None:
    identity, versions = state['identity'], state['versions']
    expected = ((identity['thread_id'], thread_id),
                (identity['backend_fingerprint'], config.backend_fingerprint()),
                (identity['principal_fingerprint'], config.principal_fingerprint()),
                (identity['runtime_config_fingerprint'], config.runtime_config_fingerprint()),
                (versions['llm_config_fingerprint'], config.llm_config.fingerprint()),
                (versions['prompt'], PROMPT_VERSION),
                (versions['llm_config'], LLM_CONFIG_VERSION),
                (versions['context_projection'], CONTEXT_VERSION))
    if any(not hmac.compare_digest(str(actual), str(wanted)) for actual, wanted in expected):
        raise RuntimeErrorCode('resume_configuration_mismatch')


def _build(config: RuntimeConfig, saver: SqliteSaver, journal: CallJournal, *,
           client=None, llm=None, clock: Callable[[], float] = time.time,
           graph_factory=None):
    from .graph import Dependencies, build_graph
    deps = Dependencies(client=client or AutoAIClient(config.backend_url, token=config.backend_token,
                                                     timeout=config.api_timeout, max_retries=0),
                        llm=llm or LLMAdapter(config.llm_config, token=config.llm_token),
                        journal=journal, clock=clock)
    return (graph_factory or build_graph)(deps, saver)


def _drive(graph, state: GraphState, *, initial: GraphState | None, wait: bool,
           clock: Callable[[], float], sleep: Callable[[float], None]) -> GraphState:
    config = _graph_config(state['identity']['thread_id'])
    incoming = initial
    max_ticks = 64 + 4 * int(state['budget']['llm_calls']['limit'] + state['budget']['api_calls']['limit'])
    for _tick in range(max_ticks):
        snapshot = graph.get_state(config)
        if initial is None and snapshot.values:
            state = StateModel.model_validate(snapshot.values).model_dump(mode='json')
        if state['lifecycle']['next_action'] is None:
            return state
        now = clock()
        wake_at = state['recovery']['next_wake_at']
        waiting = state['lifecycle']['status'] == 'waiting' and wake_at is not None and now < wake_at
        if waiting and now < state['budget']['deadline_at'] and not snapshot.next:
            if not wait:
                return state
            # This is the timer owner; a saved waiting state performs no I/O.
            while now < wake_at and now < state['budget']['deadline_at']:
                sleep(min(1.0, wake_at - now, state['budget']['deadline_at'] - now))
                now = clock()
        if incoming is None:
            incoming = None if snapshot.next else {}
        state = StateModel.model_validate(graph.invoke(incoming, config=config, durability='sync')).model_dump(mode='json')
        incoming = None
        initial = None
        if state['lifecycle']['next_action'] is None:
            return state
        if state['lifecycle']['status'] == 'waiting' and not wait:
            return state
    stopped = apply_patch(state, {'lifecycle': {'status': 'needs_attention', 'stage': 'ended',
                                               'next_action': None, 'reason_code': 'runner_progress_limit',
                                               'ended_at': clock()},
                                 'recovery': {'needs_human_review': True, 'reason_code': 'runner_progress_limit'}})
    graph.update_state(config, stopped)
    return stopped


def start_task(config: RuntimeConfig, *, dataset_id: str, allowed_models: list[str],
               storage: Path | str = DEFAULT_STORAGE, thread_id: str | None = None,
               task_id: str | None = None, wait: bool = False, seed: int = 42,
               selection_metric: str = 'macro_f1', max_llm_calls: int = 6,
               max_api_calls: int = 60, max_operation_attempts: int = 3,
               max_repair_attempts: int = 2, timeout_seconds: float = 3600.0,
               source_role: str = 'development', client=None, llm=None,
               clock: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep, graph_factory=None) -> GraphState:
    thread_id = _validate_thread(thread_id or str(uuid.uuid4()))
    directory = Path(storage)
    state = new_state(dataset_id=dataset_id, allowed_models=allowed_models,
                      backend_fingerprint=config.backend_fingerprint(),
                      principal_fingerprint=config.principal_fingerprint(),
                      llm_config_fingerprint=config.llm_config.fingerprint(),
                      runtime_config_fingerprint=config.runtime_config_fingerprint(),
                      task_id=task_id, thread_id=thread_id, seed=seed, selection_metric=selection_metric,
                      max_llm_calls=max_llm_calls, max_api_calls=max_api_calls,
                      max_operation_attempts=max_operation_attempts,
                      max_repair_attempts=max_repair_attempts, timeout_seconds=timeout_seconds,
                      now=clock(), prompt_version=PROMPT_VERSION,
                      llm_config_version=LLM_CONFIG_VERSION, source_role=source_role)
    try:
        with thread_lock(directory / 'locks', thread_id), checkpoint_store(directory) as saver:
            if _state_from_checkpoint(saver, thread_id) is not None:
                raise RuntimeErrorCode('thread_already_exists_use_resume')
            journal = CallJournal(directory / 'calls.sqlite', thread_id)
            try:
                graph = _build(config, saver, journal, client=client, llm=llm,
                               clock=clock, graph_factory=graph_factory)
                with tracing_context(enabled=False):
                    return _drive(graph, state, initial=state, wait=wait, clock=clock, sleep=sleep)
            finally:
                journal.close()
    except KeyboardInterrupt:
        raise TaskInterrupted(thread_id) from None


def resume_task(config: RuntimeConfig, *, storage: Path | str = DEFAULT_STORAGE,
                thread_id: str, wait: bool = False, client=None, llm=None,
                clock: Callable[[], float] = time.time,
                sleep: Callable[[float], None] = time.sleep, graph_factory=None) -> GraphState:
    thread_id = _validate_thread(thread_id)
    directory = Path(storage)
    if not (directory / 'checkpoints.sqlite').is_file():
        raise RuntimeErrorCode('thread_not_found')
    try:
        with thread_lock(directory / 'locks', thread_id), checkpoint_store(directory) as saver:
            state = _state_from_checkpoint(saver, thread_id)
            if state is None:
                raise RuntimeErrorCode('thread_not_found')
            _verify_binding(state, config, thread_id)
            if state['lifecycle']['next_action'] is None:
                return state
            journal = CallJournal(directory / 'calls.sqlite', thread_id)
            try:
                graph = _build(config, saver, journal, client=client, llm=llm,
                               clock=clock, graph_factory=graph_factory)
                with tracing_context(enabled=False):
                    return _drive(graph, state, initial=None, wait=wait, clock=clock, sleep=sleep)
            finally:
                journal.close()
    except KeyboardInterrupt:
        raise TaskInterrupted(thread_id) from None


def read_status(*, storage: Path | str = DEFAULT_STORAGE, thread_id: str) -> GraphState:
    """Read a validated local snapshot without requiring service/LLM settings."""
    thread_id = _validate_thread(thread_id)
    directory = Path(storage)
    if not (directory / 'checkpoints.sqlite').is_file():
        raise RuntimeErrorCode('thread_not_found')
    try:
        with checkpoint_store(directory, read_only=True) as saver:
            state = _state_from_checkpoint(saver, thread_id)
            if state is None:
                raise RuntimeErrorCode('thread_not_found')
            return state
    except (sqlite3.DatabaseError, ValidationError, KeyError, TypeError):
        raise RuntimeErrorCode('checkpoint_invalid') from None


def state_summary(state: GraphState) -> dict:
    state = StateModel.model_validate(state).model_dump(mode='json')
    return {'task_id': state['identity']['task_id'], 'thread_id': state['identity']['thread_id'],
            'session_id': state['identity']['session_id'], 'run_id': state['execution']['run_id'],
            'lifecycle': state['lifecycle'], 'run_status': state['execution']['run_status'],
            'next_wake_at': state['recovery']['next_wake_at'],
            'validation': state['feedback']['validation_metrics'],
            'selection_score': state['feedback']['selection_score'],
            'calls': {name: state['budget'][name]['actual'] for name in ('llm_calls', 'api_calls')},
            'deadline_at': state['budget']['deadline_at'], 'finalization': state['finalization']}


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        # argparse's default includes the rejected text, possibly a supplied
        # malformed URL. Do not echo CLI input into diagnostics.
        raise RuntimeErrorCode('invalid_cli_arguments')


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description='Single-host persistent Agent: start / resume / status')
    commands = parser.add_subparsers(dest='command', required=True, parser_class=_ArgumentParser)
    for name in ('start', 'resume', 'status'):
        command = commands.add_parser(name)
        command.add_argument('--storage', type=Path, default=DEFAULT_STORAGE)
        command.add_argument('--thread-id', required=name != 'start')
        command.add_argument('--full', action='store_true', help='Print the validated 21-block State')
        if name == 'status':
            continue
        command.add_argument('--backend-url')
        command.add_argument('--principal-scope')
        command.add_argument('--llm-base-url')
        command.add_argument('--llm-model')
        command.add_argument('--protocol', choices=('json_action', 'native_tools'))
        command.add_argument('--llm-timeout', type=float)
        command.add_argument('--llm-max-tokens', type=int)
        command.add_argument('--llm-max-response-bytes', type=int)
        command.add_argument('--temperature', type=float)
        command.add_argument('--top-p', type=float)
        command.add_argument('--api-timeout', type=float)
        command.add_argument('--wait', action='store_true')
        if name == 'start':
            command.add_argument('--dataset-id', required=True)
            command.add_argument('--allowed-models', default='logistic_regression,svm,random_forest')
            command.add_argument('--seed', type=int, default=42)
            command.add_argument('--selection-metric', choices=('macro_f1', 'balanced_accuracy'), default='macro_f1')
            command.add_argument('--max-llm-calls', type=int, default=6)
            command.add_argument('--max-api-calls', type=int, default=60)
            command.add_argument('--max-operation-attempts', type=int, default=3)
            command.add_argument('--max-repair-attempts', type=int, default=2)
            command.add_argument('--timeout-seconds', type=float, default=3600.0)
            command.add_argument('--source-role', choices=('development', 'benchmark', 'domain'), default='development')
    return parser


def _config_from_args(args, environ: Mapping[str, str]) -> RuntimeConfig:
    def setting(name, environment, default=None, convert=str):
        value = getattr(args, name, None)
        if value is None:
            value = environ.get(environment, default)
        if value is None or value == '':
            raise RuntimeErrorCode('required_runtime_configuration_missing')
        return convert(value)
    try:
        config = LLMConfig(
            base_url=setting('llm_base_url', 'AUTOAI_LLM_BASE_URL'),
            model=setting('llm_model', 'AUTOAI_LLM_MODEL'),
            protocol=setting('protocol', 'AUTOAI_LLM_PROTOCOL', 'json_action'),
            timeout=setting('llm_timeout', 'AUTOAI_LLM_TIMEOUT', 30.0, float),
            max_tokens=setting('llm_max_tokens', 'AUTOAI_LLM_MAX_TOKENS', 1024, int),
            max_response_bytes=setting('llm_max_response_bytes', 'AUTOAI_LLM_MAX_RESPONSE_BYTES', 32768, int),
            temperature=setting('temperature', 'AUTOAI_LLM_TEMPERATURE', 0.0, float),
            top_p=setting('top_p', 'AUTOAI_LLM_TOP_P', 1.0, float))
        return RuntimeConfig(
            backend_url=setting('backend_url', 'AUTOAI_BASE_URL', 'http://127.0.0.1:8000'),
            principal_scope=setting('principal_scope', 'AUTOAI_PRINCIPAL_SCOPE'), llm_config=config,
            backend_token=environ.get('AUTOAI_API_TOKEN') or None,
            llm_token=environ.get('AUTOAI_LLM_TOKEN') or None,
            api_timeout=setting('api_timeout', 'AUTOAI_API_TIMEOUT', 10.0, float))
    except RuntimeErrorCode:
        raise
    except Exception:
        raise RuntimeErrorCode('runtime_configuration_invalid') from None


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == 'status':
            state = read_status(storage=args.storage, thread_id=args.thread_id)
        else:
            config = _config_from_args(args, os.environ if environ is None else environ)
            if args.command == 'start':
                state = start_task(config, storage=args.storage, thread_id=args.thread_id, wait=args.wait,
                                   dataset_id=args.dataset_id,
                                   allowed_models=[part.strip() for part in args.allowed_models.split(',') if part.strip()],
                                   seed=args.seed, selection_metric=args.selection_metric,
                                   max_llm_calls=args.max_llm_calls, max_api_calls=args.max_api_calls,
                                   max_operation_attempts=args.max_operation_attempts,
                                   max_repair_attempts=args.max_repair_attempts,
                                   timeout_seconds=args.timeout_seconds, source_role=args.source_role)
            else:
                state = resume_task(config, storage=args.storage, thread_id=args.thread_id, wait=args.wait)
        print(json.dumps(state if args.full else state_summary(state), ensure_ascii=False, allow_nan=False))
        if args.command != 'status' and state['lifecycle']['status'] in {
                'failed', 'timed_out', 'cancelled', 'needs_attention'}:
            return 1
        return 0
    except TaskInterrupted as exc:
        print(json.dumps({'thread_id': exc.thread_id, 'status': 'interrupted',
                          'reason_code': 'checkpoint_preserved', 'backend_cancel_requested': False}))
        return 130
    except (RuntimeErrorCode, PersistenceError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({'error': 'orchestration_failed'}), file=sys.stderr)
        return 2
