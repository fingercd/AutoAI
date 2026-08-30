from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_poc.benchmark as benchmark_package
from agent_poc.benchmark.evaluator import (
    BenchmarkEvaluationError,
    ResultEvaluator,
)
from agent_poc.benchmark.main import (
    CodeRevisionError,
    RuntimeDependencies,
    main,
    resolve_code_revision,
)
from agent_poc.benchmark.report import (
    assert_report_safe,
    build_trial_report,
    render_trial_markdown,
    summarize_formal_trials,
    write_trial_report,
)
from agent_poc.benchmark.runner import BenchmarkRunner, DatasetUploadClient


RESULT_ROUTE_LITERAL = '/api/training/runs/{run_id}/result'
CODE_REVISION = 'd' * 40


def _result_payload(run_id: str = 'run-1') -> dict:
    return {
        'schema_version': 'run-result-v1',
        'run': {
            'run_id': run_id,
            'state': 'succeeded',
            'result_state': 'ready',
        },
        'evaluation': {
            'strategy': 'stratified_holdout',
            'primary_split': 'test',
            'primary_aggregation': 'direct',
        },
        'metrics': {
            'primary': {'macro_f1': 0.8, 'balanced_accuracy': 0.75},
        },
        'analysis': {'confusion_matrix': [[1, 0], [0, 1]]},
        'artifacts': [{'name': 'predictions.csv'}],
    }


def test_result_route_literal_is_firewalled_to_evaluator_module() -> None:
    package_root = Path(benchmark_package.__file__).resolve().parent
    hits = []
    for path in package_root.glob('*.py'):
        if RESULT_ROUTE_LITERAL in path.read_text(encoding='utf-8'):
            hits.append(path.name)
    assert hits == ['evaluator.py']


def test_evaluator_reads_once_and_returns_only_whitelisted_scalars() -> None:
    paths = []
    evaluator = ResultEvaluator(request_fn=lambda path: paths.append(path) or _result_payload())
    result = evaluator.evaluate({'status': 'finalized', 'selected_run_id': 'run-1'})
    assert evaluator.attempt_count == 1
    assert paths == ['/api/training/runs/run-1/result']
    assert result == {
        'schema_version': 'benchmark-evaluation-v1',
        'run_id': 'run-1',
        'macro_f1': 0.8,
        'balanced_accuracy': 0.75,
    }
    encoded = json.dumps(result).lower()
    assert 'confusion' not in encoded
    assert 'artifact' not in encoded
    assert 'prediction' not in encoded


@pytest.mark.parametrize(
    ('mutation', 'code'),
    [
        (lambda payload: payload.update(schema_version='wrong'), 'result_schema_mismatch'),
        (
            lambda payload: payload['run'].update(run_id='run-other'),
            'result_run_id_mismatch',
        ),
        (
            lambda payload: payload['evaluation'].update(strategy='leave_one_sample_id_cv'),
            'evaluation_contract_invalid',
        ),
        (
            lambda payload: payload['evaluation'].update(primary_aggregation='pooled_oof'),
            'evaluation_contract_invalid',
        ),
        (
            lambda payload: payload['metrics']['primary'].update(macro_f1=float('nan')),
            'macro_f1_invalid',
        ),
    ],
)
def test_evaluator_fails_closed_on_contract_drift(mutation, code) -> None:
    payload = _result_payload()
    mutation(payload)
    evaluator = ResultEvaluator(request_fn=lambda _path: payload)
    with pytest.raises(BenchmarkEvaluationError, match=code):
        evaluator.evaluate({'state': 'finalized', 'selected_run_id': 'run-1'})


@dataclass(frozen=True)
class FakeConfig:
    dataset_id: str = 'old-dataset'
    source_role: str = 'development'
    case_write: bool = True
    split_mode: str = 'leave_one_sample_id_cv'
    split_train: int = 7
    split_valid: int = 2
    split_test: int = 1
    seed: int = 7
    trace_path: Path = Path('old-trace.jsonl')
    code_revision: str = CODE_REVISION


class FakeUploadHTTP:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, _path, *, files):
        self.calls += 1
        assert files['file'][2] == 'text/csv'
        return {'dataset_id': 'uploaded-dataset'}


class FakeAutoAI:
    attempt_count = 10
    retry_attempt_count = 2


class FakeLLM:
    call_count = 3


def _verified(tmp_path: Path, role: str = 'formal_primary'):
    data = SimpleNamespace(
        name='demo',
        role=role,
        data_sha256='a' * 64,
        allowed_split='stratified_holdout',
    )
    return SimpleNamespace(dataset=lambda _name: data), data


def _runner_dependencies(tmp_path: Path, *, role: str = 'formal_primary'):
    verified, dataset = _verified(tmp_path, role)
    adapted_path = tmp_path / 'adapted.csv'
    adapted_path.write_text(
        'Index,Label,Sample_ID,Name,1\n1,A,row-1,demo,0.5\n',
        encoding='utf-8',
    )
    adapted = SimpleNamespace(
        output_path=adapted_path,
        split_strategy='stratified_holdout',
    )
    return verified, dataset, adapted


def test_runner_forces_benchmark_context_audits_then_evaluates_once(tmp_path) -> None:
    verified, _dataset, adapted = _runner_dependencies(tmp_path)
    events = []
    autoai = FakeAutoAI()
    llm = FakeLLM()

    def fake_run(cfg, _llm, _autoai, **_kwargs):
        events.append('agent')
        assert cfg.dataset_id == 'uploaded-dataset'
        assert cfg.source_role == 'benchmark'
        assert cfg.case_write is False
        assert cfg.split_mode == 'stratified_holdout'
        assert (cfg.split_train, cfg.split_valid, cfg.split_test) == (8, 1, 1)
        assert cfg.seed == 42
        autoai.attempt_count += 4
        autoai.retry_attempt_count += 1
        llm.call_count += 2
        return {
            'status': 'finalized',
            'selected_run_id': 'run-1',
            'agent_metrics': {
                'llm_call_count': 4,
                'api_call_count': 7,
                'retry_attempt_count': 3,
                'model_fit_count': 4,
            },
        }

    evaluator = ResultEvaluator(
        request_fn=lambda _path: events.append('evaluate') or _result_payload()
    )
    trace = SimpleNamespace(
        path=tmp_path / 'trace.jsonl', code_revision=CODE_REVISION
    )
    report = BenchmarkRunner(
        verify_fn=lambda *_args: verified,
        adapt_fn=lambda *_args: adapted,
        run_agent_fn=fake_run,
        trace_audit_fn=lambda _path: events.append('audit'),
        clock=iter([10.0, 12.5]).__next__,
    ).run_trial(
        benchmark_root=tmp_path,
        policy_manifest_path=tmp_path / 'manifest.json',
        dataset_name='demo',
        adapter_output_dir=tmp_path / 'adapter',
        cfg=FakeConfig(),
        llm=llm,
        autoai=autoai,
        trace=trace,
        uploader=DatasetUploadClient(FakeUploadHTTP()),
        evaluator=evaluator,
        model_key='qwen35_9b',
        model_revision='model-rev',
        code_revision=CODE_REVISION,
        seed=42,
        sleep_fn=lambda _seconds: None,
    )
    assert events == ['agent', 'audit', 'evaluate']
    assert evaluator.attempt_count == 1
    assert report['outcome'] == {
        'status': 'completed',
        'failure_code': None,
        'metrics': {'macro_f1': 0.8, 'balanced_accuracy': 0.75},
    }
    assert report['cost']['api_attempts'] == {
        'upload': 1,
        'agent': 7,
        'evaluator': 1,
    }
    # Unified process metrics include structured-output repair attempts that
    # the HTTP-only client counter cannot observe.
    assert report['cost']['llm_call_count'] == 4
    assert report['cost']['retry_attempt_count'] == 3
    assert report['cost']['model_fit_count'] == 4
    assert report['revision']['code'] == trace.code_revision == CODE_REVISION


@pytest.mark.parametrize('agent_status', ['needs_human', 'budget_exhausted'])
def test_non_finalized_outcome_never_requests_result(tmp_path, agent_status) -> None:
    verified, _dataset, adapted = _runner_dependencies(tmp_path)
    result_calls = []
    audits = []
    evaluator = ResultEvaluator(
        request_fn=lambda path: result_calls.append(path) or _result_payload()
    )
    report = BenchmarkRunner(
        verify_fn=lambda *_args: verified,
        adapt_fn=lambda *_args: adapted,
        run_agent_fn=lambda *_args, **_kwargs: {'status': agent_status},
        trace_audit_fn=lambda path: audits.append(path),
        clock=iter([1.0, 2.0]).__next__,
    ).run_trial(
        benchmark_root=tmp_path,
        policy_manifest_path=tmp_path / 'manifest.json',
        dataset_name='demo',
        adapter_output_dir=tmp_path / 'adapter',
        cfg=FakeConfig(),
        llm=FakeLLM(),
        autoai=FakeAutoAI(),
        trace=SimpleNamespace(
            path=tmp_path / 'trace.jsonl', code_revision=CODE_REVISION
        ),
        uploader=DatasetUploadClient(FakeUploadHTTP()),
        evaluator=evaluator,
        model_key='qwen35_9b',
        model_revision='model-rev',
        code_revision=CODE_REVISION,
        seed=42,
    )
    assert audits == [tmp_path / 'trace.jsonl']
    assert result_calls == []
    assert evaluator.attempt_count == 0
    assert report['outcome']['status'] == agent_status
    assert report['outcome']['metrics'] == {
        'macro_f1': None,
        'balanced_accuracy': None,
    }


def test_agent_or_trace_exception_never_requests_result(tmp_path) -> None:
    verified, _dataset, adapted = _runner_dependencies(tmp_path)
    result_calls = []
    evaluator = ResultEvaluator(
        request_fn=lambda path: result_calls.append(path) or _result_payload()
    )
    report = BenchmarkRunner(
        verify_fn=lambda *_args: verified,
        adapt_fn=lambda *_args: adapted,
        run_agent_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('/secret')),
        clock=iter([1.0, 2.0]).__next__,
    ).run_trial(
        benchmark_root=tmp_path,
        policy_manifest_path=tmp_path / 'manifest.json',
        dataset_name='demo',
        adapter_output_dir=tmp_path / 'adapter',
        cfg=FakeConfig(),
        llm=FakeLLM(),
        autoai=FakeAutoAI(),
        trace=SimpleNamespace(
            path=tmp_path / 'trace.jsonl', code_revision=CODE_REVISION
        ),
        uploader=DatasetUploadClient(FakeUploadHTTP()),
        evaluator=evaluator,
        model_key='qwen35_9b',
        model_revision='model-rev',
        code_revision=CODE_REVISION,
        seed=42,
    )
    assert result_calls == []
    assert report['outcome']['failure_code'] == 'runner_exception'
    assert '/secret' not in json.dumps(report)


def test_trace_audit_failure_blocks_hidden_evaluation(tmp_path) -> None:
    verified, _dataset, adapted = _runner_dependencies(tmp_path)
    result_calls = []
    evaluator = ResultEvaluator(
        request_fn=lambda path: result_calls.append(path) or _result_payload()
    )
    report = BenchmarkRunner(
        verify_fn=lambda *_args: verified,
        adapt_fn=lambda *_args: adapted,
        run_agent_fn=lambda *_args, **_kwargs: {
            'status': 'finalized',
            'selected_run_id': 'run-1',
        },
        trace_audit_fn=lambda _path: (_ for _ in ()).throw(
            AssertionError('unsafe trace')
        ),
        clock=iter([1.0, 2.0]).__next__,
    ).run_trial(
        benchmark_root=tmp_path,
        policy_manifest_path=tmp_path / 'manifest.json',
        dataset_name='demo',
        adapter_output_dir=tmp_path / 'adapter',
        cfg=FakeConfig(),
        llm=FakeLLM(),
        autoai=FakeAutoAI(),
        trace=SimpleNamespace(
            path=tmp_path / 'trace.jsonl', code_revision=CODE_REVISION
        ),
        uploader=DatasetUploadClient(FakeUploadHTTP()),
        evaluator=evaluator,
        model_key='qwen35_9b',
        model_revision='model-rev',
        code_revision=CODE_REVISION,
        seed=42,
    )
    assert result_calls == []
    assert evaluator.attempt_count == 0
    assert report['outcome']['failure_code'] == 'runner_exception'


def _trial(role: str, *, status: str = 'completed') -> dict:
    return build_trial_report(
        dataset_name='parity5' if role == 'pipeline_smoke_only' else 'haberman',
        dataset_role=role,
        dataset_sha256='b' * 64,
        code_revision=CODE_REVISION,
        model_key='qwen35_9b',
        model_revision='model-rev',
        seed=42,
        status=status,
        failure_code=None if status == 'completed' else 'agent_needs_human',
        macro_f1=0.7 if status == 'completed' else None,
        balanced_accuracy=0.6 if status == 'completed' else None,
        wall_clock_seconds=1.25,
        llm_call_count=2,
        upload_api_attempts=1,
        agent_api_attempts=5,
        evaluator_api_attempts=1 if status == 'completed' else 0,
        retry_attempt_count=0,
        model_fit_count=4,
    )


def test_report_is_safe_writes_json_markdown_and_excludes_smoke_from_summary(tmp_path) -> None:
    formal = _trial('formal_auxiliary')
    smoke = _trial('pipeline_smoke_only')
    assert_report_safe(formal)
    json_path, markdown_path = write_trial_report(smoke, tmp_path)
    assert json.loads(json_path.read_text(encoding='utf-8')) == smoke
    markdown = markdown_path.read_text(encoding='utf-8')
    assert '不进入正式汇总' in markdown
    assert 'total_score' not in markdown
    assert 'token' not in json.dumps(smoke).lower()
    assert 'prompt' not in json.dumps(smoke).lower()
    summary = summarize_formal_trials([formal, smoke])
    assert summary['formal_trial_count'] == 1
    assert summary['completed_formal_trial_count'] == 1
    assert summary['failure_rate'] == 0.0
    assert summary['mean_macro_f1'] == 0.7
    assert summary['mean_balanced_accuracy'] == 0.6
    assert summary['formal_trials'] == [{
        'dataset': 'haberman',
        'role': 'formal_auxiliary',
        'status': 'completed',
        'failure_code': None,
        'metrics': {'macro_f1': 0.7, 'balanced_accuracy': 0.6},
    }]
    assert summary['cost_totals'] == {
        'trial_count': 1,
        'wall_clock_seconds': 1.25,
        'llm_call_count': 2,
        'api_attempts': {'upload': 1, 'agent': 5, 'evaluator': 1},
        'retry_attempt_count': 0,
        'model_fit_count': {
            'known_total': 4,
            'known_trial_count': 1,
            'formal_trial_count': 1,
            'unknown_trial_count': 0,
        },
    }
    assert summary['smoke_trials'] == [{
        'dataset': 'parity5',
        'role': 'pipeline_smoke_only',
        'status': 'completed',
        'failure_code': None,
        'metrics': {'macro_f1': 0.7, 'balanced_accuracy': 0.6},
    }]


def test_completed_report_requires_both_independent_metrics() -> None:
    values = {
        'dataset_name': 'haberman',
        'dataset_role': 'formal_auxiliary',
        'dataset_sha256': 'b' * 64,
        'code_revision': CODE_REVISION,
        'model_key': 'qwen35_9b',
        'model_revision': 'model-rev',
        'seed': 42,
        'status': 'completed',
        'failure_code': None,
        'macro_f1': 0.7,
        'balanced_accuracy': None,
        'wall_clock_seconds': 1.25,
        'llm_call_count': 2,
        'upload_api_attempts': 1,
        'agent_api_attempts': 5,
        'evaluator_api_attempts': 1,
        'retry_attempt_count': 0,
        'model_fit_count': 4,
    }
    with pytest.raises(ValueError, match='requires both'):
        build_trial_report(**values)


def test_report_rejects_unverified_code_revision() -> None:
    report_values = {
        'dataset_name': 'haberman',
        'dataset_role': 'formal_auxiliary',
        'dataset_sha256': 'b' * 64,
        'code_revision': 'caller-supplied',
        'model_key': 'qwen35_9b',
        'model_revision': 'model-rev',
        'seed': 42,
        'status': 'completed',
        'failure_code': None,
        'macro_f1': 0.7,
        'balanced_accuracy': 0.6,
        'wall_clock_seconds': 1.25,
        'llm_call_count': 2,
        'upload_api_attempts': 1,
        'agent_api_attempts': 5,
        'evaluator_api_attempts': 1,
        'retry_attempt_count': 0,
        'model_fit_count': 4,
    }
    with pytest.raises(ValueError, match='40-character Git SHA'):
        build_trial_report(**report_values)


def test_resolve_code_revision_uses_shell_free_bounded_git_command(tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout=(CODE_REVISION + '\n') if command[1] == 'rev-parse' else ''
        )

    assert resolve_code_revision(tmp_path, run_fn=fake_run) == CODE_REVISION
    assert [item[0] for item in calls] == [
        ['git', 'rev-parse', 'HEAD'],
        ['git', 'status', '--porcelain', '--untracked-files=normal'],
    ]
    for _command, kwargs in calls:
        assert kwargs['cwd'] == tmp_path.resolve()
        assert kwargs['shell'] is False
        assert kwargs['timeout'] == 5.0
        assert kwargs['check'] is True


@pytest.mark.parametrize(
    'failure',
    [
        SimpleNamespace(stdout='not-a-sha\n'),
        subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD']),
        OSError('git unavailable'),
    ],
)
def test_resolve_code_revision_fails_closed_on_invalid_output_or_command(
    tmp_path, failure
) -> None:
    def fake_run(*_args, **_kwargs):
        if isinstance(failure, BaseException):
            raise failure
        return failure

    with pytest.raises(CodeRevisionError):
        resolve_code_revision(tmp_path, run_fn=fake_run)


def test_resolve_code_revision_rejects_dirty_checkout(tmp_path) -> None:
    def fake_run(command, **_kwargs):
        if command[1] == 'rev-parse':
            return SimpleNamespace(stdout=CODE_REVISION + '\n')
        return SimpleNamespace(stdout=' M agent_poc/benchmark/runner.py\n')

    with pytest.raises(CodeRevisionError, match='clean Git checkout'):
        resolve_code_revision(tmp_path, run_fn=fake_run)


def test_cli_verify_and_run_are_dependency_injectable(tmp_path, capsys) -> None:
    verified = SimpleNamespace(
        manifest_version='small-sample-benchmark-v1',
        suite_name='TabMini-25/PMLB',
        checksum_count=17,
        datasets=(
            SimpleNamespace(name='haberman', role='formal_auxiliary', data_sha256='c' * 64),
        ),
    )
    verify_deps = RuntimeDependencies(verify_fn=lambda *_args: verified)
    assert main(
        [
            'verify',
            '--benchmark-root', str(tmp_path),
            '--policy-manifest', str(tmp_path / 'policy.json'),
        ],
        dependencies=verify_deps,
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['status'] == 'verified'
    assert 'benchmark_root' not in output

    report = _trial('formal_auxiliary')
    run_deps = RuntimeDependencies(run_command_fn=lambda _args: report)
    assert main(
        [
            'run',
            '--benchmark-root', str(tmp_path),
            '--policy-manifest', str(tmp_path / 'policy.json'),
            '--dataset-name', 'haberman',
            '--model-key', 'qwen35_9b',
            '--models-config', str(tmp_path / 'models.toml'),
            '--agent-config', str(tmp_path / 'agent.toml'),
            '--output-dir', str(tmp_path / 'out'),
        ],
        dependencies=run_deps,
    ) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_render_markdown_refuses_report_path_or_raw_error() -> None:
    report = _trial('formal_primary')
    report['raw_error'] = '/users/fotile/private'
    with pytest.raises(ValueError):
        render_trial_markdown(report)


@pytest.mark.parametrize(
    'leaked_value',
    [
        'path=/users/fotile/private',
        'run_dir:/var/lib/autoai/run-1',
        r'cache=C:\\Users\\lenovo\\secret',
    ],
)
def test_report_rejects_assignment_embedded_server_paths(leaked_value) -> None:
    report = _trial('formal_primary')
    report['revision']['model'] = leaked_value
    with pytest.raises(ValueError, match='absolute path'):
        assert_report_safe(report)
