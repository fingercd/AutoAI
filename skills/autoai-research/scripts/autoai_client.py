"""Small resumable HTTP client. Training is always submitted through preflight.

Credentials: AUTOAI_SERVICE_TOKEN (never persisted). Service: AUTOAI_SERVICE_URL.
Each experiment owns a separate --task-dir containing state and provenance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from urllib.parse import urljoin, urlsplit
import uuid

import httpx

from prepare_dataset import fingerprint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def provenance():
    def git(*args):
        result = subprocess.run(['git', *args], capture_output=True, encoding='utf-8', errors='replace')
        return result.stdout if result.returncode == 0 else None
    untracked = git('ls-files', '--others', '--exclude-standard', '--', 'backend/app', 'skills/autoai-research') or ''
    additions = {name: Path(name).read_text(encoding='utf-8') for name in untracked.splitlines()
                 if Path(name).suffix in {'.py', '.md', '.yaml'} and Path(name).is_file()}
    return {'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'code_diff': git('diff', '--', 'backend/app', 'skills/autoai-research'),
            'untracked_code': additions,
            'python': sys.version, 'platform': platform.platform(),
            'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
            'client_sha256': fingerprint(__file__),
            'converter_sha256': fingerprint(Path(__file__).with_name('prepare_dataset.py'))}


class AutoAIClient:
    def __init__(self, service_url, task_dir=None, *, transport=None):
        parsed = urlsplit(service_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('服务地址必须是无凭据、查询参数和片段的 HTTP(S) 地址')
        self.service_url = service_url.rstrip('/') + '/'
        self.task_dir = Path(task_dir).resolve() if task_dir else None
        self.state = {}
        if self.task_dir:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            state_path = self.task_dir / 'state.json'
            if state_path.exists():
                self.state = json.loads(state_path.read_text(encoding='utf-8'))
                if self.state['service_url'] != self.service_url:
                    raise ValueError('任务记录属于另一服务地址，请使用新的任务目录')
            else:
                self.state = {'service_url': self.service_url, 'datasets': {}}
                self.save()
        token = os.environ.get('AUTOAI_SERVICE_TOKEN')
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        self.http = httpx.Client(timeout=60, headers=headers, transport=transport, follow_redirects=False)

    def save(self):
        if self.task_dir:
            write_json(self.task_dir / 'state.json', self.state)

    def url(self, path):
        target = urljoin(self.service_url, path)
        if urlsplit(target).netloc != urlsplit(self.service_url).netloc or urlsplit(target).scheme != urlsplit(self.service_url).scheme:
            raise ValueError('拒绝向其他服务地址发送凭据或下载请求')
        return target

    def request(self, method, path, **kwargs):
        # Safe retry policy: GETs and keyed training submissions only.
        retry = method == 'GET' or 'Idempotency-Key' in kwargs.get('headers', {})
        for attempt in range(3 if retry else 1):
            try:
                response = self.http.request(method, self.url(path), **kwargs)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError):
                if not retry or attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError('unreachable')

    def multipart(self, path, files, options=None):
        with ExitStack() as stack:
            uploads = [('files', (Path(file).name, stack.enter_context(Path(file).open('rb')), 'application/octet-stream')) for file in files]
            data = {key: str(value).lower() if isinstance(value, bool) else str(value)
                    for key, value in (options or {}).items() if value is not None}
            return self.request('POST', path, files=uploads, data=data).json()

    def upload(self, file, role, *, retry_uncertain=False):
        if role not in {'primary', 'external_test'}:
            raise ValueError('role 必须为 primary 或 external_test')
        file = Path(file).resolve()
        sha = fingerprint(file)
        previous = self.state['datasets'].get(role)
        if self.state.get('submission') and (not previous or previous['sha256'] != sha):
            raise ValueError('已有训练提交，不能改变数据来源；新实验请使用新任务目录')
        if previous and previous['sha256'] == sha:
            if previous.get('dataset_id'):
                return previous
            if not retry_uncertain:
                raise ValueError('上次上传结果未知；请先检查服务记录，明确重试时使用 --retry-uncertain')
        raw = self.task_dir / 'originals'
        raw.mkdir(exist_ok=True)
        copy = raw / f'{sha[:12]}-{file.name}'
        if not copy.exists():
            shutil.copy2(file, copy)
        self.state['datasets'][role] = {'sha256': sha, 'name': file.name, 'original_copy': str(copy), 'status': 'uploading'}
        self.save()
        with file.open('rb') as handle:
            result = self.request('POST', '/api/datasets/upload', files={'file': (file.name, handle, 'text/csv')}).json()
        entry = self.state['datasets'][role]
        # Curve arrays are already preserved in the source file. Do not copy
        # millions of preview values into state or online experiment config.
        entry.update(dataset_id=result['dataset_id'], summary={key: value for key, value in result['summary'].items()
                                                              if key not in {'path', 'curves'}}, status='registered')
        self.save()
        return entry

    def preprocess(self, kind, files, options, *, retry_uncertain=False):
        allowed = {'start_row', 'end_row', 'range_mode', 'x_min', 'x_max', 'baseline_method', 'hplc_interpolate'}
        if set(options) - allowed:
            raise ValueError(f'未知预处理参数：{sorted(set(options) - allowed)}')
        identities = [{'name': Path(file).name, 'sha256': fingerprint(file)} for file in files]
        if len({item['name'] for item in identities}) != len(identities):
            raise ValueError('原始文件名重复，无法按 Name 唯一映射标签和样品分组；请先明确命名')
        identity = {'kind': kind, 'files': identities, 'options': options}
        previous = self.state.get('preprocessing')
        if previous:
            if previous['request'] != identity:
                raise ValueError('同一任务目录已有不同预处理请求，请使用新目录')
            if previous.get('output'):
                if fingerprint(previous['output']) != previous['output_sha256']:
                    raise ValueError('预处理文件已变化，请使用新任务目录')
                return previous
            if not previous.get('download_url') and not retry_uncertain:
                raise ValueError('预处理结果未知；明确重试时使用 --retry-uncertain')
        else:
            raw = self.task_dir / 'originals'
            raw.mkdir(exist_ok=True)
            for file, item in zip(files, identities):
                target = raw / f"{item['sha256'][:12]}-{item['name']}"
                if not target.exists():
                    shutil.copy2(file, target)
            self.state['preprocessing'] = {'request': identity}
            self.save()
        entry = self.state['preprocessing']
        if not entry.get('download_url'):
            if kind == 'hplc':
                inspection = self.multipart('/api/preprocess/hplc/inspect', files)
                write_json(self.task_dir / 'hplc-inspection.json', inspection)
            reply = self.multipart(f'/api/preprocess/{kind}', files, options)
            write_json(self.task_dir / 'preprocessing-response.json', {key: value for key, value in reply.items() if key != 'output_path'})
            entry.update({key: value for key, value in reply.items() if key not in {'output_path', 'curves'}})
            self.save()
        response = self.request('GET', entry['download_url'])
        output = self.task_dir / 'preprocessed.csv'
        output.write_bytes(response.content)
        entry.update(output=str(output), output_sha256=fingerprint(output))
        self.save()
        return entry

    def preflight(self, plan):
        plan = dict(plan)
        plan.setdefault('dataset_id', self.state['datasets'].get('primary', {}).get('dataset_id'))
        external = self.state['datasets'].get('external_test', {}).get('dataset_id')
        if external:
            plan.setdefault('test_dataset_id', external)
        if not plan.get('dataset_id'):
            raise ValueError('请先上传主数据，或在计划中提供 dataset_id')
        result = self.request('POST', '/api/training/preflight', json=plan).json()
        if self.state.get('submission') and self.state.get('plan') != plan:
            raise ValueError('已提交或正在提交的任务不能改配置；新实验请使用新任务目录')
        self.state.update(plan=plan, preflight=result)
        self.save()
        write_json(self.task_dir / 'preflight.json', result)
        return result

    def submit(self):
        previous = self.state.get('submission')
        if previous and previous.get('id'):
            return self.status()
        # If the first response was lost, replay the saved payload/key first.
        if previous is None:
            if 'plan' not in self.state:
                raise ValueError('请先执行 preflight --plan，核验实验方案后再提交')
            preview = self.preflight(self.state['plan'])
            if not preview['runnable']:
                raise ValueError('训练预检未通过，请检查 preflight.json')
            worker = preview['worker']
            if not worker.get('available') or not worker.get('compatible'):
                raise ValueError('Worker 当前不可用或不兼容；已保存方案，请恢复后重试提交')
            task_type = self.state['plan'].get('task_type', 'run')
            self.state['submission'] = {'key': uuid.uuid4().hex, 'task_type': task_type,
                                        'payload': preview['submit_payload']}
            self.state['provenance'] = provenance()
            self.state['cloud_id'] = uuid.uuid4().hex[:8]
            self.save()
            self.log({'experiment/submission_started': 1})
            self.sync_cloud(verify=False)
        entry = self.state['submission']
        path = '/api/training/runs' if entry['task_type'] == 'run' else '/api/training/batches'
        response = self.request('POST', path, json=entry['payload'], headers={'Idempotency-Key': entry['key']}).json()
        entry['id'] = response['run_id' if entry['task_type'] == 'run' else 'batch_id']
        entry['response'] = response
        self.save()
        self.log({'experiment/submitted': 1})
        self.sync_cloud(verify=False)
        return response

    def status(self):
        entry = self.state['submission']
        if not entry.get('id'):
            raise ValueError('提交响应未知；运行 submit 用原请求键恢复，不要新建任务')
        path = '/api/training/runs' if entry['task_type'] == 'run' else '/api/training/batches'
        result = self.request('GET', f"{path}/{entry['id']}").json()
        self.state['latest_status'] = result
        self.save()
        return result

    def stop(self):
        entry = self.state['submission']
        path = '/api/training/runs' if entry['task_type'] == 'run' else '/api/training/batches'
        # No blind POST retry; query status before an explicit retry.
        result = self.request('POST', f"{path}/{entry['id']}/stop").json()
        self.state['latest_status'] = result
        self.save()
        self.log({'experiment/cancel_requested': 1})
        self.sync_cloud(verify=False)
        return result

    def log(self, metrics):
        path = self.task_dir / 'events.jsonl'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'time': time.time(), 'metrics': metrics}, ensure_ascii=False, allow_nan=False) + '\n')

    def sync_cloud(self, *, verify=True):
        """Replay local events to one online experiment; failures keep the journal."""
        if 'cloud_id' not in self.state:
            return {'status': 'not_started'}
        try:
            import swanlab
            datasets = {role: {**item, 'summary': {key: value for key, value in item.get('summary', {}).items()
                                                  if key != 'curves'}} for role, item in self.state['datasets'].items()}
            metadata = {**self.state.get('provenance', {}), 'datasets': datasets,
                        'plan': self.state.get('plan'), 'submission': self.state.get('submission'),
                        'effective_configs': self.state.get('preflight', {}).get('normalized_configs'),
                        'weights_version': 'no_pretrained_weights'}
            cloud = swanlab.init(project='AutoAI-Skill', name=self.task_dir.name,
                                 id=self.state['cloud_id'], resume='allow', mode='online',
                                 log_dir=str(self.task_dir / 'swanlab'), config=metadata)
            self.state['cloud_url'] = cloud.url
            self.save()
            events = [json.loads(line) for line in (self.task_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
            for step, event in enumerate(events):
                if step >= self.state.get('cloud_synced_count', 0) and event['metrics']:
                    swanlab.log(event['metrics'], step=step)
            swanlab.finish()
            self.state.pop('cloud_error_type', None)
            self.state['cloud_synced_count'] = len(events)
            self.state['cloud_status'] = 'uploaded_unverified'
            if verify:
                path = cloud.url.replace('https://swanlab.cn/@', '').replace('/runs/', '/')
                run = swanlab.Api().run(path)
                expected = {}
                for event in events:
                    expected.update(event['metrics'])
                remote = run.summary(keys=list(expected))
                remote_config = run.profile.get('config', {})
                missing_config = [key for key, value in metadata.items() if remote_config.get(key, {}).get('value') != value]
                metrics_match = all(remote.get(key, {}).get('value') == value for key, value in expected.items())
                write_json(self.task_dir / 'cloud-verification.json', {'url': cloud.url, 'expected': expected,
                           'remote_summary': remote, 'config_mismatches': missing_config})
                self.state['cloud_status'] = 'verified' if metrics_match and not missing_config else 'uploaded_unverified'
                if not metrics_match:
                    # SDK upload failures can be logged without raising. Retain
                    # replay eligibility until the remote metrics are confirmed.
                    self.state['cloud_synced_count'] = 0
        except Exception as exc:
            # Do not print third-party exception strings which may include credentials.
            self.state['cloud_status'] = 'pending_upload_or_verification'
            self.state['cloud_error_type'] = type(exc).__name__
        self.save()
        return {key: self.state.get(key) for key in ('cloud_status', 'cloud_url', 'cloud_error_type')}

    def watch(self, seconds=60, interval=3):
        end = time.monotonic() + seconds
        while True:
            status = self.status()
            runs = status.get('runs', [status])
            metrics = {}
            for run in runs:
                prefix = run.get('model_type') or run.get('run_id', 'run')
                progress = run.get('progress', run)
                for key in ('current_epoch', 'completed_folds', 'search_completed', 'feature_scheme_index'):
                    if isinstance(progress.get(key), (int, float)):
                        metrics[f'progress/{prefix}/{key}'] = progress[key]
                metrics[f'progress/{prefix}/finished'] = int(run.get('state') in {'succeeded', 'failed', 'cancelled'})
            self.log(metrics)
            if status['state'] in {'succeeded', 'failed', 'cancelled', 'partial'} or time.monotonic() >= end:
                break
            time.sleep(min(interval, max(0, end - time.monotonic())))
        self.sync_cloud(verify=False)
        return status

    def result(self, *, download=False):
        entry = self.state['submission']
        exports = []
        if entry['task_type'] == 'batch':
            status = self.status()
            run_ids = [run['run_id'] for run in status['runs']]
            comparison = self.request('GET', f"/api/training/batches/{entry['id']}/comparison").json()
            write_json(self.task_dir / 'comparison.json', comparison)
            if download and comparison.get('comparable'):
                content = self.request('GET', f"/api/training/batches/{entry['id']}/predictions.xlsx").content
                target = self.task_dir / 'predictions.xlsx'
                target.write_bytes(content)
                exports.append(str(target))
                archive = self.request('GET', f"/api/training/batches/{entry['id']}/archive").json()
                write_json(self.task_dir / 'archive-status.json', archive)
                if archive.get('state') == 'ready':
                    content = self.request('GET', f"/api/training/batches/{entry['id']}/archive/files/all.zip").content
                    target = self.task_dir / 'comparison.zip'
                    target.write_bytes(content)
                    exports.append(str(target))
            link = f"{self.service_url}#/comparison?batch_id={entry['id']}"
        else:
            run_ids = [entry['id']]
            link = f"{self.service_url}#/results?run_id={entry['id']}"
        summaries = []
        for run_id in run_ids:
            result = self.request('GET', f'/api/training/runs/{run_id}/result').json()
            write_json(self.task_dir / f'result-{run_id}.json', result)
            summary = {**result['run'], **{key: result.get(key) for key in ('model', 'evaluation', 'warnings')}}
            # Verify the underlying metric files through the public download route,
            # whose Manifest boundary hashes contents rather than only checking size.
            for artifact in result.get('artifacts', []):
                if artifact['name'] in {'metrics.json', 'cv_metrics.json'} and artifact.get('download_url'):
                    self.download_artifact(artifact, run_id, save=download)
            summary['metrics'] = result.get('metrics', {}).get('primary', {})
            summaries.append(summary)
            strategy = summary['evaluation']['strategy']
            metric_group = 'pooled_oof' if strategy == 'leave_one_sample_id_cv' else 'external_test' if 'external_test' in strategy else 'test'
            model = summary['model']['type']
            self.log({f"{metric_group}/{model}/{key}": value for key, value in summary['metrics'].items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
            if download:
                for artifact in result.get('artifacts', []):
                    if artifact.get('integrity') != 'ok' or not artifact.get('download_url'):
                        continue
                    if artifact['name'] not in {'metrics.json', 'cv_metrics.json'}:
                        self.download_artifact(artifact, run_id, save=True)
        report = {'result_url': link, 'runs': summaries, 'exports': exports, 'cloud': self.sync_cloud()}
        write_json(self.task_dir / 'summary.json', report)
        return report

    def download_artifact(self, artifact, run_id, *, save):
        name = artifact['name']
        if Path(name).name != name:
            raise ValueError('产物名称不是安全文件名')
        content = self.request('GET', artifact['download_url']).content
        if artifact.get('size_bytes') is not None and len(content) != artifact['size_bytes']:
            raise ValueError('下载产物大小不匹配')
        if artifact.get('sha256') and hashlib.sha256(content).hexdigest() != artifact['sha256']:
            raise ValueError('下载产物 SHA-256 不匹配')
        if save:
            target = self.task_dir / 'artifacts' / run_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-url', default=os.environ.get('AUTOAI_SERVICE_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--task-dir', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('capabilities')
    sub.add_parser('health')
    upload = sub.add_parser('upload')
    upload.add_argument('file', type=Path)
    upload.add_argument('--role', choices=['primary', 'external_test'], default='primary')
    upload.add_argument('--retry-uncertain', action='store_true')
    inspect_hplc = sub.add_parser('inspect-hplc')
    inspect_hplc.add_argument('files', type=Path, nargs='+')
    preprocess = sub.add_parser('preprocess')
    preprocess.add_argument('--kind', choices=['raman', 'hplc'], required=True)
    preprocess.add_argument('--options', type=Path, required=True)
    preprocess.add_argument('--retry-uncertain', action='store_true')
    preprocess.add_argument('files', type=Path, nargs='+')
    preflight = sub.add_parser('preflight')
    preflight.add_argument('--plan', type=Path, required=True)
    for name in ('submit', 'status', 'stop', 'sync-cloud'):
        sub.add_parser(name)
    watch = sub.add_parser('watch')
    watch.add_argument('--seconds', type=float, default=60)
    watch.add_argument('--interval', type=float, default=3)
    result = sub.add_parser('result')
    result.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.command not in {'capabilities', 'health', 'inspect-hplc'} and not args.task_dir:
        parser.error('此操作需要 --task-dir，以保存来源和恢复记录')
    client = AutoAIClient(args.service_url, args.task_dir)
    try:
        command = args.command
        if command in {'capabilities', 'health'}:
            output = client.request('GET', '/api/models' if command == 'capabilities' else '/health').json()
        elif command == 'inspect-hplc':
            output = client.multipart('/api/preprocess/hplc/inspect', args.files)
        elif command == 'upload':
            output = client.upload(args.file, args.role, retry_uncertain=args.retry_uncertain)
        elif command == 'preprocess':
            options = json.loads(args.options.read_text(encoding='utf-8-sig'))
            output = client.preprocess(args.kind, args.files, options, retry_uncertain=args.retry_uncertain)
        elif command == 'preflight':
            output = client.preflight(json.loads(args.plan.read_text(encoding='utf-8-sig')))
        elif command == 'result':
            output = client.result(download=args.download)
        elif command == 'watch':
            if args.seconds <= 0 or args.interval <= 0:
                raise ValueError('等待时间和间隔必须大于零')
            output = client.watch(args.seconds, args.interval)
        else:
            output = getattr(client, command.replace('-', '_'))()
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError, httpx.HTTPError) as exc:
        token = os.environ.get('AUTOAI_SERVICE_TOKEN', '')
        message = str(exc).replace(token, '[redacted]') if token else str(exc)
        print(json.dumps({'error': message, 'recovery': '保留任务目录；提交未知时使用同目录 submit 恢复'}, ensure_ascii=False))
        raise SystemExit(1)
    finally:
        client.http.close()


if __name__ == '__main__':
    main()
