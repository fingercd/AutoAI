"""Crash-visible trial facts and monotonic stage spans for one training Run."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import csv
import hashlib
import io
import json
import os
import pickle
import tempfile
import time


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False, suffix='.tmp') as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    # Windows readers and indexers can briefly hold the destination without
    # delete sharing.  A bounded retry keeps the atomic write contract.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))


class SearchAccounting:
    def __init__(self, run_dir: Path, plan: dict[str, Any], run_id: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.run_id = run_id
        self.trials: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self._starts: dict[str, float] = {}
        # A restarted worker may not replay a fit without a verified checkpoint.
        # Leave the earlier ledger intact and fail the Run explicitly.
        if (self.run_dir / 'search_trials.json').exists():
            raise ValueError('interrupted search has no verified trial checkpoint')
        _atomic_json(self.run_dir / 'search_plan.json', plan)
        self._flush()

    def _flush(self) -> None:
        _atomic_json(self.run_dir / 'search_trials.json', self.trials)
        _atomic_json(self.run_dir / 'search_timeline.json', dict(
            schema_version='search-timeline-v1', clock='utc-plus-process-monotonic-v1',
            run_id=self.run_id, spans=self.spans))
        # This partial projection remains readable after cancellation or a worker
        # crash.  A completed Run replaces it only after every planned fit.
        counts = {state: sum(row['state'] == state for row in self.trials)
                  for state in ('prepared', 'running', 'succeeded', 'failed', 'interrupted')}
        _atomic_json(self.run_dir / 'search_summary.json', dict(
            schema_version='search-summary-v1', run_id=self.run_id,
            plan_digest=self.plan['plan_digest'], mode=self.plan['mode'],
            planned_trials=None, started_trials=len(self.trials),
            completed_trials=counts['succeeded'], failed_trials=counts['failed'],
            interrupted_trials=counts['interrupted'], pending_trials=counts['prepared'] + counts['running'],
            effective_search=self.plan['effective_search'], selected=None,
            selection_metric=self.plan['selection_metric'],
            candidate_fit_count=counts['succeeded'] + counts['failed'] + counts['interrupted'],
            final_refit_count=None, total_fit_count=None,
            actual_epochs=sum((row['actual_epochs'] or 0) for row in self.trials),
            training_batches=sum((row['training_batches'] or 0) for row in self.trials),
            integrity='incomplete'))

    def record_queue_wait(self, created_at: str | None, started_at: str | None) -> None:
        """Persist the repository's queue interval without mixing process clocks."""
        row = dict(span_id=f'span-{len(self.spans)}', parent_span_id=None,
                   run_id=self.run_id, fold_index=None, trial_index=None,
                   stage='queue_wait', status='unknown', started_at=created_at,
                   ended_at=started_at, duration_seconds=None,
                   duration_clock='utc_wall_same_host_estimate',
                   limitation='Repository timestamps may include cross-process wall clock skew.')
        try:
            if created_at and started_at:
                start=datetime.fromisoformat(created_at)
                end=datetime.fromisoformat(started_at)
                if start.tzinfo is not None and end.tzinfo is not None and end>=start:
                    row['duration_seconds']=(end-start).total_seconds()
                    row['status']='succeeded'
        except ValueError:
            pass
        self.spans.append(row)
        self._flush()

    def start_stage(self, name: str, *, fold_index: int | None = None,
                    parent_span: str | None = None) -> dict[str, Any]:
        row = dict(span_id=f'span-{len(self.spans)}', parent_span_id=parent_span,
                   run_id=self.run_id, fold_index=fold_index, trial_index=None,
                   stage=name, status='running', started_at=_utc(), ended_at=None,
                   duration_seconds=None, duration_clock='process_monotonic')
        self.spans.append(row)
        self._starts[row['span_id']] = time.monotonic()
        self._flush()
        return row

    def end_stage(self, row: dict[str, Any], status: str = 'succeeded') -> None:
        row['status'] = status
        row['ended_at'] = _utc()
        row['duration_seconds'] = time.monotonic() - self._starts.pop(row['span_id'])
        self._flush()

    @contextmanager
    def span(self, name: str, *, fold_index: int | None = None,
             trial_index: int | None = None, parent_span: str | None = None,
             applicable: bool = True) -> Iterator[dict[str, Any]]:
        span_id = f'span-{len(self.spans)}'
        row = dict(span_id=span_id, parent_span_id=parent_span, run_id=self.run_id,
                   fold_index=fold_index, trial_index=trial_index, stage=name,
                   status='running' if applicable else 'not_applicable',
                   started_at=_utc() if applicable else None,
                   ended_at=None, duration_seconds=None,
                   duration_clock='process_monotonic' if applicable else None)
        self.spans.append(row)
        self._flush()
        if not applicable:
            yield row
            return
        start = time.monotonic()
        try:
            yield row
        except BaseException as exc:
            row['status'] = 'interrupted' if type(exc).__name__ in ('InvalidRunTransition', 'TrainingRunReplaced') else 'failed'
            row['error_type'] = type(exc).__name__
            raise
        else:
            row['status'] = 'succeeded'
        finally:
            row['ended_at'] = _utc()
            row['duration_seconds'] = time.monotonic() - start
            self._flush()

    @contextmanager
    def trial(self, fold_index: int, candidate: dict[str, Any], *, parent_span: str) -> Iterator[dict[str, Any]]:
        index = candidate['index']
        row = dict(run_id=self.run_id, fold_index=fold_index, trial_index=index,
                   params=candidate['params'], params_digest=candidate['params_digest'],
                   state='prepared', started_at=None, ended_at=None, duration_seconds=None,
                   actual_epochs=None, training_batches=None, selection_metric=self.plan['selection_metric'],
                   selection_score=None, error_type=None, artifact_digest=None)
        if any((item['fold_index'], item['trial_index']) == (fold_index, index) for item in self.trials):
            raise ValueError('duplicate trial execution')
        self.trials.append(row)
        self._flush()
        with self.span('trial_train_validation', fold_index=fold_index, trial_index=index,
                       parent_span=parent_span) as span:
            row['state'] = 'running'
            row['started_at'] = span['started_at']
            self._flush()
            started = time.monotonic()
            try:
                yield row
            except BaseException as exc:
                row['state'] = 'interrupted' if type(exc).__name__ in ('InvalidRunTransition', 'TrainingRunReplaced') else 'failed'
                row['error_type'] = type(exc).__name__
                raise
            else:
                row['state'] = 'succeeded'
            finally:
                row['ended_at'] = _utc()
                row['duration_seconds'] = time.monotonic() - started
                self._flush()

    def sync_trial_durations(self) -> None:
        spans = {(s['fold_index'], s['trial_index']): s for s in self.spans
                 if s['stage'] == 'trial_train_validation'}
        for row in self.trials:
            span = spans.get((row['fold_index'], row['trial_index']))
            if span is not None:
                row['duration_seconds'] = span['duration_seconds']
        self._flush()

    def save_traditional_candidate(self, *, fold_index: int, trial_index: int,
                                   model: Any) -> str:
        """Persist a private fitted candidate before its ledger row can succeed."""
        folder=self.run_dir/f'trial_{fold_index}_{trial_index}'
        folder.mkdir(exist_ok=True)
        target=folder/'model.pkl'
        with tempfile.NamedTemporaryFile('wb',dir=folder,delete=False,suffix='.tmp') as handle:
            temporary=Path(handle.name)
            try:
                pickle.dump(model,handle,protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary,target)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def finish(self, *, selected: list[dict[str, Any]], refit_count: int) -> dict[str, Any]:
        self.sync_trial_durations()
        if any(row['state'] != 'succeeded' for row in self.trials):
            raise ValueError('search trials incomplete')
        planned = self.plan['effective_trials'] * len(selected)
        if len(self.trials) != planned:
            raise ValueError('search trial count differs from frozen plan')
        body = dict(schema_version='search-summary-v1', run_id=self.run_id,
                    plan_digest=self.plan['plan_digest'], mode=self.plan['mode'],
                    planned_trials=planned, started_trials=len(self.trials),
                    completed_trials=len(self.trials), failed_trials=0, interrupted_trials=0,
                    effective_search=self.plan['effective_search'], selected=selected,
                    selection_metric=self.plan['selection_metric'],
                    candidate_fit_count=len(self.trials), final_refit_count=refit_count,
                    total_fit_count=len(self.trials)+refit_count,
                    actual_epochs=sum((row['actual_epochs'] or 0) for row in self.trials),
                    training_batches=sum((row['training_batches'] or 0) for row in self.trials),
                    integrity='complete')
        _atomic_json(self.run_dir / 'search_summary.json', body)
        fields = ('run_id','fold_index','trial_index','state','params_digest','selection_metric',
                  'selection_score','actual_epochs','training_batches','started_at','ended_at',
                  'duration_seconds','error_type')
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in self.trials)
        (self.run_dir / 'search_trials.csv').write_text(output.getvalue(), encoding='utf-8-sig')
        return body
