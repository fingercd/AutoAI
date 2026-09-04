"""Agent reservation 与 Run durable submission mapping 的显式安全对账。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from ..runs.contracts import Principal, RunRecord
from ..runs.repository import InvalidRunTransition, RunNotFound, RunRepository
from .contracts import (
    AGENT_API_CONTRACT_VERSION,
    AGENT_RESERVATION_PROTOCOL_VERSION,
    AgentDomainError,
)
from .repository import AgentExperimentRecord, AgentSessionRepository


DEFAULT_STALE_SECONDS = 300

_MANUAL_CODES = {
    'legacy_protocol_unknown',
    'submission_scope_mismatch',
    'submission_mapping_invalid',
    'mapped_run_unavailable',
    'run_reference_missing',
    'run_state_uncertain',
    'cancellation_uncertain',
    'reservation_timestamp_invalid',
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace('Z', '+00:00')))
    except ValueError:
        return None


class AgentReconciliationService:
    """无 FastAPI 依赖、由 SQLite 条件更新保证并发正确性的领域服务。"""

    def __init__(
        self,
        *,
        session_repository: AgentSessionRepository,
        run_repository: RunRepository,
    ) -> None:
        self.sessions = session_repository
        self.runs = run_repository

    def reconcile_session(
        self,
        *,
        session_id: str,
        principal: Principal,
        now: datetime | None = None,
        stale_after_seconds: int = DEFAULT_STALE_SECONDS,
    ) -> dict[str, object]:
        if stale_after_seconds < 0:
            raise ValueError('stale_after_seconds must be non-negative')
        moment = _utc(now or datetime.now(timezone.utc))
        stale_before = moment - timedelta(seconds=stale_after_seconds)
        try:
            self.sessions.get_session_scoped(session_id, principal=principal)
            records = self.sessions.list_experiments_scoped(
                session_id, principal=principal, include_released=True
            )
        except AgentDomainError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('Agent reservation 暂时不可读取') from exc

        summary = {
            'examined': 0,
            'recovered_bound': 0,
            'released': 0,
            'unchanged': 0,
            'manual_review': 0,
        }
        items: list[dict[str, object]] = []
        for record in records:
            if record.state == 'reserved':
                updated_at = _parse_timestamp(record.updated_at)
                if updated_at is None:
                    summary['examined'] += 1
                    result, changed = self._transition(
                        record,
                        new_state='compensation_required',
                        run_id=None,
                        resolution_code='reservation_timestamp_invalid',
                        principal=principal,
                        now=moment,
                    )
                    self._collect(summary, items, result, changed)
                    continue
                if updated_at is not None and updated_at > stale_before:
                    summary['unchanged'] += 1
                    items.append(self._item(record, 'reservation_fresh', manual=False))
                    continue
                summary['examined'] += 1
                result, changed = self._reconcile_reserved(
                    record, principal=principal, now=moment
                )
                self._collect(summary, items, result, changed)
            elif record.state == 'compensation_required':
                summary['examined'] += 1
                result, changed = self._reconcile_compensation(
                    record, principal=principal, now=moment
                )
                self._collect(summary, items, result, changed)

        status = 'unchanged'
        if summary['manual_review']:
            status = 'manual_review_required'
        elif summary['recovered_bound'] or summary['released']:
            status = 'reconciled'
        return {
            'contract_version': AGENT_API_CONTRACT_VERSION,
            'session_id': session_id,
            'status': status,
            'summary': summary,
            'items': items,
        }

    def _mapping(self, reservation: AgentExperimentRecord, principal: Principal):
        try:
            return self.runs.lookup_submission_mapping(
                reservation.experiment_id, principal=principal
            )
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('训练关联暂时不可读取') from exc

    def _reconcile_reserved(
        self,
        reservation: AgentExperimentRecord,
        *,
        principal: Principal,
        now: datetime,
    ) -> tuple[AgentExperimentRecord, bool]:
        if reservation.protocol_version != AGENT_RESERVATION_PROTOCOL_VERSION:
            return self._transition(
                reservation,
                new_state='compensation_required',
                run_id=None,
                resolution_code='legacy_protocol_unknown',
                principal=principal,
                now=now,
            )
        mapping = self._mapping(reservation, principal)
        if mapping.status == 'missing':
            return self._transition(
                reservation,
                new_state='released',
                run_id=None,
                resolution_code='stale_without_run',
                principal=principal,
                now=now,
            )
        if mapping.status == 'scope_mismatch':
            return self._transition(
                reservation,
                new_state='compensation_required',
                run_id=None,
                resolution_code='submission_scope_mismatch',
                principal=principal,
                now=now,
            )
        if mapping.submission_source != 'agent' or not mapping.run_id:
            return self._transition(
                reservation,
                new_state='compensation_required',
                run_id=None,
                resolution_code='submission_mapping_invalid',
                principal=principal,
                now=now,
            )
        try:
            self.runs.get_scoped(mapping.run_id, principal=principal)
        except RunNotFound:
            return self._transition(
                reservation,
                new_state='compensation_required',
                run_id=None,
                resolution_code='mapped_run_unavailable',
                principal=principal,
                now=now,
            )
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('训练记录暂时不可读取') from exc
        return self._transition(
            reservation,
            new_state='bound',
            run_id=mapping.run_id,
            resolution_code='recovered_binding',
            principal=principal,
            now=now,
        )

    def _reconcile_compensation(
        self,
        reservation: AgentExperimentRecord,
        *,
        principal: Principal,
        now: datetime,
    ) -> tuple[AgentExperimentRecord, bool]:
        mapping = self._mapping(reservation, principal)
        if mapping.status == 'scope_mismatch':
            return self._transition(
                reservation, new_state='compensation_required', run_id=None,
                resolution_code='submission_scope_mismatch', principal=principal, now=now,
            )
        if mapping.status == 'found' and mapping.submission_source != 'agent':
            return self._transition(
                reservation, new_state='compensation_required', run_id=None,
                resolution_code='submission_mapping_invalid', principal=principal, now=now,
            )
        if reservation.run_id and mapping.status == 'found' and mapping.run_id != reservation.run_id:
            return self._transition(
                reservation, new_state='compensation_required', run_id=None,
                resolution_code='submission_mapping_invalid', principal=principal, now=now,
            )
        run_id = reservation.run_id or (mapping.run_id if mapping.status == 'found' else None)
        if not run_id:
            return self._transition(
                reservation, new_state='compensation_required', run_id=None,
                resolution_code='run_reference_missing', principal=principal, now=now,
            )
        try:
            run = self.runs.get_scoped(run_id, principal=principal)
        except RunNotFound:
            return self._transition(
                reservation, new_state='compensation_required', run_id=None,
                resolution_code='mapped_run_unavailable', principal=principal, now=now,
            )
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('训练记录暂时不可读取') from exc

        if run.state == 'queued' and run.started_at is None:
            return self._cancel_unstarted(
                reservation, run=run, principal=principal, now=now
            )
        if run.state == 'running':
            return self._transition(
                reservation, new_state='compensation_required', run_id=run.run_id,
                resolution_code='run_still_active', principal=principal, now=now,
            )
        if run.state in {'succeeded', 'failed'}:
            return self._transition(
                reservation, new_state='bound', run_id=run.run_id,
                resolution_code='recovered_terminal_binding', principal=principal, now=now,
            )
        if run.state == 'cancelled':
            if run.started_at is None:
                return self._transition(
                    reservation, new_state='released', run_id=run.run_id,
                    resolution_code='cancelled_before_start', principal=principal, now=now,
                )
            return self._transition(
                reservation, new_state='bound', run_id=run.run_id,
                resolution_code='recovered_cancelled_binding', principal=principal, now=now,
            )
        return self._transition(
            reservation, new_state='compensation_required', run_id=run.run_id,
            resolution_code='run_state_uncertain', principal=principal, now=now,
        )

    def _cancel_unstarted(
        self,
        reservation: AgentExperimentRecord,
        *,
        run: RunRecord,
        principal: Principal,
        now: datetime,
    ) -> tuple[AgentExperimentRecord, bool]:
        # 先落 Agent DB 审计 claim；若该写入失败，绝不触碰 Run 终态。
        claimed, changed = self._transition(
            reservation, new_state='compensation_required', run_id=run.run_id,
            resolution_code='cancellation_pending', principal=principal, now=now,
        )
        if not changed:
            return claimed, False
        try:
            cancelled = self.runs.cancel_queued_unstarted_scoped(
                run.run_id, now=now, principal=principal,
            )
        except (RunNotFound, InvalidRunTransition):
            try:
                current_run = self.runs.get_scoped(run.run_id, principal=principal)
            except RunNotFound:
                return self._transition(
                    claimed, new_state='compensation_required', run_id=run.run_id,
                    resolution_code='mapped_run_unavailable', principal=principal,
                    now=now, increment_attempt=False,
                )
            except (sqlite3.Error, OSError) as exc:
                raise self._unavailable('训练记录暂时不可读取') from exc
            if current_run.state == 'running':
                return self._transition(
                    claimed, new_state='compensation_required', run_id=run.run_id,
                    resolution_code='run_still_active', principal=principal,
                    now=now, increment_attempt=False,
                )
            if current_run.state == 'cancelled' and current_run.started_at is None:
                cancelled = current_run
            else:
                return self._transition(
                    claimed, new_state='compensation_required', run_id=run.run_id,
                    resolution_code='cancellation_uncertain', principal=principal,
                    now=now, increment_attempt=False,
                )
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('Run 取消暂时不可执行') from exc
        if cancelled.state != 'cancelled' or cancelled.started_at is not None:
            return self._transition(
                claimed, new_state='compensation_required', run_id=run.run_id,
                resolution_code='cancellation_uncertain', principal=principal,
                now=now, increment_attempt=False,
            )
        return self._transition(
            claimed, new_state='released', run_id=run.run_id,
            resolution_code='cancelled_before_start', principal=principal,
            now=now, increment_attempt=False,
        )

    def _transition(
        self,
        reservation: AgentExperimentRecord,
        *,
        new_state: str,
        run_id: str | None,
        resolution_code: str,
        principal: Principal,
        now: datetime,
        increment_attempt: bool = True,
    ) -> tuple[AgentExperimentRecord, bool]:
        try:
            return self.sessions.reconcile_transition(
                reservation.experiment_id,
                expected_state=reservation.state,
                expected_updated_at=reservation.updated_at,
                new_state=new_state,
                run_id=run_id,
                resolution_code=resolution_code,
                reconciled_at=now,
                principal=principal,
                increment_attempt=increment_attempt,
            )
        except AgentDomainError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise self._unavailable('Agent reservation 暂时不可更新') from exc

    @staticmethod
    def _item(
        reservation: AgentExperimentRecord,
        resolution_code: str,
        *,
        manual: bool,
    ) -> dict[str, object]:
        return {
            'attempt': reservation.attempt,
            'state': reservation.state,
            'resolution_code': resolution_code,
            'requires_manual_review': manual,
        }

    def _collect(
        self,
        summary: dict[str, int],
        items: list[dict[str, object]],
        reservation: AgentExperimentRecord,
        changed: bool,
    ) -> None:
        code = reservation.resolution_code or 'concurrent_state_changed'
        manual = code in _MANUAL_CODES
        items.append(self._item(reservation, code, manual=manual))
        if not changed:
            summary['unchanged'] += 1
        elif reservation.state == 'released':
            summary['released'] += 1
        elif reservation.state == 'bound':
            summary['recovered_bound'] += 1
        else:
            summary['unchanged'] += 1
        if manual:
            summary['manual_review'] += 1

    @staticmethod
    def _unavailable(message: str) -> AgentDomainError:
        return AgentDomainError(
            'agent_reconciliation_unavailable', message,
            status_code=503, retryable=True,
            allowed_actions=('inspect_ml_session',),
        )
