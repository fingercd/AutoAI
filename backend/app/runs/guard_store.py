"""Guard records in the existing Runs DB. No independent state or usage ledger."""
import json

from .guard import GuardError, GuardPolicy, GuardReport, digest


def initialize(connection):
    # Acquire the write reservation before schema reads; deferred read-to-write
    # upgrades can deadlock concurrent HTTP initialization in WAL mode.
    own_transaction = not connection.in_transaction
    if own_transaction:
        connection.execute('BEGIN IMMEDIATE')
    connection.execute('SAVEPOINT guard_schema')
    columns = {row[1] for row in connection.execute('PRAGMA table_info(runs)')}
    for name in ('guard_policy_json', 'publication_report_id'):
        if name not in columns:
            connection.execute(f'ALTER TABLE runs ADD COLUMN {name} TEXT')
    connection.execute('''CREATE TABLE IF NOT EXISTS run_guard_reports_v1 (
        report_id TEXT PRIMARY KEY, run_id TEXT, scope_digest TEXT NOT NULL,
        stage TEXT NOT NULL, report_json TEXT NOT NULL)''')
    connection.execute('CREATE INDEX IF NOT EXISTS guard_run_stage ON run_guard_reports_v1(run_id,stage)')
    connection.execute('DROP TRIGGER IF EXISTS budget_run_claim_contract_v1')
    connection.execute('''CREATE TRIGGER budget_run_claim_contract_v1
        BEFORE UPDATE OF state ON runs
        WHEN NEW.state='running' AND OLD.state='queued'
          AND json_extract(NEW.config_json,'$.execution_budget_task_id') IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM worker_heartbeats WHERE worker_id=NEW.worker_id
            AND contract_version IN ('training-worker-budget-v1','training-worker-guard-v1'))
        BEGIN SELECT RAISE(ABORT,'budget_worker_contract_required'); END''')
    connection.execute('''CREATE TRIGGER IF NOT EXISTS guard_run_claim_contract_v1
        BEFORE UPDATE OF state ON runs
        WHEN NEW.state='running' AND OLD.state='queued' AND NEW.guard_policy_json IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM worker_heartbeats WHERE worker_id=NEW.worker_id
            AND contract_version='training-worker-guard-v1')
        BEGIN SELECT RAISE(ABORT,'guard_worker_contract_required'); END''')
    connection.execute('''CREATE TRIGGER IF NOT EXISTS guard_success_contract_v1
        BEFORE UPDATE OF state ON runs
        WHEN NEW.state='succeeded' AND OLD.state!='succeeded' AND NEW.guard_policy_json IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM run_guard_reports_v1
            WHERE report_id=NEW.publication_report_id AND run_id=NEW.run_id
              AND stage='publication' AND json_extract(report_json,'$.status')='passed')
        BEGIN SELECT RAISE(ABORT,'guard_publication_required'); END''')
    connection.execute('''CREATE TRIGGER IF NOT EXISTS guard_policy_immutable_v1
        BEFORE UPDATE OF guard_policy_json ON runs
        WHEN OLD.guard_policy_json IS NOT NULL AND NEW.guard_policy_json IS NOT OLD.guard_policy_json
        BEGIN SELECT RAISE(ABORT,'guard_policy_immutable'); END''')
    connection.execute('RELEASE guard_schema')
    if own_transaction:
        connection.commit()


def insert(connection, value):
    value = GuardReport.model_validate(value.model_dump() if isinstance(value, GuardReport) else value)
    body = value.model_dump_json()
    old = connection.execute('SELECT report_json FROM run_guard_reports_v1 WHERE report_id=?', (value.report_id,)).fetchone()
    if old is not None:
        # created_at is audit only; replay preserves the original report.
        prior = GuardReport.model_validate_json(old[0])
        if prior.model_dump(exclude={'created_at'}) != value.model_dump(exclude={'created_at'}):
            raise GuardError('guard_report_binding_mismatch')
        return
    connection.execute('INSERT INTO run_guard_reports_v1 VALUES(?,?,?,?,?)',
        (value.report_id, value.bindings.run_id, value.bindings.scope_digest, value.stage, body))


def validate_binding(value, record, *, publication=False):
    expected = (record.run_id, digest([record.owner_id, record.tenant_id]),
                digest(record.config), digest(record.guard_policy), record.dataset_snapshot.get('sha256'))
    bound = value.bindings
    if (bound.run_id, bound.scope_digest, bound.config_digest, bound.policy_digest, bound.dataset_digest) != expected:
        raise GuardError('guard_report_binding_mismatch')
    GuardPolicy.model_validate(record.guard_policy)
    if publication and (value.stage != 'publication' or value.status != 'passed'
            or value.eligibility != 'eligible' or not bound.manifest_digest
            or bound.fence_digest != digest(record.claim_token)):
        raise GuardError('guard_report_binding_mismatch')
