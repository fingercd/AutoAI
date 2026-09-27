# Step 9: isolated database backup and rollback acceptance

This is an acceptance procedure for synthetic databases, not a production restore command.
The production application and the SQLite schemas are unchanged by this closeout.

## Run and scope

Run `python -m pytest agent_poc/tests/test_step9_backup.py -q`. The checkout must contain
Git commits `39743e455cdfa5f7ba2c8f1f44f17553f38dfe4e` (pre-Step9) and
`7502267a2e673bcdb8d4583e5a9dea68ce48796c` (pre-assessment repair).
The test extracts those tracked sources into pytest temporary directories and executes
those actual historical programs using the current dependency environment. This verifies
source compatibility in this environment; it does not certify every historic dependency stack.

For machine-readable receipts, set `AUTOAI_T29_EVIDENCE` to a new evidence directory.
A repeated run needs a different directory: receipts are opened exclusively and never replaced.
No real canonical store, dataset, model, endpoint or production database is an input.
Socket dispatch and construction of the executing orchestration graph are forbidden.
Synthetic confirmed calls exercise accounting; they are not physical LLM/API requests.

## Supported boundary

The fixture database set is `calls.sqlite`, `checkpoints.sqlite`, and `agent.sqlite3`.
It represents a task with no training Run, so there is no Runs DB, artifact tree or model to
restore. Existing scope/Run migration tests cover their own contracts; this test does not
claim production-wide or artifact-inclusive disaster recovery.

The test uses `scripts.task_cost_report.readonly_snapshot`, the existing SQLite backup API
reader, then materializes that consistent snapshot to a new database. It never relies on
copying an open main file without WAL. All fixture writers are quiescent before the cutoff;
`BEGIN IMMEDIATE` is held on every database throughout the entire backup interval. Actual
contending writers are rejected. Consequently the serial backups share a stable cutoff;
this is not a claim that sequential backups of actively changing databases are atomic.

Restore means inspection in a newly created directory. SQLite integrity checks, FK checks,
full table/column/value digests, input/proposal/report binding, task/session/principal/policy
linkage, confirmed calls and cost rows must agree. The restored task retains its original
canonical path. Direct historical inspection is possible, but canonical diagnosis export,
execution preflight and a new metered call are rejected. The test does not authorize a
canonical rebinding or resume from a copied store.

Rollback program scope is deliberately narrow:

- The pre-Step9 executable can read its supported v8 history and settled journal on the
  additive schema without changing any historical table values or accounting.
- It rejects v9 checkpoints with `unknown State version`. No version marker or task binding
  is rewritten to disguise the new task as an old one. Running old code on v9 tasks is prohibited.
- The pre-assessment executable creates an authentic historical report without assessment;
  two current initializations preserve raw proposal, report JSON/digest, identity and costs.
  The missing historical assessment is never manufactured by migration.
- After a backup, the test commits a later call, later session, and later checkpoint. Restoring
  that older backup over the existing directory raises `FileExistsError` before any file
  write. All newer rows remain. Replacing live files after stopping services is not supported
  or certified by this procedure; a real in-place rollback requires a separate approved plan.

## Evidence map

| Test | Receipt | Observable result |
|---|---|---|
| `test_old_schema_double_migration_and_program_rollback` | `legacy-migration.json` | Actual pre-Step9 schema, migration twice, old rows/proposal/cost/identity unchanged; old executable reads supported v8 after additive migration. |
| `test_pre_assessment_report_keeps_original_proposal_and_hash` | `old-report.json` | Actual old report has no assessment, confirmed proposal retains insufficient_evidence; repeated initialization preserves all values and digests. |
| `test_wal_set_restore_canonical_and_post_backup_write_protection` | `wal-restore.json` | Nonempty committed WAL in all three DBs; main-only negative control misses report; consistent backup and restore preserve all data; canonical copies/new-version downgrade/old-backup overwrite rejected. |

T29's migration, scope, concurrency and canonical-binding tests remain supplementary evidence;
none alone is presented as a complete backup/restore test. Production version integration,
guardian, full historical Manifest audit and live backup/release/rollback remain separate release gates.

## Accepted non-blocking quality limitation

R2-Q2 at source `511c9ad1306a593c92bc8c5676f8fb6bc3b29ab1`, task
`step9-repair2-real-r2-q2`, response `chatcmpl-f0663d1feebc4b94a456474794ab8f1d`, is a
fixed negative quality example. Its explanation says "significantly lower" for
Train macro-F1 0.3088235294 versus Valid 0.2857142857, without a statistical test or interval.
The raw response is retained in the original acceptance evidence, not rewritten here.
`tentative` and `no_statistical_interval` do not cancel an unsupported assertion in the prose.

The architecture review of 2026-09-28 closed R-S9-01 through R-S9-04 and accepted this as a
non-blocking limitation for Step9 engineering sign-off, not as a passing quality sample.
T11 must separately retain this failed manual quality judgment; T32 must distinguish
successful real pipelines from sample quality. R2-Q3's acceptable rejection explanation
does not overwrite Q3 or R-Q3 failures. No new real batch, statistical test, seed change,
Test-based selection, deployment or push is part of this acceptance procedure.
