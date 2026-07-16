# ADR 0001: Stable Run Architecture

- Status: Accepted
- Scope: Run persistence, worker execution, state projection and artifact publication
- Last reviewed: 2026-07-16

## Decision

- SQLite RunRepository is the command-side authority.
- A separate local worker claims queued runs using a lease.
- status.json is a legacy projection, never a state-machine source.
- A Run becomes succeeded only after its Manifest is committed; Manifest v2 uses an explicit public catalog and integrity metadata.
- Local requests use an empty Principal. Server deployment requires Bearer authentication, and owner/tenant are injected by the server rather than accepted in request bodies.
- Server-scoped Run/Dataset reads and mutations are filtered by Principal; unowned historical Runs require an explicit audited migration.
- `GET /api/training/runs/{run_id}/result` is the versioned result-page projection; raw artifacts remain the audit source.
- Worker heartbeats are persisted separately from Run state and exposed only as an anonymous health summary.
- At the time of this decision, the architecture release preserved master model behavior. Later classification-v2 model work is governed by its independent model plan and acceptance tests; it does not change this ADR's Run-state guarantees.

## Consequences

- Restart recovery is possible without Redis/Celery.
- Existing clients receive legacy status strings through an adapter.
- Generic Run-directory downloads are removed.
- A corrupt or partial result can be reported without treating status.json as authoritative or returning server paths.
