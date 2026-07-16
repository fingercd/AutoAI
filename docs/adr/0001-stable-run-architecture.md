# ADR 0001: Stable Run Architecture

- Status: Accepted
- Scope: Run persistence, worker execution, state projection and artifact publication
- Last reviewed: 2026-07-15

## Decision

- SQLite RunRepository is the command-side authority.
- A separate local worker claims queued runs using a lease.
- status.json is a legacy projection, never a state-machine source.
- A Run becomes succeeded only after its downloadable Manifest is committed.
- Local requests do not accept owner_id or tenant_id.
- At the time of this decision, the architecture release preserved master model behavior. Later classification-v2 model work is governed by its independent model plan and acceptance tests; it does not change this ADR's Run-state guarantees.

## Consequences

- Restart recovery is possible without Redis/Celery.
- Existing clients receive legacy status strings through an adapter.
- Generic Run-directory downloads are removed.
