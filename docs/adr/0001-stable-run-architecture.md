# ADR 0001: Stable Run Architecture

## Decision

- SQLite RunRepository is the command-side authority.
- A separate local worker claims queued runs using a lease.
- status.json is a legacy projection, never a state-machine source.
- A Run becomes succeeded only after its downloadable Manifest is committed.
- Local requests do not accept owner_id or tenant_id.
- The architecture release preserves master model behavior; new models and algorithm changes require a separate ADR and acceptance dataset.

## Consequences

- Restart recovery is possible without Redis/Celery.
- Existing clients receive legacy status strings through an adapter.
- Generic Run-directory downloads are removed.
