# Data Retention Policy

This policy applies to every table in the warehouse and to all backups.

## Retention periods

| Data set         | Retention | Notes                                    |
|------------------|-----------|------------------------------------------|
| events_raw       | 30 days   | Deleted by partition drop, nightly       |
| events_rejected  | 7 days    | Includes the original payload            |
| events_curated   | 400 days  | PII columns hashed at write time         |
| sessions_daily   | 400 days  | Derived; no raw identifiers              |
| billing_usage    | 7 years   | Statutory retention for financial records|
| Backups          | 35 days   | Encrypted, restored only via change ticket|

Deletion is enforced by the retention sweeper, which runs nightly at 02:00 UTC
and emits a report to the `data-governance` channel. A table that misses two
consecutive sweeps raises a compliance alert.

## Personal data

PII columns are hashed with a per-organisation salt before they are written to
`events_curated`. The salt is stored in the key management service and rotated
annually. Raw identifiers exist only in `events_raw`, which is why its 30 day
retention is a hard limit rather than a default.

## Deletion requests

A verified subject deletion request must be completed within 30 calendar days.
The request is executed by the `dsr-runner` job, which deletes from
`events_curated` and `sessions_daily` and records a tombstone. Because
`events_raw` expires within 30 days on its own, it is not rewritten.

Deletion requests cannot remove rows from `billing_usage`; statutory retention
takes precedence, and the records there carry no raw identifiers.

## Exceptions

Retention exceptions require written approval from both the data protection
officer and the owning team, are time-boxed to 90 days, and are tracked in the
retention exception register.
