# Warehouse Table Ownership

Every table in the warehouse has exactly one writer. If you need data in a
table you do not own, ask the owning job's team for a view.

## events_raw

Written by the Pelican ingestion service. Partitioned by hour on
`received_at`. This table is append-only; corrections are published as
compensating events rather than updates.

## events_curated

Written by the Heron transform job, which reads `events_raw`, applies the
current schema version, hashes PII columns and deduplicates on
`(client_id, event_uuid)`. Heron runs every 15 minutes and is owned by the
Data Platform team.

Downstream consumers must read `events_curated`, never `events_raw`. The
Kestrel search service is the largest consumer.

## events_rejected

Written by Pelican. Holds events that failed schema validation, with the
validation error in the `error_detail` column.

## sessions_daily

Written by the Heron transform job as a second output. Sessionisation uses a
30 minute inactivity gap.

## billing_usage

Written by the Ledger job, owned by the Billing team. Ledger reads
`events_curated` and is the only job allowed to write to `billing_usage`.
Because billing is restated monthly, `billing_usage` is the one table in the
warehouse that permits updates.
