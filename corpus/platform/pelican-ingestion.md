# Pelican Ingestion Service

Pelican is the public event intake service. Every client SDK writes to Pelican,
and Pelican is the only writer of the `events_raw` table.

## Ownership

Pelican is owned by the Platform Ingest team. The service runs in the
`ingest-prod` cluster across three regions: `eu-west-1`, `us-east-1` and
`ap-south-1`.

## Interface

Clients POST batches to `/v2/events`. A batch may contain at most 500 events
and 1 MiB of compressed payload. Pelican acknowledges a batch only after the
events are durable in `events_raw`, so a 200 response is a durability
guarantee, not a receipt.

## Rate limits

Each API key is limited to 2,000 events per second, averaged over a 10 second
window. Exceeding the limit returns HTTP 429 with the error code `PEL-429` and
a `Retry-After` header. Clients must honour `Retry-After`; SDKs older than
v4.2 ignore it and are the usual cause of retry storms.

A second limit applies per organisation: 20,000 events per second across all
keys. Breaching the organisation limit returns `PEL-430`, which is not
retryable and requires a quota change request.

## Schema handling

Pelican validates each event against the registered schema for its event type.
Unknown fields are preserved in the `extras` JSON column rather than rejected,
because dropping data silently was the root cause of incident INC-2291.
Events that fail validation outright are written to `events_rejected` with the
validation error attached, and are retained for 7 days.

## Failure modes

* `PEL-429` / `PEL-430`: rate limited. See the on-call runbook.
* `PEL-503`: downstream storage unavailable; Pelican sheds load and clients
  must retry with backoff.
* Schema registry unavailable: Pelican fails open and accepts events into
  `events_raw` with `schema_validated = false`.
