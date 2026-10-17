# Kestrel Search Service

Kestrel serves the product's search and analytics queries. It is owned by the
Discovery team.

## Data

Kestrel indexes `events_curated` incrementally. The indexer lags the table by
at most 5 minutes under normal load; the lag is exported as the
`kestrel_index_lag_seconds` metric and alerts at 900 seconds.

## Latency budget

The p95 latency budget for `/search` is 180 ms and the p99 budget is 450 ms.
Queries that exceed 5 seconds are cancelled server-side and return `KES-504`.

## Query limits

A single query may request at most 10,000 rows. Aggregations are capped at
50 buckets per dimension and 3 nested levels. Requests that exceed these caps
return `KES-400` with the offending parameter named.

## Caching

Kestrel caches query results for 60 seconds, keyed by the normalised query and
the organisation id. Cache entries are invalidated early when the indexer
commits a batch that touches the queried time range, so a search run twice in
a minute can legitimately return different counts.

## Escalation

Kestrel paging goes to the Discovery team's rotation. Index lag alerts are
routed to the Data Platform team instead, because lag is almost always caused
by the upstream transform job rather than by Kestrel.
