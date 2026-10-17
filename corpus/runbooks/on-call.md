# On-Call Runbook

## Rotations

| Service / job        | Rotation       | Business hours only |
|----------------------|----------------|---------------------|
| Pelican              | Platform Ingest| no                  |
| Heron transform job  | Data Platform  | no                  |
| Kestrel              | Discovery      | no                  |
| Ledger               | Billing        | yes                 |

The primary on-call acknowledges within 5 minutes. If the primary does not
acknowledge within 10 minutes the page escalates to the secondary, and after
20 minutes to the engineering manager on duty.

## Handling PEL-429 rate limiting

1. Confirm the spike is a single API key: check the `pelican_rate_limited`
   metric grouped by `api_key_hash`.
2. If one key dominates, contact the customer before raising the limit. A
   temporary limit increase is applied with
   `pelicanctl quota set --key <hash> --eps <value> --ttl 2h`.
3. If many keys are limited at once, suspect a retry storm from SDKs older
   than v4.2, which ignore `Retry-After`. Enable the global shed valve with
   `pelicanctl shed --percent 10`.
4. Never raise the organisation limit (`PEL-430`) during an incident. That
   limit protects the warehouse write path.

## Handling Heron lag

Heron lag shows up as Kestrel index lag, so the first page often reaches the
wrong team. Check `heron_last_success_timestamp` before investigating Kestrel.

1. If Heron has not succeeded in two cycles (30 minutes), restart the job with
   `heronctl rerun --from-last-success`.
2. Heron is idempotent: it deduplicates on `(client_id, event_uuid)`, so a
   rerun over an already-processed window is safe.
3. If the schema registry is unavailable, Heron blocks rather than failing
   open. This is intentional: curated data must never contain unvalidated
   events.

## Incident severities

* SEV1: data loss, or the public API is down in more than one region.
* SEV2: a single region is degraded, or curated data is more than 2 hours
  stale.
* SEV3: elevated error rates with no customer-visible impact.

A SEV1 or SEV2 requires a written postmortem within five working days.
