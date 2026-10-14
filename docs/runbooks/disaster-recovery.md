# Runbook: disaster recovery

| Objective | Target | Mechanism |
|---|---|---|
| RPO | <= 5 minutes | Managed PostgreSQL point-in-time recovery (backup retention 14 days) |
| RTO | <= 1 hour | Terraform + Helm redeploy in the recovery region |

## State that matters

- **PostgreSQL**: collections, documents, chunks with embeddings and terms,
  entities, assertions, communities, and `chunk_vectors` (pgvector). This is
  the only state needed to serve traffic.
- **Retrieval indexes and graph views**: caches rebuilt automatically from
  PostgreSQL when a replica first serves a collection; nothing to restore.
- **Audit signing key and provider settings**: in Secrets Manager.
- **Audit chains**: WORM S3 bucket (Object Lock, compliance mode). Not needed to
  serve traffic.

## Procedure

1. Restore PostgreSQL to the latest point before the incident (PITR); note
   the new endpoint.
2. Update the database URL secret if the endpoint changed.
3. `terraform apply` in the recovery region if the cluster region is lost.
4. `helm upgrade --install` with the same values.
5. Verify `/readyz`, list collections, and run the retrieval golden set for
   one collection (`POST /v1/collections/{c}/evaluations/retrieval`).
6. Documents ingested after the restore point are missing: re-ingest them
   from their sources (ingestion is idempotent by content digest).
7. Erasures performed after the restore point must be replayed: list
   `document.erase` audit events since the restore time from the WORM bucket
   and repeat each `DELETE`.
8. Record the cutover; new audit chains start at sequence 0 on new volumes.

Recovery is exercised quarterly by restoring a PITR copy into a staging
namespace and running `tests/e2e` and the integration suite against it.
