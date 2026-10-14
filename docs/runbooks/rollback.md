# Runbook: rollback

The schema is additive within a minor version (`schema_version` table, tables
created with `CREATE TABLE IF NOT EXISTS`). Version 0.1.0 stores BM25 terms
per chunk in `chunks.terms`; a later version that adds columns keeps them
nullable or defaulted so the previous image can still read the database.

```bash
helm -n context-engineering history rag
helm -n context-engineering rollback rag <revision> --wait
kubectl -n context-engineering rollout status statefulset/rag-context-engineering-graphrag
curl -sf "$GRAPHRAG/readyz"
```

In-memory indexes and graph views are caches keyed by collection revision;
rolled-back pods rebuild them from PostgreSQL on first use.

Policy-only regressions: revert the commit touching
`policies/context-governance.yaml`, release, and confirm `policy_digest` in
new `authorize:*` and `answer.release` audit events matches the reverted file.

Release-gate regressions (too many withheld answers after a generator or
judge change): set `graphrag.generator=extractive` and `graphrag.judge=lexical`
and `helm upgrade`; both need no model provider.

Every release keeps its image and chart in GHCR, so any previous version can
be reinstalled with `--version`.
