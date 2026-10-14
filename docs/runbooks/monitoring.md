# Runbook: monitoring

## Signals

| Signal | Source | Normal |
|---|---|---|
| `graphrag:request_errors:ratio_5m` | Recording rule | < 0.001 |
| `graphrag_http_request_duration_seconds{route}` | Histogram | p95 search < 500 ms, context < 1 s, query < 3 s |
| `graphrag_stage_duration_seconds{stage}` | Histogram | Stages: `graph.load`, `retrieve`, `graph.local`, `graph.global`, `budget`, `generate`, `evaluate`, `search` |
| `graphrag_answer_groundedness` | Histogram | Median >= 0.8 |
| `graphrag_answers_total{disposition,mode}` | Counter | `withheld` + `pending_approval` < 25% |
| `graphrag_context_tokens` | Histogram | Below the configured budget |
| `graphrag_ingested_chunks_total` / `graphrag_quarantined_chunks_total{collection}` | Counters | Quarantine near zero for curated sources |
| `graphrag_authorization_decisions_total{decision,source}` | Counter | Deny share stable per release |
| `graphrag_index_reload_seconds` | Histogram | Reloads follow ingestion, not every request |
| `graphrag_graph_builds_total{outcome}` | Counter | No `failed` |
| `graphrag_agent_calls_total{protocol,operation,outcome}` | Counter | MCP and A2A outcomes |
| Traces | OTLP (`GRAPHRAG_OTLP_ENDPOINT`) | One span per request; model calls carry `gen_ai.*` attributes |
| Logs | JSON on stdout | `correlation_id` on every line |

Alerts: `configs/prometheus/alerts.yaml`. Dashboard: `configs/grafana/context-engineering-graphrag.json`.

## Latency

1. Find the slow route on "p95 latency by route", then the slow stage on
   "p95 pipeline stage latency".
2. `retrieve` slow after ingestion bursts: index rebuilds. Check
   `graphrag_index_reload_seconds`; large collections benefit from
   `graphrag.vectorBackend=pgvector`. Batch ingestion outside peak hours.
3. `retrieve` slow with the HTTP reranker: check the reranker service; callers
   can send `"rerank": false` to degrade gracefully.
4. `generate`/`evaluate` slow: model-backed features. Check provider latency
   in traces (`gen_ai.client.operation.duration`), lower `max_answer_tokens`,
   or switch `GRAPHRAG_JUDGE` to `lexical`.
5. Everything slow: PostgreSQL CPU and connections, then pod CPU throttling;
   raise the HPA ceiling if CPU-bound.

## Groundedness

A falling median or a rise in withheld answers means answers are not
supported by the retrieved sources.

1. Break down `graphrag_answers_total` by `mode` to see whether local, global
   or passage retrieval is affected.
2. Run the collection's golden set through
   `POST /v1/collections/{c}/evaluations/retrieval`; falling MRR or recall
   points at retrieval (new documents, chunking change, embedding change).
3. Stable retrieval but low groundedness with `GRAPHRAG_GENERATOR=model`:
   a provider or prompt regression. Compare the `generator` and
   `policy_digest` fields of recent `answer.release` audit events with the
   previous release.
4. Stale community summaries after large ingestions: rebuild with
   `POST /v1/collections/{c}/graph/build`.

## Tracing a request

Every response carries `X-Correlation-ID` (callers may supply it). Search
logs and the audit chain for it:
`jq 'select(.correlation_id=="<id>")' /var/lib/graphrag/audit/<pod>.jsonl`.
