# Runbook: incident response

## Elevated 5xx

Alert `GraphRAGErrorBudgetBurnFast`.

1. `kubectl -n context-engineering get pods`; check `/readyz` per pod.
2. Readiness failing: PostgreSQL. Check RDS status and failover events.
3. 503 `temporarily_unavailable`: the identity fabric, embedding provider or
   reranker is unreachable. The platform fails closed by design; restore the
   dependency (check NetworkPolicy egress and the provider's status).
4. 503 `provider_unavailable`: every eligible foundation model failed. Check
   Bedrock quotas and the circuit-breaker events in traces.
5. Pods crash-looping: configuration. `kubectl logs` shows the setting the
   service refused.
6. If a release caused it, follow `rollback.md`.

## Corpus poisoning

Alert `GraphRAGQuarantineSpike`: many chunks in one collection matched
prompt-injection signals.

1. Find the documents: filter audit events `action="document.ingest"` for the
   collection with non-empty `details.quarantine_reasons`; note `actor`.
2. Quarantined chunks are never indexed or placed in context, so no
   immediate exposure. Decide whether the source is compromised.
3. If the source is untrusted, erase the documents
   (`DELETE /v1/collections/{c}/documents/{id}`) and rebuild the graph
   (`POST /v1/collections/{c}/graph/build`).
4. If the uploading agent is compromised, revoke it in the fabric (below).

## Deny spike

Alert `GraphRAGDenySpike`. After a policy change this is usually the policy:
compare `policy_digest` in recent `authorize:*` audit events with the
previous release and revert the commit if unintended. Otherwise group denials
by `actor` to find a misconfigured or probing agent.

## Replicas down

Alert `GraphRAGNotReady`: fewer than two replicas scraped. Check node
pressure, PDB status and audit volume capacity; the service tolerates one
replica for capacity but loses zone redundancy.

## Compromised agent credential

1. Suspend the agent in agent-identity-fabric (`POST /v1/suspend`); every
   outstanding credential stops validating, so this platform's
   `/v1/authorize` calls deny immediately.
2. Pull this platform's audit events with `actor` equal to the agent's
   SPIFFE ID: what it searched (`retrieval.search` hits), ingested
   (`document.ingest`) and erased (`document.erase`).
3. Erase or re-ingest documents the agent wrote if their integrity is in doubt.

## Data leak through an answer

1. Identify the answer from the `answer.release` audit event
   (`citations` lists chunk ids, `answer_digest` the released text).
2. Erase the source document (`DELETE ...`), which also deletes all community
   summaries of the collection, then rebuild the graph.
3. If the classification was wrong, re-ingest into a collection with the right
   classification and review the callers' ceilings in the fabric.
