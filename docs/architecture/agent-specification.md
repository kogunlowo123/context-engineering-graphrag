# Agent specification

The platform is an agent-facing service: it plans retrieval, decides what
enters an agent's context and decides whether an answer may be released. This
document states its mandate in the same terms the identity fabric requires of
every registered agent, and defines the contract callers rely on.

## Role and mission

- **Purpose:** give enterprise agents the right context, of the right
  version, within a token budget, with every statement traceable to a source.
- **Business objective:** let agents answer and act on enterprise knowledge
  without hallucinated facts, stale facts or injected instructions reaching
  the decision.
- **Protected resources:** collections and their documents, chunks,
  embeddings, knowledge graph and community summaries; the audit chain.
- **Operating boundaries:** reads only collections the caller is authorized
  for; never executes actions on other systems; never follows instructions
  found in retrieved content; never releases an answer the release policy
  denies; stores queries and answers only as digests.
- **Delegated responsibility:** access decisions are delegated to
  agent-identity-fabric (or the local policy when no fabric is configured);
  answer release is decided by the versioned `policies/context-governance.yaml`
  owned by AI governance.

## Delegated scope

| Allowed | Disallowed | Risk | Escalation |
|---|---|---|---|
| Retrieve, budget and cite content from authorized collections | Access above the caller's classification ceiling | MODERATE | Denied and audited (`authorize:*`, outcome `deny`) |
| Answer when groundedness >= 0.5, no contradictions, no fabricated citations | Release ungrounded, contradicted or falsely cited answers | HIGH | Disposition `withheld`; `GraphRAGWithheldAnswerSpike` alert |
| Answer from restricted sources with groundedness >= 0.8 | Release weakly grounded restricted answers | HIGH | Disposition `pending_approval`; human reviewer |
| Ingest and version documents | Store credentials or retrievable injected instructions | HIGH | Redaction and quarantine; `GraphRAGQuarantineSpike` pages security |
| Erase documents on request of an accountable principal | Erasure by a delegated agent without sign-off | HIGH | `approval_required` (rule `approve-delegated-erasure`) |
| Serve restricted collections with a declared purpose of use | Restricted access without purpose or through more than one delegation hop | HIGH | Denied (`deny-restricted-without-purpose`, `deny-deep-delegation-on-restricted`) |
| Serve callers of any risk tier up to HIGH | Unreviewed access by CRITICAL-risk agents | CRITICAL | `approval_required` (rule `approve-critical-risk-callers`) |

## Objectives

| Type | Definition | Target |
|---|---|---|
| SLO availability | Non-5xx share of requests (`graphrag:request_errors:ratio_5m`) | 99.9% over 30 days |
| SLO latency | p95 `/v1/collections/{collection}/search` | < 500 ms |
| SLO latency | p95 `/v1/collections/{collection}/context` | < 1 s |
| SLO latency | p95 `/v1/collections/{collection}/query` | < 3 s |
| SLO quality | Median `graphrag_answer_groundedness` of evaluated answers | >= 0.8 |
| SLI | `graphrag_http_request_duration_seconds`, `graphrag_stage_duration_seconds`, `graphrag_answer_groundedness`, `graphrag_answers_total` | Prometheus |
| KPI | Share of answers released (`answered` + `abstained`) | Tracked per collection |
| KPI | Retrieval MRR and nDCG on each collection's golden set | Must not regress release over release |
| KPI | Quarantined chunks per ingested chunk | Trend reviewed by security |
| Completion | A request is complete when its decision is in the signed audit chain | Enforced: an audit failure fails the request before any write |

## Permissions

| Mechanism | Use |
|---|---|
| RBAC (scopes) | `rag:collection.create`, `rag:collection.read`, `rag:document.write`, `rag:document.read`, `rag:document.delete`, `rag:search`, `rag:context`, `rag:query`, `rag:graph.build`, `rag:graph.read`, `rag:evaluate`; wildcards such as `rag:*` are matched as globs |
| ABAC | Subject: status, scopes, `action_in_scope`, `max_classification`, risk, `delegated`, `delegation_depth`. Resource: type (`collection`, `evaluation`, `answer`), id, classification; for answers groundedness, unsupported and contradicted claims, citations, abstained, invalid citations. Environment: `purpose_of_use` |
| Classification ceiling | Enforced outside policy: a resource above the caller's `max_cls` is denied regardless of rules |
| Agent identity | SPIFFE IDs (`spiffe://<trust-domain>/agent/<namespace>/<name>`) issued by agent-identity-fabric; delegation chains from RFC 8693 `act` claims |
| OAuth 2.0 / OIDC | Bearer JWT credentials; fabric `/v1/authorize` in production; local JWKS verification (EdDSA, ES256, RS256; issuer, audience, expiry with injected clock) without a fabric |
| Workload identity | IRSA (AWS), Azure Workload Identity, GKE Workload Identity for the platform's own access to Bedrock, Azure OpenAI, Vertex AI and audit storage |

## Inputs and outputs

All bodies are Pydantic models with JSON Schemas in `schemas/`; inputs reject
unknown fields.

| Schema | Use |
|---|---|
| `create-collection.request` | Collection name, description, classification, owner |
| `ingest-document.request` / `.response` | Document version with text, source URI, classification, trust, metadata, validity interval, chunking; result with version, status, chunk, quarantine, redaction and assertion counts |
| `search.request` / `.response` | Query, k, candidate depth, rerank, MMR lambda, filters; hits with chunk and score breakdown (BM25, dense, fused, rerank) |
| `context.request` / `.response` | Query, mode, budget policy, filters, `as_of`, purpose; plan, context pack, rendered context, trace |
| `query.request` / `.response` | Context request plus answer token limit; disposition, answer, citations, groundedness report, release decision, plan, trace |
| `graph-build.request` / `.response` | Louvain resolution and levels; merged entities, levels with modularity, communities |
| `neighbourhood.response` | Centre entity, entities and relations with validity intervals |
| `groundedness.request` / `.response` | Answer and labelled sources; per-claim verdicts, groundedness, citation precision and coverage |
| `retrieval-evaluation.request` / `.response` | Golden queries with relevant documents; recall, precision, MRR, nDCG |

### Validation

- Collection names: 2–63 characters of `a-z`, `0-9`, `-`. Document ids: up to
  200 characters of `A-Z a-z 0-9 . _ : / -`, no `..`.
- Metadata: at most 32 keys matching `^[a-z][a-z0-9_]{0,63}$`, string values
  up to 512 characters.
- Document text up to 2,000,000 characters; queries up to 4,000.
- Timestamps must carry a timezone; `expires_at` must follow `effective_at`;
  a new version may not take effect before the current one.
- Callers cannot assert `trust: SYSTEM`; document classification may not
  exceed the collection's.
- Chunking: 32–2048 tokens, overlap at most half the chunk size. Budget:
  200–200,000 tokens.
- Bearer credentials longer than 8,192 characters are rejected.

## Provenance

- Every document version carries the SHA-256 digest of the submitted text,
  `source_uri`, `source_type`, trust level, classification and who ingested it.
- Every chunk carries exact character offsets into the stored text, its own
  digest, the document version and its quarantine status.
- Every assertion carries the evidence sentence, the chunk and document
  version it came from and its validity interval.
- Every context source and citation carries document id and version, chunk
  id, digest, source URI and (for graph facts) evidence chunk ids.
- Every release decision carries the matched rules, policy version and policy
  digest.

## Audit metadata

Events are appended to the hash-chained, Ed25519-signed agentic-core audit
log with actor, action, resource, outcome, correlation ID and details. Every
event's details include `channel` (`http`, `mcp`, `a2a`).

| Action | Emitted by | Details |
|---|---|---|
| `authorize:<action>` | every use case | matched rules, policy version and digest, delegation chain, decision source |
| `collection.create` | create collection | classification, owner |
| `document.ingest` | ingest | version, digest, chunks, quarantined count and reasons, redacted secrets, assertions, classification |
| `document.erase` | erasure | chunks, mentions, assertions, entities, communities removed |
| `retrieval.search` | search | `query_digest`, hit chunk ids, revision |
| `context.build` | context | `query_digest`, mode, source keys, used tokens |
| `answer.release` | query | `query_digest`, `answer_digest`, mode, groundedness, cited chunks, release rules, policy digest, generator |
| `graph.build` | graph build | merged entities, communities, levels, summarizer |
| `evaluation.groundedness`, `evaluation.retrieval` | evaluations | groundedness and claim count; query count, MRR, mean nDCG |
| `mcp.tool:<name>` | MCP server | outcome (`ok` or platform error code); actor is the authorized principal or `unauthenticated` |
