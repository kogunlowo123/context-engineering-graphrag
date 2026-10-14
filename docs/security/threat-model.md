# Threat model

## Assets

| Asset | Impact if compromised |
|---|---|
| Collections, documents and chunks (up to RESTRICTED) | Disclosure of confidential or regulated content through search, context packs or answers |
| Knowledge graph (entities, temporal assertions, community summaries) | Disclosure of relationships; poisoned facts steering agent decisions |
| Answer-release decisions | Ungrounded or fabricated answers reaching users and downstream agents |
| Governance policy (`policies/context-governance.yaml`) | Excess access or silently disabled release gate |
| Audit chains | Loss of accountability for ingestion, erasure and answers |
| Audit signing key, database URL, provider credentials | Forged audit, direct data access, model spend |

## Trust boundaries

1. Agents and users -> REST API, `/mcp`, `/a2a` (network, untrusted callers with bearer credentials).
2. Platform -> agent-identity-fabric `/v1/authorize` (decision authority for access).
3. Ingested document content -> chunk store and model context (content is EXTERNAL/USER/VERIFIED trust, never instructions).
4. Platform -> embedding, reranker and foundation-model providers (Bedrock, Azure OpenAI, Vertex AI, self-hosted).
5. Platform -> PostgreSQL (pgvector), secret store and audit storage.

## STRIDE

| ID | Threat | Category | Control | Test |
|---|---|---|---|---|
| T1 | Anonymous calls to REST, MCP or A2A | Spoofing | Bearer credential required on every `/v1` route; `RequireBearer` wrapper on `/mcp`; `/a2a` authenticates before parsing; oversized credentials rejected | `tests/security/test_threats.py::test_t1_unauthenticated_requests_are_rejected_everywhere` |
| T2 | Forged signature, `alg: none`, HS256 key confusion | Spoofing | `JwtVerifier` accepts only EdDSA/ES256/RS256 with a `kid` resolved from the issuer JWKS | `test_t2_forged_and_confused_tokens` |
| T3 | Replay of expired tokens or tokens minted for another service | Spoofing | `exp`/`iat`/`nbf` checked against the injected clock (30 s leeway); issuer and audience verified | `test_t3_expired_and_wrong_audience_tokens` |
| T4 | Reader credential used to ingest, erase, build the graph or create collections | Elevation of privilege | Each use case authorizes its own action (`rag:document.write`, `rag:document.delete`, `rag:graph.build`, `rag:collection.create`); `allow-scoped-access` requires `action_in_scope` | `test_t4_scope_escalation_is_denied` |
| T5 | Caller reads data above its clearance or without a declared purpose | Information disclosure | Classification ceiling enforced outside policy (`max_cls`); `deny-restricted-without-purpose` | `test_t5_classification_ceiling_and_purpose` |
| T6 | Indirect prompt injection planted in documents (override phrases, role delimiters, invisible characters) | Tampering | `PromptInjectionDetector` per chunk at ingestion with trust-dependent thresholds; matches are quarantined, never indexed or placed in context; all context spotlighted with `isolate_untrusted` | `test_t6_indirect_injection_never_reaches_context` |
| T7 | Direct injection or credentials in the question | Tampering / disclosure | `AgenticRetrievalWorkflow.guard_query` rejects injection (USER threshold) and secret material with `content_rejected` | `test_t7_direct_injection_and_credentials_in_queries` |
| T8 | Secrets inside documents leak through storage, embeddings or answers | Information disclosure | `SecretScanner` findings redacted before chunking, storage and embedding; output guard on model answers | `test_t8_secrets_are_redacted_before_storage_and_embedding` |
| T9 | Ungrounded, contradicted or fabricated-citation answers released | Tampering (integrity of output) | Groundedness evaluation per claim; release policy rules `deny-ungrounded-answer`, `deny-contradicted-answer`, `deny-fabricated-citations`, `approve-restricted-weak-grounding` | `test_t9_ungrounded_or_fabricated_answers_are_withheld`, `tests/unit/test_workflow.py` |
| T10 | Cross-collection data access through filters | Information disclosure | Indexes and queries are scoped to one collection; document filters apply inside it | `test_t10_collections_are_isolated` |
| T11 | Audit events altered, removed or reordered | Repudiation | agentic-core hash-chained, Ed25519-signed `AuditLog`; `verify_chain`; WORM storage | `test_t11_audit_tampering_is_detected` |
| T12 | DNS rebinding against `/mcp` | Spoofing | MCP transport security with allowed hosts from `GRAPHRAG_PUBLIC_URL` and `GRAPHRAG_ADDITIONAL_HOSTS` | `test_t12_mcp_rejects_foreign_host_headers` |
| T13 | Long delegation chains and delegated destructive actions | Elevation of privilege | `deny-deep-delegation-on-restricted`; `approve-delegated-erasure` returns `approval_required` | `test_t13_delegation_limits` |
| T14 | Oversized or malformed input, path traversal in document ids, mass assignment | Tampering / DoS | Pydantic bounds and `extra="forbid"`; document id pattern; query length 4000; paging limit 500 | `test_t14_input_bounds` |
| T15 | Collection existence oracle | Information disclosure | Unknown collections are authorized as RESTRICTED before a 404 is returned | `test_t15_no_existence_oracle` |

### Fail-closed behaviour (chaos suite)

| Failure | Behaviour | Test |
|---|---|---|
| Database unavailable | `/readyz` 503, liveness unaffected | `tests/chaos/test_dependency_failures.py::test_database_outage_flips_readiness` |
| Identity fabric unreachable | 503 `temporarily_unavailable`, never allow | `test_identity_fabric_outage_fails_closed` |
| Embedding provider down | Ingestion fails, nothing written, revision unchanged | `test_embedding_outage_writes_nothing` |
| Reranker down | 503; caller may explicitly disable reranking | `test_reranker_outage_returns_503` |
| Model provider down | 503 `provider_unavailable` | `test_model_provider_outage_returns_503` |
| Audit sink failure | Authorization audit fails first, so no state change is written | `test_audit_outage_blocks_writes` |
| Writes on another replica | Revision check rebuilds caches; erased content disappears on all replicas | `test_replicas_see_each_others_writes` |

## MITRE ATT&CK (Enterprise)

| Technique | Mitigation |
|---|---|
| T1078 Valid Accounts | Short-lived fabric credentials, per-action authorization, classification ceiling |
| T1528 Steal Application Access Token | Audience binding, expiry, no tokens in logs or audit |
| T1530 Data from Cloud Storage | Collection-scoped access, KMS encryption, private networking |
| T1552 Unsecured Credentials | Secret redaction at ingestion; provider keys only by environment variable name |
| T1565.001 Stored Data Manipulation | Document digests, versioned assertions, signed audit of ingestion |
| T1562.008 Impair Defenses: Disable Cloud Logs | WORM audit storage; audit failure blocks writes |

## MITRE ATLAS

| Technique | Relevance | Control |
|---|---|---|
| AML.T0051 LLM Prompt Injection (direct and indirect) | Questions and retrieved documents | Query screening, chunk quarantine, spotlighting, release gate |
| AML.T0054 LLM Jailbreak | Role and mode-switch phrasing in questions | Injection detector signals (`role_reassignment`, `mode_switch`) |
| AML.T0057 LLM Data Leakage | Answers quoting restricted or secret content | Classification ceiling, secret redaction, output guard, citations to authorized chunks only |
| AML.T0070 RAG Poisoning | Planted documents shaping retrieval and graph facts | Quarantine, trust levels, provenance digests, erasure cascade, quarantine-spike alert |

## OWASP Top 10 for LLM Applications (2025)

| Risk | Relevance | Control |
|---|---|---|
| LLM01 Prompt Injection | Primary (indirect via corpus) | Quarantine, query screening, spotlighting, extractive default generator |
| LLM02 Sensitive Information Disclosure | Primary | Classification ceiling, purpose of use, secret redaction, PII redaction on model output |
| LLM03 Supply Chain | Models and embedders | Providers declared in `configs/providers.yaml` with classification ceilings; pinned images, SBOM, Grype |
| LLM04 Data and Model Poisoning | Corpus and graph | Quarantine, document versioning and digests, temporal assertions, erasure |
| LLM05 Improper Output Handling | Answers consumed by agents | Structured outputs, citations validated against the context pack, invalid labels stripped and counted |
| LLM06 Excessive Agency | Platform tools | Read-only MCP tools except ingestion; erasure by delegated agents needs approval |
| LLM07 System Prompt Leakage | Model generator | Exfiltration phrasing rejected at query; no secrets in prompts |
| LLM08 Vector and Embedding Weaknesses | Primary | Per-collection indexes, classification filters before ranking, embedding model recorded per collection, quarantined chunks never embedded by providers |
| LLM09 Misinformation | Primary | Groundedness evaluation (numbers, negation, named entities), release policy, abstention |
| LLM10 Unbounded Consumption | Model and retrieval cost | Token budgets, `max_answer_tokens`, request bounds, HPA, latency alerts |

## OWASP API Security Top 10 (2023)

| Risk | Control |
|---|---|
| API1 Broken Object Level Authorization | Every collection-scoped call authorized against that collection and its classification |
| API2 Broken Authentication | Signature, algorithm allow-list, issuer, audience and time checks; fabric introspection of credential state |
| API3 Broken Object Property Level Authorization | `extra="forbid"` on inputs; responses carry no credentials or internal secrets |
| API4 Unrestricted Resource Consumption | Field bounds, paging limit, token budgets, HPA |
| API5 Broken Function Level Authorization | Distinct actions per use case (`rag:search`, `rag:query`, `rag:document.delete`, ...) |
| API7 Server Side Request Forgery | No caller-supplied URLs are fetched; `source_uri` is stored as metadata only |
| API8 Security Misconfiguration | Settings fail closed; Helm `required`; restricted Pod Security; `X-Content-Type-Options: nosniff` |
| API9 Improper Inventory Management | OpenAPI and JSON Schemas generated and drift-checked in CI |
| API10 Unsafe Consumption of APIs | Provider responses validated (dimension, finiteness, index coverage) and mapped to 503 |

## Risk register

| ID | Risk | Likelihood | Impact | Owner | Treatment | Residual |
|---|---|---|---|---|---|---|
| R1 | Novel injection phrasing evades the heuristic detector | Medium | High | AI Security | Spotlighting, extractive default generator, release gate, quarantine alert; add a classifier via `extra_signals` | Medium |
| R2 | Lexical groundedness judge accepts a paraphrased false claim | Medium | Medium | AI Governance | Numeric, negation and named-entity checks; model judge option; restricted answers below 0.8 need review | Medium |
| R3 | Policy change disables the release gate | Low | High | AI Governance | CODEOWNERS on `policies/`, policy digest in every decision, policy tests in CI | Low |
| R4 | Erased content survives in community summaries | Low | High | Data Protection | Erasure deletes all communities of the collection; rebuild runbook | Low |
| R5 | Provider outage blocks ingestion or answers | Medium | Medium | Platform | Fail closed with 503; extractive and lexical modes need no provider | Low |
| R6 | Entity resolution merges distinct entities | Medium | Medium | Knowledge Engineering | Type compatibility, 0.88 threshold, aliases retained, evaluation on golden sets | Medium |
