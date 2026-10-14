# Context Engineering GraphRAG

Agentic RAG and GraphRAG context engineering for enterprise agents. Documents
are chunked with exact provenance, screened for prompt injection and
credentials, embedded, and turned into a temporal knowledge graph. Agents get
hybrid retrieval, graph-aware context packs that fit a token budget, and
answers in which every sentence is cited, checked for groundedness and
released only when the governance policy allows it. Every decision and state
change is recorded in a signed, tamper-evident audit chain.

```
Query -> Retrieval -> Reranking -> Context Budgeting -> Grounded Action
```

## Capabilities

| Capability | Implementation |
|---|---|
| Ingestion | Fixed, sentence and recursive chunking with token overlap and exact character offsets; SHA-256 digests per document and chunk; document versions with `effective_at` / `expires_at` |
| Content controls | Prompt-injection screening per chunk with trust-dependent thresholds (quarantined chunks are never retrievable); credentials redacted before storage or embedding |
| Hybrid retrieval | Okapi BM25 + dense vectors fused with reciprocal rank fusion; cross-encoder reranking (lexical or hosted `/rerank`); MMR diversification; metadata, document, classification and `as_of` filters |
| Vector backends | In-process exact cosine search rebuilt from PostgreSQL, or pgvector HNSW (`GRAPHRAG_VECTOR_BACKEND=pgvector`) |
| Knowledge graph | Pattern or model-grounded entity and relation extraction; entity resolution (blocking, Jaro-Winkler, acronyms, union-find); assertions with `valid_from` / `valid_to` |
| GraphRAG | Hierarchical Louvain communities with extractive or model summaries; local search (personalised PageRank) and global search over community summaries |
| Context engineering | Query-focused compression, near-duplicate removal, section budgets for passages, graph facts and community summaries; every source keeps provenance; spotlighted rendering |
| Agentic workflow | Planner (passages, local, global), iterative retrieval with query expansion, attributed generation, groundedness evaluation, policy-gated release (answered, abstained, withheld, pending approval) |
| Evaluation | Groundedness per claim (supported / partial / contradicted / unsupported), citation precision and coverage; retrieval recall, precision, MRR and nDCG against golden sets |
| Identity and authorization | Every call authorized by [agent-identity-fabric](https://github.com/kogunlowo123/agent-identity-fabric) `/v1/authorize`, or by local JWT verification plus policy; never allow-all |
| Agent protocols | MCP server (official SDK, streamable HTTP) with six typed tools; A2A Agent Card and JSON-RPC `message/send` |
| Audit and observability | Hash-chained Ed25519-signed audit events; Prometheus metrics, OpenTelemetry traces, JSON logs with correlation IDs, SLO alerts, Grafana dashboard |

Shared controls (policy engine, audit chain, content guard, model router,
logging, telemetry) come from
[agentic-core](https://github.com/kogunlowo123/agentic-core).

## Quick start (local)

Requires Docker. No API keys or cloud credentials are needed: the stack uses
the deterministic hashing embedder, PostgreSQL with pgvector, a mock OIDC
issuer and an OpenTelemetry collector.

```bash
docker compose up --build

TOKEN=$(curl -s -X POST localhost:8081/agents/token \
  -d grant_type=client_credentials -d client_id=analyst -d client_secret=local \
  -d scope="rag:*" | jq -r .access_token)

curl -s -X POST localhost:8080/v1/collections -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name":"supplier-risk","classification":2,"owner":"risk-team"}'
curl -s localhost:8080/.well-known/agent-card.json | jq .skills
```

## Development

```bash
uv sync
make check        # ruff, black, mypy --strict, bandit, contracts, tests (coverage >= 90%)
make integration  # PostgreSQL + pgvector tests (needs GRAPHRAG_TEST_DATABASE_URL)
```

Test suites: `tests/unit`, `tests/security` (attacks mapped to the threat
model, `test_t<n>_...`), `tests/chaos` (dependency failures and replica
consistency), `tests/performance` (latency budgets; `locustfile.py` for load),
`tests/e2e` (HTTP, MCP and A2A lifecycle plus a property-based OpenAPI contract
test), `tests/integration` (PostgreSQL with both vector backends).

## API

All `/v1` routes require `Authorization: Bearer <agent credential>`; the
optional `X-Purpose-Of-Use` header declares a purpose of use (required for
restricted collections).

| Endpoint | Scope | Purpose |
|---|---|---|
| `POST /v1/collections` | `rag:collection.create` | Create a collection with a classification and owner |
| `GET /v1/collections`, `GET /v1/collections/{collection}` | `rag:collection.read` | Inventory |
| `POST /v1/collections/{collection}/documents` | `rag:document.write` | Ingest a document version |
| `GET /v1/collections/{collection}/documents`, `GET .../documents/{document_id}` | `rag:document.read` | Documents and their chunks with offsets and digests |
| `DELETE /v1/collections/{collection}/documents/{document_id}` | `rag:document.delete` | Right-to-erasure cascade |
| `POST /v1/collections/{collection}/search` | `rag:search` | Hybrid retrieval with score breakdown |
| `POST /v1/collections/{collection}/context` | `rag:context` | Budgeted, attributed context pack for the caller's own model |
| `POST /v1/collections/{collection}/query` | `rag:query` | Agentic RAG answer with citations, groundedness and release decision |
| `POST /v1/collections/{collection}/graph/build` | `rag:graph.build` | Entity resolution, communities and summaries |
| `GET /v1/collections/{collection}/entities` | `rag:graph.read` | Entity lookup by name or alias |
| `GET /v1/collections/{collection}/entities/{entity_id}/neighbourhood` | `rag:graph.read` | k-hop neighbourhood, optionally `as_of` a past instant |
| `GET /v1/collections/{collection}/communities` | `rag:graph.read` | Community summaries by level |
| `POST /v1/evaluations/groundedness` | `rag:evaluate` | Judge an attributed answer against its sources |
| `POST /v1/collections/{collection}/evaluations/retrieval` | `rag:evaluate` | Recall, precision, MRR, nDCG on a golden set |
| `POST /mcp` | per tool | MCP streamable HTTP endpoint |
| `GET /.well-known/agent-card.json` | none | A2A Agent Card |
| `POST /a2a` | `rag:query` / `rag:context` | A2A JSON-RPC `message/send` |
| `GET /healthz`, `/readyz`, `/metrics` | none | Operations |

Full contract: [docs/api/openapi.json](docs/api/openapi.json); JSON Schemas in [schemas/](schemas/).

### MCP tools

| Tool | Purpose |
|---|---|
| `search_passages` | Hybrid retrieval over one collection |
| `build_context` | Planned, budgeted, attributed context pack |
| `answer_question` | Grounded answer with citations, groundedness report and release decision |
| `explore_entity` | Find an entity by name and return its graph neighbourhood |
| `evaluate_groundedness` | Per-claim groundedness of an attributed answer |
| `ingest_document` | Chunk, screen, embed and graph-extract a document version |

## Deployment

| Layer | Location |
|---|---|
| Container image | `Dockerfile` (non-root uid 10001, read-only root filesystem compatible) |
| Helm chart | `deployments/helm/context-engineering-graphrag` (StatefulSet with per-replica audit volume, PDB, HPA, NetworkPolicy, ServiceMonitor) |
| Namespace and secret sync | `deployments/kubernetes` (restricted Pod Security, default-deny, External Secrets) |
| Cloud infrastructure | `deployments/terraform/{aws,azure,gcp}`: managed PostgreSQL with pgvector, KMS, secret store, WORM audit storage, workload identity, model access |

AWS is the reference deployment (EKS, RDS for PostgreSQL, Bedrock, S3 Object
Lock). See the [startup runbook](docs/runbooks/startup.md).

## Documentation

- [Architecture](docs/architecture/overview.md) and [agent specification](docs/architecture/agent-specification.md)
- ADRs: [hybrid retrieval](docs/architecture/adr/0001-hybrid-retrieval-with-rank-fusion.md), [temporal assertions](docs/architecture/adr/0002-temporal-assertions-for-graph-facts.md), [revision-checked indexes](docs/architecture/adr/0003-revision-checked-in-process-indexes.md), [groundedness-gated release](docs/architecture/adr/0004-groundedness-gated-answer-release.md), [local implementations behind interfaces](docs/architecture/adr/0005-deterministic-local-implementations-behind-interfaces.md), [ingestion content controls](docs/architecture/adr/0006-ingestion-content-controls.md)
- [Threat model and risk register](docs/security/threat-model.md)
- [Compliance control mapping](docs/compliance/control-mapping.md)
- Runbooks: [startup](docs/runbooks/startup.md), [monitoring](docs/runbooks/monitoring.md), [incident response](docs/runbooks/incident-response.md), [rollback](docs/runbooks/rollback.md), [disaster recovery](docs/runbooks/disaster-recovery.md), [reindex and erasure](docs/runbooks/reindex-and-erasure.md)
- [Configuration reference](docs/operations/configuration.md)
- [Article: engineering context for enterprise agents](docs/linkedin-article.md)

## License

Apache-2.0
