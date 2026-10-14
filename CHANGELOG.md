# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-14

### Added

- Ingestion with fixed, sentence and recursive chunking (token overlap, exact
  offsets), SHA-256 provenance digests, document versioning, prompt-injection
  quarantine and secret redaction before storage or embedding.
- Hybrid retrieval: Okapi BM25 and dense vectors (in-process exact search or
  pgvector HNSW) fused with reciprocal rank fusion, cross-encoder reranking
  (lexical or hosted `/rerank`) and MMR diversification; revision-aware index
  caches shared consistently across replicas.
- Knowledge graph: pattern or model-grounded entity and relation extraction,
  entity resolution (blocking, Jaro-Winkler, acronyms, union-find), temporal
  assertions with validity intervals and `as_of` queries.
- GraphRAG: hierarchical Louvain communities, extractive or model community
  summaries, local search (personalised PageRank) and global search.
- Context engineering: extractive compression, de-duplication and section
  token budgeting with full provenance and spotlighted rendering.
- Agentic workflow: planner (passages, local, global), iterative retrieval
  with query expansion, attributed answers, groundedness evaluation and a
  policy-gated release decision.
- Retrieval evaluation (recall, precision, MRR, nDCG) and groundedness API.
- Identity-fabric authorization (local JWT + policy fallback), MCP server
  (6 tools), A2A Agent Card and `message/send`, signed audit chain,
  Prometheus metrics and OpenTelemetry traces.
- Helm chart, Kubernetes namespace and secret sync, Terraform for AWS, Azure
  and GCP; CI with PostgreSQL/pgvector integration, Trivy, Syft, Grype,
  Semgrep and a container smoke test.
