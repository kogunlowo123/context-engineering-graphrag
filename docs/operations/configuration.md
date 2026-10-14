# Configuration reference

All service settings are `GRAPHRAG_*` environment variables (see
`src/context_engineering_graphrag/config.py`). The service refuses to start
with an incomplete or unsafe configuration.

## Service

| Variable | Required | Default | Description |
|---|---|---|---|
| `GRAPHRAG_ENVIRONMENT` | no | `production` | `production` requires an audit key and forbids the hashing embedder |
| `GRAPHRAG_PUBLIC_URL` | yes | | External base URL; used in the A2A Agent Card and as an allowed MCP `Host` |
| `GRAPHRAG_ADDITIONAL_HOSTS` | no | `[]` | JSON list of extra `Host` values accepted on `/mcp` (e.g. `["graphrag.context-engineering.svc:80"]`) |
| `GRAPHRAG_DATABASE_URL` | yes | | SQLAlchemy async URL, `postgresql+asyncpg://...` in production |
| `GRAPHRAG_VECTOR_BACKEND` | no | `memory` | `memory` (exact search in process) or `pgvector` (HNSW in PostgreSQL) |
| `GRAPHRAG_AUDIT_LOG_PATH` | no | `/var/lib/graphrag/audit/audit.jsonl` | Signed audit chain file |
| `GRAPHRAG_AUDIT_KEY_FILE` | production | | Ed25519 PEM that signs the audit chain; the file stem is the key id |
| `GRAPHRAG_POLICY_FILE` | no | `policies/context-governance.yaml` | Access (local mode) and answer-release policy |
| `GRAPHRAG_LOG_LEVEL` | no | `INFO` | |
| `GRAPHRAG_OTLP_ENDPOINT` | no | | OTLP gRPC endpoint for traces |
| `GRAPHRAG_HOST` / `GRAPHRAG_PORT` | no | `127.0.0.1` / `8080` | The container image sets host `0.0.0.0` |
| `GRAPHRAG_FORWARDED_ALLOW_IPS` | no | `127.0.0.1` | Proxies trusted for `X-Forwarded-*` |

## Identity

Set the fabric URL (production) or all three local issuer settings.

| Variable | Required | Description |
|---|---|---|
| `GRAPHRAG_IDENTITY_FABRIC_URL` | one of | agent-identity-fabric base URL; every request is authorized by its `/v1/authorize` |
| `GRAPHRAG_LOCAL_ISSUER` | one of | Trusted JWT issuer when no fabric is configured |
| `GRAPHRAG_LOCAL_AUDIENCE` | one of | Audience bearer tokens must carry |
| `GRAPHRAG_LOCAL_JWKS_URL` | one of | JWKS of the local issuer (EdDSA, ES256 or RS256 keys) |

## Retrieval and graph

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_EMBEDDING_PROVIDER` | `bedrock` | `bedrock`, `openai_compatible`, or `hashing` (non-production) |
| `GRAPHRAG_EMBEDDING_MODEL` | `amazon.titan-embed-text-v2:0` | Model or deployment name |
| `GRAPHRAG_EMBEDDING_DIMENSION` | `1024` | Titan V2 supports 256, 512, 1024 |
| `GRAPHRAG_AWS_REGION` | | Required for Bedrock embeddings |
| `GRAPHRAG_EMBEDDING_BASE_URL` | | Required for `openai_compatible` |
| `GRAPHRAG_EMBEDDING_PATH` | `/v1/embeddings` | Request path for `openai_compatible` |
| `GRAPHRAG_EMBEDDING_AUTH` | `bearer` | `bearer`, `api-key` or `none` |
| `GRAPHRAG_EMBEDDING_API_KEY_ENV` | | Name of the variable holding the embedding key |
| `GRAPHRAG_RERANKER` | `lexical` | `lexical` or `http` (hosted cross-encoder, `/rerank`) |
| `GRAPHRAG_RERANKER_URL` | | Required for `http` |
| `GRAPHRAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reported in search results |
| `GRAPHRAG_RERANKER_API_KEY_ENV` | | Name of the variable holding the reranker key |
| `GRAPHRAG_CHUNK_STRATEGY` | `recursive` | `fixed`, `sentence` or `recursive` |
| `GRAPHRAG_CHUNK_MAX_TOKENS` | `256` | 32–2048 |
| `GRAPHRAG_CHUNK_OVERLAP_TOKENS` | `32` | At most half of the maximum |
| `GRAPHRAG_RESOLUTION_THRESHOLD` | `0.88` | Entity-resolution match threshold |
| `GRAPHRAG_GAZETTEER_FILE` | | YAML mapping entity name to type for the pattern extractor |

## Model-backed features

`extractive`, `pattern` and `lexical` need no model. Any `model` setting
requires `GRAPHRAG_PROVIDERS_FILE` (normally `configs/providers.yaml`).

| Variable | Default | Values |
|---|---|---|
| `GRAPHRAG_GENERATOR` | `extractive` | `extractive`, `model` |
| `GRAPHRAG_EXTRACTOR` | `pattern` | `pattern`, `model` |
| `GRAPHRAG_SUMMARIZER` | `extractive` | `extractive`, `model` |
| `GRAPHRAG_JUDGE` | `lexical` | `lexical`, `model` |
| `GRAPHRAG_PROVIDERS_FILE` | | Path to the provider configuration |

### Variables referenced by `configs/providers.yaml`

Loading fails closed if any referenced variable is unset or empty.

| Variable | Provider | Meaning |
|---|---|---|
| `AWS_REGION` | Bedrock | Region of the Bedrock runtime endpoint |
| `BEDROCK_CLAUDE_MODEL_ID` | Bedrock | Claude model or inference-profile id enabled in the account |
| `BEDROCK_CLAUDE_INPUT_USD_PER_MTOK` / `BEDROCK_CLAUDE_OUTPUT_USD_PER_MTOK` | Bedrock | Contracted prices per million tokens |
| `NOVA_INPUT_USD_PER_MTOK` / `NOVA_OUTPUT_USD_PER_MTOK` | Bedrock | Nova Pro prices |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Resource endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI | Deployment name |
| `AZURE_OPENAI_REGION` | Azure OpenAI | Region (residency filter) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | API key (from the secret store) |
| `AZURE_GPT_INPUT_USD_PER_MTOK` / `AZURE_GPT_OUTPUT_USD_PER_MTOK` | Azure OpenAI | Prices |
| `GCP_PROJECT_ID` | Vertex AI | Project |
| `GCP_REGION` | Vertex AI | Region |
| `GCP_ACCESS_TOKEN` | Vertex AI | OAuth access token (workload identity token exchange) |
| `GEMINI_INPUT_USD_PER_MTOK` / `GEMINI_OUTPUT_USD_PER_MTOK` | Vertex AI | Prices |
