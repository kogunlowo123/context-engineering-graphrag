# Runbook: first deployment and startup (AWS reference)

Azure and GCP follow the same steps with `deployments/terraform/azure` or
`gcp` and the matching secret-manager CLI.

## Prerequisites

- EKS 1.27 or later with an IAM OIDC provider, External Secrets Operator and a
  `ClusterSecretStore` named `platform-secrets` bound to AWS Secrets Manager.
- agent-identity-fabric deployed and reachable from the cluster.
- Bedrock model access enabled in the account for
  `amazon.titan-embed-text-v2:0` (embeddings) and, if model-backed features are
  used, the generation models listed in `configs/providers.yaml`.

## 1. Provision cloud resources

```bash
cd deployments/terraform/aws
terraform init                      # configure an encrypted remote backend first
terraform apply -var-file=<environment>.tfvars
terraform output
```

Outputs: `service_account_role_arn` (IRSA annotation), `database_url_secret_name`,
`kms_key_arn`, `audit_bucket`.

## 2. Create the audit signing key

The key never passes through Terraform state.

```bash
openssl genpkey -algorithm ed25519 \
  | aws secretsmanager create-secret --name context-engineering-graphrag-audit-key-2026-10 \
      --kms-key-id "$(terraform output -raw kms_key_arn)" --secret-string file:///dev/stdin
```

External Secrets maps it to `audit-2026-10.pem`; set `secrets.auditKeyFile`
to that name. Provider prices, endpoints and keys referenced by
`configs/providers.yaml` are stored as `context-engineering-graphrag-env-<VARIABLE>`.

## 3. Database extension

The service runs `CREATE EXTENSION IF NOT EXISTS vector` at start-up when
`graphrag.vectorBackend=pgvector`. On RDS this requires a role holding
`rds_superuser`. Either start the first release with the master user, or
grant the application role once:

```sql
GRANT rds_superuser TO graphrag;   -- or run CREATE EXTENSION vector; as the master user
```

## 4. Namespace and secret sync

```bash
kubectl apply -k deployments/kubernetes
kubectl -n context-engineering get externalsecret   # all READY=True
```

## 5. Register callers in the identity fabric

Agents that call this platform need scopes for the actions they perform:
`rag:search`, `rag:context`, `rag:query`, `rag:document.read`,
`rag:document.write`, `rag:document.delete`, `rag:graph.build`,
`rag:graph.read`, `rag:collection.create`, `rag:collection.read`,
`rag:evaluate`. Register them through the fabric's `POST /v1/agents` with
the classification ceiling each agent needs.

## 6. Install the chart

Create `<environment>.yaml` with every value marked REQUIRED in `values.yaml`
(`graphrag.publicUrl`, `graphrag.identityFabricUrl`,
`graphrag.embedding.awsRegion`, `secrets.auditKeyFile`) plus
`serviceAccount.annotations: {eks.amazonaws.com/role-arn: <service_account_role_arn>}`.

```bash
helm upgrade --install rag oci://ghcr.io/kogunlowo123/charts/context-engineering-graphrag \
  --version 0.1.0 -n context-engineering -f <environment>.yaml --wait
```

## 7. Verify

```bash
kubectl -n context-engineering rollout status statefulset/rag-context-engineering-graphrag
kubectl -n context-engineering port-forward svc/rag-context-engineering-graphrag 8080:80
curl -sf localhost:8080/readyz
curl -s localhost:8080/.well-known/agent-card.json | jq '.name, .skills[].id'
curl -s localhost:8080/metrics | grep graphrag_http_request_duration_seconds_count
```

Expected: `{"status":"ready"}`, the agent name and skills `grounded-answer` and
`context-pack`, and metric samples. Then create a collection, ingest a known
document and run `POST /v1/collections/{c}/evaluations/retrieval` with a small
golden set as the post-install check.
