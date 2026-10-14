"""Runtime settings, read from GRAPHRAG_* environment variables.

The service refuses to start with an incomplete or unsafe configuration
(fail closed): production requires a persistent audit signing key and a
provider-backed embedder; without an identity fabric URL a local JWT issuer
must be configured; any model-backed feature requires a provider file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from context_engineering_graphrag.domain.models import ChunkingStrategy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPHRAG_", env_nested_delimiter="__")

    environment: str = "production"
    public_url: HttpUrl = Field(description="External base URL; used in the A2A agent card")
    additional_hosts: list[str] = Field(
        default_factory=list,
        description="Extra Host header values accepted on /mcp, e.g. the cluster service DNS name",
    )
    database_url: str = Field(description="SQLAlchemy async URL, e.g. postgresql+asyncpg://...")
    vector_backend: Literal["memory", "pgvector"] = "memory"

    audit_log_path: Path = Path("/var/lib/graphrag/audit/audit.jsonl")
    audit_key_file: Path | None = Field(
        default=None, description="Ed25519 PEM signing the audit chain; required in production"
    )
    policy_file: Path = Path("policies/context-governance.yaml")

    identity_fabric_url: HttpUrl | None = Field(
        default=None,
        description="agent-identity-fabric base URL; inbound calls are authorized there",
    )
    local_issuer: str | None = Field(
        default=None, description="Trusted JWT issuer without a fabric"
    )
    local_audience: str | None = None
    local_jwks_url: str | None = None

    embedding_provider: Literal["hashing", "bedrock", "openai_compatible"] = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimension: int = Field(default=1024, ge=16, le=8192)
    embedding_base_url: str | None = None
    embedding_path: str = "/v1/embeddings"
    embedding_auth: Literal["bearer", "api-key", "none"] = "bearer"
    embedding_api_key_env: str | None = Field(
        default=None, description="Name of the variable holding the embedding API key"
    )
    aws_region: str | None = None

    reranker: Literal["lexical", "http"] = "lexical"
    reranker_url: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_key_env: str | None = None

    generator: Literal["extractive", "model"] = "extractive"
    extractor: Literal["pattern", "model"] = "pattern"
    summarizer: Literal["extractive", "model"] = "extractive"
    judge: Literal["lexical", "model"] = "lexical"
    providers_file: Path | None = Field(
        default=None, description="agentic-core provider config (configs/providers.yaml)"
    )
    gazetteer_file: Path | None = Field(
        default=None,
        description="YAML mapping of entity name to type used by the pattern extractor",
    )

    chunk_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_max_tokens: int = Field(default=256, ge=32, le=2048)
    chunk_overlap_tokens: int = Field(default=32, ge=0, le=512)
    resolution_threshold: float = Field(default=0.88, ge=0.5, le=1.0)

    log_level: str = "INFO"
    otlp_endpoint: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Settings:
        problems: list[str] = []
        if self.environment == "production":
            if self.audit_key_file is None:
                problems.append("GRAPHRAG_AUDIT_KEY_FILE is required in production")
            if self.embedding_provider == "hashing":
                problems.append("the hashing embedder is not allowed in production")
        if self.identity_fabric_url is None and not (
            self.local_issuer and self.local_audience and self.local_jwks_url
        ):
            problems.append(
                "set GRAPHRAG_IDENTITY_FABRIC_URL or all of GRAPHRAG_LOCAL_ISSUER, "
                "GRAPHRAG_LOCAL_AUDIENCE and GRAPHRAG_LOCAL_JWKS_URL"
            )
        if self.uses_models and self.providers_file is None:
            problems.append("GRAPHRAG_PROVIDERS_FILE is required for model-backed features")
        if self.vector_backend == "pgvector" and not self.database_url.startswith("postgresql"):
            problems.append("vector_backend=pgvector requires a PostgreSQL database_url")
        if self.embedding_provider == "bedrock" and not self.aws_region:
            problems.append("GRAPHRAG_AWS_REGION is required for bedrock embeddings")
        if self.embedding_provider == "openai_compatible" and not self.embedding_base_url:
            problems.append(
                "GRAPHRAG_EMBEDDING_BASE_URL is required for openai_compatible embeddings"
            )
        if self.reranker == "http" and not self.reranker_url:
            problems.append("GRAPHRAG_RERANKER_URL is required for the http reranker")
        if self.chunk_overlap_tokens * 2 > self.chunk_max_tokens:
            problems.append("chunk overlap must be at most half of chunk_max_tokens")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def uses_models(self) -> bool:
        return "model" in (self.generator, self.extractor, self.summarizer, self.judge)

    @property
    def public_url_str(self) -> str:
        return str(self.public_url).rstrip("/")

    @property
    def allowed_hosts(self) -> list[str]:
        parsed = urlsplit(self.public_url_str)
        host = parsed.netloc
        hosts = {host, "127.0.0.1:*", "localhost:*", *self.additional_hosts}
        if ":" not in host:
            hosts.add(f"{host}:*")
        return sorted(hosts)
