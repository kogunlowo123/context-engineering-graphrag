"""Composition root: builds every component once per process from Settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
import yaml
from agentic_core.audit import AuditLog, JsonlAuditSink
from agentic_core.models.factory import ProvidersConfig, build_providers
from agentic_core.models.router import ModelRouter
from agentic_core.observability import ModelTelemetryObserver
from agentic_core.policy import PolicyEngine, PolicySet
from agentic_core.security import ContentGuard
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy.ext.asyncio import AsyncEngine

from context_engineering_graphrag.application.knowledge_graph import KnowledgeGraph
from context_engineering_graphrag.application.service import ContextPlatform
from context_engineering_graphrag.config import Settings
from context_engineering_graphrag.context.budget import ContextBudgeter
from context_engineering_graphrag.domain.models import ChunkingConfig
from context_engineering_graphrag.domain.ports import Clock, Embedder, SystemClock
from context_engineering_graphrag.domain.text import HeuristicTokenizer
from context_engineering_graphrag.embeddings.hashing import HashingEmbedder
from context_engineering_graphrag.embeddings.providers import (
    BedrockTitanEmbedder,
    OpenAICompatibleEmbedder,
)
from context_engineering_graphrag.evaluation.groundedness import (
    EntailmentJudge,
    GroundednessEvaluator,
    LexicalEntailmentJudge,
    ModelEntailmentJudge,
)
from context_engineering_graphrag.generation.answer import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    ModelAnswerGenerator,
)
from context_engineering_graphrag.graph.extraction import (
    EntityExtractor,
    ModelEntityExtractor,
    PatternEntityExtractor,
)
from context_engineering_graphrag.graph.resolution import EntityResolver
from context_engineering_graphrag.graph.summarize import (
    CommunitySummarizer,
    ExtractiveCommunitySummarizer,
    ModelCommunitySummarizer,
)
from context_engineering_graphrag.infrastructure.database import (
    SqlStore,
    create_schema,
    make_engine,
)
from context_engineering_graphrag.ingestion.service import IngestionPipeline
from context_engineering_graphrag.integrations.identity_fabric import IdentityFabricClient
from context_engineering_graphrag.observability.metrics import PlatformMetrics
from context_engineering_graphrag.orchestration.workflow import AgenticRetrievalWorkflow
from context_engineering_graphrag.retrieval.hybrid import HybridRetriever
from context_engineering_graphrag.retrieval.rerank import (
    CrossEncoder,
    HttpCrossEncoder,
    LexicalCrossEncoder,
)
from context_engineering_graphrag.security.authorization import (
    Authorizer,
    FabricAuthorizer,
    JwtVerifier,
    KeyResolver,
    LocalAuthorizer,
    RemoteJwks,
)


@dataclass
class Container:
    settings: Settings
    platform: ContextPlatform
    metrics: PlatformMetrics
    store: SqlStore
    engine: AsyncEngine
    http: httpx.AsyncClient
    embedder: Embedder
    reranker: CrossEncoder
    router: ModelRouter | None

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await self.reranker.aclose()
        if self.router is not None:
            await self.router.aclose()
        await self.http.aclose()
        await self.engine.dispose()


def _audit_signer(settings: Settings) -> tuple[Ed25519PrivateKey, str]:
    if settings.audit_key_file is None:
        return Ed25519PrivateKey.generate(), "audit-ephemeral"
    key = serialization.load_pem_private_key(settings.audit_key_file.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("audit key must be Ed25519")
    return key, settings.audit_key_file.stem


def _secret(name: str | None, env: Mapping[str, str]) -> str | None:
    if name is None:
        return None
    value = env.get(name)
    if not value:
        raise ValueError(f"environment variable {name} is not set")
    return value


def build_embedder(
    settings: Settings,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Embedder:
    env = os.environ if env is None else env
    match settings.embedding_provider:
        case "hashing":
            return HashingEmbedder(settings.embedding_dimension)
        case "bedrock":
            return BedrockTitanEmbedder.from_region(
                settings.aws_region or "",
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
            )
        case "openai_compatible":
            return OpenAICompatibleEmbedder(
                base_url=settings.embedding_base_url or "",
                path=settings.embedding_path,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                api_key=_secret(settings.embedding_api_key_env, env),
                auth=settings.embedding_auth,
                transport=transport,
            )


def build_reranker(
    settings: Settings,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CrossEncoder:
    env = os.environ if env is None else env
    if settings.reranker == "http":
        return HttpCrossEncoder(
            base_url=settings.reranker_url or "",
            model=settings.reranker_model,
            api_key=_secret(settings.reranker_api_key_env, env),
            transport=transport,
        )
    return LexicalCrossEncoder()


def build_router(settings: Settings, *, env: Mapping[str, str] | None = None) -> ModelRouter:
    if settings.providers_file is None:
        raise ValueError("providers_file is not configured")
    config = ProvidersConfig.from_file(settings.providers_file, env=env)
    return ModelRouter(build_providers(config, env=env), observers=[ModelTelemetryObserver()])


def load_gazetteer(settings: Settings) -> dict[str, str]:
    if settings.gazetteer_file is None:
        return {}
    loaded = yaml.safe_load(settings.gazetteer_file.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and 0 < len(v) <= 32 for k, v in loaded.items()
    ):
        raise ValueError("gazetteer must map entity names to entity types")
    return dict(loaded)


async def build_container(
    settings: Settings,
    *,
    clock: Clock | None = None,
    http: httpx.AsyncClient | None = None,
    jwks_resolver: KeyResolver | None = None,
    embedder: Embedder | None = None,
    reranker: CrossEncoder | None = None,
    router: ModelRouter | None = None,
) -> Container:
    clock = clock or SystemClock()
    http = http or httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    metrics = PlatformMetrics()
    tokenizer = HeuristicTokenizer()

    embedder = embedder or build_embedder(settings)
    reranker = reranker or build_reranker(settings)
    if settings.uses_models and router is None:
        router = build_router(settings)
    guard = ContentGuard()

    engine = make_engine(settings.database_url)
    await create_schema(
        engine,
        pgvector_dimension=embedder.dimension if settings.vector_backend == "pgvector" else None,
    )
    store = SqlStore(engine, pgvector=settings.vector_backend == "pgvector")

    signer, key_id = _audit_signer(settings)
    audit = AuditLog(JsonlAuditSink(settings.audit_log_path), signer, key_id=key_id)
    policy = PolicyEngine(PolicySet.from_file(settings.policy_file))

    authorizer: Authorizer
    if settings.identity_fabric_url is not None:
        authorizer = FabricAuthorizer(
            IdentityFabricClient(str(settings.identity_fabric_url), http=http)
        )
    else:
        authorizer = LocalAuthorizer(
            JwtVerifier(
                jwks_resolver or RemoteJwks(settings.local_jwks_url or "", client=http),
                issuer=settings.local_issuer or "",
                audience=settings.local_audience or "",
                clock=clock,
            ),
            policy,
        )

    extractor: EntityExtractor
    summarizer: CommunitySummarizer
    generator: AnswerGenerator
    judge: EntailmentJudge
    if router is not None and settings.extractor == "model":
        extractor = ModelEntityExtractor(router, guard)
    else:
        extractor = PatternEntityExtractor(load_gazetteer(settings))
    extractive_summaries = ExtractiveCommunitySummarizer()
    if router is not None and settings.summarizer == "model":
        summarizer = ModelCommunitySummarizer(router, guard, extractive_summaries)
    else:
        summarizer = extractive_summaries
    if router is not None and settings.generator == "model":
        generator = ModelAnswerGenerator(router, guard)
    else:
        generator = ExtractiveAnswerGenerator(tokenizer)
    if router is not None and settings.judge == "model":
        judge = ModelEntailmentJudge(router, guard)
    else:
        judge = LexicalEntailmentJudge()

    resolver = EntityResolver(settings.resolution_threshold)
    retriever = HybridRetriever(
        store,
        embedder,
        reranker,
        clock,
        on_reload=lambda _collection, seconds: metrics.index_reloads.observe(seconds),
    )
    graph = KnowledgeGraph(store, resolver, summarizer, clock)
    evaluator = GroundednessEvaluator(judge)
    workflow = AgenticRetrievalWorkflow(
        store,
        retriever,
        graph,
        budgeter=ContextBudgeter(tokenizer),
        generator=generator,
        evaluator=evaluator,
        release_policy=policy,
        metrics=metrics,
    )
    pipeline = IngestionPipeline(
        store,
        embedder,
        extractor=extractor,
        resolver=resolver,
        tokenizer=tokenizer,
        clock=clock,
        default_chunking=ChunkingConfig(
            strategy=settings.chunk_strategy,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
    )
    platform = ContextPlatform(
        store=store,
        authorizer=authorizer,
        audit=audit,
        pipeline=pipeline,
        retriever=retriever,
        graph=graph,
        workflow=workflow,
        evaluator=evaluator,
        metrics=metrics,
        clock=clock,
    )
    return Container(
        settings=settings,
        platform=platform,
        metrics=metrics,
        store=store,
        engine=engine,
        http=http,
        embedder=embedder,
        reranker=reranker,
        router=router,
    )
