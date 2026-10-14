"""Use cases. Every call is authorized (identity fabric or local policy), every
decision and state change is written to the signed audit chain."""

from __future__ import annotations

import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from agentic_core.audit import AuditLog
from agentic_core.domain.classification import DataClassification, Decision
from agentic_core.domain.provenance import content_digest
from agentic_core.observability import correlation_id
from agentic_core.policy import Resource
from pydantic import BaseModel, ConfigDict, Field

from context_engineering_graphrag.application.knowledge_graph import (
    GraphBuildOptions,
    GraphBuildReport,
    KnowledgeGraph,
    Neighbourhood,
)
from context_engineering_graphrag.domain.errors import (
    ApprovalRequiredError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
)
from context_engineering_graphrag.domain.models import (
    Chunk,
    Collection,
    CollectionName,
    Community,
    Document,
    Entity,
    Metadata,
)
from context_engineering_graphrag.domain.ports import Clock
from context_engineering_graphrag.evaluation.groundedness import (
    GroundednessEvaluator,
    GroundednessReport,
)
from context_engineering_graphrag.evaluation.retrieval_metrics import (
    GoldenQuery,
    RetrievalReport,
    score_query,
    summarize,
)
from context_engineering_graphrag.infrastructure.database import ErasureReport, SqlStore
from context_engineering_graphrag.ingestion.service import (
    IngestDocument,
    IngestionPipeline,
    IngestResult,
)
from context_engineering_graphrag.observability.metrics import PlatformMetrics
from context_engineering_graphrag.orchestration.workflow import (
    AgenticRetrievalWorkflow,
    ContextRequest,
    ContextResult,
    QueryRequest,
    QueryResult,
)
from context_engineering_graphrag.retrieval.hybrid import (
    HybridRetriever,
    RetrievalFilters,
    RetrievalOptions,
    ScoredChunk,
)
from context_engineering_graphrag.security.authorization import AccessDecision, Authorizer

CHANNEL: ContextVar[str] = ContextVar("graphrag_channel", default="http")
"""Protocol the current call arrived on (http, mcp, a2a); recorded in every audit event."""

PRINCIPAL: ContextVar[str | None] = ContextVar("graphrag_principal", default=None)
"""Principal of the most recent authorization decision in this call."""


class CreateCollection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: CollectionName
    description: str = Field(default="", max_length=2000)
    classification: DataClassification
    owner: str = Field(min_length=1, max_length=200)


class SearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=8, ge=1, le=100)
    candidates: int = Field(default=50, ge=1, le=500)
    rerank: bool = True
    diversify: bool = True
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Metadata = Field(default_factory=dict)
    document_ids: frozenset[str] | None = Field(default=None, max_length=500)
    max_classification: DataClassification | None = None
    as_of: datetime | None = None
    purpose_of_use: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    bm25: float | None
    bm25_rank: int | None
    dense: float | None
    dense_rank: int | None
    fused: float
    rerank: float | None

    @classmethod
    def of(cls, hit: ScoredChunk) -> SearchHit:
        return cls(
            chunk=hit.chunk,
            score=hit.score,
            bm25=hit.bm25,
            bm25_rank=hit.bm25_rank,
            dense=hit.dense,
            dense_rank=hit.dense_rank,
            fused=round(hit.fused, 8),
            rerank=hit.rerank,
        )


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    revision: int
    hits: tuple[SearchHit, ...]
    reranker: str
    embedding_model: str


class DocumentView(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: Document
    chunks: tuple[Chunk, ...]


class GroundednessRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(min_length=1, max_length=20_000)
    sources: dict[str, str] = Field(
        min_length=1, max_length=100, description="Citation label (S1, F1, C1...) -> source text"
    )


class RetrievalEvaluationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: tuple[GoldenQuery, ...] = Field(min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True


class Caller(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    purpose_of_use: str | None = None


class ContextPlatform:
    def __init__(
        self,
        *,
        store: SqlStore,
        authorizer: Authorizer,
        audit: AuditLog,
        pipeline: IngestionPipeline,
        retriever: HybridRetriever,
        graph: KnowledgeGraph,
        workflow: AgenticRetrievalWorkflow,
        evaluator: GroundednessEvaluator,
        metrics: PlatformMetrics,
        clock: Clock,
    ) -> None:
        self.store = store
        self.authorizer = authorizer
        self.audit = audit
        self.pipeline = pipeline
        self.retriever = retriever
        self.graph = graph
        self.workflow = workflow
        self.evaluator = evaluator
        self.metrics = metrics
        self.clock = clock

    # --------------------------------------------------------- authorization

    async def authorize(
        self,
        caller: Caller,
        action: str,
        *,
        resource_type: str,
        resource_id: str,
        classification: DataClassification,
        attributes: dict[str, Any] | None = None,
        purpose_of_use: str | None = None,
    ) -> AccessDecision:
        purpose = purpose_of_use or caller.purpose_of_use
        decision = await self.authorizer.authorize(
            caller.token,
            action=action,
            resource=Resource(
                type=resource_type,
                id=resource_id,
                classification=classification,
                attributes=attributes or {},
            ),
            environment={"purpose_of_use": purpose} if purpose else {},
        )
        self.metrics.authz.labels(decision.decision.value, decision.source).inc()
        PRINCIPAL.set(decision.principal)
        await self.record(
            decision.principal,
            f"authorize:{action}",
            decision.resource,
            decision.decision.value,
            {
                "rules": list(decision.matched_rules),
                "policy_version": decision.policy_version,
                "policy_digest": decision.policy_digest,
                "chain": list(decision.delegation_chain),
                "source": decision.source,
            },
        )
        if decision.decision is Decision.DENY:
            raise ForbiddenError(f"{action} denied: {decision.reason}")
        if decision.decision is Decision.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(
                f"{action} requires human approval: {decision.reason}", rules=decision.matched_rules
            )
        return decision

    async def _collection_access(
        self, caller: Caller, action: str, name: str, *, purpose_of_use: str | None = None
    ) -> tuple[Collection, AccessDecision]:
        collection = await self.store.get_collection(name)
        if collection is None:
            # Authorize against the most restrictive label before revealing absence.
            await self.authorize(
                caller,
                action,
                resource_type="collection",
                resource_id=name,
                classification=DataClassification.RESTRICTED,
                purpose_of_use=purpose_of_use,
            )
            raise NotFoundError(f"collection {name} not found")
        decision = await self.authorize(
            caller,
            action,
            resource_type="collection",
            resource_id=name,
            classification=collection.classification,
            purpose_of_use=purpose_of_use,
        )
        if collection.embedding_model != self.retriever.embedder.model:
            raise ConflictError(
                f"collection {name} was embedded with {collection.embedding_model}; this service "
                f"uses {self.retriever.embedder.model}. Re-ingest into a new collection."
            )
        return collection, decision

    async def record(
        self, actor: str, action: str, resource: str, outcome: str, details: dict[str, Any]
    ) -> None:
        await self.audit.record(
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            correlation_id=correlation_id(),
            details={**details, "channel": CHANNEL.get()},
        )

    # ----------------------------------------------------------- collections

    async def create_collection(self, caller: Caller, body: CreateCollection) -> Collection:
        decision = await self.authorize(
            caller,
            "rag:collection.create",
            resource_type="collection",
            resource_id=body.name,
            classification=body.classification,
        )
        now = self.clock.now()
        collection = Collection(
            name=body.name,
            description=body.description,
            classification=body.classification,
            owner=body.owner,
            revision=0,
            embedding_model=self.retriever.embedder.model,
            embedding_dimension=self.retriever.embedder.dimension,
            created_at=now,
            updated_at=now,
        )
        await self.store.create_collection(collection)
        await self.record(
            decision.principal,
            "collection.create",
            f"collection:{body.name}",
            "created",
            {"classification": body.classification.name, "owner": body.owner},
        )
        return collection

    async def get_collection(self, caller: Caller, name: str) -> Collection:
        collection, _ = await self._collection_access(caller, "rag:collection.read", name)
        return collection

    async def list_collections(self, caller: Caller) -> list[Collection]:
        await self.authorize(
            caller,
            "rag:collection.read",
            resource_type="collection",
            resource_id="*",
            classification=DataClassification.INTERNAL,
        )
        return await self.store.list_collections()

    # ------------------------------------------------------------- documents

    async def ingest(self, caller: Caller, name: str, body: IngestDocument) -> IngestResult:
        collection, decision = await self._collection_access(caller, "rag:document.write", name)
        prepared = await self.pipeline.prepare(collection, body, decision.principal)
        if prepared.write is not None:
            await self.store.write_document(prepared.write)
            self.metrics.ingested_chunks.labels(name).inc(prepared.result.chunks)
            if prepared.result.quarantined_chunks:
                self.metrics.quarantined_chunks.labels(name).inc(prepared.result.quarantined_chunks)
        result = prepared.result
        await self.record(
            decision.principal,
            "document.ingest",
            f"collection:{name}/document:{body.document_id}",
            result.status,
            {
                "version": result.version,
                "digest": result.digest,
                "chunks": result.chunks,
                "quarantined": result.quarantined_chunks,
                "quarantine_reasons": sorted(set(prepared.quarantine_reasons)),
                "redacted_secrets": result.redacted_secrets,
                "assertions": result.assertions,
                "classification": body.classification.name,
            },
        )
        return result

    async def get_document(self, caller: Caller, name: str, document_id: str) -> DocumentView:
        await self._collection_access(caller, "rag:document.read", name)
        document = await self.store.get_document(name, document_id)
        if document is None:
            raise NotFoundError(f"document {document_id} not found in {name}")
        return DocumentView(
            document=document, chunks=tuple(await self.store.document_chunks(name, document_id))
        )

    async def list_documents(
        self, caller: Caller, name: str, *, limit: int, offset: int
    ) -> list[Document]:
        await self._collection_access(caller, "rag:document.read", name)
        return await self.store.list_documents(name, limit=limit, offset=offset)

    async def erase_document(self, caller: Caller, name: str, document_id: str) -> ErasureReport:
        _, decision = await self._collection_access(caller, "rag:document.delete", name)
        report = await self.store.erase_document(name, document_id, self.clock.now())
        await self.record(
            decision.principal,
            "document.erase",
            f"collection:{name}/document:{document_id}",
            "erased",
            {
                "chunks": report.chunks,
                "mentions": report.mentions,
                "assertions": report.assertions,
                "entities": report.entities,
                "communities": report.communities,
            },
        )
        return report

    # ------------------------------------------------------------- retrieval

    async def search(self, caller: Caller, name: str, body: SearchRequest) -> SearchResult:
        _, decision = await self._collection_access(
            caller, "rag:search", name, purpose_of_use=body.purpose_of_use
        )
        self.workflow.guard_query(body.query)
        started = time.perf_counter()
        hits = await self.retriever.search(
            name,
            body.query,
            filters=RetrievalFilters(
                metadata=body.metadata,
                document_ids=body.document_ids,
                max_classification=body.max_classification,
                as_of=body.as_of,
            ),
            options=RetrievalOptions(
                k=body.k,
                candidates=body.candidates,
                rerank=body.rerank,
                diversify=body.diversify,
                mmr_lambda=body.mmr_lambda,
            ),
        )
        self.metrics.stage.labels("search").observe(time.perf_counter() - started)
        revision = await self.store.revision(name)
        await self.record(
            decision.principal,
            "retrieval.search",
            f"collection:{name}",
            "ok",
            {
                "query_digest": content_digest(body.query),
                "hits": [h.chunk.chunk_id for h in hits],
                "revision": revision,
            },
        )
        return SearchResult(
            collection=name,
            revision=revision,
            hits=tuple(SearchHit.of(h) for h in hits),
            reranker=self.retriever.reranker.name if body.rerank else "none",
            embedding_model=self.retriever.embedder.model,
        )

    async def context(self, caller: Caller, name: str, body: ContextRequest) -> ContextResult:
        collection, decision = await self._collection_access(
            caller, "rag:context", name, purpose_of_use=body.purpose_of_use
        )
        result = await self.workflow.context(collection, body)
        await self.record(
            decision.principal,
            "context.build",
            f"collection:{name}",
            "ok",
            {
                "query_digest": content_digest(body.query),
                "mode": result.plan.mode.value,
                "sources": [s.key for s in result.context.sources],
                "used_tokens": result.context.used_tokens,
            },
        )
        return result

    async def query(self, caller: Caller, name: str, body: QueryRequest) -> QueryResult:
        collection, decision = await self._collection_access(
            caller, "rag:query", name, purpose_of_use=body.purpose_of_use
        )
        result = await self.workflow.answer(
            collection,
            body,
            principal=decision.principal,
            delegated=bool(decision.delegation_chain),
        )
        await self.record(
            decision.principal,
            "answer.release",
            f"collection:{name}",
            result.disposition.value,
            {
                "query_digest": content_digest(body.query),
                "answer_digest": content_digest(result.answer) if result.answer else None,
                "mode": result.plan.mode.value,
                "groundedness": result.groundedness.groundedness,
                "citations": [c.chunk_id or c.label for c in result.citations],
                "release_rules": list(result.release.matched_rules),
                "policy_digest": result.release.policy_digest,
                "generator": result.generator,
            },
        )
        return result

    # ----------------------------------------------------------------- graph

    async def build_graph(
        self, caller: Caller, name: str, options: GraphBuildOptions
    ) -> GraphBuildReport:
        collection, decision = await self._collection_access(caller, "rag:graph.build", name)
        try:
            report = await self.graph.build(collection, options)
        except Exception:
            self.metrics.graph_builds.labels("failed").inc()
            raise
        self.metrics.graph_builds.labels("succeeded").inc()
        await self.record(
            decision.principal,
            "graph.build",
            f"collection:{name}",
            "built",
            {
                "merged_entities": report.merged_entities,
                "communities": report.communities,
                "levels": len(report.levels),
                "summarizer": report.summarizer,
            },
        )
        return report

    async def entities(
        self, caller: Caller, name: str, *, query: str | None, limit: int
    ) -> list[Entity]:
        await self._collection_access(caller, "rag:graph.read", name)
        return await self.graph.find_entities(name, name=query, limit=limit)

    async def neighbourhood(
        self, caller: Caller, name: str, entity_id: str, *, hops: int, as_of: datetime | None
    ) -> Neighbourhood:
        await self._collection_access(caller, "rag:graph.read", name)
        found = await self.graph.neighbourhood(name, entity_id, hops=hops, as_of=as_of)
        if found is None:
            raise NotFoundError(f"entity {entity_id} not found in {name}")
        return found

    async def communities(self, caller: Caller, name: str, *, level: int | None) -> list[Community]:
        await self._collection_access(caller, "rag:graph.read", name)
        return await self.store.communities(name, level=level)

    # ------------------------------------------------------------ evaluation

    async def evaluate_groundedness(
        self, caller: Caller, body: GroundednessRequest
    ) -> GroundednessReport:
        decision = await self.authorize(
            caller,
            "rag:evaluate",
            resource_type="evaluation",
            resource_id="groundedness",
            classification=DataClassification.INTERNAL,
        )
        report = await self.evaluator.evaluate(body.answer, body.sources)
        await self.record(
            decision.principal,
            "evaluation.groundedness",
            "evaluation:groundedness",
            "ok",
            {"groundedness": report.groundedness, "claims": len(report.claims)},
        )
        return report

    async def evaluate_retrieval(
        self, caller: Caller, name: str, body: RetrievalEvaluationRequest
    ) -> RetrievalReport:
        _, decision = await self._collection_access(caller, "rag:evaluate", name)
        results = []
        for golden in body.queries:
            hits = await self.retriever.search(
                name,
                golden.query,
                options=RetrievalOptions(
                    k=body.k, candidates=max(50, body.k * 3), rerank=body.rerank, diversify=False
                ),
            )
            results.append(score_query(golden, [h.chunk.document_id for h in hits], body.k))
        report = summarize(body.k, results)
        await self.record(
            decision.principal,
            "evaluation.retrieval",
            f"collection:{name}",
            "ok",
            {"queries": len(results), "mrr": report.mrr, "mean_ndcg": report.mean_ndcg},
        )
        return report
