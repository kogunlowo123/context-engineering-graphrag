"""Agentic RAG workflow.

Query -> guard -> plan -> retrieve (hybrid + graph, iterative) -> rerank ->
context budgeting -> attributed generation -> groundedness evaluation ->
release decision (policy) -> grounded answer.

The planner chooses a retrieval mode from the query and the graph:

* ``local``   the query names entities in the graph: personalised PageRank
              around them adds relation facts and their evidence chunks,
* ``global``  the query asks for themes across the corpus and community
              summaries exist: the best summaries are mapped in,
* ``passages`` otherwise: hybrid passage retrieval only.

If the first round leaves query terms uncovered, the workflow reformulates
once: it adds linked entity aliases and pseudo-relevance terms from the top
passages and retrieves again, fusing both rounds.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar

from agentic_core.domain.classification import DataClassification, Decision, TrustLevel
from agentic_core.policy import PolicyEngine, PolicyRequest, Resource, Subject
from agentic_core.security import GuardConfig, PromptInjectionDetector, SecretScanner
from pydantic import BaseModel, ConfigDict, Field

from context_engineering_graphrag.application.knowledge_graph import KnowledgeGraph
from context_engineering_graphrag.context.budget import (
    BudgetPolicy,
    Candidate,
    ContextBudgeter,
    ContextPack,
    SourceKind,
)
from context_engineering_graphrag.domain.errors import ContentRejectedError
from context_engineering_graphrag.domain.models import Collection, Metadata
from context_engineering_graphrag.domain.text import STOPWORDS, terms, words
from context_engineering_graphrag.evaluation.groundedness import (
    GroundednessEvaluator,
    GroundednessReport,
)
from context_engineering_graphrag.generation.answer import AnswerGenerator
from context_engineering_graphrag.graph.search import (
    GraphView,
    LocalSearchResult,
    RankedCommunity,
    local_search,
    phrase,
    rank_communities,
)
from context_engineering_graphrag.infrastructure.database import SqlStore
from context_engineering_graphrag.observability.metrics import PlatformMetrics
from context_engineering_graphrag.retrieval.fusion import reciprocal_rank_fusion
from context_engineering_graphrag.retrieval.hybrid import (
    HybridRetriever,
    RetrievalFilters,
    RetrievalOptions,
    ScoredChunk,
)

T = TypeVar("T")

_THEMATIC = re.compile(
    r"\b(?:main|key|major|overall|common|recurring|dominant)\s+(?:themes?|topics?|trends?|issues?|"
    r"risks?|findings?|patterns?)\b|\bsummar(?:y|ise|ize)\b|\bacross\s+(?:the\s+)?(?:corpus|"
    r"collection|documents|portfolio)\b|\bwhat\s+are\s+the\s+(?:themes?|trends?)\b|\boverview\b",
    re.I,
)


class RetrievalMode(StrEnum):
    AUTO = "auto"
    PASSAGES = "passages"
    LOCAL = "local"
    GLOBAL = "global"


class Disposition(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    WITHHELD = "withheld"
    PENDING_APPROVAL = "pending_approval"


class ContextRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode = RetrievalMode.AUTO
    k: int = Field(default=8, ge=1, le=50)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    as_of: datetime | None = Field(
        default=None, description="Answer from documents and graph facts valid at this instant"
    )
    metadata: Metadata = Field(default_factory=dict)
    document_ids: frozenset[str] | None = Field(default=None, max_length=500)
    max_classification: DataClassification | None = None
    community_level: int | None = Field(default=None, ge=0, le=8)
    rerank: bool = True
    purpose_of_use: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class QueryRequest(ContextRequest):
    max_answer_tokens: int = Field(default=512, ge=32, le=8000)


class TraceStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: str
    duration_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: RetrievalMode
    rationale: str
    linked_entities: tuple[str, ...]
    rounds: int


class ContextResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    plan: Plan
    context: ContextPack
    rendered: str
    trace: tuple[TraceStep, ...]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    kind: SourceKind
    title: str
    source_uri: str | None
    document_id: str | None
    document_version: int | None
    chunk_id: str | None
    digest: str | None
    evidence_chunks: tuple[str, ...]
    excerpt: str


class ReleaseDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    matched_rules: tuple[str, ...]
    policy_version: str
    policy_digest: str
    reason: str


class QueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    disposition: Disposition
    answer: str | None = Field(
        description="Released answer; null unless disposition is answered/abstained"
    )
    citations: tuple[Citation, ...]
    groundedness: GroundednessReport
    release: ReleaseDecision
    plan: Plan
    generator: str
    context_tokens: int
    context_budget: int
    sources: int
    dropped_sources: int
    trace: tuple[TraceStep, ...]


class AgenticRetrievalWorkflow:
    def __init__(
        self,
        store: SqlStore,
        retriever: HybridRetriever,
        graph: KnowledgeGraph,
        *,
        budgeter: ContextBudgeter,
        generator: AnswerGenerator,
        evaluator: GroundednessEvaluator,
        release_policy: PolicyEngine,
        metrics: PlatformMetrics,
        max_rounds: int = 2,
        coverage_target: float = 0.6,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._graph = graph
        self._budgeter = budgeter
        self._generator = generator
        self._evaluator = evaluator
        self._release = release_policy
        self._metrics = metrics
        self._max_rounds = max_rounds
        self._coverage_target = coverage_target
        self._injection = PromptInjectionDetector()
        self._secrets = SecretScanner()
        self._threshold = GuardConfig().block_threshold[TrustLevel.USER]

    @property
    def generator(self) -> AnswerGenerator:
        return self._generator

    async def context(self, collection: Collection, request: ContextRequest) -> ContextResult:
        trace: list[TraceStep] = []
        plan, pack = await self._build_context(collection, request, trace)
        return ContextResult(
            collection=collection.name,
            plan=plan,
            context=pack,
            rendered=pack.render(),
            trace=tuple(trace),
        )

    async def answer(
        self, collection: Collection, request: QueryRequest, *, principal: str, delegated: bool
    ) -> QueryResult:
        trace: list[TraceStep] = []
        plan, pack = await self._build_context(collection, request, trace)
        draft = await self._timed(
            trace,
            "generate",
            lambda: self._generator.generate(
                request.query, pack, max_tokens=request.max_answer_tokens
            ),
            lambda d: {
                "generator": d.generator,
                "abstained": d.abstained,
                "citations": len(d.citations),
            },
        )
        report = await self._timed(
            trace,
            "evaluate",
            lambda: self._evaluator.evaluate(
                draft.text, {s.source_id: s.text for s in pack.sources}
            ),
            lambda r: {"groundedness": r.groundedness, "claims": len(r.claims)},
        )
        decision = self._release.evaluate(
            PolicyRequest(
                subject=Subject(
                    id=principal,
                    kind="agent",
                    attributes={"delegated": delegated},
                ),
                action="answer:release",
                resource=Resource(
                    type="answer",
                    id=collection.name,
                    classification=pack.classification,
                    attributes={
                        "groundedness": report.groundedness,
                        "unsupported_claims": report.unsupported_claims,
                        "contradicted_claims": report.contradicted_claims,
                        "citations": len(draft.citations),
                        "abstained": draft.abstained,
                        "invalid_citations": draft.invalid_citations,
                    },
                ),
                environment=(
                    {"purpose_of_use": request.purpose_of_use} if request.purpose_of_use else {}
                ),
            )
        )
        if decision.decision is Decision.ALLOW:
            disposition = Disposition.ABSTAINED if draft.abstained else Disposition.ANSWERED
        elif decision.decision is Decision.REQUIRE_APPROVAL:
            disposition = Disposition.PENDING_APPROVAL
        else:
            disposition = Disposition.WITHHELD
        released = disposition in (Disposition.ANSWERED, Disposition.ABSTAINED)
        by_id = pack.by_id()
        citations = tuple(
            Citation(
                label=label,
                kind=by_id[label].kind,
                title=by_id[label].title,
                source_uri=by_id[label].source_uri,
                document_id=by_id[label].document_id,
                document_version=by_id[label].document_version,
                chunk_id=by_id[label].chunk_id,
                digest=by_id[label].digest,
                evidence_chunks=by_id[label].evidence_chunks,
                excerpt=by_id[label].text[:400],
            )
            for label in draft.citations
            if label in by_id
        )
        self._metrics.answers.labels(disposition.value, plan.mode.value).inc()
        self._metrics.groundedness.observe(report.groundedness)
        return QueryResult(
            collection=collection.name,
            disposition=disposition,
            answer=draft.text if released else None,
            citations=citations,
            groundedness=report,
            release=ReleaseDecision(
                decision=decision.decision,
                matched_rules=decision.matched_rules,
                policy_version=decision.policy_version,
                policy_digest=decision.policy_digest,
                reason=decision.reason,
            ),
            plan=plan,
            generator=draft.generator,
            context_tokens=pack.used_tokens,
            context_budget=pack.budget_tokens,
            sources=len(pack.sources),
            dropped_sources=len(pack.dropped),
            trace=tuple(trace),
        )

    # ---------------------------------------------------------------- stages

    def guard_query(self, query: str) -> None:
        if self._secrets.detect(query):
            raise ContentRejectedError("query contains credential material")
        assessment = self._injection.assess(query)
        if assessment.score >= self._threshold:
            raise ContentRejectedError(
                "query rejected by prompt-injection screening: " + ", ".join(assessment.kinds)
            )

    async def _build_context(
        self, collection: Collection, request: ContextRequest, trace: list[TraceStep]
    ) -> tuple[Plan, ContextPack]:
        self.guard_query(request.query)
        view = await self._timed(
            trace,
            "graph.load",
            lambda: self._graph.view(collection.name, as_of=request.as_of),
            lambda v: {"entities": len(v.entities), "relations": len(v.relations)},
        )
        communities = await self._store.communities(collection.name)
        linked = view.link(request.query)
        mode, rationale = self._plan(request, linked, bool(communities))
        filters = RetrievalFilters(
            metadata=request.metadata,
            document_ids=request.document_ids,
            max_classification=request.max_classification,
            as_of=request.as_of,
        )
        options = RetrievalOptions(k=request.k, rerank=request.rerank)
        passages = await self._timed(
            trace,
            "retrieve",
            lambda: self._retriever.search(
                collection.name, request.query, filters=filters, options=options
            ),
            lambda r: {"round": 1, "hits": len(r)},
        )
        rounds = 1
        coverage = _coverage(request.query, passages)
        if coverage < self._coverage_target and self._max_rounds > 1:
            expansion = _expansion_terms(request.query, passages, view, linked)
            if expansion:
                rounds = 2
                expanded = request.query + " " + " ".join(expansion)
                second = await self._timed(
                    trace,
                    "retrieve",
                    lambda: self._retriever.search(
                        collection.name, expanded, filters=filters, options=options
                    ),
                    lambda r: {
                        "round": 2,
                        "hits": len(r),
                        "expansion": expansion,
                        "coverage": coverage,
                    },
                )
                passages = _fuse_rounds(passages, second, request.k)

        local: LocalSearchResult | None = None
        ranked_communities: list[RankedCommunity] = []
        if mode is RetrievalMode.LOCAL:
            local = await self._timed(
                trace,
                "graph.local",
                lambda: _async(local_search(view, linked)),
                lambda r: {"entities": len(r.entities), "relations": len(r.relations)},
            )
        elif mode is RetrievalMode.GLOBAL:
            level = request.community_level if request.community_level is not None else 0
            at_level = [c for c in communities if c.level == level] or communities
            names = {e: view.entities[e].name for e in view.entities}
            ranked_communities = await self._timed(
                trace,
                "graph.global",
                lambda: _async(rank_communities(request.query, at_level, names)),
                lambda r: {"level": level, "communities": len(r)},
            )

        candidates = await self._candidates(
            collection=collection,
            passages=passages,
            local=local,
            communities=ranked_communities,
            view=view,
            filters=filters,
        )
        pack = await self._timed(
            trace,
            "budget",
            lambda: _async(self._budgeter.build(request.query, candidates, request.budget)),
            lambda p: {"used": p.used_tokens, "sources": len(p.sources), "dropped": len(p.dropped)},
        )
        self._metrics.context_tokens.observe(pack.used_tokens)
        plan = Plan(
            mode=mode,
            rationale=rationale,
            linked_entities=tuple(
                view.entities[e].name for e in sorted(linked, key=lambda e: -linked[e])
            ),
            rounds=rounds,
        )
        return plan, pack

    @staticmethod
    def _plan(
        request: ContextRequest, linked: dict[str, float], has_communities: bool
    ) -> tuple[RetrievalMode, str]:
        if request.mode is not RetrievalMode.AUTO:
            if request.mode is RetrievalMode.LOCAL and not linked:
                return RetrievalMode.PASSAGES, "local requested but no query entity is in the graph"
            if request.mode is RetrievalMode.GLOBAL and not has_communities:
                return RetrievalMode.PASSAGES, "global requested but communities are not built"
            return request.mode, "requested by caller"
        if _THEMATIC.search(request.query) and has_communities:
            return RetrievalMode.GLOBAL, "thematic question and community summaries are available"
        if linked:
            return RetrievalMode.LOCAL, f"query mentions {len(linked)} graph entities"
        return RetrievalMode.PASSAGES, "no graph entities linked; passage retrieval"

    async def _candidates(
        self,
        *,
        collection: Collection,
        passages: list[ScoredChunk],
        local: LocalSearchResult | None,
        communities: list[RankedCommunity],
        view: GraphView,
        filters: RetrievalFilters,
    ) -> list[Candidate]:
        out = [
            Candidate(
                key=p.chunk.chunk_id,
                kind=SourceKind.PASSAGE,
                text=p.chunk.text,
                score=p.score,
                title=p.chunk.title,
                source_uri=p.chunk.source_uri,
                document_id=p.chunk.document_id,
                document_version=p.chunk.document_version,
                chunk_id=p.chunk.chunk_id,
                digest=p.chunk.digest,
                classification=p.chunk.classification,
                trust=p.chunk.trust,
            )
            for p in passages
        ]
        if local is not None:
            seen = {c.key for c in out}
            admits = await self._retriever.admits(collection.name, filters)
            evidence = await self._store.get_chunks(
                c for c in local.evidence_chunks if c not in seen and admits(c)
            )
            top = max((r.score for r in local.relations), default=1.0) or 1.0
            for position, chunk_id in enumerate(local.evidence_chunks):
                chunk = evidence.get(chunk_id)
                if chunk is None:
                    continue
                out.append(
                    Candidate(
                        key=chunk.chunk_id,
                        kind=SourceKind.PASSAGE,
                        text=chunk.text,
                        score=round(0.5 / (1 + position), 6),
                        title=chunk.title,
                        source_uri=chunk.source_uri,
                        document_id=chunk.document_id,
                        document_version=chunk.document_version,
                        chunk_id=chunk.chunk_id,
                        digest=chunk.digest,
                        classification=chunk.classification,
                        trust=chunk.trust,
                    )
                )
            for ranked in local.relations:
                r = ranked.relation
                if r.relation_type == "related_to":
                    continue
                source, target = view.entities[r.source_id], view.entities[r.target_id]
                validity = f"valid from {r.valid_from.date().isoformat()}"
                if r.valid_to is not None:
                    validity += f" to {r.valid_to.date().isoformat()}"
                out.append(
                    Candidate(
                        key=f"{r.source_id}|{r.relation_type}|{r.target_id}",
                        kind=SourceKind.FACT,
                        text=f"{source.name} {phrase(r.relation_type)} {target.name} ({validity}).",
                        score=round(ranked.score / top, 6),
                        title=f"{source.name} / {target.name}",
                        classification=collection.classification,
                        trust=TrustLevel.EXTERNAL,
                        evidence_chunks=r.evidence_chunks,
                    )
                )
        if communities:
            top_score = communities[0].score or 1.0
            for ranked_community in communities:
                c = ranked_community.community
                out.append(
                    Candidate(
                        key=c.community_id,
                        kind=SourceKind.COMMUNITY,
                        text=c.summary,
                        score=round(ranked_community.score / top_score, 6),
                        title=f"Community level {c.level}: "
                        + ", ".join(
                            view.entities[e].name for e in c.key_entities[:3] if e in view.entities
                        ),
                        classification=collection.classification,
                        trust=TrustLevel.EXTERNAL,
                    )
                )
        return out

    async def _timed(
        self,
        trace: list[TraceStep],
        step: str,
        run: Callable[[], Awaitable[T]],
        describe: Callable[[T], dict[str, Any]],
    ) -> T:
        started = time.perf_counter()
        result = await run()
        elapsed = time.perf_counter() - started
        self._metrics.stage.labels(step).observe(elapsed)
        trace.append(
            TraceStep(step=step, duration_ms=round(elapsed * 1000, 3), detail=describe(result))
        )
        return result


async def _async[T](value: T) -> T:
    return value


def _coverage(query: str, passages: list[ScoredChunk]) -> float:
    wanted = set(terms(query))
    if not wanted:
        return 1.0
    present: set[str] = set()
    for p in passages[:3]:
        present |= set(terms(p.chunk.text))
    return len(wanted & present) / len(wanted)


def _expansion_terms(
    query: str, passages: list[ScoredChunk], view: GraphView, linked: dict[str, float]
) -> list[str]:
    """Linked-entity aliases plus pseudo-relevance feedback terms (RM3-style, top 3 passages)."""
    expansion: list[str] = []
    for entity_id in linked:
        entity = view.entities[entity_id]
        expansion.extend(a for a in entity.aliases if a.casefold() not in query.casefold())
    present = {w.casefold() for w in words(query)}
    counts: Counter[str] = Counter()
    for p in passages[:3]:
        for word in words(p.chunk.text):
            lowered = word.casefold()
            if (
                lowered not in STOPWORDS
                and lowered not in present
                and len(lowered) > 3
                and not lowered.isdigit()
            ):
                counts[lowered] += 1
    expansion.extend(w for w, _ in counts.most_common(5))
    return list(dict.fromkeys(expansion))[:8]


def _fuse_rounds(first: list[ScoredChunk], second: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    by_id = {p.chunk.chunk_id: p for p in [*second, *first]}
    fused = reciprocal_rank_fusion(
        {"round1": [p.chunk.chunk_id for p in first], "round2": [p.chunk.chunk_id for p in second]}
    )
    return [by_id[c] for c, _ in fused[:k]]
