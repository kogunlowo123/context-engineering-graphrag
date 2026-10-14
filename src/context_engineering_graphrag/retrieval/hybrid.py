"""Hybrid retrieval: BM25 + dense vectors, reciprocal rank fusion, cross-encoder
reranking and MMR diversification.

Indexes are built per collection from the database and cached together with
the collection revision they reflect. Before serving a query the retriever
reads the current revision (one indexed primary-key lookup) and rebuilds the
cache if another replica has written since, so every replica serves the same
content without a message bus.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from agentic_core.domain.classification import DataClassification

from context_engineering_graphrag.domain.models import Chunk
from context_engineering_graphrag.domain.ports import Clock, Embedder
from context_engineering_graphrag.domain.text import terms
from context_engineering_graphrag.infrastructure.database import IndexedChunk, SqlStore
from context_engineering_graphrag.retrieval.bm25 import BM25Index
from context_engineering_graphrag.retrieval.fusion import (
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from context_engineering_graphrag.retrieval.rerank import CrossEncoder


@dataclass(frozen=True)
class RetrievalFilters:
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    document_ids: frozenset[str] | None = None
    max_classification: DataClassification | None = None
    as_of: datetime | None = None


@dataclass(frozen=True)
class RetrievalOptions:
    k: int = 8
    candidates: int = 50
    rerank_depth: int = 30
    rerank: bool = True
    diversify: bool = True
    mmr_lambda: float = 0.7
    rrf_k: int = 60
    weights: Mapping[str, float] = field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0})


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    bm25: float | None
    bm25_rank: int | None
    dense: float | None
    dense_rank: int | None
    fused: float
    rerank: float | None
    score: float


@dataclass
class CollectionIndex:
    revision: int
    items: list[IndexedChunk]
    by_id: dict[str, int]
    bm25: BM25Index
    matrix: npt.NDArray[np.float32]
    built_ms: float


class HybridRetriever:
    def __init__(
        self,
        store: SqlStore,
        embedder: Embedder,
        reranker: CrossEncoder,
        clock: Clock,
        *,
        on_reload: Callable[[str, float], None] | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._clock = clock
        self._cache: dict[str, CollectionIndex] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._on_reload = on_reload

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        return self._reranker

    async def index(self, collection: str) -> CollectionIndex:
        revision = await self._store.revision(collection)
        cached = self._cache.get(collection)
        if cached is not None and cached.revision == revision:
            return cached
        lock = self._locks.setdefault(collection, asyncio.Lock())
        async with lock:
            cached = self._cache.get(collection)
            revision = await self._store.revision(collection)
            if cached is not None and cached.revision == revision:
                return cached
            started = time.perf_counter()
            items = await self._store.indexed_chunks(collection)
            dimension = self._embedder.dimension
            matrix = (
                np.vstack([i.embedding for i in items]).astype(np.float32)
                if items
                else np.zeros((0, dimension), dtype=np.float32)
            )
            built = CollectionIndex(
                revision=revision,
                items=items,
                by_id={i.chunk.chunk_id: n for n, i in enumerate(items)},
                bm25=BM25Index((i.chunk.chunk_id, i.terms) for i in items),
                matrix=matrix,
                built_ms=(time.perf_counter() - started) * 1000,
            )
            self._cache[collection] = built
            if self._on_reload is not None:
                self._on_reload(collection, built.built_ms / 1000)
            return built

    async def admits(self, collection: str, filters: RetrievalFilters) -> Callable[[str], bool]:
        """Predicate applying the same filters as search (validity, metadata, ids,
        classification); unknown or quarantined chunk ids are never admitted."""
        idx = await self.index(collection)
        allow = self._allow(idx, filters)
        return lambda chunk_id: chunk_id in idx.by_id and allow(chunk_id)

    def invalidate(self, collection: str) -> None:
        self._cache.pop(collection, None)

    async def embed_query(self, query: str) -> npt.NDArray[np.float32]:
        return np.asarray((await self._embedder.embed([query]))[0], dtype=np.float32)

    async def search(
        self,
        collection: str,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        options: RetrievalOptions | None = None,
    ) -> list[ScoredChunk]:
        filters = filters or RetrievalFilters()
        options = options or RetrievalOptions()
        idx = await self.index(collection)
        if not idx.items:
            return []
        allow = self._allow(idx, filters)
        depth = max(options.candidates, options.k)
        lexical = idx.bm25.search(terms(query), depth, allow=allow)
        qvec = await self.embed_query(query)
        dense = await self._dense(collection, idx, qvec, depth, allow)
        fused = reciprocal_rank_fusion(
            {"bm25": [c for c, _ in lexical], "dense": [c for c, _ in dense]},
            k=options.rrf_k,
            weights=options.weights,
        )[:depth]
        if not fused:
            return []
        lex = {c: (s, r) for r, (c, s) in enumerate(lexical, start=1)}
        den = {c: (s, r) for r, (c, s) in enumerate(dense, start=1)}
        top_fused = fused[0][1]
        scored: list[ScoredChunk] = []
        rerank_scores: dict[str, float] = {}
        if options.rerank:
            head = [c for c, _ in fused[: options.rerank_depth]]
            values = await self._reranker.score(
                query, [idx.items[idx.by_id[c]].chunk.text for c in head]
            )
            rerank_scores = dict(zip(head, values, strict=True))
        for chunk_id, fused_score in fused:
            relative = fused_score / top_fused
            rr = rerank_scores.get(chunk_id)
            final = (
                relative
                if not options.rerank
                else (0.8 * rr + 0.2 * relative if rr is not None else 0.2 * relative)
            )
            scored.append(
                ScoredChunk(
                    chunk=idx.items[idx.by_id[chunk_id]].chunk,
                    bm25=lex[chunk_id][0] if chunk_id in lex else None,
                    bm25_rank=lex[chunk_id][1] if chunk_id in lex else None,
                    dense=den[chunk_id][0] if chunk_id in den else None,
                    dense_rank=den[chunk_id][1] if chunk_id in den else None,
                    fused=fused_score,
                    rerank=rr,
                    score=round(final, 6),
                )
            )
        scored.sort(key=lambda s: (-s.score, -s.fused, s.chunk.chunk_id))
        if not options.diversify or len(scored) <= options.k:
            return scored[: options.k]
        vectors = np.vstack([idx.matrix[idx.by_id[s.chunk.chunk_id]] for s in scored])
        order = maximal_marginal_relevance(
            [s.score for s in scored], vectors, k=options.k, lambda_=options.mmr_lambda
        )
        return [scored[i] for i in order]

    async def _dense(
        self,
        collection: str,
        idx: CollectionIndex,
        qvec: npt.NDArray[np.float32],
        depth: int,
        allow: Callable[[str], bool],
    ) -> list[tuple[str, float]]:
        if self._store.pgvector:
            hits = await self._store.vector_search(collection, qvec.tolist(), depth * 3)
            return [(c, s) for c, s in hits if c in idx.by_id and allow(c)][:depth]
        sims = idx.matrix @ qvec
        order = np.argsort(-sims, kind="stable")
        out: list[tuple[str, float]] = []
        for row in order:
            chunk_id = idx.items[int(row)].chunk.chunk_id
            if allow(chunk_id):
                out.append((chunk_id, float(sims[int(row)])))
                if len(out) >= depth:
                    break
        return out

    def _allow(self, idx: CollectionIndex, filters: RetrievalFilters) -> Callable[[str], bool]:
        moment = filters.as_of or self._clock.now()

        def allow(chunk_id: str) -> bool:
            item = idx.items[idx.by_id[chunk_id]]
            chunk = item.chunk
            if filters.document_ids is not None and chunk.document_id not in filters.document_ids:
                return False
            if (
                filters.max_classification is not None
                and chunk.classification > filters.max_classification
            ):
                return False
            if item.effective_at > moment or (
                item.expires_at is not None and item.expires_at <= moment
            ):
                return False
            return all(_meta_equal(item.metadata.get(k), v) for k, v in filters.metadata.items())

        return allow


def _meta_equal(actual: Any, expected: Any) -> bool:
    return actual is not None and type(actual) is type(expected) and actual == expected
