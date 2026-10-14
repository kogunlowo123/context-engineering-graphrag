"""Offline retrieval quality metrics against a golden set."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class GoldenQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    relevant_documents: frozenset[str] = Field(min_length=1, max_length=100)


class QueryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    retrieved_documents: tuple[str, ...]
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float


class RetrievalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: int
    queries: tuple[QueryMetrics, ...]
    mean_recall: float
    mean_precision: float
    mrr: float
    mean_ndcg: float


def dedupe(ranked: Sequence[str]) -> list[str]:
    """Collapse chunk-level hits to document-level ranks (first occurrence wins)."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    head = ranked[:k]
    return len(set(head) & relevant) / len(head) if head else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance nDCG with the log2(rank + 1) discount."""
    dcg = sum(
        1.0 / math.log2(r + 1) for r, item in enumerate(ranked[:k], start=1) if item in relevant
    )
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def score_query(query: GoldenQuery, ranked_documents: Sequence[str], k: int) -> QueryMetrics:
    ranked = dedupe(ranked_documents)
    return QueryMetrics(
        query=query.query,
        retrieved_documents=tuple(ranked[:k]),
        recall=round(recall_at_k(ranked, query.relevant_documents, k), 4),
        precision=round(precision_at_k(ranked, query.relevant_documents, k), 4),
        reciprocal_rank=round(reciprocal_rank(ranked[:k], query.relevant_documents), 4),
        ndcg=round(ndcg_at_k(ranked, query.relevant_documents, k), 4),
    )


def summarize(k: int, results: Sequence[QueryMetrics]) -> RetrievalReport:
    n = len(results) or 1
    return RetrievalReport(
        k=k,
        queries=tuple(results),
        mean_recall=round(sum(r.recall for r in results) / n, 4),
        mean_precision=round(sum(r.precision for r in results) / n, 4),
        mrr=round(sum(r.reciprocal_rank for r in results) / n, 4),
        mean_ndcg=round(sum(r.ndcg for r in results) / n, 4),
    )
