"""Second-stage rerankers behind a cross-encoder interface.

A cross-encoder scores each (query, passage) pair jointly instead of
comparing independently computed vectors. ``HttpCrossEncoder`` calls a
hosted cross-encoder model (the Text Embeddings Inference ``/rerank`` wire
format, e.g. ``BAAI/bge-reranker-v2-m3``). ``LexicalCrossEncoder`` is a
deterministic pairwise scorer used offline and in tests: query-term
coverage weighted by idf, term proximity and phrase overlap.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from context_engineering_graphrag.domain.errors import DependencyUnavailableError
from context_engineering_graphrag.domain.text import terms


class CrossEncoder(Protocol):
    @property
    def name(self) -> str: ...

    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    async def aclose(self) -> None: ...


class LexicalCrossEncoder:
    """Scores in [0, 1]: ``0.6 coverage + 0.25 proximity + 0.15 phrase``."""

    @property
    def name(self) -> str:
        return "lexical-cross-encoder-v1"

    async def aclose(self) -> None:
        return None

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_terms = list(dict.fromkeys(terms(query)))
        if not query_terms:
            return [0.0 for _ in passages]
        passage_terms = [terms(p) for p in passages]
        n = len(passages)
        df = {t: sum(1 for pt in passage_terms if t in pt) for t in query_terms}
        # idf over the candidate set, smoothed so a term in every passage still counts.
        weights = {t: math.log(1.0 + (n + 1) / (df[t] + 0.5)) for t in query_terms}
        total = sum(weights.values())
        query_bigrams = set(itertools.pairwise(query_terms))
        return [self._pair(query_terms, weights, total, query_bigrams, pt) for pt in passage_terms]

    @staticmethod
    def _pair(
        query_terms: list[str],
        weights: dict[str, float],
        total: float,
        query_bigrams: set[tuple[str, str]],
        passage: list[str],
    ) -> float:
        present = {t for t in query_terms if t in passage}
        if not present:
            return 0.0
        coverage = sum(weights[t] for t in present) / total
        proximity = _proximity(passage, present)
        if query_bigrams:
            passage_bigrams = set(itertools.pairwise(passage))
            phrase = len(query_bigrams & passage_bigrams) / len(query_bigrams)
        else:
            phrase = 1.0
        return round(0.6 * coverage + 0.25 * proximity + 0.15 * phrase, 6)


def _proximity(passage: list[str], present: set[str]) -> float:
    """``|present| / length of the shortest window containing every present term``."""
    if len(present) == 1:
        return 1.0
    counts: dict[str, int] = {}
    best = math.inf
    left = 0
    covered = 0
    for right, term in enumerate(passage):
        if term in present:
            counts[term] = counts.get(term, 0) + 1
            if counts[term] == 1:
                covered += 1
        while covered == len(present):
            best = min(best, right - left + 1)
            leaving = passage[left]
            if leaving in present:
                counts[leaving] -= 1
                if counts[leaving] == 0:
                    covered -= 1
            left += 1
    return min(1.0, len(present) / best)


class HttpCrossEncoder:
    """Client for a hosted cross-encoder exposing ``POST /rerank``."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            transport=transport,
        )
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        try:
            response = await self._client.post(
                "/rerank", json={"query": query, "texts": list(passages), "raw_scores": False}
            )
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError(f"reranker: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise DependencyUnavailableError(f"reranker: provider returned {response.status_code}")
        try:
            items = response.json()
            scores = [0.0] * len(passages)
            seen = set()
            for item in items:
                index = int(item["index"])
                if not 0 <= index < len(passages) or index in seen:
                    raise ValueError("index out of range")
                seen.add(index)
                scores[index] = float(item["score"])
        except (ValueError, KeyError, TypeError) as exc:
            raise DependencyUnavailableError("reranker: malformed response") from exc
        if len(seen) != len(passages):
            raise DependencyUnavailableError("reranker: response did not score every passage")
        return scores
