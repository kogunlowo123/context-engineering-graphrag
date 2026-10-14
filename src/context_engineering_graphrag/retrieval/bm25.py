"""Okapi BM25 over stemmed, stop-word-filtered terms with an inverted index."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence


class BM25Index:
    """Immutable BM25 index. Build once per collection revision.

    ``k1`` controls term-frequency saturation and ``b`` length
    normalisation; the defaults are the values most systems use.
    The idf variant is the Lucene one, ``ln(1 + (N - df + 0.5) / (df + 0.5))``,
    which is never negative.
    """

    def __init__(
        self,
        documents: Iterable[tuple[str, Sequence[str]]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc_terms in documents:
            index = len(self._ids)
            self._ids.append(doc_id)
            self._lengths.append(len(doc_terms))
            for term, tf in Counter(doc_terms).items():
                self._postings[term].append((index, tf))
        total = sum(self._lengths)
        self._avgdl = total / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def idf(self, term: str) -> float:
        n = len(self._ids)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query_terms: Sequence[str],
        k: int,
        *,
        allow: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._ids or not query_terms or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(query_terms).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for index, tf in postings:
                norm = 1.0 - self.b + self.b * self._lengths[index] / (self._avgdl or 1.0)
                scores[index] += qtf * idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        candidates = (
            (score, index)
            for index, score in scores.items()
            if allow is None or allow(self._ids[index])
        )
        best = heapq.nlargest(k, candidates, key=lambda item: (item[0], -item[1]))
        return [(self._ids[index], score) for score, index in best]
