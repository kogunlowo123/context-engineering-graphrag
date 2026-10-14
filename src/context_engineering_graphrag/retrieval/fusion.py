"""Rank fusion and diversity selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Cormack, Clarke and Buettcher (2009): ``score(d) = sum_i w_i / (k + rank_i(d))``.

    ``rankings`` maps a retriever name to its ranked list of ids (best
    first, rank 1). Ties are broken by best individual rank, then id, so the
    output is deterministic.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for name, ranked in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(ranked, start=1):
            scores[item] += weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


def maximal_marginal_relevance(
    relevance: Sequence[float],
    vectors: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    *,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Carbonell and Goldstein (1998) MMR over L2-normalised vectors.

    Greedily picks the candidate maximising
    ``lambda * rel(i) - (1 - lambda) * max_{j in selected} cos(i, j)``.
    Relevance is min-max normalised first so both terms share a scale.
    Returns indices into ``relevance`` in selection order.
    """
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be within [0, 1]")
    rel = np.asarray(relevance, dtype=np.float64)
    spread = float(rel.max() - rel.min())
    rel = (rel - rel.min()) / spread if spread > 0 else np.ones(n)
    sims = np.asarray(vectors, dtype=np.float64) @ np.asarray(vectors, dtype=np.float64).T
    selected: list[int] = []
    max_sim = np.full(n, -np.inf)
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_index = -1
        best_value = -np.inf
        for i in sorted(remaining):
            redundancy = 0.0 if not selected else float(max_sim[i])
            value = lambda_ * float(rel[i]) - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value = i, value
        selected.append(best_index)
        remaining.discard(best_index)
        max_sim = np.maximum(max_sim, sims[best_index])
    return selected
