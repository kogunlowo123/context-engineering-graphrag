"""GraphRAG search over the temporal knowledge graph.

* **Local search** links query mentions to entities, spreads relevance over
  the graph with personalised PageRank from those seeds, and returns the
  highest-ranked entities, the relations among them and the chunks that
  evidence them.
* **Global search** ranks community summaries against the query with BM25
  (map), then the caller reduces the best summaries into context.

The graph is materialised for one point in time (``as_of``) from assertions
whose validity interval contains it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from context_engineering_graphrag.domain.models import Assertion, Community, Entity, Relation
from context_engineering_graphrag.domain.text import normalize, terms
from context_engineering_graphrag.graph.resolution import jaro_winkler
from context_engineering_graphrag.retrieval.bm25 import BM25Index

RELATION_PHRASES: Mapping[str, str] = {
    "acquired": "acquired",
    "merged_with": "merged with",
    "partner_of": "is a partner of",
    "part_of": "is part of",
    "owns": "owns",
    "depends_on": "depends on",
    "located_in": "is located in",
    "founded": "founded",
    "leads": "leads",
    "works_for": "works for",
    "supplies": "supplies",
    "regulates": "regulates",
    "invested_in": "invested in",
    "replaced": "replaced",
    "competes_with": "competes with",
    "reports_to": "reports to",
    "related_to": "is mentioned with",
}


def phrase(relation_type: str) -> str:
    return RELATION_PHRASES.get(relation_type, relation_type.replace("_", " "))


def aggregate_relations(assertions: Iterable[Assertion], at: datetime) -> list[Relation]:
    grouped: dict[tuple[str, str, str], list[Assertion]] = defaultdict(list)
    for item in assertions:
        if item.valid_at(at):
            grouped[(item.source_id, item.target_id, item.relation_type)].append(item)
    out: list[Relation] = []
    for (source, target, kind), items in sorted(grouped.items()):
        evidence = tuple(dict.fromkeys(i.evidence for i in items))[:3]
        open_ended = [i.valid_to for i in items]
        out.append(
            Relation(
                source_id=source,
                target_id=target,
                relation_type=kind,
                weight=sum(i.weight for i in items),
                evidence_chunks=tuple(sorted({i.chunk_id for i in items})),
                evidence=evidence,
                valid_from=min(i.valid_from for i in items),
                valid_to=(
                    None
                    if any(v is None for v in open_ended)
                    else max(v for v in open_ended if v is not None)
                ),
            )
        )
    return out


class GraphView:
    def __init__(self, entities: Sequence[Entity], relations: Sequence[Relation]) -> None:
        self.entities = {e.entity_id: e for e in entities}
        self.relations = [
            r for r in relations if r.source_id in self.entities and r.target_id in self.entities
        ]
        self.adjacency: dict[str, dict[str, float]] = {e: {} for e in self.entities}
        for r in self.relations:
            self.adjacency[r.source_id][r.target_id] = (
                self.adjacency[r.source_id].get(r.target_id, 0.0) + r.weight
            )
            self.adjacency[r.target_id][r.source_id] = (
                self.adjacency[r.target_id].get(r.source_id, 0.0) + r.weight
            )
        self._names: dict[str, list[str]] = defaultdict(list)
        for entity in self.entities.values():
            for name in {entity.normalized, *(normalize(a) for a in entity.aliases)}:
                if name:
                    self._names[name].append(entity.entity_id)

    def edges(self) -> list[tuple[str, str, float]]:
        return [
            (r.source_id, r.target_id, r.weight)
            for r in self.relations
            if r.source_id != r.target_id
        ]

    def link(self, query: str, *, fuzzy_threshold: float = 0.93) -> dict[str, float]:
        """Entities mentioned in the query, scored by match quality and n-gram length."""
        tokens = normalize(query).split()
        found: dict[str, float] = {}
        max_n = min(6, len(tokens))
        for n in range(max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                if n == 1 and len(gram) < 3:
                    continue
                exact = self._names.get(gram)
                if exact:
                    for entity_id in exact:
                        found[entity_id] = max(found.get(entity_id, 0.0), 1.0 + 0.1 * n)
                    continue
                if len(gram) < 5:
                    continue
                for name, ids in self._names.items():
                    if len(name.split()) == n and jaro_winkler(gram, name) >= fuzzy_threshold:
                        for entity_id in ids:
                            found.setdefault(entity_id, 0.8 + 0.1 * n)
        return found


def personalized_pagerank(
    adjacency: Mapping[str, Mapping[str, float]],
    seeds: Mapping[str, float],
    *,
    damping: float = 0.85,
    iterations: int = 60,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Power iteration of PageRank with restarts to the (normalised) seed distribution.

    Dangling nodes return their mass to the seeds, so scores always sum to 1.
    """
    nodes = list(adjacency)
    seed_total = sum(w for n, w in seeds.items() if n in adjacency)
    if not nodes or seed_total <= 0:
        return {}
    restart = {n: seeds.get(n, 0.0) / seed_total for n in nodes}
    out_weight = {n: sum(adjacency[n].values()) for n in nodes}
    rank = dict(restart)
    for _ in range(iterations):
        dangling = sum(rank[n] for n in nodes if out_weight[n] == 0)
        nxt = {n: (1 - damping) * restart[n] + damping * dangling * restart[n] for n in nodes}
        for node in nodes:
            if out_weight[node] == 0:
                continue
            share = damping * rank[node] / out_weight[node]
            for other, weight in adjacency[node].items():
                nxt[other] += share * weight
        delta = sum(abs(nxt[n] - rank[n]) for n in nodes)
        rank = nxt
        if delta < tolerance:
            break
    return rank


@dataclass(frozen=True)
class RankedEntity:
    entity: Entity
    score: float
    seed: bool


@dataclass(frozen=True)
class RankedRelation:
    relation: Relation
    score: float


@dataclass(frozen=True)
class LocalSearchResult:
    entities: tuple[RankedEntity, ...]
    relations: tuple[RankedRelation, ...]
    evidence_chunks: tuple[str, ...]


def local_search(
    view: GraphView,
    seeds: Mapping[str, float],
    *,
    max_entities: int = 12,
    max_relations: int = 20,
    max_chunks: int = 12,
) -> LocalSearchResult:
    if not seeds:
        return LocalSearchResult((), (), ())
    ranks = personalized_pagerank(view.adjacency, seeds)
    ordered = sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [(e, s) for e, s in ordered if s > 0][:max_entities]
    chosen = {e for e, _ in top}
    ranked_entities = tuple(RankedEntity(view.entities[e], s, e in seeds) for e, s in top)
    scored_relations = sorted(
        (
            RankedRelation(
                r, r.weight * (ranks.get(r.source_id, 0.0) + ranks.get(r.target_id, 0.0))
            )
            for r in view.relations
            if r.source_id in chosen and r.target_id in chosen
        ),
        key=lambda rr: (
            -rr.score,
            rr.relation.source_id,
            rr.relation.target_id,
            rr.relation.relation_type,
        ),
    )[:max_relations]
    chunk_scores: dict[str, float] = defaultdict(float)
    for rr in scored_relations:
        for chunk_id in rr.relation.evidence_chunks:
            chunk_scores[chunk_id] += rr.score
    evidence = tuple(
        c for c, _ in sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:max_chunks]
    )
    return LocalSearchResult(ranked_entities, tuple(scored_relations), evidence)


@dataclass(frozen=True)
class RankedCommunity:
    community: Community
    score: float


def rank_communities(
    query: str, communities: Sequence[Community], entity_names: Mapping[str, str], *, top_k: int = 5
) -> list[RankedCommunity]:
    """BM25 over each community's summary plus its key entity names."""
    if not communities:
        return []
    by_id = {c.community_id: c for c in communities}
    index = BM25Index(
        (
            c.community_id,
            terms(c.summary + " " + " ".join(entity_names.get(e, "") for e in c.key_entities)),
        )
        for c in communities
    )
    hits = index.search(terms(query), top_k)
    return [RankedCommunity(by_id[cid], score) for cid, score in hits if score > 0]
