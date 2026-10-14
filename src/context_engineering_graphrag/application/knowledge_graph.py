"""Knowledge-graph use cases: materialised views, build pipeline, neighbourhoods."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from context_engineering_graphrag.domain.models import Collection, Community, Entity, Relation
from context_engineering_graphrag.domain.ports import Clock
from context_engineering_graphrag.domain.text import normalize
from context_engineering_graphrag.graph.community import build_graph, louvain, modularity
from context_engineering_graphrag.graph.resolution import GENERIC_TYPE, EntityResolver, NamedItem
from context_engineering_graphrag.graph.search import GraphView, aggregate_relations
from context_engineering_graphrag.graph.summarize import CommunityFacts, CommunitySummarizer
from context_engineering_graphrag.infrastructure.database import SqlStore


class GraphBuildOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution: float = Field(default=1.0, gt=0.0, le=10.0, description="Louvain resolution")
    max_levels: int = Field(default=4, ge=1, le=8)
    min_community_size: int = Field(default=2, ge=2, le=100)


class LevelReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: int
    communities: int
    modularity: float


class GraphBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    merged_entities: int
    entities: int
    relations: int
    levels: tuple[LevelReport, ...]
    communities: int
    summarizer: str
    duration_ms: float


class Neighbourhood(BaseModel):
    model_config = ConfigDict(frozen=True)

    center: Entity
    hops: int
    as_of: datetime
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


class KnowledgeGraph:
    def __init__(
        self,
        store: SqlStore,
        resolver: EntityResolver,
        summarizer: CommunitySummarizer,
        clock: Clock,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._summarizer = summarizer
        self._clock = clock
        self._views: dict[str, tuple[int, GraphView]] = {}
        self._lock = asyncio.Lock()

    @property
    def summarizer(self) -> CommunitySummarizer:
        return self._summarizer

    async def view(self, collection: str, *, as_of: datetime | None = None) -> GraphView:
        """Graph valid at ``as_of`` (default now). Current views are cached per revision."""
        if as_of is not None:
            return await self._materialise(collection, as_of)
        revision = await self._store.revision(collection)
        cached = self._views.get(collection)
        if cached is not None and cached[0] == revision:
            return cached[1]
        async with self._lock:
            view = await self._materialise(collection, self._clock.now())
            self._views[collection] = (revision, view)
            return view

    async def _materialise(self, collection: str, at: datetime) -> GraphView:
        entities = await self._store.entities(collection)
        relations = aggregate_relations(await self._store.assertions(collection, as_of=at), at)
        return GraphView(entities, relations)

    async def neighbourhood(
        self, collection: str, entity_id: str, *, hops: int, as_of: datetime | None
    ) -> Neighbourhood | None:
        view = await self.view(collection, as_of=as_of)
        center = view.entities.get(entity_id)
        if center is None:
            return None
        depth = {entity_id: 0}
        queue = deque([entity_id])
        while queue:
            node = queue.popleft()
            if depth[node] >= hops:
                continue
            for other in sorted(view.adjacency[node]):
                if other not in depth:
                    depth[other] = depth[node] + 1
                    queue.append(other)
        members = set(depth)
        return Neighbourhood(
            center=center,
            hops=hops,
            as_of=as_of or self._clock.now(),
            entities=tuple(
                sorted(
                    (view.entities[e] for e in members), key=lambda e: (depth[e.entity_id], e.name)
                )
            ),
            relations=tuple(
                r for r in view.relations if r.source_id in members and r.target_id in members
            ),
        )

    async def find_entities(self, collection: str, *, name: str | None, limit: int) -> list[Entity]:
        entities = await self._store.entities(collection)
        if name:
            wanted = normalize(name)
            entities = [
                e
                for e in entities
                if wanted in e.normalized or any(wanted in normalize(a) for a in e.aliases)
            ]
        entities.sort(key=lambda e: (-e.mention_count, e.name))
        return entities[:limit]

    async def build(self, collection: Collection, options: GraphBuildOptions) -> GraphBuildReport:
        started = time.perf_counter()
        merged = await self._resolve(collection.name)
        now = self._clock.now()
        entities = await self._store.entities(collection.name)
        relations = aggregate_relations(
            await self._store.assertions(collection.name, as_of=now), now
        )
        view = GraphView(entities, relations)
        graph = build_graph(view.edges(), sorted(view.entities))
        levels = louvain(graph, resolution=options.resolution, max_levels=options.max_levels)
        revision = await self._store.revision(collection.name)
        communities: list[Community] = []
        reports: list[LevelReport] = []
        for level, partition in enumerate(levels):
            groups: dict[int, list[str]] = {}
            for node, community in partition.items():
                groups.setdefault(community, []).append(node)
            kept = 0
            for members in groups.values():
                if len(members) < options.min_community_size:
                    continue
                member_set = set(members)
                internal = tuple(
                    r
                    for r in view.relations
                    if r.source_id in member_set and r.target_id in member_set
                )
                if not internal:
                    continue
                facts = CommunityFacts(
                    members=tuple(view.entities[m] for m in sorted(members)),
                    relations=internal,
                    classification=collection.classification,
                )
                key_entities = tuple(e.entity_id for e, _ in facts.centrality()[:5])
                summary = await self._summarizer.summarize(facts)
                digest = hashlib.sha256(
                    f"{collection.name}\x1f{level}\x1f{','.join(sorted(members))}".encode()
                ).hexdigest()[:32]
                communities.append(
                    Community(
                        community_id=f"com_{digest}",
                        collection=collection.name,
                        level=level,
                        entity_ids=tuple(sorted(members)),
                        key_entities=key_entities,
                        summary=summary,
                        weight=round(sum(r.weight for r in internal), 6),
                        built_revision=revision + 1,
                        built_at=now,
                    )
                )
                kept += 1
            reports.append(
                LevelReport(
                    level=level,
                    communities=kept,
                    modularity=round(
                        modularity(graph, partition, resolution=options.resolution), 6
                    ),
                )
            )
        await self._store.replace_communities(collection.name, communities, now)
        return GraphBuildReport(
            collection=collection.name,
            merged_entities=merged,
            entities=len(view.entities),
            relations=len(view.relations),
            levels=tuple(reports),
            communities=len(communities),
            summarizer=self._summarizer.name,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def _resolve(self, collection: str) -> int:
        entities = {e.entity_id: e for e in await self._store.entities(collection)}
        items = [
            NamedItem(e.entity_id, e.name, e.entity_type, e.mention_count)
            for e in entities.values()
        ]
        merges: dict[str, str] = {}
        survivors: list[Entity] = []
        for cluster in self._resolver.cluster(items):
            if len(cluster) < 2:
                continue
            head = entities[cluster[0].key]
            aliases = set(head.aliases)
            mentions = head.mention_count
            entity_type = head.entity_type
            for other in cluster[1:]:
                merges[other.key] = head.entity_id
                absorbed = entities[other.key]
                aliases |= {absorbed.name, *absorbed.aliases}
                mentions += absorbed.mention_count
                if entity_type == GENERIC_TYPE:
                    entity_type = absorbed.entity_type
            aliases.discard(head.name)
            survivors.append(
                head.model_copy(
                    update={
                        "aliases": tuple(sorted(aliases)),
                        "mention_count": mentions,
                        "entity_type": entity_type,
                    }
                )
            )
        await self._store.merge_entities(collection, merges, survivors, self._clock.now())
        return len(merges)
