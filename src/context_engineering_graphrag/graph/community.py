"""Hierarchical community detection with the Louvain method.

Blondel, Guillaume, Lambiotte and Lefebvre (2008). Phase one moves single
nodes to the neighbouring community with the largest modularity gain
``k_i,in - gamma * Sigma_tot * k_i / 2m`` until no move improves modularity;
phase two collapses communities into super-nodes (internal edges become
self-loops) and repeats. Every pass yields one level of the hierarchy: level
0 is the finest partition, each higher level coarser. Nodes are visited in
sorted order, so results are deterministic for a given graph.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping

Graph = Mapping[str, Mapping[str, float]]


def build_graph(
    edges: list[tuple[str, str, float]], nodes: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Undirected weighted adjacency; parallel edges add up, self-loops are dropped."""
    adj: dict[str, dict[str, float]] = {n: {} for n in nodes or ()}
    for u, v, w in edges:
        if w <= 0:
            raise ValueError("edge weights must be positive")
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0.0) + w
        adj[v][u] = adj[v].get(u, 0.0) + w
    return adj


def _degree[N: Hashable](adj: Mapping[N, Mapping[N, float]], node: N) -> float:
    neighbours = adj[node]
    return sum(neighbours.values()) + neighbours.get(node, 0.0)


def modularity(
    graph: Graph, partition: Mapping[str, Hashable], *, resolution: float = 1.0
) -> float:
    m2 = sum(_degree(graph, n) for n in graph)
    if m2 == 0:
        return 0.0
    internal: dict[Hashable, float] = defaultdict(float)
    total: dict[Hashable, float] = defaultdict(float)
    for node, neighbours in graph.items():
        community = partition[node]
        total[community] += _degree(graph, node)
        for other, weight in neighbours.items():
            if partition[other] == community:
                internal[community] += weight * (2.0 if other == node else 1.0)
    return sum(internal[c] / m2 - resolution * (total[c] / m2) ** 2 for c in total)


def _one_level(adj: dict[int, dict[int, float]], resolution: float) -> dict[int, int]:
    nodes = sorted(adj)
    node_to_com = {n: n for n in nodes}
    degree = {n: _degree(adj, n) for n in nodes}
    total = {n: degree[n] for n in nodes}
    m2 = sum(degree.values())
    if m2 == 0:
        return node_to_com
    improved = True
    while improved:
        improved = False
        for node in nodes:
            current = node_to_com[node]
            links: dict[int, float] = defaultdict(float)
            for other, weight in adj[node].items():
                if other != node:
                    links[node_to_com[other]] += weight
            total[current] -= degree[node]
            best = current
            best_gain = links.get(current, 0.0) - resolution * total[current] * degree[node] / m2
            for community in sorted(links):
                gain = links[community] - resolution * total[community] * degree[node] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = community, gain
            total[best] += degree[node]
            if best != current:
                node_to_com[node] = best
                improved = True
    return node_to_com


def _aggregate(
    adj: dict[int, dict[int, float]], node_to_com: dict[int, int]
) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {c: {} for c in set(node_to_com.values())}
    for u, neighbours in adj.items():
        cu = node_to_com[u]
        for v, w in neighbours.items():
            if v < u:
                continue
            cv = node_to_com[v]
            if cu == cv:
                out[cu][cu] = out[cu].get(cu, 0.0) + w
            else:
                out[cu][cv] = out[cu].get(cv, 0.0) + w
                out[cv][cu] = out[cv].get(cu, 0.0) + w
    return out


def louvain(graph: Graph, *, resolution: float = 1.0, max_levels: int = 6) -> list[dict[str, int]]:
    """Partitions of the original nodes, finest first.

    Community ids are dense and ordered by each community's smallest node
    name. A graph where no move improves modularity yields one level in
    which every node is its own community.
    """
    names = sorted(graph)
    if not names:
        return []
    index = {n: i for i, n in enumerate(names)}
    adj: dict[int, dict[int, float]] = {
        index[u]: {index[v]: w for v, w in graph[u].items()} for u in names
    }
    super_of = {n: index[n] for n in names}
    levels: list[dict[str, int]] = []
    for _ in range(max_levels):
        node_to_com = _one_level(adj, resolution)
        moved = any(node_to_com[s] != s for s in adj)
        if not moved and levels:
            break
        raw = {n: node_to_com[super_of[n]] for n in names}
        dense = _renumber(raw, names)
        label_to_dense = {raw[n]: dense[n] for n in names}
        adj = _aggregate(adj, {s: label_to_dense[c] for s, c in node_to_com.items()})
        super_of = dense
        levels.append(dense)
        if not moved or len(adj) == 1:
            break
    return levels


def _renumber(membership: dict[str, int], names: list[str]) -> dict[str, int]:
    ids: dict[int, int] = {}
    out: dict[str, int] = {}
    for name in names:  # names are sorted, so ids follow each community's smallest member
        community = membership[name]
        if community not in ids:
            ids[community] = len(ids)
        out[name] = ids[community]
    return out
