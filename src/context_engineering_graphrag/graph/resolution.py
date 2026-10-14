"""Entity resolution: blocking, pairwise similarity, clustering.

1. **Blocking** keeps the comparison count near-linear: only names sharing a
   key are compared. Keys: the normalised name, the name with legal or
   organisational suffixes removed, the sorted token set, the acronym of a
   multi-word name (``International Business Machines`` -> ``ibm``) and a
   distinctive first token.
2. **Similarity** combines exact core-name equality, acronym expansion,
   Jaro-Winkler on the core names and stemmed-token Jaccard; incompatible
   specific types (person vs organisation) never match.
3. **Clustering** is union-find over pairs above the threshold, i.e. the
   connected components of the match graph. The canonical name of a cluster
   is its most-mentioned, then longest, then alphabetically first name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from context_engineering_graphrag.domain.text import STOPWORDS, normalize, stem

_SUFFIXES = frozenset(
    [
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
        "ltd",
        "llc",
        "plc",
        "gmbh",
        "ag",
        "sa",
        "group",
        "holdings",
        "limited",
        "incorporated",
        "the",
    ]
)
_GENERIC_FIRST = frozenset(
    {"new", "north", "south", "east", "west", "united", "national", "general"}
)
GENERIC_TYPE = "entity"


def core_name(name: str) -> str:
    tokens = normalize(name).split()
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    while len(tokens) > 1 and tokens[0] in _SUFFIXES:
        tokens.pop(0)
    return " ".join(tokens)


def acronym(name: str) -> str | None:
    """Acronym of a multi-word name, or the name itself when it already is one."""
    raw = name.strip()
    compact = raw.replace(".", "").replace("&", "")
    if 2 <= len(compact) <= 10 and compact.isalnum() and compact.isupper():
        return compact.casefold()
    tokens = [t for t in core_name(name).split() if t not in STOPWORDS]
    if len(tokens) < 2:
        return None
    return "".join(t[0] for t in tokens)


def blocking_keys(name: str) -> set[str]:
    norm = normalize(name)
    core = core_name(name)
    keys = {f"n:{norm}", f"c:{core}", "t:" + " ".join(sorted(set(core.split())))}
    abbreviation = acronym(name)
    if abbreviation and len(abbreviation) >= 2:
        keys.add(f"a:{abbreviation}")
    first = core.split()[0] if core else ""
    if len(first) >= 4 and first not in _GENERIC_FIRST and first not in STOPWORDS:
        keys.add(f"f:{first}")
    return keys


def jaro_winkler(left: str, right: str, *, prefix_scale: float = 0.1) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    window = max(0, max(len(left), len(right)) // 2 - 1)
    left_flags = [False] * len(left)
    right_flags = [False] * len(right)
    matches = 0
    for i, char in enumerate(left):
        lo, hi = max(0, i - window), min(len(right), i + window + 1)
        for j in range(lo, hi):
            if not right_flags[j] and right[j] == char:
                left_flags[i] = right_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    left_matched = [c for c, f in zip(left, left_flags, strict=True) if f]
    right_matched = [c for c, f in zip(right, right_flags, strict=True) if f]
    transpositions = sum(a != b for a, b in zip(left_matched, right_matched, strict=True)) / 2
    jaro = (matches / len(left) + matches / len(right) + (matches - transpositions) / matches) / 3
    prefix = 0
    for a, b in zip(left[:4], right[:4], strict=False):
        if a != b:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1 - jaro)


def types_compatible(left: str, right: str) -> bool:
    return left == right or GENERIC_TYPE in (left, right)


def name_similarity(left: str, right: str) -> float:
    left_core, right_core = core_name(left), core_name(right)
    if not left_core or not right_core:
        return 0.0
    if left_core == right_core:
        return 1.0
    left_acr, right_acr = acronym(left), acronym(right)
    left_is_acr = left_core.replace(" ", "") == left_acr
    right_is_acr = right_core.replace(" ", "") == right_acr
    if (left_is_acr and left_acr == right_acr and not right_is_acr) or (
        right_is_acr and right_acr == left_acr and not left_is_acr
    ):
        return 0.92
    left_terms = {stem(t) for t in left_core.split()}
    right_terms = {stem(t) for t in right_core.split()}
    jaccard = len(left_terms & right_terms) / len(left_terms | right_terms)
    jw = jaro_winkler(left_core, right_core)
    return max(
        jaccard, 0.5 * jw + 0.5 * jaccard, jw if min(len(left_core), len(right_core)) >= 8 else 0.0
    )


@dataclass(frozen=True, slots=True)
class NamedItem:
    key: str
    name: str
    entity_type: str
    weight: int = 1


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {i: i for i in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[max(a, b)] = min(a, b)


class EntityResolver:
    def __init__(self, threshold: float = 0.88) -> None:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0.5, 1.0]")
        self.threshold = threshold

    def similarity(self, left: NamedItem, right: NamedItem) -> float:
        if not types_compatible(left.entity_type, right.entity_type):
            return 0.0
        return name_similarity(left.name, right.name)

    def best_match(self, item: NamedItem, candidates: Sequence[NamedItem]) -> NamedItem | None:
        scored = [(self.similarity(item, c), c) for c in candidates]
        eligible = [(s, c) for s, c in scored if s >= self.threshold]
        if not eligible:
            return None
        return max(eligible, key=lambda sc: (sc[0], sc[1].weight, -len(sc[1].key), sc[1].key))[1]

    def cluster(self, items: Sequence[NamedItem]) -> list[list[NamedItem]]:
        """Connected components of the match graph, each sorted canonical-first."""
        by_key = {i.key: i for i in items}
        blocks: dict[str, list[str]] = defaultdict(list)
        for item in items:
            for block in blocking_keys(item.name):
                blocks[block].append(item.key)
        forest = _UnionFind(by_key)
        compared: set[tuple[str, str]] = set()
        for members in blocks.values():
            if len(members) < 2:
                continue
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    pair = (min(left, right), max(left, right))
                    if pair in compared:
                        continue
                    compared.add(pair)
                    if self.similarity(by_key[left], by_key[right]) >= self.threshold:
                        forest.union(left, right)
        groups: dict[str, list[NamedItem]] = defaultdict(list)
        for item in items:
            groups[forest.find(item.key)].append(item)
        return sorted(
            (sorted(g, key=canonical_order) for g in groups.values()),
            key=lambda g: g[0].key,
        )


def canonical_order(item: NamedItem) -> tuple[int, int, str]:
    return (-item.weight, -len(item.name), item.name)
