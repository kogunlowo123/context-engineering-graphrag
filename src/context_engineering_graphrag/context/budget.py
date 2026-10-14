"""Context engineering: compression, de-duplication and token budgeting.

Candidates come from three sources: retrieved passages, knowledge-graph
facts and community summaries. The budgeter

1. compresses each passage to at most ``per_item_tokens`` by keeping the
   sentences that cover the most query terms (in original order),
2. drops near-duplicates within a section (word 3-shingle containment >= 0.7),
3. allocates the token budget by section (facts and summaries are capped at
   a share; passages take the rest), packing items greedily in rank order,
   compressing an item further when only part of it fits,
4. hands budget left over by one section to items another section dropped.

Every source keeps its provenance; every dropped item is reported with the
reason, so the context an agent saw is fully explainable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from agentic_core.domain.classification import DataClassification, TrustLevel
from agentic_core.security import isolate_untrusted
from pydantic import BaseModel, ConfigDict, Field

from context_engineering_graphrag.domain.text import Tokenizer, shingles, split_sentences, terms


class SourceKind(StrEnum):
    PASSAGE = "passage"
    FACT = "fact"
    COMMUNITY = "community"


class BudgetPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(default=3000, ge=200, le=200_000)
    per_item_tokens: int = Field(default=320, ge=40, le=8000)
    fact_share: float = Field(default=0.2, ge=0.0, le=1.0)
    community_share: float = Field(default=0.3, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    min_fragment_tokens: int = Field(default=40, ge=8, le=1000)


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: SourceKind
    text: str
    score: float
    title: str
    source_uri: str | None = None
    document_id: str | None = None
    document_version: int | None = None
    chunk_id: str | None = None
    digest: str | None = None
    classification: DataClassification = DataClassification.INTERNAL
    trust: TrustLevel = TrustLevel.EXTERNAL
    evidence_chunks: tuple[str, ...] = ()


class ContextSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Citation label: S<n> passage, F<n> fact, C<n> community")
    kind: SourceKind
    text: str
    tokens: int
    score: float
    compressed: bool
    title: str
    key: str
    source_uri: str | None = None
    document_id: str | None = None
    document_version: int | None = None
    chunk_id: str | None = None
    digest: str | None = Field(
        default=None, description="sha256 of the full chunk the text came from"
    )
    classification: DataClassification
    trust: TrustLevel
    evidence_chunks: tuple[str, ...] = ()


class DroppedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    kind: SourceKind
    reason: str


class ContextPack(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    sources: tuple[ContextSource, ...]
    dropped: tuple[DroppedSource, ...]
    budget_tokens: int
    used_tokens: int

    @property
    def classification(self) -> DataClassification:
        return max((s.classification for s in self.sources), default=DataClassification.PUBLIC)

    def by_id(self) -> dict[str, ContextSource]:
        return {s.source_id: s for s in self.sources}

    def render(self, *, isolate: bool = True) -> str:
        """Prompt-ready context.

        With ``isolate`` every source is spotlighted as untrusted data between
        random boundaries. Callers that hand the context to the agentic-core
        content guard as EXTERNAL content render without it, because the guard
        isolates the whole block itself.
        """
        blocks = []
        for s in self.sources:
            header = f"[{s.source_id}] {s.kind.value}: {s.title}"
            if s.source_uri:
                header += f" <{s.source_uri}>"
            body = isolate_untrusted(s.text, source=s.source_id) if isolate else s.text
            blocks.append(header + "\n" + body)
        return "\n\n".join(blocks)


def compress(
    query_terms: set[str], text: str, limit: int, tokenizer: Tokenizer
) -> tuple[str, bool]:
    """Keep the most query-relevant sentences, in document order, within ``limit`` tokens."""
    if tokenizer.count(text) <= limit:
        return text, False
    sentences = split_sentences(text)
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: (-len(query_terms & set(terms(sentences[i].text))), i),
    )
    keep: list[int] = []
    used = 0
    for i in ranked:
        cost = tokenizer.count(sentences[i].text)
        if used + cost <= limit:
            keep.append(i)
            used += cost
    if not keep:
        words = sentences[ranked[0]].text.split() if sentences else text.split()
        clipped: list[str] = []
        for word in words:
            if tokenizer.count(" ".join([*clipped, word])) > limit:
                break
            clipped.append(word)
        return " ".join(clipped), True
    return " ".join(sentences[i].text for i in sorted(keep)), True


class ContextBudgeter:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        policy: BudgetPolicy,
    ) -> ContextPack:
        query_terms = set(terms(query))
        dropped: list[DroppedSource] = []
        prepared: list[tuple[Candidate, str, bool, int]] = []
        kept_shingles: dict[SourceKind, list[frozenset[tuple[str, ...]]]] = {
            k: [] for k in SourceKind
        }
        for cand in sorted(candidates, key=lambda c: (_section_order(c.kind), -c.score, c.key)):
            text, compressed = compress(query_terms, cand.text, policy.per_item_tokens, self._tok)
            grams = shingles(text)
            if any(
                _containment(grams, other) >= policy.duplicate_threshold
                for other in kept_shingles[cand.kind]
            ):
                dropped.append(DroppedSource(key=cand.key, kind=cand.kind, reason="duplicate"))
                continue
            kept_shingles[cand.kind].append(grams)
            prepared.append((cand, text, compressed, self._tok.count(text)))

        caps = {
            SourceKind.FACT: int(policy.max_tokens * policy.fact_share),
            SourceKind.COMMUNITY: int(policy.max_tokens * policy.community_share),
        }
        used = dict.fromkeys(SourceKind, 0)
        chosen: list[tuple[Candidate, str, bool, int]] = []
        leftovers: list[tuple[Candidate, str, bool, int]] = []
        total = 0
        for kind in (SourceKind.FACT, SourceKind.COMMUNITY, SourceKind.PASSAGE):
            for item in (p for p in prepared if p[0].kind is kind):
                cap = caps.get(kind, policy.max_tokens)
                room = min(cap - used[kind], policy.max_tokens - total)
                fitted = self._fit(query_terms, item, room, policy)
                if fitted is None:
                    leftovers.append(item)
                    continue
                chosen.append(fitted)
                used[kind] += fitted[3]
                total += fitted[3]
        for item in sorted(leftovers, key=lambda p: -p[0].score):
            fitted = self._fit(query_terms, item, policy.max_tokens - total, policy)
            if fitted is None:
                dropped.append(DroppedSource(key=item[0].key, kind=item[0].kind, reason="budget"))
                continue
            chosen.append(fitted)
            total += fitted[3]

        counters = {SourceKind.PASSAGE: 0, SourceKind.FACT: 0, SourceKind.COMMUNITY: 0}
        prefix = {SourceKind.PASSAGE: "S", SourceKind.FACT: "F", SourceKind.COMMUNITY: "C"}
        sources: list[ContextSource] = []
        for cand, text, compressed, tokens in sorted(
            chosen, key=lambda p: (_section_order(p[0].kind), -p[0].score, p[0].key)
        ):
            counters[cand.kind] += 1
            sources.append(
                ContextSource(
                    source_id=f"{prefix[cand.kind]}{counters[cand.kind]}",
                    kind=cand.kind,
                    text=text,
                    tokens=tokens,
                    score=round(cand.score, 6),
                    compressed=compressed,
                    title=cand.title,
                    key=cand.key,
                    source_uri=cand.source_uri,
                    document_id=cand.document_id,
                    document_version=cand.document_version,
                    chunk_id=cand.chunk_id,
                    digest=cand.digest,
                    classification=cand.classification,
                    trust=cand.trust,
                    evidence_chunks=cand.evidence_chunks,
                )
            )
        return ContextPack(
            query=query,
            sources=tuple(sources),
            dropped=tuple(dropped),
            budget_tokens=policy.max_tokens,
            used_tokens=total,
        )

    def _fit(
        self,
        query_terms: set[str],
        item: tuple[Candidate, str, bool, int],
        room: int,
        policy: BudgetPolicy,
    ) -> tuple[Candidate, str, bool, int] | None:
        cand, text, _compressed, tokens = item
        if tokens <= room:
            return item
        if room < policy.min_fragment_tokens or cand.kind is SourceKind.FACT:
            return None
        shorter, _ = compress(query_terms, text, room, self._tok)
        cost = self._tok.count(shorter)
        if not shorter or cost > room:
            return None
        return cand, shorter, True, cost


def _section_order(kind: SourceKind) -> int:
    return {SourceKind.PASSAGE: 0, SourceKind.FACT: 1, SourceKind.COMMUNITY: 2}[kind]


def _containment(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
