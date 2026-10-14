"""Ingestion pipeline: content controls, chunking, embedding, graph extraction,
incremental entity resolution and versioned, temporally valid assertions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from agentic_core.domain.classification import DataClassification, TrustLevel
from agentic_core.domain.provenance import content_digest
from agentic_core.security import GuardConfig, PromptInjectionDetector, SecretScanner
from agentic_core.security.patterns import redact
from pydantic import BaseModel, ConfigDict, Field, model_validator

from context_engineering_graphrag.domain.errors import ConflictError, PlatformError
from context_engineering_graphrag.domain.models import (
    Assertion,
    Chunk,
    ChunkingConfig,
    Collection,
    Document,
    DocumentId,
    Entity,
    Mention,
    Metadata,
)
from context_engineering_graphrag.domain.ports import Clock, Embedder
from context_engineering_graphrag.domain.text import Tokenizer, normalize
from context_engineering_graphrag.graph.extraction import EntityExtractor, ExtractedEntity
from context_engineering_graphrag.graph.resolution import EntityResolver, NamedItem, blocking_keys
from context_engineering_graphrag.infrastructure.database import DocumentWrite, SqlStore
from context_engineering_graphrag.ingestion.chunking import Chunker

MAX_DOCUMENT_CHARS = 2_000_000


class IngestDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: DocumentId
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    source_uri: str = Field(min_length=1, max_length=2000)
    source_type: str = Field(default="document", pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    classification: DataClassification = DataClassification.INTERNAL
    trust: TrustLevel = Field(
        default=TrustLevel.EXTERNAL,
        description="Origin trust. SYSTEM is reserved for platform-authored content.",
    )
    metadata: Metadata = Field(default_factory=dict)
    effective_at: datetime | None = Field(
        default=None, description="When the facts in this version take effect (default: now)"
    )
    expires_at: datetime | None = Field(default=None, description="When the facts stop being valid")
    chunking: ChunkingConfig | None = None

    @model_validator(mode="after")
    def _interval(self) -> IngestDocument:
        if self.trust is TrustLevel.SYSTEM:
            raise ValueError("trust SYSTEM cannot be asserted by a caller")
        for value in (self.effective_at, self.expires_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("timestamps must include a timezone offset")
        if self.effective_at and self.expires_at and self.expires_at <= self.effective_at:
            raise ValueError("expires_at must be after effective_at")
        return self


class IngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: str
    document_id: str
    version: int
    digest: str
    status: str = Field(description="created, updated or unchanged")
    chunks: int
    quarantined_chunks: int
    redacted_secrets: int
    entities_created: int
    mentions: int
    assertions: int


@dataclass(frozen=True)
class PreparedIngest:
    write: DocumentWrite | None
    result: IngestResult
    quarantine_reasons: tuple[str, ...]


def chunk_id(collection: str, document_id: str, version: int, ordinal: int) -> str:
    raw = f"{collection}\x1f{document_id}\x1f{version}\x1f{ordinal}".encode()
    return "chk_" + hashlib.sha256(raw).hexdigest()[:32]


def entity_id(collection: str, normalized: str, entity_type: str) -> str:
    raw = f"{collection}\x1f{normalized}\x1f{entity_type}".encode()
    return "ent_" + hashlib.sha256(raw).hexdigest()[:32]


def assertion_id(*parts: object) -> str:
    raw = "\x1f".join(str(p) for p in parts).encode()
    return "asr_" + hashlib.sha256(raw).hexdigest()[:32]


class IngestionPipeline:
    def __init__(
        self,
        store: SqlStore,
        embedder: Embedder,
        *,
        extractor: EntityExtractor,
        resolver: EntityResolver,
        tokenizer: Tokenizer,
        clock: Clock,
        default_chunking: ChunkingConfig,
        guard_config: GuardConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._extractor = extractor
        self._resolver = resolver
        self._tokenizer = tokenizer
        self._clock = clock
        self._default_chunking = default_chunking
        self._thresholds = (guard_config or GuardConfig()).block_threshold
        self._injection = PromptInjectionDetector()
        self._secrets = SecretScanner()

    async def prepare(
        self, collection: Collection, doc: IngestDocument, actor: str
    ) -> PreparedIngest:
        if doc.classification > collection.classification:
            raise PlatformError(
                f"document classification {doc.classification.name} exceeds collection "
                f"classification {collection.classification.name}"
            )
        now = self._clock.now()
        digest = content_digest(doc.text)
        existing = await self._store.get_document(collection.name, doc.document_id)
        effective_at = doc.effective_at or now
        if existing is not None:
            if (
                existing.digest == digest
                and existing.title == doc.title
                and existing.metadata == doc.metadata
                and existing.classification == doc.classification
                and existing.expires_at == doc.expires_at
            ):
                return PreparedIngest(
                    None,
                    IngestResult(
                        collection=collection.name,
                        document_id=doc.document_id,
                        version=existing.version,
                        digest=digest,
                        status="unchanged",
                        chunks=existing.chunk_count,
                        quarantined_chunks=0,
                        redacted_secrets=0,
                        entities_created=0,
                        mentions=0,
                        assertions=0,
                    ),
                    (),
                )
            if effective_at < existing.effective_at:
                raise ConflictError(
                    "effective_at of a new version must not precede the current version "
                    f"({existing.effective_at.isoformat()})"
                )
        version = 1 if existing is None else existing.version + 1

        secret_findings = self._secrets.detect(doc.text)
        body = redact(doc.text, secret_findings) if secret_findings else doc.text
        drafts = Chunker(doc.chunking or self._default_chunking, self._tokenizer).chunk(body)

        chunks: list[Chunk] = []
        reasons: list[str] = []
        threshold = self._thresholds[doc.trust]
        for draft in drafts:
            assessment = self._injection.assess(draft.text)
            quarantined = assessment.score >= threshold
            reason = f"prompt_injection:{','.join(assessment.kinds)}" if quarantined else None
            if reason:
                reasons.append(reason)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(collection.name, doc.document_id, version, draft.ordinal),
                    collection=collection.name,
                    document_id=doc.document_id,
                    document_version=version,
                    ordinal=draft.ordinal,
                    text=draft.text,
                    start=draft.start,
                    end=draft.end,
                    token_count=draft.token_count,
                    digest=content_digest(draft.text),
                    classification=doc.classification,
                    trust=doc.trust,
                    source_uri=doc.source_uri,
                    title=doc.title,
                    quarantined=quarantined,
                    quarantine_reason=reason,
                )
            )

        clean = [c for c in chunks if not c.quarantined]
        vectors = (
            await self._embedder.embed([c.title + "\n" + c.text for c in clean]) if clean else []
        )
        by_chunk = dict(zip((c.chunk_id for c in clean), vectors, strict=True))
        zero = [0.0] * self._embedder.dimension
        embeddings = [by_chunk.get(c.chunk_id, zero) for c in chunks]

        graph = await self._extract_graph(collection, doc, version, clean, effective_at)
        document = Document(
            collection=collection.name,
            document_id=doc.document_id,
            version=version,
            title=doc.title,
            source_uri=doc.source_uri,
            source_type=doc.source_type,
            classification=doc.classification,
            trust=doc.trust,
            metadata=doc.metadata,
            digest=digest,
            effective_at=effective_at,
            expires_at=doc.expires_at,
            ingested_at=now,
            ingested_by=actor,
            chunk_count=len(chunks),
        )
        write = DocumentWrite(
            document=document,
            chunks=chunks,
            embeddings=embeddings,
            new_entities=graph.new_entities,
            entity_keys=graph.keys,
            mention_count_delta=graph.mention_delta,
            alias_additions=graph.aliases,
            mentions=graph.mentions,
            assertions=graph.assertions,
            supersede_at=effective_at if existing is not None else None,
        )
        result = IngestResult(
            collection=collection.name,
            document_id=doc.document_id,
            version=version,
            digest=digest,
            status="created" if existing is None else "updated",
            chunks=len(chunks),
            quarantined_chunks=len(chunks) - len(clean),
            redacted_secrets=len(secret_findings),
            entities_created=len(graph.new_entities),
            mentions=len(graph.mentions),
            assertions=len(graph.assertions),
        )
        return PreparedIngest(write, result, tuple(reasons))

    async def _extract_graph(
        self,
        collection: Collection,
        doc: IngestDocument,
        version: int,
        chunks: list[Chunk],
        effective_at: datetime,
    ) -> _GraphDelta:
        delta = _GraphDelta()
        seen_mentions: set[tuple[str, int]] = set()
        seen_assertions: set[str] = set()
        for chunk in chunks:
            extraction = await self._extractor.extract(
                chunk.text, classification=doc.classification
            )
            surface_to_id: dict[str, str] = {}
            for found in extraction.entities:
                resolved = await self._resolve(collection.name, found, delta)
                surface_to_id[found.surface.casefold()] = resolved
                absolute = chunk.start + found.start
                if (resolved, absolute) in seen_mentions:
                    continue
                seen_mentions.add((resolved, absolute))
                delta.mention_delta[resolved] += 1
                delta.mentions.append(
                    Mention(
                        entity_id=resolved,
                        chunk_id=chunk.chunk_id,
                        document_id=doc.document_id,
                        surface=found.surface,
                        start=absolute,
                        end=chunk.start + found.end,
                    )
                )
            for relation in extraction.relations:
                source = surface_to_id.get(relation.source.casefold())
                target = surface_to_id.get(relation.target.casefold())
                if source is None or target is None or source == target:
                    continue
                identifier = assertion_id(
                    collection.name,
                    doc.document_id,
                    version,
                    source,
                    target,
                    relation.relation_type,
                    relation.evidence,
                )
                if identifier in seen_assertions:
                    continue
                seen_assertions.add(identifier)
                delta.assertions.append(
                    Assertion(
                        assertion_id=identifier,
                        collection=collection.name,
                        source_id=source,
                        target_id=target,
                        relation_type=relation.relation_type,
                        weight=relation.weight,
                        document_id=doc.document_id,
                        document_version=version,
                        chunk_id=chunk.chunk_id,
                        evidence=relation.evidence[:2000],
                        valid_from=effective_at,
                        valid_to=doc.expires_at,
                    )
                )
        for entity in delta.new_entities:
            delta.keys[entity.entity_id] |= blocking_keys(entity.name)
        delta.new_entities = [
            e.model_copy(
                update={
                    "mention_count": delta.mention_delta.get(e.entity_id, 0),
                    "aliases": tuple(sorted(delta.aliases.get(e.entity_id, set()))),
                }
            )
            for e in delta.new_entities
        ]
        return delta

    async def _resolve(self, collection: str, found: ExtractedEntity, delta: _GraphDelta) -> str:
        item = NamedItem(key="", name=found.surface, entity_type=found.entity_type)
        keys = blocking_keys(found.surface)
        local = [
            NamedItem(e.entity_id, e.name, e.entity_type, delta.mention_delta.get(e.entity_id, 0))
            for e in delta.new_entities
            if keys & delta.keys[e.entity_id]
        ]
        stored = [
            NamedItem(e.entity_id, e.name, e.entity_type, e.mention_count)
            for e in await self._store.entity_candidates(collection, keys)
        ]
        match = self._resolver.best_match(item, local + stored)
        if match is not None:
            if normalize(found.surface) != normalize(match.name):
                delta.aliases[match.key].add(found.surface)
            delta.keys[match.key] |= keys
            return match.key
        normalized = normalize(found.surface)
        new_id = entity_id(collection, normalized, found.entity_type)
        delta.new_entities.append(
            Entity(
                entity_id=new_id,
                collection=collection,
                name=found.surface,
                normalized=normalized,
                entity_type=found.entity_type,
            )
        )
        delta.keys[new_id] |= keys
        return new_id


class _GraphDelta:
    def __init__(self) -> None:
        self.new_entities: list[Entity] = []
        self.keys: defaultdict[str, set[str]] = defaultdict(set)
        self.mention_delta: defaultdict[str, int] = defaultdict(int)
        self.aliases: defaultdict[str, set[str]] = defaultdict(set)
        self.mentions: list[Mention] = []
        self.assertions: list[Assertion] = []
