"""Domain model: collections, documents, chunks, the knowledge graph and communities."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from agentic_core.domain.classification import DataClassification, TrustLevel
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

_COLLECTION = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_META_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _collection_name(value: str) -> str:
    if not _COLLECTION.fullmatch(value):
        raise ValueError("collection names are 2-63 chars of a-z, 0-9 and '-'")
    return value


def _document_id(value: str) -> str:
    if not _DOCUMENT_ID.fullmatch(value) or ".." in value:
        raise ValueError("document ids are 1-200 chars of A-Z a-z 0-9 . _ : / -")
    return value


def _metadata(value: dict[str, str | int | float | bool]) -> dict[str, str | int | float | bool]:
    if len(value) > 32:
        raise ValueError("at most 32 metadata keys")
    for key, item in value.items():
        if not _META_KEY.fullmatch(key):
            raise ValueError(f"invalid metadata key {key!r}")
        if isinstance(item, str) and len(item) > 512:
            raise ValueError(f"metadata value for {key!r} exceeds 512 characters")
    return value


CollectionName = Annotated[str, AfterValidator(_collection_name)]
DocumentId = Annotated[str, AfterValidator(_document_id)]
Metadata = Annotated[dict[str, str | int | float | bool], AfterValidator(_metadata)]


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"  # sliding token window with overlap
    SENTENCE = "sentence"  # pack whole sentences, overlap by sentences
    RECURSIVE = "recursive"  # split on sections, paragraphs, sentences, words


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    max_tokens: int = Field(default=256, ge=32, le=2048)
    overlap_tokens: int = Field(default=32, ge=0, le=512)

    @model_validator(mode="after")
    def _overlap(self) -> ChunkingConfig:
        if self.overlap_tokens * 2 > self.max_tokens:
            raise ValueError("overlap_tokens must be at most half of max_tokens")
        return self


class Collection(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: CollectionName
    description: str = Field(default="", max_length=2000)
    classification: DataClassification
    owner: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=0, ge=0)
    embedding_model: str
    embedding_dimension: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: CollectionName
    document_id: DocumentId
    version: int = Field(ge=1)
    title: str
    source_uri: str
    source_type: str
    classification: DataClassification
    trust: TrustLevel
    metadata: Metadata = Field(default_factory=dict)
    digest: str
    effective_at: datetime
    expires_at: datetime | None = None
    ingested_at: datetime
    ingested_by: str
    chunk_count: int = Field(ge=0)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    collection: str
    document_id: str
    document_version: int
    ordinal: int = Field(ge=0)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    digest: str
    classification: DataClassification
    trust: TrustLevel
    source_uri: str
    title: str
    quarantined: bool = False
    quarantine_reason: str | None = None


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    collection: str
    name: str
    normalized: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    mention_count: int = Field(default=0, ge=0)


class Mention(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    chunk_id: str
    document_id: str
    surface: str
    start: int
    end: int


class Assertion(BaseModel):
    """One relation stated by one document version, with its validity interval.

    ``valid_to`` is ``None`` while the assertion is current. A newer version of
    the source document closes it, so historical (``as_of``) queries still see
    what was known at the time.
    """

    model_config = ConfigDict(frozen=True)

    assertion_id: str
    collection: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = Field(gt=0)
    document_id: str
    document_version: int
    chunk_id: str
    evidence: str
    valid_from: datetime
    valid_to: datetime | None = None

    def valid_at(self, moment: datetime) -> bool:
        return self.valid_from <= moment and (self.valid_to is None or moment < self.valid_to)


class Relation(BaseModel):
    """Assertions with the same endpoints and type aggregated for one point in time."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    relation_type: str
    weight: float
    evidence_chunks: tuple[str, ...]
    evidence: tuple[str, ...]
    valid_from: datetime
    valid_to: datetime | None


class Community(BaseModel):
    model_config = ConfigDict(frozen=True)

    community_id: str
    collection: str
    level: int = Field(ge=0)
    entity_ids: tuple[str, ...]
    key_entities: tuple[str, ...]
    summary: str
    weight: float = Field(ge=0)
    built_revision: int
    built_at: datetime
