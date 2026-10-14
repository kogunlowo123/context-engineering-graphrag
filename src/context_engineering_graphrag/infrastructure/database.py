"""SQLAlchemy Core persistence. PostgreSQL in production, SQLite in tests.

Embeddings are stored with each chunk as little-endian float32 bytes, so any
replica can rebuild its in-memory indexes from the database. With
``vector_backend=pgvector`` they are also written to a ``chunk_vectors``
table with an HNSW cosine index, and dense search runs in PostgreSQL.

Every write that changes what retrieval can see increments the collection
``revision`` in the same transaction; replicas compare revisions to know when
their caches are stale.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from agentic_core.domain.classification import DataClassification, TrustLevel
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.types import UserDefinedType

from context_engineering_graphrag.domain.errors import ConflictError, NotFoundError
from context_engineering_graphrag.domain.models import (
    Assertion,
    Chunk,
    Collection,
    Community,
    Document,
    Entity,
    Mention,
)
from context_engineering_graphrag.domain.text import terms

metadata = MetaData()

collections = Table(
    "collections",
    metadata,
    Column("name", String(63), primary_key=True),
    Column("description", Text, nullable=False),
    Column("classification", Integer, nullable=False),
    Column("owner", String(200), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    Column("embedding_dimension", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

documents = Table(
    "documents",
    metadata,
    Column("collection", String(63), primary_key=True),
    Column("document_id", String(200), primary_key=True),
    Column("version", Integer, nullable=False),
    Column("title", Text, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("source_type", String(64), nullable=False),
    Column("classification", Integer, nullable=False),
    Column("trust", Integer, nullable=False),
    Column("metadata", Text, nullable=False),
    Column("digest", String(80), nullable=False),
    Column("effective_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("ingested_by", Text, nullable=False),
    Column("chunk_count", Integer, nullable=False),
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", String(40), primary_key=True),
    Column("collection", String(63), nullable=False),
    Column("document_id", String(200), nullable=False),
    Column("document_version", Integer, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("digest", String(80), nullable=False),
    Column("classification", Integer, nullable=False),
    Column("trust", Integer, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("quarantined", Boolean, nullable=False),
    Column("quarantine_reason", Text, nullable=True),
    Column("terms", Text, nullable=False),
    Column("embedding", LargeBinary, nullable=False),
    Index("ix_chunks_collection_document", "collection", "document_id"),
)

entities = Table(
    "entities",
    metadata,
    Column("entity_id", String(40), primary_key=True),
    Column("collection", String(63), nullable=False),
    Column("name", Text, nullable=False),
    Column("normalized", Text, nullable=False),
    Column("entity_type", String(32), nullable=False),
    Column("aliases", Text, nullable=False),
    Column("mention_count", Integer, nullable=False),
    Index("ix_entities_collection", "collection"),
)

entity_keys = Table(
    "entity_keys",
    metadata,
    Column("collection", String(63), primary_key=True),
    Column("key", String(200), primary_key=True),
    Column("entity_id", String(40), primary_key=True),
)

mentions = Table(
    "mentions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("collection", String(63), nullable=False),
    Column("entity_id", String(40), nullable=False),
    Column("chunk_id", String(40), nullable=False),
    Column("document_id", String(200), nullable=False),
    Column("surface", Text, nullable=False),
    Column("start_offset", Integer, nullable=False),
    Column("end_offset", Integer, nullable=False),
    Index("ix_mentions_entity", "collection", "entity_id"),
    Index("ix_mentions_document", "collection", "document_id"),
)

assertions = Table(
    "assertions",
    metadata,
    Column("assertion_id", String(40), primary_key=True),
    Column("collection", String(63), nullable=False),
    Column("source_id", String(40), nullable=False),
    Column("target_id", String(40), nullable=False),
    Column("relation_type", String(64), nullable=False),
    Column("weight", Float, nullable=False),
    Column("document_id", String(200), nullable=False),
    Column("document_version", Integer, nullable=False),
    Column("chunk_id", String(40), nullable=False),
    Column("evidence", Text, nullable=False),
    Column("valid_from", DateTime(timezone=True), nullable=False),
    Column("valid_to", DateTime(timezone=True), nullable=True),
    Index("ix_assertions_collection", "collection", "valid_to"),
    Index("ix_assertions_document", "collection", "document_id"),
)

communities = Table(
    "communities",
    metadata,
    Column("community_id", String(40), primary_key=True),
    Column("collection", String(63), nullable=False),
    Column("level", Integer, nullable=False),
    Column("entity_ids", Text, nullable=False),
    Column("key_entities", Text, nullable=False),
    Column("summary", Text, nullable=False),
    Column("weight", Float, nullable=False),
    Column("built_revision", Integer, nullable=False),
    Column("built_at", DateTime(timezone=True), nullable=False),
    Index("ix_communities_collection", "collection", "level"),
)

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
)

SCHEMA_VERSION = 1  # bump with an additive migration when a column is added


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


async def create_schema(engine: AsyncEngine, *, pgvector_dimension: int | None = None) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        current = (await conn.execute(select(schema_version.c.version))).scalar()
        if current is None:
            await conn.execute(insert(schema_version).values(version=SCHEMA_VERSION))
        if pgvector_dimension is not None:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(vector_table(pgvector_dimension).metadata.create_all)


class Vector(UserDefinedType[Any]):
    """pgvector column type ``vector(n)``."""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        if not 1 <= dimension <= 16_000:
            raise ValueError("vector dimension out of range")
        self.dimension = dimension

    def get_col_spec(self, **_: Any) -> str:
        return f"vector({self.dimension})"


def vector_table(dimension: int) -> Table:
    """``chunk_vectors`` with an HNSW cosine index (PostgreSQL + pgvector only)."""
    table = Table(
        "chunk_vectors",
        MetaData(),
        Column("chunk_id", String(40), primary_key=True),
        Column("collection", String(63), nullable=False),
        Column("embedding", Vector(dimension), nullable=False),
    )
    Index(
        "ix_chunk_vectors_hnsw",
        table.c.embedding,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    return table


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _opt_aware(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


def encode_vector(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def decode_vector(blob: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.frombuffer(blob, dtype="<f4")


def _pg_vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]"


@dataclass
class DocumentWrite:
    """Everything one ingestion writes, applied atomically."""

    document: Document
    chunks: list[Chunk]
    embeddings: list[list[float]]
    new_entities: list[Entity]
    entity_keys: dict[str, set[str]]
    mention_count_delta: dict[str, int]
    alias_additions: dict[str, set[str]]
    mentions: list[Mention]
    assertions: list[Assertion]
    supersede_at: datetime | None = None


class ErasureReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: int
    mentions: int
    assertions: int
    entities: int
    communities: int


def index_terms(chunk: Chunk) -> list[str]:
    """BM25 terms of a chunk (title and body), computed once at ingestion."""
    return terms(chunk.title + "\n" + chunk.text)


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    terms: list[str]
    embedding: np.ndarray[Any, np.dtype[np.float32]]
    metadata: dict[str, Any]
    effective_at: datetime
    expires_at: datetime | None


class SqlStore:
    def __init__(self, engine: AsyncEngine, *, pgvector: bool = False) -> None:
        self._engine = engine
        self._pgvector = pgvector

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def pgvector(self) -> bool:
        return self._pgvector

    # ----------------------------------------------------------- collections

    async def create_collection(self, collection: Collection) -> None:
        async with self._engine.begin() as conn:
            exists = (
                await conn.execute(
                    select(collections.c.name).where(collections.c.name == collection.name)
                )
            ).first()
            if exists:
                raise ConflictError(f"collection {collection.name} already exists")
            await conn.execute(
                insert(collections).values(
                    name=collection.name,
                    description=collection.description,
                    classification=int(collection.classification),
                    owner=collection.owner,
                    revision=collection.revision,
                    embedding_model=collection.embedding_model,
                    embedding_dimension=collection.embedding_dimension,
                    created_at=collection.created_at,
                    updated_at=collection.updated_at,
                )
            )

    async def get_collection(self, name: str) -> Collection | None:
        async with self._engine.connect() as conn:
            row = (
                (await conn.execute(select(collections).where(collections.c.name == name)))
                .mappings()
                .first()
            )
        return _collection(row) if row else None

    async def require_collection(self, name: str) -> Collection:
        found = await self.get_collection(name)
        if found is None:
            raise NotFoundError(f"collection {name} not found")
        return found

    async def list_collections(self) -> list[Collection]:
        async with self._engine.connect() as conn:
            rows = (
                (await conn.execute(select(collections).order_by(collections.c.name)))
                .mappings()
                .all()
            )
        return [_collection(r) for r in rows]

    async def revision(self, name: str) -> int:
        async with self._engine.connect() as conn:
            value = (
                await conn.execute(select(collections.c.revision).where(collections.c.name == name))
            ).scalar()
        if value is None:
            raise NotFoundError(f"collection {name} not found")
        return int(value)

    @staticmethod
    async def _bump(conn: AsyncConnection, name: str, at: datetime) -> None:
        await conn.execute(
            update(collections)
            .where(collections.c.name == name)
            .values(revision=collections.c.revision + 1, updated_at=at)
        )

    # ------------------------------------------------------------- documents

    async def get_document(self, collection: str, document_id: str) -> Document | None:
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(documents).where(
                            documents.c.collection == collection,
                            documents.c.document_id == document_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
        return _document(row) if row else None

    async def list_documents(self, collection: str, *, limit: int, offset: int) -> list[Document]:
        query = (
            select(documents)
            .where(documents.c.collection == collection)
            .order_by(documents.c.document_id)
            .limit(limit)
            .offset(offset)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [_document(r) for r in rows]

    async def write_document(self, write: DocumentWrite) -> None:
        doc = write.document
        async with self._engine.begin() as conn:
            existing = (
                await conn.execute(
                    select(documents.c.version).where(
                        documents.c.collection == doc.collection,
                        documents.c.document_id == doc.document_id,
                    )
                )
            ).scalar()
            if existing is not None and int(existing) != doc.version - 1:
                raise ConflictError(
                    f"document {doc.document_id} changed concurrently (version {existing})"
                )
            if existing is not None:
                await self._remove_document_rows(conn, doc.collection, doc.document_id)
                await conn.execute(
                    update(assertions)
                    .where(
                        assertions.c.collection == doc.collection,
                        assertions.c.document_id == doc.document_id,
                        assertions.c.valid_to.is_(None),
                    )
                    .values(valid_to=write.supersede_at or doc.effective_at)
                )
                await conn.execute(
                    delete(documents).where(
                        documents.c.collection == doc.collection,
                        documents.c.document_id == doc.document_id,
                    )
                )
            await conn.execute(
                insert(documents).values(
                    collection=doc.collection,
                    document_id=doc.document_id,
                    version=doc.version,
                    title=doc.title,
                    source_uri=doc.source_uri,
                    source_type=doc.source_type,
                    classification=int(doc.classification),
                    trust=int(doc.trust),
                    metadata=json.dumps(doc.metadata, sort_keys=True),
                    digest=doc.digest,
                    effective_at=doc.effective_at,
                    expires_at=doc.expires_at,
                    ingested_at=doc.ingested_at,
                    ingested_by=doc.ingested_by,
                    chunk_count=doc.chunk_count,
                )
            )
            if write.chunks:
                await conn.execute(
                    insert(chunks),
                    [
                        {
                            "chunk_id": c.chunk_id,
                            "collection": c.collection,
                            "document_id": c.document_id,
                            "document_version": c.document_version,
                            "ordinal": c.ordinal,
                            "text": c.text,
                            "start_offset": c.start,
                            "end_offset": c.end,
                            "token_count": c.token_count,
                            "digest": c.digest,
                            "classification": int(c.classification),
                            "trust": int(c.trust),
                            "source_uri": c.source_uri,
                            "title": c.title,
                            "quarantined": c.quarantined,
                            "quarantine_reason": c.quarantine_reason,
                            "terms": " ".join(index_terms(c)),
                            "embedding": encode_vector(e),
                        }
                        for c, e in zip(write.chunks, write.embeddings, strict=True)
                    ],
                )
                if self._pgvector:
                    for c, e in zip(write.chunks, write.embeddings, strict=True):
                        if c.quarantined:
                            continue
                        await conn.execute(
                            text(
                                "INSERT INTO chunk_vectors (chunk_id, collection, embedding)"
                                " VALUES (:chunk_id, :collection, CAST(:embedding AS vector))"
                            ),
                            {
                                "chunk_id": c.chunk_id,
                                "collection": c.collection,
                                "embedding": _pg_vector_literal(e),
                            },
                        )
            await self._write_entities(conn, doc.collection, write)
            if write.mentions:
                await conn.execute(
                    insert(mentions),
                    [
                        {
                            "collection": doc.collection,
                            "entity_id": m.entity_id,
                            "chunk_id": m.chunk_id,
                            "document_id": m.document_id,
                            "surface": m.surface,
                            "start_offset": m.start,
                            "end_offset": m.end,
                        }
                        for m in write.mentions
                    ],
                )
            if write.assertions:
                await conn.execute(
                    insert(assertions), [_assertion_row(a) for a in write.assertions]
                )
            await self._bump(conn, doc.collection, doc.ingested_at)

    async def _write_entities(
        self, conn: AsyncConnection, collection: str, w: DocumentWrite
    ) -> None:
        if w.new_entities:
            await conn.execute(
                insert(entities),
                [
                    {
                        "entity_id": e.entity_id,
                        "collection": e.collection,
                        "name": e.name,
                        "normalized": e.normalized,
                        "entity_type": e.entity_type,
                        "aliases": json.dumps(sorted(e.aliases)),
                        "mention_count": e.mention_count,
                    }
                    for e in w.new_entities
                ],
            )
        new_ids = {e.entity_id for e in w.new_entities}
        for entity_id, delta in w.mention_count_delta.items():
            if entity_id in new_ids:
                continue
            await conn.execute(
                update(entities)
                .where(entities.c.entity_id == entity_id)
                .values(mention_count=entities.c.mention_count + delta)
            )
        for entity_id, added in w.alias_additions.items():
            if entity_id in new_ids or not added:
                continue
            current = (
                await conn.execute(
                    select(entities.c.aliases).where(entities.c.entity_id == entity_id)
                )
            ).scalar()
            merged = sorted(set(json.loads(current or "[]")) | added)
            await conn.execute(
                update(entities)
                .where(entities.c.entity_id == entity_id)
                .values(aliases=json.dumps(merged))
            )
        await self._add_keys(conn, collection, w.entity_keys)

    @staticmethod
    async def _add_keys(conn: AsyncConnection, collection: str, keys: dict[str, set[str]]) -> None:
        for entity_id, entity_key_set in keys.items():
            if not entity_key_set:
                continue
            present = set(
                (
                    await conn.execute(
                        select(entity_keys.c.key).where(
                            entity_keys.c.collection == collection,
                            entity_keys.c.entity_id == entity_id,
                        )
                    )
                ).scalars()
            )
            rows = [
                {"collection": collection, "key": k, "entity_id": entity_id}
                for k in sorted(entity_key_set - present)
            ]
            if rows:
                await conn.execute(insert(entity_keys), rows)

    async def _remove_document_rows(
        self, conn: AsyncConnection, collection: str, document_id: str
    ) -> tuple[int, int]:
        """Delete chunks and mentions of a document and decrement entity mention counts."""
        counts = (
            await conn.execute(
                select(mentions.c.entity_id, func.count())
                .where(mentions.c.collection == collection, mentions.c.document_id == document_id)
                .group_by(mentions.c.entity_id)
            )
        ).all()
        for entity_id, count in counts:
            await conn.execute(
                update(entities)
                .where(entities.c.entity_id == entity_id)
                .values(mention_count=entities.c.mention_count - int(count))
            )
        removed_mentions = (
            await conn.execute(
                delete(mentions).where(
                    mentions.c.collection == collection, mentions.c.document_id == document_id
                )
            )
        ).rowcount
        if self._pgvector:
            await conn.execute(
                text(
                    "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT chunk_id FROM chunks"
                    " WHERE collection = :collection AND document_id = :document_id)"
                ),
                {"collection": collection, "document_id": document_id},
            )
        removed_chunks = (
            await conn.execute(
                delete(chunks).where(
                    chunks.c.collection == collection, chunks.c.document_id == document_id
                )
            )
        ).rowcount
        return int(removed_chunks or 0), int(removed_mentions or 0)

    async def erase_document(
        self, collection: str, document_id: str, at: datetime
    ) -> ErasureReport:
        """Right-to-erasure cascade: chunks, vectors, mentions, assertions, orphaned
        entities and every community summary of the collection (summaries may
        quote erased text)."""
        async with self._engine.begin() as conn:
            exists = (
                await conn.execute(
                    select(documents.c.document_id).where(
                        documents.c.collection == collection,
                        documents.c.document_id == document_id,
                    )
                )
            ).first()
            if not exists:
                raise NotFoundError(f"document {document_id} not found in {collection}")
            removed_chunks, removed_mentions = await self._remove_document_rows(
                conn, collection, document_id
            )
            removed_assertions = (
                await conn.execute(
                    delete(assertions).where(
                        assertions.c.collection == collection,
                        assertions.c.document_id == document_id,
                    )
                )
            ).rowcount
            orphans = list(
                (
                    await conn.execute(
                        select(entities.c.entity_id).where(
                            entities.c.collection == collection,
                            entities.c.mention_count <= 0,
                        )
                    )
                ).scalars()
            )
            if orphans:
                await conn.execute(
                    delete(assertions).where(
                        assertions.c.collection == collection,
                        or_(
                            assertions.c.source_id.in_(orphans), assertions.c.target_id.in_(orphans)
                        ),
                    )
                )
                await conn.execute(
                    delete(entity_keys).where(
                        entity_keys.c.collection == collection,
                        entity_keys.c.entity_id.in_(orphans),
                    )
                )
                await conn.execute(delete(entities).where(entities.c.entity_id.in_(orphans)))
            removed_communities = (
                await conn.execute(
                    delete(communities).where(communities.c.collection == collection)
                )
            ).rowcount
            await conn.execute(
                delete(documents).where(
                    documents.c.collection == collection, documents.c.document_id == document_id
                )
            )
            await self._bump(conn, collection, at)
        return ErasureReport(
            chunks=removed_chunks,
            mentions=removed_mentions,
            assertions=int(removed_assertions or 0),
            entities=len(orphans),
            communities=int(removed_communities or 0),
        )

    # ---------------------------------------------------------------- chunks

    async def indexed_chunks(self, collection: str) -> list[IndexedChunk]:
        """Retrievable (non-quarantined) chunks with embeddings and document metadata."""
        query = (
            select(
                chunks,
                documents.c.metadata.label("doc_metadata"),
                documents.c.effective_at,
                documents.c.expires_at,
            )
            .join(
                documents,
                and_(
                    documents.c.collection == chunks.c.collection,
                    documents.c.document_id == chunks.c.document_id,
                ),
            )
            .where(chunks.c.collection == collection, chunks.c.quarantined.is_(False))
            .order_by(chunks.c.document_id, chunks.c.ordinal)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [
            IndexedChunk(
                chunk=_chunk(r),
                terms=r["terms"].split(),
                embedding=decode_vector(r["embedding"]),
                metadata=json.loads(r["doc_metadata"]),
                effective_at=_aware(r["effective_at"]),
                expires_at=_opt_aware(r["expires_at"]),
            )
            for r in rows
        ]

    async def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = sorted(set(chunk_ids))
        if not ids:
            return {}
        async with self._engine.connect() as conn:
            rows = (
                (await conn.execute(select(chunks).where(chunks.c.chunk_id.in_(ids))))
                .mappings()
                .all()
            )
        return {r["chunk_id"]: _chunk(r) for r in rows}

    async def document_chunks(self, collection: str, document_id: str) -> list[Chunk]:
        query = (
            select(chunks)
            .where(chunks.c.collection == collection, chunks.c.document_id == document_id)
            .order_by(chunks.c.ordinal)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [_chunk(r) for r in rows]

    async def vector_search(
        self, collection: str, vector: Sequence[float], k: int
    ) -> list[tuple[str, float]]:
        """pgvector cosine search; returns (chunk_id, cosine similarity)."""
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT chunk_id, 1 - (embedding <=> CAST(:q AS vector)) AS similarity"
                        " FROM chunk_vectors WHERE collection = :collection"
                        " ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
                    ),
                    {"q": _pg_vector_literal(vector), "collection": collection, "k": k},
                )
            ).all()
        return [(str(r[0]), float(r[1])) for r in rows]

    # ------------------------------------------------------------------ graph

    async def entity_candidates(self, collection: str, keys: Iterable[str]) -> list[Entity]:
        key_list = sorted(set(keys))
        if not key_list:
            return []
        query = (
            select(entities)
            .where(
                entities.c.collection == collection,
                entities.c.entity_id.in_(
                    select(entity_keys.c.entity_id).where(
                        entity_keys.c.collection == collection, entity_keys.c.key.in_(key_list)
                    )
                ),
            )
            .order_by(entities.c.entity_id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [_entity(r) for r in rows]

    async def entities(self, collection: str) -> list[Entity]:
        query = (
            select(entities)
            .where(entities.c.collection == collection)
            .order_by(entities.c.entity_id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [_entity(r) for r in rows]

    async def get_entity(self, collection: str, entity_id: str) -> Entity | None:
        query = select(entities).where(
            entities.c.collection == collection, entities.c.entity_id == entity_id
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(query)).mappings().first()
        return _entity(row) if row else None

    async def mentions_for(self, collection: str, entity_ids: Iterable[str]) -> list[Mention]:
        ids = sorted(set(entity_ids))
        if not ids:
            return []
        query = (
            select(mentions)
            .where(mentions.c.collection == collection, mentions.c.entity_id.in_(ids))
            .order_by(mentions.c.id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [
            Mention(
                entity_id=r["entity_id"],
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                surface=r["surface"],
                start=r["start_offset"],
                end=r["end_offset"],
            )
            for r in rows
        ]

    async def assertions(
        self, collection: str, *, as_of: datetime | None = None
    ) -> list[Assertion]:
        query = select(assertions).where(assertions.c.collection == collection)
        if as_of is not None:
            query = query.where(
                assertions.c.valid_from <= as_of,
                or_(assertions.c.valid_to.is_(None), assertions.c.valid_to > as_of),
            )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query.order_by(assertions.c.assertion_id))).mappings().all()
        return [_assertion(r) for r in rows]

    async def merge_entities(
        self, collection: str, merges: dict[str, str], survivors: list[Entity], at: datetime
    ) -> None:
        """Repoint mentions, assertions and blocking keys of merged entities to survivors."""
        if not merges:
            return
        async with self._engine.begin() as conn:
            for old, new in sorted(merges.items()):
                await conn.execute(
                    update(mentions)
                    .where(mentions.c.collection == collection, mentions.c.entity_id == old)
                    .values(entity_id=new)
                )
                await conn.execute(
                    update(assertions)
                    .where(assertions.c.collection == collection, assertions.c.source_id == old)
                    .values(source_id=new)
                )
                await conn.execute(
                    update(assertions)
                    .where(assertions.c.collection == collection, assertions.c.target_id == old)
                    .values(target_id=new)
                )
                moved = set(
                    (
                        await conn.execute(
                            select(entity_keys.c.key).where(
                                entity_keys.c.collection == collection,
                                entity_keys.c.entity_id == old,
                            )
                        )
                    ).scalars()
                )
                await conn.execute(
                    delete(entity_keys).where(
                        entity_keys.c.collection == collection, entity_keys.c.entity_id == old
                    )
                )
                await self._add_keys(conn, collection, {new: moved})
            await conn.execute(delete(entities).where(entities.c.entity_id.in_(sorted(merges))))
            for survivor in survivors:
                await conn.execute(
                    update(entities)
                    .where(entities.c.entity_id == survivor.entity_id)
                    .values(
                        name=survivor.name,
                        normalized=survivor.normalized,
                        entity_type=survivor.entity_type,
                        aliases=json.dumps(sorted(survivor.aliases)),
                        mention_count=survivor.mention_count,
                    )
                )
            # Self-loops created by merging two endpoints of one relation carry no information.
            await conn.execute(
                delete(assertions).where(
                    assertions.c.collection == collection,
                    assertions.c.source_id == assertions.c.target_id,
                )
            )
            await self._bump(conn, collection, at)

    async def replace_communities(
        self, collection: str, items: Sequence[Community], at: datetime
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(delete(communities).where(communities.c.collection == collection))
            if items:
                await conn.execute(
                    insert(communities),
                    [
                        {
                            "community_id": c.community_id,
                            "collection": c.collection,
                            "level": c.level,
                            "entity_ids": json.dumps(list(c.entity_ids)),
                            "key_entities": json.dumps(list(c.key_entities)),
                            "summary": c.summary,
                            "weight": c.weight,
                            "built_revision": c.built_revision,
                            "built_at": c.built_at,
                        }
                        for c in items
                    ],
                )
            await self._bump(conn, collection, at)

    async def communities(self, collection: str, *, level: int | None = None) -> list[Community]:
        query = select(communities).where(communities.c.collection == collection)
        if level is not None:
            query = query.where(communities.c.level == level)
        query = query.order_by(
            communities.c.level, communities.c.weight.desc(), communities.c.community_id
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [
            Community(
                community_id=r["community_id"],
                collection=r["collection"],
                level=r["level"],
                entity_ids=tuple(json.loads(r["entity_ids"])),
                key_entities=tuple(json.loads(r["key_entities"])),
                summary=r["summary"],
                weight=r["weight"],
                built_revision=r["built_revision"],
                built_at=_aware(r["built_at"]),
            )
            for r in rows
        ]

    async def ping(self) -> None:
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))


# ---------------------------------------------------------------- row mapping


def _collection(r: Any) -> Collection:
    return Collection(
        name=r["name"],
        description=r["description"],
        classification=DataClassification(r["classification"]),
        owner=r["owner"],
        revision=r["revision"],
        embedding_model=r["embedding_model"],
        embedding_dimension=r["embedding_dimension"],
        created_at=_aware(r["created_at"]),
        updated_at=_aware(r["updated_at"]),
    )


def _document(r: Any) -> Document:
    return Document(
        collection=r["collection"],
        document_id=r["document_id"],
        version=r["version"],
        title=r["title"],
        source_uri=r["source_uri"],
        source_type=r["source_type"],
        classification=DataClassification(r["classification"]),
        trust=TrustLevel(r["trust"]),
        metadata=json.loads(r["metadata"]),
        digest=r["digest"],
        effective_at=_aware(r["effective_at"]),
        expires_at=_opt_aware(r["expires_at"]),
        ingested_at=_aware(r["ingested_at"]),
        ingested_by=r["ingested_by"],
        chunk_count=r["chunk_count"],
    )


def _chunk(r: Any) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"],
        collection=r["collection"],
        document_id=r["document_id"],
        document_version=r["document_version"],
        ordinal=r["ordinal"],
        text=r["text"],
        start=r["start_offset"],
        end=r["end_offset"],
        token_count=r["token_count"],
        digest=r["digest"],
        classification=DataClassification(r["classification"]),
        trust=TrustLevel(r["trust"]),
        source_uri=r["source_uri"],
        title=r["title"],
        quarantined=bool(r["quarantined"]),
        quarantine_reason=r["quarantine_reason"],
    )


def _entity(r: Any) -> Entity:
    return Entity(
        entity_id=r["entity_id"],
        collection=r["collection"],
        name=r["name"],
        normalized=r["normalized"],
        entity_type=r["entity_type"],
        aliases=tuple(json.loads(r["aliases"])),
        mention_count=r["mention_count"],
    )


def _assertion(r: Any) -> Assertion:
    return Assertion(
        assertion_id=r["assertion_id"],
        collection=r["collection"],
        source_id=r["source_id"],
        target_id=r["target_id"],
        relation_type=r["relation_type"],
        weight=r["weight"],
        document_id=r["document_id"],
        document_version=r["document_version"],
        chunk_id=r["chunk_id"],
        evidence=r["evidence"],
        valid_from=_aware(r["valid_from"]),
        valid_to=_opt_aware(r["valid_to"]),
    )


def _assertion_row(a: Assertion) -> dict[str, Any]:
    return {
        "assertion_id": a.assertion_id,
        "collection": a.collection,
        "source_id": a.source_id,
        "target_id": a.target_id,
        "relation_type": a.relation_type,
        "weight": a.weight,
        "document_id": a.document_id,
        "document_version": a.document_version,
        "chunk_id": a.chunk_id,
        "evidence": a.evidence,
        "valid_from": a.valid_from,
        "valid_to": a.valid_to,
    }
