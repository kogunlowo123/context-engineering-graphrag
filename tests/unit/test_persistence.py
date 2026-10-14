"""SqlStore behaviour not covered through the platform."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentic_core.domain.classification import DataClassification, TrustLevel

from context_engineering_graphrag.domain.errors import ConflictError, NotFoundError
from context_engineering_graphrag.domain.models import Collection, Document
from context_engineering_graphrag.infrastructure.database import (
    DocumentWrite,
    SqlStore,
    _pg_vector_literal,
    create_schema,
    decode_vector,
    encode_vector,
    make_engine,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def collection(name: str = "c1") -> Collection:
    return Collection(
        name=name,
        description="",
        classification=DataClassification.INTERNAL,
        owner="o",
        embedding_model="m",
        embedding_dimension=4,
        created_at=NOW,
        updated_at=NOW,
    )


def document(version: int) -> Document:
    return Document(
        collection="c1",
        document_id="d1",
        version=version,
        title="t",
        source_uri="https://docs.corp.test/d1",
        source_type="document",
        classification=DataClassification.INTERNAL,
        trust=TrustLevel.VERIFIED,
        digest="sha256:" + "a" * 64,
        effective_at=NOW,
        ingested_at=NOW,
        ingested_by="agent",
        chunk_count=0,
    )


def write(version: int) -> DocumentWrite:
    return DocumentWrite(
        document=document(version),
        chunks=[],
        embeddings=[],
        new_entities=[],
        entity_keys={},
        mention_count_delta={},
        alias_additions={},
        mentions=[],
        assertions=[],
    )


@pytest.fixture
async def store(tmp_path: Path) -> SqlStore:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 's.db'}")
    await create_schema(engine)
    await create_schema(engine)  # idempotent
    return SqlStore(engine)


async def test_collections(store: SqlStore) -> None:
    await store.create_collection(collection())
    with pytest.raises(ConflictError):
        await store.create_collection(collection())
    assert (await store.require_collection("c1")).name == "c1"
    with pytest.raises(NotFoundError):
        await store.require_collection("nope")
    with pytest.raises(NotFoundError):
        await store.revision("nope")
    assert await store.revision("c1") == 0
    await store.ping()
    assert store.pgvector is False
    assert store.engine is not None
    await store.engine.dispose()


async def test_versioning_detects_concurrent_writers(store: SqlStore) -> None:
    await store.create_collection(collection())
    await store.write_document(write(1))
    with pytest.raises(ConflictError, match="concurrently"):
        await store.write_document(write(3))
    await store.write_document(write(2))
    assert (await store.get_document("c1", "d1")).version == 2  # type: ignore[union-attr]
    assert await store.revision("c1") == 2
    await store.engine.dispose()


async def test_empty_lookups(store: SqlStore) -> None:
    assert await store.get_chunks([]) == {}
    assert await store.entity_candidates("c1", []) == []
    assert await store.mentions_for("c1", []) == []
    assert await store.get_entity("c1", "ent_x") is None
    await store.create_collection(collection())
    await store.merge_entities("c1", {}, [], NOW)
    assert await store.revision("c1") == 0
    await store.replace_communities("c1", [], NOW)
    assert await store.communities("c1") == []
    await store.engine.dispose()


def test_vector_codecs() -> None:
    blob = encode_vector([0.5, -1.0, 2.0])
    assert decode_vector(blob).tolist() == [0.5, -1.0, 2.0]
    assert _pg_vector_literal([0.5, 1.0]) == "[0.5,1]"


def test_pgvector_table_ddl() -> None:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    from context_engineering_graphrag.infrastructure.database import Vector, vector_table

    table = vector_table(256)
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert "embedding vector(256) NOT NULL" in ddl
    (index,) = table.indexes
    assert str(CreateIndex(index).compile(dialect=postgresql.dialect())).endswith(
        "USING hnsw (embedding vector_cosine_ops)"
    )
    with pytest.raises(ValueError, match="out of range"):
        Vector(0)
