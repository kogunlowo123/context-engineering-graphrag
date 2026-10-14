"""Platform behaviour against a real PostgreSQL with pgvector (CI service container).

Runs when GRAPHRAG_TEST_DATABASE_URL points at PostgreSQL, e.g.
postgresql+asyncpg://graphrag:graphrag@localhost:5432/graphrag
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from context_engineering_graphrag.application.knowledge_graph import GraphBuildOptions
from context_engineering_graphrag.application.service import Caller, SearchRequest
from context_engineering_graphrag.orchestration.workflow import Disposition, QueryRequest
from context_engineering_graphrag.runtime.container import build_container
from context_engineering_graphrag.security.authorization import StaticJwks
from tests.conftest import FakeClock, FakeIdp, corpus_document, make_settings, seed

URL = os.environ.get("GRAPHRAG_TEST_DATABASE_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL.startswith("postgresql"), reason="PostgreSQL not configured"),
]


@pytest.mark.parametrize("backend", ["memory", "pgvector"])
async def test_full_platform_on_postgres(tmp_path: Path, backend: str) -> None:
    clock = FakeClock()
    idp = FakeIdp(clock)
    settings = make_settings(tmp_path, database_url=URL, vector_backend=backend)
    container = await build_container(settings, clock=clock, jwks_resolver=StaticJwks(idp.jwks()))
    caller = Caller(token=idp.mint(), purpose_of_use="research")
    name = f"pg-{backend[:3]}-{uuid.uuid4().hex[:8]}"
    try:
        await seed(container.platform, caller, name)
        hits = await container.platform.search(
            caller, name, SearchRequest(query="FDA audit cartridges", k=3)
        )
        assert hits.hits[0].chunk.document_id == "quality-audit"
        if backend == "pgvector":
            dense = await container.store.vector_search(
                name, (await container.embedder.embed(["FDA audit"]))[0], 3
            )
            assert dense
            assert all(-1.0 <= score <= 1.0 for _, score in dense)
        answer = await container.platform.query(
            caller, name, QueryRequest(query="Which company acquired Beta Reagents?")
        )
        assert answer.disposition is Disposition.ANSWERED
        assert "Northwind" in (answer.answer or "")

        clock.advance(days=200)
        caller = Caller(token=idp.mint(), purpose_of_use="research")
        await container.platform.ingest(
            caller,
            name,
            corpus_document(
                "acquisition-2024",
                text="Initech Partners acquired Beta Reagents from Northwind Holdings "
                "in June 2026.",
                effective_at=datetime(2026, 6, 1, tzinfo=UTC),
            ),
        )
        past = await container.store.assertions(name, as_of=datetime(2026, 3, 1, tzinfo=UTC))
        now = await container.store.assertions(name, as_of=datetime(2026, 7, 1, tzinfo=UTC))
        assert {a.document_version for a in past if a.document_id == "acquisition-2024"} == {1}
        assert {a.document_version for a in now if a.document_id == "acquisition-2024"} == {2}

        report = await container.platform.build_graph(caller, name, GraphBuildOptions())
        assert report.communities >= 1
        erased = await container.platform.erase_document(caller, name, "security-services")
        assert erased.chunks >= 1
        assert erased.communities >= 1
        after = await container.platform.search(
            caller, name, SearchRequest(query="threat monitoring")
        )
        assert all(h.chunk.document_id != "security-services" for h in after.hits)
    finally:
        await container.aclose()
