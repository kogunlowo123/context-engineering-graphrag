"""Fault injection: the platform fails closed, reports unreadiness, and keeps
replicas consistent when dependencies break or other replicas write."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from agentic_core.domain.classification import DataClassification
from fastapi.testclient import TestClient

from context_engineering_graphrag.api.app import create_app
from context_engineering_graphrag.application.service import Caller, CreateCollection, SearchRequest
from context_engineering_graphrag.domain.errors import DependencyUnavailableError
from context_engineering_graphrag.embeddings.providers import OpenAICompatibleEmbedder
from context_engineering_graphrag.infrastructure.database import make_engine
from context_engineering_graphrag.retrieval.rerank import HttpCrossEncoder
from context_engineering_graphrag.runtime.container import Container, build_container
from context_engineering_graphrag.security.authorization import StaticJwks
from tests.conftest import FakeClock, FakeIdp, auth, corpus_document, make_settings, seed
from tests.unit.test_context_generation_evaluation import model_router

pytestmark = pytest.mark.chaos


def down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_database_outage_flips_readiness(client: TestClient) -> None:
    container: Container = client.app.state.container  # type: ignore[attr-defined]
    assert client.get("/readyz").status_code == 200
    healthy = container.store._engine
    container.store._engine = make_engine("sqlite+aiosqlite:////nonexistent/dir/db.sqlite")
    try:
        assert client.get("/readyz").status_code == 503
        assert client.get("/healthz").status_code == 200  # liveness unaffected
    finally:
        container.store._engine = healthy


@pytest.fixture
def fabric_down_client(tmp_path: Path, clock: FakeClock) -> Iterator[TestClient]:
    settings = make_settings(
        tmp_path, identity_fabric_url="https://identity.corp.test", local_issuer=None
    )

    async def factory() -> Container:
        return await build_container(
            settings, clock=clock, http=httpx.AsyncClient(transport=httpx.MockTransport(down))
        )

    with TestClient(create_app(factory, allowed_hosts=settings.allowed_hosts)) as client:
        yield client


def test_identity_fabric_outage_fails_closed(fabric_down_client: TestClient, idp: FakeIdp) -> None:
    response = fabric_down_client.get("/v1/collections", headers=auth(idp))
    assert response.status_code == 503
    assert response.json()["error"] == "temporarily_unavailable"


async def test_embedding_outage_writes_nothing(
    tmp_path: Path, idp: FakeIdp, clock: FakeClock
) -> None:
    embedder = OpenAICompatibleEmbedder(
        base_url="https://tei.corp.test",
        model="bge-m3",
        dimension=8,
        auth="none",
        transport=httpx.MockTransport(down),
    )
    container = await build_container(
        make_settings(tmp_path),
        clock=clock,
        jwks_resolver=StaticJwks(idp.jwks()),
        embedder=embedder,
    )
    caller = Caller(token=idp.mint(), purpose_of_use="research")
    try:
        await container.platform.create_collection(
            caller,
            CreateCollection(
                name="docs", classification=DataClassification.CONFIDENTIAL, owner="o"
            ),
        )
        with pytest.raises(DependencyUnavailableError):
            await container.platform.ingest(caller, "docs", corpus_document("quality-audit"))
        assert await container.store.revision("docs") == 0
        assert await container.store.get_document("docs", "quality-audit") is None
    finally:
        await container.aclose()


def test_reranker_outage_returns_503(tmp_path: Path, idp: FakeIdp, clock: FakeClock) -> None:
    settings = make_settings(tmp_path)

    async def factory() -> Container:
        return await build_container(
            settings,
            clock=clock,
            jwks_resolver=StaticJwks(idp.jwks()),
            reranker=HttpCrossEncoder(
                base_url="http://rr.svc", model="m", transport=httpx.MockTransport(down)
            ),
        )

    with TestClient(create_app(factory, allowed_hosts=settings.allowed_hosts)) as client:
        h = auth(idp)
        client.post(
            "/v1/collections", json={"name": "docs", "classification": 2, "owner": "o"}, headers=h
        )
        client.post(
            "/v1/collections/docs/documents",
            json=corpus_document("quality-audit").model_dump(mode="json"),
            headers=h,
        )
        broken = client.post("/v1/collections/docs/search", json={"query": "audit"}, headers=h)
        assert broken.status_code == 503
        degraded = client.post(
            "/v1/collections/docs/search", json={"query": "audit", "rerank": False}, headers=h
        )
        assert degraded.status_code == 200
        assert degraded.json()["hits"]


def test_model_provider_outage_returns_503(tmp_path: Path, idp: FakeIdp, clock: FakeClock) -> None:
    root = Path(__file__).resolve().parents[2]
    settings = make_settings(
        tmp_path, generator="model", providers_file=root / "configs" / "providers.yaml"
    )

    async def factory() -> Container:
        return await build_container(
            settings,
            clock=clock,
            jwks_resolver=StaticJwks(idp.jwks()),
            router=model_router("x", status=503),
        )

    with TestClient(create_app(factory, allowed_hosts=settings.allowed_hosts)) as client:
        h = auth(idp)
        client.post(
            "/v1/collections", json={"name": "docs", "classification": 2, "owner": "o"}, headers=h
        )
        client.post(
            "/v1/collections/docs/documents",
            json=corpus_document("quality-audit").model_dump(mode="json"),
            headers=h,
        )
        response = client.post("/v1/collections/docs/query", json={"query": "FDA audit"}, headers=h)
        assert response.status_code == 503
        assert response.json()["error"] == "provider_unavailable"


async def test_audit_outage_blocks_writes(tmp_path: Path, idp: FakeIdp, clock: FakeClock) -> None:
    blocked = tmp_path / "audit-dir"
    blocked.mkdir()
    container = await build_container(
        make_settings(tmp_path, audit_log_path=blocked),
        clock=clock,
        jwks_resolver=StaticJwks(idp.jwks()),
    )
    caller = Caller(token=idp.mint(), purpose_of_use="research")
    try:
        with pytest.raises((IsADirectoryError, PermissionError)):
            await container.platform.create_collection(
                caller,
                CreateCollection(
                    name="docs", classification=DataClassification.INTERNAL, owner="o"
                ),
            )
        assert await container.store.get_collection("docs") is None
    finally:
        await container.aclose()


async def test_replicas_see_each_others_writes(
    tmp_path: Path, idp: FakeIdp, clock: FakeClock
) -> None:
    settings = make_settings(tmp_path)
    replica_a = await build_container(settings, clock=clock, jwks_resolver=StaticJwks(idp.jwks()))
    replica_b = await build_container(
        make_settings(tmp_path, audit_log_path=tmp_path / "audit-b.jsonl"),
        clock=clock,
        jwks_resolver=StaticJwks(idp.jwks()),
    )
    caller = Caller(token=idp.mint(), purpose_of_use="research")
    try:
        await seed(replica_a.platform, caller, build_graph=False)
        before = await replica_b.platform.search(
            caller, "supplier-risk", SearchRequest(query="Vega roadmap")
        )
        assert all(h.chunk.document_id != "vega" for h in before.hits)
        await replica_a.platform.ingest(
            caller,
            "supplier-risk",
            corpus_document("vega", title="Vega", text="The Vega roadmap adds reagent tracking."),
        )
        after = await replica_b.platform.search(
            caller, "supplier-risk", SearchRequest(query="Vega roadmap")
        )
        assert after.hits[0].chunk.document_id == "vega"
        await replica_a.platform.erase_document(caller, "supplier-risk", "vega")
        erased = await replica_b.platform.search(
            caller, "supplier-risk", SearchRequest(query="Vega roadmap")
        )
        assert all(h.chunk.document_id != "vega" for h in erased.hits)
    finally:
        await replica_a.aclose()
        await replica_b.aclose()
