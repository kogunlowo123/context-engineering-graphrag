"""Shared fixtures: an in-process token issuer, a synthetic corpus, a platform wired
to SQLite with the deterministic embedder, and an ASGI app."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from agentic_core.domain.classification import DataClassification, TrustLevel
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from context_engineering_graphrag.api.app import create_app
from context_engineering_graphrag.application.service import (
    Caller,
    ContextPlatform,
    CreateCollection,
)
from context_engineering_graphrag.config import Settings
from context_engineering_graphrag.ingestion.service import IngestDocument
from context_engineering_graphrag.runtime.container import Container, build_container
from context_engineering_graphrag.security.authorization import StaticJwks

ROOT = Path(__file__).resolve().parents[1]
ISSUER = "https://login.corp.test/tenant"
AUDIENCE = "context-engineering-graphrag"
AGENT = "spiffe://agents.corp.test/agent/research/analyst"


@dataclass
class FakeClock:
    current: datetime = field(default_factory=lambda: datetime(2026, 10, 1, 12, 0, tzinfo=UTC))

    def now(self) -> datetime:
        return self.current

    def advance(self, **delta: float) -> None:
        self.current += timedelta(**delta)


@dataclass
class FakeIdp:
    """Token issuer standing in for the enterprise IdP or the identity fabric (ES256)."""

    clock: FakeClock
    issuer: str = ISSUER
    kid: str = "k1"
    key: ec.EllipticCurvePrivateKey = field(
        default_factory=lambda: ec.generate_private_key(ec.SECP256R1())
    )

    def jwks(self) -> dict[str, Any]:
        jwk = jwt.algorithms.ECAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        jwk.update({"kid": self.kid, "alg": "ES256", "use": "sig"})
        return {"keys": [jwk]}

    def mint(
        self,
        *,
        scopes: tuple[str, ...] = ("rag:*",),
        sub: str = AGENT,
        max_cls: int = 3,
        ttl: int = 600,
        audience: str = AUDIENCE,
        **extra: Any,
    ) -> str:
        now = int(self.clock.now().timestamp())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "sub": sub,
            "aud": audience,
            "iat": now,
            "exp": now + ttl,
            "scope": " ".join(scopes),
            "max_cls": max_cls,
        }
        claims.update(extra)
        return jwt.encode(claims, self.key, algorithm="ES256", headers={"kid": self.kid})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def idp(clock: FakeClock) -> FakeIdp:
    return FakeIdp(clock)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "public_url": "http://testserver",
        "database_url": os.environ.get(
            "GRAPHRAG_TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'g.db'}"
        ),
        "audit_log_path": tmp_path / "audit.jsonl",
        "policy_file": ROOT / "policies" / "context-governance.yaml",
        "local_issuer": ISSUER,
        "local_audience": AUDIENCE,
        "local_jwks_url": "https://login.corp.test/jwks",
        "embedding_provider": "hashing",
        "embedding_dimension": 256,
        "chunk_max_tokens": 96,
        "chunk_overlap_tokens": 16,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
async def container(settings: Settings, idp: FakeIdp, clock: FakeClock) -> AsyncIterator[Container]:
    built = await build_container(settings, clock=clock, jwks_resolver=StaticJwks(idp.jwks()))
    yield built
    await built.aclose()


@pytest.fixture
def platform(container: Container) -> ContextPlatform:
    return container.platform


@pytest.fixture
def caller(idp: FakeIdp) -> Caller:
    return Caller(token=idp.mint(), purpose_of_use="research")


# --------------------------------------------------------------- the corpus
#
# Synthetic supplier-risk corpus. All organisations and people are invented.

CORPUS: dict[str, tuple[str, str]] = {
    "supplier-overview": (
        "Supplier overview",
        "Acme Diagnostics depends on Beta Reagents for assay reagents. Beta Reagents is based in "
        "Lyon. Acme Diagnostics partnered with Globex Logistics to ship reagents across Europe. "
        "Globex Logistics operates in Rotterdam.\n\nThe reagent contract renews every two years "
        "and covers cartridges for the Orion Analyzer.",
    ),
    "acquisition-2024": (
        "Northwind acquisition",
        "Northwind Holdings acquired Beta Reagents in March 2024. Dr. Maria Chen leads Northwind "
        "Holdings. Northwind Holdings is headquartered in Boston. The acquisition raised "
        "concentration risk for Acme Diagnostics, because Beta Reagents supplies 70% of its "
        "reagent volume.",
    ),
    "quality-audit": (
        "Quality audit 2025",
        "The FDA audited Beta Reagents in 2025. The audit found no critical deviations in "
        "cartridge manufacturing. Acme Diagnostics uses the Orion Analyzer platform. Orion "
        "Analyzer depends on Beta Reagents cartridges for every assay run.",
    ),
    "logistics-risk": (
        "Logistics risk",
        "Globex Logistics competes with Contoso Freight. Contoso Freight is based in Hamburg. "
        "Port congestion in Rotterdam delayed reagent shipments for six weeks in 2024. Contoso "
        "Freight supplies cold-chain capacity to Acme Diagnostics during peak season.",
    ),
    "security-services": (
        "Security services",
        "Fabrikam Security provides threat monitoring for Acme Diagnostics. Fabrikam Security "
        "reports incidents to the Acme Diagnostics security office within four hours. Fabrikam "
        "Security is based in Dublin.",
    ),
}


def corpus_document(doc_id: str, **overrides: Any) -> IngestDocument:
    title, text = CORPUS.get(doc_id, (doc_id, ""))
    values: dict[str, Any] = {
        "document_id": doc_id,
        "title": title,
        "text": text,
        "source_uri": f"https://docs.corp.test/supplier-risk/{doc_id}",
        "classification": DataClassification.CONFIDENTIAL,
        "trust": TrustLevel.VERIFIED,
        "metadata": {"domain": "supplier-risk", "year": 2025},
        "effective_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(overrides)
    return IngestDocument(**values)


async def seed(
    platform: ContextPlatform,
    caller: Caller,
    name: str = "supplier-risk",
    *,
    classification: DataClassification = DataClassification.CONFIDENTIAL,
    build_graph: bool = True,
) -> None:
    from context_engineering_graphrag.application.knowledge_graph import GraphBuildOptions

    await platform.create_collection(
        caller,
        CreateCollection(
            name=name, description="Supplier risk", classification=classification, owner="risk-team"
        ),
    )
    for doc_id in CORPUS:
        await platform.ingest(caller, name, corpus_document(doc_id))
    if build_graph:
        await platform.build_graph(caller, name, GraphBuildOptions())


@pytest.fixture
async def seeded(platform: ContextPlatform, caller: Caller) -> ContextPlatform:
    await seed(platform, caller)
    return platform


@pytest.fixture
def client(settings: Settings, idp: FakeIdp, clock: FakeClock) -> Iterator[TestClient]:
    """App whose container is built inside the app's own event loop by its lifespan."""

    async def factory() -> Container:
        return await build_container(settings, clock=clock, jwks_resolver=StaticJwks(idp.jwks()))

    with TestClient(create_app(factory, allowed_hosts=settings.allowed_hosts)) as test_client:
        yield test_client


def auth(idp: FakeIdp, **kw: Any) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {idp.mint(**kw)}"}
    headers["X-Purpose-Of-Use"] = "research"
    return headers
