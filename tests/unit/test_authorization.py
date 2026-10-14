"""Identity-fabric client, fabric-backed and local authorizers, JWT verification."""

from __future__ import annotations

import json
from typing import Any

import httpx
import jwt
import pytest
from agentic_core.domain.classification import DataClassification, Decision
from agentic_core.policy import PolicyEngine, PolicySet, Resource

from context_engineering_graphrag.domain.errors import DependencyUnavailableError, UnauthorizedError
from context_engineering_graphrag.integrations.identity_fabric import (
    JWT_BEARER,
    AuthorizeCall,
    IdentityFabricClient,
)
from context_engineering_graphrag.security.authorization import (
    FabricAuthorizer,
    JwtVerifier,
    LocalAuthorizer,
    RemoteJwks,
    StaticJwks,
)
from tests.conftest import AUDIENCE, ISSUER, ROOT, FakeClock, FakeIdp

FABRIC = "https://identity.corp.test"
DECISION = {
    "decision": "allow",
    "agent": "spiffe://agents.corp.test/agent/research/analyst",
    "action": "rag:query",
    "resource": "collection:supplier-risk",
    "matched_rules": ["allow-scoped-actions"],
    "policy_version": "2026.10.0",
    "policy_digest": "sha256:" + "1" * 64,
    "reason": "explicit allow",
    "delegation_chain": [],
}


def fabric(handler: Any) -> IdentityFabricClient:
    return IdentityFabricClient(
        FABRIC + "/", http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


class TestIdentityFabricClient:
    async def test_wire_formats(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/oauth2/token":
                return httpx.Response(
                    200,
                    json={
                        "access_token": "t",
                        "token_type": "Bearer",
                        "expires_in": 600,
                        "scope": "rag:query",
                    },
                )
            if request.url.path == "/oauth2/introspect":
                return httpx.Response(
                    200, json={"active": True, "sub": "agent", "scope": "rag:query"}
                )
            return httpx.Response(200, json=DECISION)

        client = fabric(handler)
        issued = await client.token(
            client_id="spiffe://a/agent/x/y", assertion="wl", scopes=frozenset({"rag:query"})
        )
        assert issued.expires_in == 600
        form = dict(httpx.QueryParams(seen[0].content.decode()))
        assert form == {
            "grant_type": JWT_BEARER,
            "client_id": "spiffe://a/agent/x/y",
            "assertion": "wl",
            "scope": "rag:query",
        }
        assert (await client.introspect("t")).active
        decision = await client.authorize(
            "agent-token",
            AuthorizeCall(
                action="rag:query", resource_type="collection", resource_id="supplier-risk"
            ),
        )
        assert decision.decision is Decision.ALLOW
        assert seen[2].headers["authorization"] == "Bearer agent-token"
        assert json.loads(seen[2].content)["resource_classification"] == 1
        assert str(seen[2].url) == FABRIC + "/v1/authorize"

    async def test_error_mapping(self) -> None:
        async def call(response: httpx.Response, kind: str) -> None:
            client = fabric(lambda _: response)
            if kind == "token":
                await client.token(client_id="c", assertion="a")
            else:
                await client.authorize("t", AuthorizeCall(action="a", resource_type="r"))

        with pytest.raises(UnauthorizedError, match="invalid_grant: bad"):
            await call(
                httpx.Response(400, json={"error": "invalid_grant", "error_description": "bad"}),
                "token",
            )
        with pytest.raises(UnauthorizedError, match="invalid_client"):
            await call(httpx.Response(401, json={"error": "invalid_client"}), "token")
        with pytest.raises(UnauthorizedError, match="rejected the request"):
            await call(httpx.Response(400, text="nope"), "token")
        with pytest.raises(UnauthorizedError, match="rejected the request"):
            await call(httpx.Response(400, json=["x"]), "token")
        with pytest.raises(UnauthorizedError, match="credential rejected"):
            await call(httpx.Response(401, json={"error": "unauthorized"}), "authorize")
        with pytest.raises(DependencyUnavailableError, match="returned 503"):
            await call(httpx.Response(503), "authorize")
        with pytest.raises(DependencyUnavailableError, match="non-JSON"):
            await call(httpx.Response(200, text="<html>"), "authorize")
        with pytest.raises(DependencyUnavailableError, match="unexpected body"):
            await call(httpx.Response(200, json=[1]), "authorize")
        with pytest.raises(DependencyUnavailableError, match="does not match"):
            await call(httpx.Response(200, json={"decision": "maybe"}), "authorize")

    async def test_unreachable_fails_closed(self) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = fabric(down)
        with pytest.raises(DependencyUnavailableError, match="unreachable"):
            await client.authorize("t", AuthorizeCall(action="a", resource_type="r"))
        with pytest.raises(DependencyUnavailableError, match="unreachable"):
            await client.introspect("t")


async def test_fabric_authorizer_maps_decision() -> None:
    authorizer = FabricAuthorizer(
        fabric(lambda _: httpx.Response(200, json={**DECISION, "delegation_chain": ["spiffe://o"]}))
    )
    decision = await authorizer.authorize(
        "t",
        action="rag:query",
        resource=Resource(
            type="collection", id="supplier-risk", classification=DataClassification.RESTRICTED
        ),
        environment={"purpose_of_use": "TREAT"},
    )
    assert decision.source == "identity-fabric"
    assert authorizer.name == "identity-fabric"
    assert decision.principal == DECISION["agent"]
    assert decision.delegation_chain == ("spiffe://o",)


def verifier(idp: FakeIdp, clock: FakeClock) -> JwtVerifier:
    return JwtVerifier(StaticJwks(idp.jwks()), issuer=ISSUER, audience=AUDIENCE, clock=clock)


class TestJwtVerifier:
    async def test_valid_token_claims(self, idp: FakeIdp, clock: FakeClock) -> None:
        token = idp.mint(
            scopes=("rag:query", "rag:search"),
            max_cls=9,
            risk=7,
            act={"sub": "o1", "act": {"sub": "o2"}},
        )
        verified = await verifier(idp, clock).verify(token)
        assert verified.scopes == {"rag:query", "rag:search"}
        assert verified.max_classification is DataClassification.RESTRICTED
        assert int(verified.risk) == 4
        assert verified.delegation_chain == ("o1", "o2")
        listed = await verifier(idp, clock).verify(idp.mint(scope=["a", "b"]))
        assert listed.scopes == {"a", "b"}
        odd = await verifier(idp, clock).verify(idp.mint(scope=7))
        assert odd.scopes == frozenset()

    async def test_rejections(self, idp: FakeIdp, clock: FakeClock) -> None:
        check = verifier(idp, clock)
        with pytest.raises(UnauthorizedError, match="malformed"):
            await check.verify("not-a-jwt")
        none_alg = jwt.encode({"sub": "x"}, None, algorithm="none")  # type: ignore[arg-type]
        with pytest.raises(UnauthorizedError, match="algorithm"):
            await check.verify(none_alg)
        hs = jwt.encode({"sub": "x"}, "k" * 32, algorithm="HS256", headers={"kid": "k1"})
        with pytest.raises(UnauthorizedError, match="algorithm"):
            await check.verify(hs)
        with pytest.raises(UnauthorizedError, match="InvalidAudience"):
            await check.verify(idp.mint(audience="other"))
        with pytest.raises(UnauthorizedError, match="unknown signing key"):
            await check.verify(FakeIdp(clock, kid="k2").mint())
        forged = FakeIdp(clock)  # same kid, different key
        with pytest.raises(UnauthorizedError, match="InvalidSignature"):
            await check.verify(forged.mint())
        expired = idp.mint(ttl=10)
        clock.advance(seconds=60)
        with pytest.raises(UnauthorizedError, match="expired"):
            await check.verify(expired)
        future = idp.mint(iat=int(clock.now().timestamp()) + 3600)
        with pytest.raises(UnauthorizedError, match="not yet valid"):
            await check.verify(future)


class TestRemoteJwks:
    async def test_caching_refresh_and_failure(self, idp: FakeIdp) -> None:
        calls = {"n": 0}
        state = {"ok": True}
        ticks = {"t": 0.0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=idp.jwks()) if state["ok"] else httpx.Response(500)

        jwks = RemoteJwks(
            "https://login.corp.test/jwks",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ttl_s=100,
            min_refresh_interval_s=10,
            monotonic=lambda: ticks["t"],
        )
        assert await jwks.key_for("k1") is not None
        assert await jwks.key_for("k1") is not None
        assert calls["n"] == 1
        with pytest.raises(UnauthorizedError, match="unknown signing key"):
            await jwks.key_for("zz")  # within min refresh interval: no refetch
        assert calls["n"] == 1
        ticks["t"] = 200.0
        state["ok"] = False
        assert await jwks.key_for("k1") is not None  # stale cache tolerated for known kid
        with pytest.raises(UnauthorizedError, match="cannot fetch"):
            await RemoteJwks(
                "https://x/jwks", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ).key_for("k1")


async def test_local_authorizer_enforces_policy_and_ceiling(idp: FakeIdp, clock: FakeClock) -> None:
    engine = PolicyEngine(PolicySet.from_file(ROOT / "policies" / "context-governance.yaml"))
    authorizer = LocalAuthorizer(verifier(idp, clock), engine)
    assert authorizer.name == "local-policy"
    restricted = Resource(type="collection", id="c", classification=DataClassification.RESTRICTED)
    allowed = await authorizer.authorize(
        idp.mint(scopes=("rag:query",)),
        action="rag:query",
        resource=restricted,
        environment={"purpose_of_use": "R"},
    )
    assert allowed.decision is Decision.ALLOW
    assert allowed.matched_rules == ("allow-scoped-access",)
    ceiling = await authorizer.authorize(
        idp.mint(max_cls=2),
        action="rag:query",
        resource=restricted,
        environment={"purpose_of_use": "R"},
    )
    assert ceiling.decision is Decision.DENY
    assert ceiling.reason == "resource exceeds caller classification"
    out_of_scope = await authorizer.authorize(
        idp.mint(scopes=("rag:search",)),
        action="rag:query",
        resource=restricted,
        environment={"purpose_of_use": "R"},
    )
    assert out_of_scope.decision is Decision.DENY
