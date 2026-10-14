"""Inbound authorization of agent and user requests.

``FabricAuthorizer`` (production) delegates every decision to
agent-identity-fabric ``/v1/authorize``: the fabric validates the bearer
credential (signature, expiry, credential epoch, revocation, delegation
chain) and evaluates its policy. If the fabric is unreachable the request
fails closed with 503.

``LocalAuthorizer`` is used when no fabric URL is configured. It verifies
the bearer JWT against a configured issuer's JWKS and evaluates the local
policy set with the agentic-core PolicyEngine. There is no mode that allows
unauthenticated access.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import jwt
from agentic_core.domain.classification import DataClassification, Decision, RiskLevel
from agentic_core.policy import PolicyEngine, PolicyRequest, Resource, Subject
from pydantic import BaseModel, ConfigDict

from context_engineering_graphrag.domain.errors import UnauthorizedError
from context_engineering_graphrag.domain.ports import Clock
from context_engineering_graphrag.integrations.identity_fabric import (
    AuthorizeCall,
    IdentityFabricClient,
)


class AccessDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    principal: str
    action: str
    resource: str
    matched_rules: tuple[str, ...]
    policy_version: str
    policy_digest: str
    reason: str
    delegation_chain: tuple[str, ...] = ()
    source: str


class Authorizer(Protocol):
    @property
    def name(self) -> str: ...

    async def authorize(
        self, token: str, *, action: str, resource: Resource, environment: dict[str, Any]
    ) -> AccessDecision: ...


class FabricAuthorizer:
    def __init__(self, client: IdentityFabricClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "identity-fabric"

    async def authorize(
        self, token: str, *, action: str, resource: Resource, environment: dict[str, Any]
    ) -> AccessDecision:
        decision = await self._client.authorize(
            token,
            AuthorizeCall(
                action=action,
                resource_type=resource.type,
                resource_id=resource.id,
                resource_classification=resource.classification,
                resource_attributes=dict(resource.attributes),
                environment=environment,
            ),
        )
        return AccessDecision(
            decision=decision.decision,
            principal=decision.agent,
            action=action,
            resource=f"{resource.type}:{resource.id}",
            matched_rules=decision.matched_rules,
            policy_version=decision.policy_version,
            policy_digest=decision.policy_digest,
            reason=decision.reason,
            delegation_chain=decision.delegation_chain,
            source=self.name,
        )


class KeyResolver(Protocol):
    async def key_for(self, kid: str) -> Any: ...


class StaticJwks:
    def __init__(self, jwks: dict[str, Any]) -> None:
        self._keys = {k["kid"]: jwt.PyJWK(k).key for k in jwks.get("keys", []) if "kid" in k}

    async def key_for(self, kid: str) -> Any:
        key = self._keys.get(kid)
        if key is None:
            raise UnauthorizedError(f"unknown signing key {kid!r}")
        return key


class RemoteJwks:
    """JWKS cache refreshed on TTL expiry or unknown ``kid``, rate limited."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient,
        ttl_s: float = 300.0,
        min_refresh_interval_s: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._client = client
        self._ttl = ttl_s
        self._min_refresh = min_refresh_interval_s
        self._monotonic = monotonic
        self._keys: dict[str, Any] = {}
        self._fetched_at: float | None = None

    async def key_for(self, kid: str) -> Any:
        now = self._monotonic()
        stale = self._fetched_at is None or now - self._fetched_at > self._ttl
        may_refresh = self._fetched_at is None or now - self._fetched_at >= self._min_refresh
        if stale or (kid not in self._keys and may_refresh):
            try:
                response = await self._client.get(self._url)
                response.raise_for_status()
                document = response.json()
                self._keys = {
                    k["kid"]: jwt.PyJWK(k).key for k in document.get("keys", []) if "kid" in k
                }
                self._fetched_at = now
            except (httpx.HTTPError, ValueError, KeyError, jwt.PyJWKError) as exc:
                if kid not in self._keys:
                    raise UnauthorizedError("cannot fetch issuer signing keys") from exc
        key = self._keys.get(kid)
        if key is None:
            raise UnauthorizedError(f"unknown signing key {kid!r}")
        return key


class VerifiedToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str
    scopes: frozenset[str]
    max_classification: DataClassification
    risk: RiskLevel
    delegation_chain: tuple[str, ...]


class JwtVerifier:
    ALGORITHMS = ("EdDSA", "ES256", "RS256")
    LEEWAY_S = 30

    def __init__(self, resolver: KeyResolver, *, issuer: str, audience: str, clock: Clock) -> None:
        self._resolver = resolver
        self._issuer = issuer
        self._audience = audience
        self._clock = clock

    async def verify(self, token: str) -> VerifiedToken:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise UnauthorizedError("malformed bearer token") from exc
        algorithm = header.get("alg")
        kid = header.get("kid")
        if algorithm not in self.ALGORITHMS or not isinstance(kid, str):
            raise UnauthorizedError("unsupported token algorithm or missing kid")
        key = await self._resolver.key_for(kid)
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                    "require": ["exp", "iat", "sub"],
                },
            )
        except jwt.PyJWTError as exc:
            raise UnauthorizedError(f"invalid bearer credential ({type(exc).__name__})") from exc
        now = self._clock.now().timestamp()
        if int(claims["exp"]) + self.LEEWAY_S <= now:
            raise UnauthorizedError("bearer token expired")
        if (
            int(claims["iat"]) - self.LEEWAY_S > now
            or int(claims.get("nbf", 0)) - self.LEEWAY_S > now
        ):
            raise UnauthorizedError("bearer token not yet valid")
        return VerifiedToken(
            subject=str(claims["sub"]),
            scopes=_scopes(claims),
            max_classification=DataClassification(
                max(0, min(3, int(claims.get("max_cls", DataClassification.INTERNAL))))
            ),
            risk=RiskLevel(max(1, min(4, int(claims.get("risk", RiskLevel.LOW))))),
            delegation_chain=tuple(_chain(claims.get("act"))),
        )


def _scopes(claims: dict[str, Any]) -> frozenset[str]:
    raw = claims.get("scope", claims.get("scp", ""))
    if isinstance(raw, str):
        return frozenset(raw.split())
    if isinstance(raw, list):
        return frozenset(str(s) for s in raw)
    return frozenset()


def _chain(act: Any) -> list[str]:
    out: list[str] = []
    depth = 0
    while isinstance(act, dict) and isinstance(act.get("sub"), str) and depth < 10:
        out.append(act["sub"])
        act = act.get("act")
        depth += 1
    return out


class LocalAuthorizer:
    def __init__(self, verifier: JwtVerifier, policy: PolicyEngine) -> None:
        self._verifier = verifier
        self._policy = policy

    @property
    def name(self) -> str:
        return "local-policy"

    async def authorize(
        self, token: str, *, action: str, resource: Resource, environment: dict[str, Any]
    ) -> AccessDecision:
        verified = await self._verifier.verify(token)
        subject = Subject(
            id=verified.subject,
            kind="agent",
            attributes={
                "status": "active",
                "scopes": sorted(verified.scopes),
                "max_classification": int(verified.max_classification),
                "delegation_depth": len(verified.delegation_chain),
                "delegated": bool(verified.delegation_chain),
                "action_in_scope": any(fnmatch.fnmatchcase(action, s) for s in verified.scopes),
            },
        )
        decision = self._policy.evaluate(
            PolicyRequest(
                subject=subject,
                action=action,
                resource=resource,
                risk=verified.risk,
                environment=environment,
            )
        )
        outcome, reason, rules = decision.decision, decision.reason, decision.matched_rules
        if resource.classification > verified.max_classification and outcome is not Decision.DENY:
            outcome, reason, rules = Decision.DENY, "resource exceeds caller classification", ()
        return AccessDecision(
            decision=outcome,
            principal=verified.subject,
            action=action,
            resource=f"{resource.type}:{resource.id}",
            matched_rules=rules,
            policy_version=decision.policy_version,
            policy_digest=decision.policy_digest,
            reason=reason,
            delegation_chain=verified.delegation_chain,
            source=self.name,
        )
