"""Typed client for agent-identity-fabric.

Wire formats follow the fabric's published contract
(github.com/kogunlowo123/agent-identity-fabric, docs/api/openapi.json):

* ``POST /oauth2/token``       RFC 7523 jwt-bearer grant (form encoded)
* ``POST /oauth2/introspect``  RFC 7662 introspection (form encoded)
* ``POST /v1/authorize``       policy decision for the bearer's credential
"""

from __future__ import annotations

from typing import Any

import httpx
from agentic_core.domain.classification import DataClassification, Decision
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from context_engineering_graphrag.domain.errors import (
    DependencyUnavailableError,
    UnauthorizedError,
)

JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"


class IssuedCredential(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    token_type: str
    expires_in: int
    scope: str = ""


class Introspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    active: bool
    sub: str | None = None
    scope: str | None = None
    exp: int | None = None
    act: dict[str, Any] | None = None
    agent_status: str | None = None


class FabricDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    agent: str
    action: str
    resource: str
    matched_rules: tuple[str, ...] = ()
    policy_version: str
    policy_digest: str
    reason: str
    delegation_chain: tuple[str, ...] = ()


class AuthorizeCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: str
    resource_type: str
    resource_id: str = "*"
    resource_classification: DataClassification = DataClassification.INTERNAL
    resource_attributes: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)


class IdentityFabricClient:
    def __init__(self, base_url: str, *, http: httpx.AsyncClient) -> None:
        self._base = base_url.rstrip("/")
        self._http = http

    async def token(
        self, *, client_id: str, assertion: str, scopes: frozenset[str] | None = None
    ) -> IssuedCredential:
        form = {"grant_type": JWT_BEARER, "client_id": client_id, "assertion": assertion}
        if scopes:
            form["scope"] = " ".join(sorted(scopes))
        body = await self._post_form("/oauth2/token", form)
        return self._parse(IssuedCredential, body)

    async def introspect(self, token: str) -> Introspection:
        body = await self._post_form("/oauth2/introspect", {"token": token})
        return self._parse(Introspection, body)

    async def authorize(self, bearer: str, call: AuthorizeCall) -> FabricDecision:
        try:
            response = await self._http.post(
                f"{self._base}/v1/authorize",
                json=call.model_dump(mode="json"),
                headers={"Authorization": f"Bearer {bearer}"},
            )
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError(
                f"identity fabric unreachable: {type(exc).__name__}"
            ) from exc
        if response.status_code == 401:
            raise UnauthorizedError("agent credential rejected by identity fabric")
        return self._parse(FabricDecision, self._json(response))

    async def _post_form(self, path: str, form: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._http.post(f"{self._base}{path}", data=form)
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError(
                f"identity fabric unreachable: {type(exc).__name__}"
            ) from exc
        if response.status_code in (400, 401):
            raise UnauthorizedError(_oauth_error(response))
        return self._json(response)

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise DependencyUnavailableError(f"identity fabric returned {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise DependencyUnavailableError("identity fabric returned non-JSON") from exc
        if not isinstance(body, dict):
            raise DependencyUnavailableError("identity fabric returned an unexpected body")
        return body

    @staticmethod
    def _parse[M: BaseModel](model: type[M], body: dict[str, Any]) -> M:
        try:
            return model.model_validate(body)
        except ValidationError as exc:
            raise DependencyUnavailableError(
                f"identity fabric response does not match {model.__name__}"
            ) from exc


def _oauth_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"identity fabric rejected the request ({response.status_code})"
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        description = body.get("error_description")
        return f"{body['error']}: {description}" if isinstance(description, str) else body["error"]
    return f"identity fabric rejected the request ({response.status_code})"
