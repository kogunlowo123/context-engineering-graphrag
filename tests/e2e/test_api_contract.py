"""Property-based API contract test: no generated input may cause a 5xx,
whether the caller is anonymous or fully authorized."""

from __future__ import annotations

from typing import Any

import pytest
import schemathesis
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, assume, settings

from tests.conftest import FakeIdp, auth

pytestmark = pytest.mark.e2e


@pytest.fixture
def api_schema(client: TestClient) -> Any:
    return schemathesis.openapi.from_dict(client.get("/openapi.json").json())


schema = schemathesis.pytest.from_fixture("api_schema")


def _send(client: TestClient, case: Any, extra_headers: dict[str, str]) -> Any:
    kwargs = case.as_transport_kwargs(base_url="")
    headers = {**(kwargs.get("headers") or {}), **extra_headers}
    return client.request(
        kwargs["method"],
        kwargs["url"],
        params=kwargs.get("params"),
        headers=headers,
        json=kwargs.get("json"),
        data=kwargs.get("data"),
        content=kwargs.get("content"),
    )


@schema.parametrize()
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_no_server_errors(case: Any, client: TestClient, idp: FakeIdp) -> None:
    # Header values must be transmittable over HTTP/1.1 at all.
    assume(all(str(v).isascii() for v in (case.headers or {}).values()))
    anonymous = _send(client, case, {})
    assert anonymous.status_code < 500, (case.method, case.path, anonymous.text)
    authorized = _send(client, case, auth(idp))
    assert authorized.status_code < 500, (case.method, case.path, authorized.text)
