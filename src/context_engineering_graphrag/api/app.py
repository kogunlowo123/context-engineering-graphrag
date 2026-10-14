"""FastAPI application factory: REST API, MCP server at /mcp, A2A at /a2a."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from agentic_core.domain.errors import (
    AgenticCoreError,
    PolicyDeniedError,
    ProviderUnavailableError,
    SecurityViolationError,
)
from agentic_core.observability import bind_correlation_id, get_logger
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from context_engineering_graphrag import __version__
from context_engineering_graphrag.api import a2a
from context_engineering_graphrag.api.mcp import build_mcp_server
from context_engineering_graphrag.api.routes import router
from context_engineering_graphrag.domain.errors import ApprovalRequiredError, PlatformError
from context_engineering_graphrag.runtime.container import Container

ContainerFactory = Callable[[], Awaitable[Container]]

log = get_logger("context_engineering_graphrag.api")


class RequireBearer:
    """Rejects MCP transport requests that carry no bearer credential at all.

    Per-tool authorization still runs for every call; this only stops
    anonymous probing of the MCP endpoint (including tool discovery).
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            value = headers.get(b"authorization", b"").decode("latin-1")
            if not value.lower().startswith("bearer ") or not value[7:].strip():
                response = JSONResponse(
                    {"error": "unauthorized", "error_description": "bearer credential required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


def create_app(factory: ContainerFactory, *, allowed_hosts: list[str] | None = None) -> FastAPI:
    holder: dict[str, Container] = {}
    mcp = build_mcp_server(lambda: holder["container"])
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts or ["127.0.0.1:*", "localhost:*"],
            allowed_origins=[],
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await factory()
        holder["container"] = container
        app.state.container = container
        log.info("startup", version=__version__)
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await container.aclose()

    app = FastAPI(
        title="Context Engineering GraphRAG",
        version=__version__,
        description=(
            "Agentic RAG and GraphRAG context engineering: ingestion with provenance, hybrid "
            "retrieval, temporal knowledge graph, community search, reranking, token budgeting, "
            "attributed answers and groundedness-gated release."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(a2a.router)
    for route in mcp_app.routes:
        if isinstance(route, Route):
            app.router.routes.append(Route(route.path, endpoint=RequireBearer(route.app)))

    @app.middleware("http")
    async def correlation_and_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-correlation-id")
        cid = incoming if incoming and len(incoming) <= 64 and incoming.isprintable() else None
        started = time.perf_counter()
        with bind_correlation_id(cid) as bound:
            response = await call_next(request)
        response.headers["X-Correlation-ID"] = bound
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        container: Container | None = getattr(request.app.state, "container", None)
        if container is not None:
            container.metrics.requests.labels(
                request.method, path, str(response.status_code)
            ).observe(time.perf_counter() - started)
        return response

    @app.exception_handler(PlatformError)
    async def platform_error(_: Request, exc: PlatformError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        body: dict[str, Any] = {"error": exc.code, "error_description": exc.message}
        if isinstance(exc, ApprovalRequiredError):
            body["matched_rules"] = list(exc.rules)
        return JSONResponse(status_code=exc.status, content=body, headers=headers)

    @app.exception_handler(AgenticCoreError)
    async def core_error(_: Request, exc: AgenticCoreError) -> JSONResponse:
        if isinstance(exc, PolicyDeniedError):
            status = 403
        elif isinstance(exc, SecurityViolationError):
            status = 422
        elif isinstance(exc, ProviderUnavailableError):
            status = 503
        else:
            status = 502
        return JSONResponse(
            status_code=status, content={"error": exc.code, "error_description": str(exc)[:500]}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "error_description": "request failed validation",
                "details": [
                    {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", ""))}
                    for e in exc.errors()
                ],
            },
        )

    return app
