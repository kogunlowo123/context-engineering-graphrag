"""MCP server (official ``mcp`` SDK, streamable HTTP, stateless JSON responses).

Each tool has typed Pydantic input and output schemas, authenticates the
caller from the HTTP ``Authorization`` header of the MCP request, is
authorized through the same platform use case as the REST API (identity
fabric or local policy), and writes an audit event for its outcome.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, TypeVar

from agentic_core.domain.errors import AgenticCoreError
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from context_engineering_graphrag import __version__
from context_engineering_graphrag.api.routes import bearer_token
from context_engineering_graphrag.application.knowledge_graph import Neighbourhood
from context_engineering_graphrag.application.service import (
    CHANNEL,
    PRINCIPAL,
    Caller,
    GroundednessRequest,
    SearchRequest,
    SearchResult,
)
from context_engineering_graphrag.domain.errors import NotFoundError, PlatformError
from context_engineering_graphrag.domain.models import CollectionName
from context_engineering_graphrag.evaluation.groundedness import GroundednessReport
from context_engineering_graphrag.ingestion.service import IngestDocument, IngestResult
from context_engineering_graphrag.orchestration.workflow import (
    ContextRequest,
    ContextResult,
    QueryRequest,
    QueryResult,
)
from context_engineering_graphrag.runtime.container import Container

T = TypeVar("T")
ContainerRef = Callable[[], Container]

_INSTRUCTIONS = (
    "Enterprise context engineering for agents. Use search_passages for ranked evidence, "
    "build_context for a token-budgeted, attributed context pack to feed your own model, "
    "answer_question for a grounded answer with citations and a groundedness score, and "
    "explore_entity for knowledge-graph neighbourhoods (optionally as of a past date)."
)


def build_mcp_server(container: ContainerRef) -> MCPServer:
    server: MCPServer = MCPServer(
        "context-engineering-graphrag",
        version=__version__,
        instructions=_INSTRUCTIONS,
        log_level="WARNING",
    )

    async def call(ctx: Context, operation: str, run: Callable[[Caller], Awaitable[T]]) -> T:
        c = container()
        CHANNEL.set("mcp")
        PRINCIPAL.set(None)
        headers = ctx.headers or {}
        outcome = "ok"
        try:
            caller = Caller(
                token=bearer_token(headers.get("authorization")),
                purpose_of_use=headers.get("x-purpose-of-use") or None,
            )
            return await run(caller)
        except PlatformError as exc:
            outcome = exc.code
            raise ToolError(f"{exc.code}: {exc.message}") from exc
        except AgenticCoreError as exc:
            outcome = exc.code
            raise ToolError(f"{exc.code}: {exc}") from exc
        finally:
            c.metrics.tool_calls.labels("mcp", operation, outcome).inc()
            await c.platform.record(
                PRINCIPAL.get() or "unauthenticated",
                f"mcp.tool:{operation}",
                "mcp",
                outcome,
                {},
            )

    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)

    @server.tool(
        name="search_passages",
        description="Hybrid BM25 + dense retrieval with reciprocal rank fusion, cross-encoder "
        "reranking and MMR diversity over one collection.",
        annotations=read_only,
    )
    async def search_passages(
        collection: CollectionName, request: SearchRequest, ctx: Context
    ) -> SearchResult:
        return await call(
            ctx,
            "search_passages",
            lambda who: container().platform.search(who, collection, request),
        )

    @server.tool(
        name="build_context",
        description="Plan retrieval (passages, local graph or global communities), then compress, "
        "de-duplicate and pack sources into a token budget with full provenance.",
        annotations=read_only,
    )
    async def build_context(
        collection: CollectionName, request: ContextRequest, ctx: Context
    ) -> ContextResult:
        return await call(
            ctx, "build_context", lambda who: container().platform.context(who, collection, request)
        )

    @server.tool(
        name="answer_question",
        description="Grounded answer with per-sentence citations, a groundedness report and the "
        "governance release decision. Ungrounded answers are withheld.",
        annotations=read_only,
    )
    async def answer_question(
        collection: CollectionName, request: QueryRequest, ctx: Context
    ) -> QueryResult:
        return await call(
            ctx, "answer_question", lambda who: container().platform.query(who, collection, request)
        )

    @server.tool(
        name="explore_entity",
        description="Find an entity by name and return its knowledge-graph neighbourhood with "
        "relation validity intervals.",
        annotations=read_only,
    )
    async def explore_entity(
        collection: CollectionName,
        entity: Annotated[str, Field(min_length=1, max_length=200)],
        ctx: Context,
        hops: Annotated[int, Field(ge=1, le=3)] = 1,
    ) -> Neighbourhood:
        async def run(who: Caller) -> Neighbourhood:
            platform = container().platform
            found = await platform.entities(who, collection, query=entity, limit=1)
            if not found:
                raise NotFoundError(f"no entity matching {entity!r} in {collection}")
            return await platform.neighbourhood(
                who, collection, found[0].entity_id, hops=hops, as_of=None
            )

        return await call(ctx, "explore_entity", run)

    @server.tool(
        name="evaluate_groundedness",
        description="Judge each sentence of an attributed answer against the sources it cites.",
        annotations=read_only,
    )
    async def evaluate_groundedness(
        request: GroundednessRequest, ctx: Context
    ) -> GroundednessReport:
        return await call(
            ctx,
            "evaluate_groundedness",
            lambda who: container().platform.evaluate_groundedness(who, request),
        )

    @server.tool(
        name="ingest_document",
        description="Chunk, screen, embed and graph-extract one document version into a "
        "collection.",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True
        ),
    )
    async def ingest_document(
        collection: CollectionName, document: IngestDocument, ctx: Context
    ) -> IngestResult:
        return await call(
            ctx,
            "ingest_document",
            lambda who: container().platform.ingest(who, collection, document),
        )

    return server
