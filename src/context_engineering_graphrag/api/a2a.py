"""A2A (Agent2Agent protocol 0.3) surface: Agent Card and JSON-RPC ``message/send``.

A peer agent sends a user message whose text part is the question and whose
``metadata`` names the collection and the skill:

* ``grounded-answer`` (default) -> agentic RAG answer with citations,
* ``context-pack``              -> budgeted, attributed context.

The call is synchronous: the result is a completed ``Task`` with a text
artifact and a data artifact carrying citations, groundedness and the
release decision. Credentials are the same bearer tokens as the REST API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from agentic_core.domain.errors import AgenticCoreError
from agentic_core.domain.identifiers import new_id
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from context_engineering_graphrag import __version__
from context_engineering_graphrag.api.routes import Deps, bearer_token
from context_engineering_graphrag.application.service import CHANNEL, Caller
from context_engineering_graphrag.domain.errors import PlatformError
from context_engineering_graphrag.domain.models import CollectionName
from context_engineering_graphrag.orchestration.workflow import (
    ContextRequest,
    QueryRequest,
    RetrievalMode,
)
from context_engineering_graphrag.runtime.container import Container

router = APIRouter()

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SERVER_ERROR = -32000
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005

SKILLS = (
    {
        "id": "grounded-answer",
        "name": "Grounded answer",
        "description": "Answer a question from an enterprise collection with per-sentence "
        "citations, a groundedness score and a governance release decision.",
        "tags": ["rag", "graphrag", "citations", "groundedness"],
        "examples": ["Which supplier does the assay platform depend on, and since when?"],
        "inputModes": ["text/plain"],
        "outputModes": ["text/plain", "application/json"],
    },
    {
        "id": "context-pack",
        "name": "Context pack",
        "description": "Token-budgeted, de-duplicated, attributed context (passages, graph facts, "
        "community summaries) for the caller's own model.",
        "tags": ["context-engineering", "retrieval", "token-budget"],
        "examples": ["Build a 2,000-token context about vendor concentration risk."],
        "inputModes": ["text/plain"],
        "outputModes": ["application/json"],
    },
)


def agent_card(public_url: str) -> dict[str, Any]:
    return {
        "protocolVersion": "0.3.0",
        "name": "context-engineering-graphrag",
        "description": "Agentic RAG and GraphRAG context engineering: hybrid retrieval, temporal "
        "knowledge graph, community search, reranking, token budgeting and grounded answers.",
        "url": f"{public_url}/a2a",
        "preferredTransport": "JSONRPC",
        "version": __version__,
        "provider": {
            "organization": "kogunlowo123",
            "url": "https://github.com/kogunlowo123/context-engineering-graphrag",
        },
        "documentationUrl": "https://github.com/kogunlowo123/context-engineering-graphrag#readme",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": list(SKILLS),
        "securitySchemes": {
            "bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Agent credential issued by agent-identity-fabric",
            }
        },
        "security": [{"bearer": []}],
    }


class TextPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4000)


class SkillMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    collection: CollectionName
    skill: Literal["grounded-answer", "context-pack"] = "grounded-answer"
    mode: RetrievalMode = RetrievalMode.AUTO
    k: int = Field(default=8, ge=1, le=50)
    purpose_of_use: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class A2AMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "agent"]
    parts: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    messageId: str = Field(min_length=1, max_length=200)  # noqa: N815 - A2A wire name
    contextId: str | None = Field(default=None, max_length=200)  # noqa: N815
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: A2AMessage
    metadata: dict[str, Any] = Field(default_factory=dict)


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _error(rpc_id: Any, exc: RpcError) -> JSONResponse:
    error: dict[str, Any] = {"code": exc.code, "message": exc.message}
    if exc.data:
        error["data"] = exc.data
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": error})


@router.get("/.well-known/agent-card.json", tags=["a2a"])
async def card(c: Deps) -> dict[str, Any]:
    return agent_card(c.settings.public_url_str)


def _parse(envelope: Any) -> tuple[SendParams, SkillMetadata, str]:
    if (
        not isinstance(envelope, dict)
        or envelope.get("jsonrpc") != "2.0"
        or "method" not in envelope
    ):
        raise RpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 request")
    method = envelope["method"]
    if method in {"message/stream", "tasks/resubscribe", "tasks/pushNotificationConfig/set"}:
        raise RpcError(UNSUPPORTED_OPERATION, f"{method} is not supported by this agent")
    if method != "message/send":
        raise RpcError(METHOD_NOT_FOUND, f"unknown method {str(method)[:100]!r}")
    try:
        params = SendParams.model_validate(envelope.get("params"))
        meta = SkillMetadata.model_validate({**params.metadata, **params.message.metadata})
        texts = [
            TextPart.model_validate(p).text for p in params.message.parts if p.get("kind") == "text"
        ]
    except ValidationError as exc:
        raise RpcError(
            INVALID_PARAMS,
            "invalid message/send params",
            {"errors": exc.errors(include_url=False, include_context=False, include_input=False)},
        ) from exc
    if not texts:
        raise RpcError(CONTENT_TYPE_NOT_SUPPORTED, "message needs a text part")
    return params, meta, "\n".join(texts)[:4000]


@router.post("/a2a", tags=["a2a"], response_model=None)
async def jsonrpc(
    request: Request,
    c: Deps,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """JSON-RPC 2.0 endpoint; supports ``message/send``."""
    CHANNEL.set("a2a")
    token = bearer_token(authorization)
    try:
        envelope = await request.json()
    except ValueError:
        return _error(None, RpcError(PARSE_ERROR, "invalid JSON"))
    rpc_id = envelope.get("id") if isinstance(envelope, dict) else None
    try:
        params, meta, question = _parse(envelope)
    except RpcError as exc:
        return _error(rpc_id, exc)
    outcome = "ok"
    try:
        task = await _run(
            c,
            Caller(token=token, purpose_of_use=meta.purpose_of_use),
            meta,
            question,
            params.message,
        )
    except PlatformError as exc:
        outcome = exc.code
        if exc.status == 401:
            raise
        return _error(rpc_id, RpcError(SERVER_ERROR, exc.message, {"error": exc.code}))
    except ValidationError as exc:
        outcome = "invalid_params"
        return _error(rpc_id, RpcError(INVALID_PARAMS, str(exc)[:500]))
    except AgenticCoreError as exc:
        outcome = exc.code
        return _error(rpc_id, RpcError(SERVER_ERROR, str(exc)[:500], {"error": exc.code}))
    finally:
        c.metrics.tool_calls.labels("a2a", meta.skill, outcome).inc()
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task})


async def _run(
    c: Container, caller: Caller, meta: SkillMetadata, question: str, message: A2AMessage
) -> dict[str, Any]:
    context_id = message.contextId or new_id()
    artifacts: list[dict[str, Any]]
    if meta.skill == "context-pack":
        context = await c.platform.context(
            caller,
            meta.collection,
            ContextRequest(
                query=question, mode=meta.mode, k=meta.k, purpose_of_use=meta.purpose_of_use
            ),
        )
        artifacts = [
            {
                "artifactId": new_id(),
                "name": "context-pack",
                "parts": [{"kind": "data", "data": context.model_dump(mode="json")}],
            }
        ]
        state = "completed"
    else:
        result = await c.platform.query(
            caller,
            meta.collection,
            QueryRequest(
                query=question, mode=meta.mode, k=meta.k, purpose_of_use=meta.purpose_of_use
            ),
        )
        data = result.model_dump(mode="json", exclude={"trace"})
        parts: list[dict[str, Any]] = []
        if result.answer is not None:
            parts.append({"kind": "text", "text": result.answer})
        parts.append({"kind": "data", "data": data})
        artifacts = [{"artifactId": new_id(), "name": "grounded-answer", "parts": parts}]
        state = "completed" if result.answer is not None else "rejected"
    return {
        "kind": "task",
        "id": new_id(),
        "contextId": context_id,
        "status": {"state": state, "timestamp": datetime.now(UTC).isoformat()},
        "artifacts": artifacts,
        "history": [message.model_dump(mode="json")],
    }
