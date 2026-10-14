"""REST routes. Every /v1 route requires an agent or user bearer credential."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict

from context_engineering_graphrag.application.knowledge_graph import (
    GraphBuildOptions,
    GraphBuildReport,
    Neighbourhood,
)
from context_engineering_graphrag.application.service import (
    Caller,
    CreateCollection,
    DocumentView,
    GroundednessRequest,
    RetrievalEvaluationRequest,
    SearchRequest,
    SearchResult,
)
from context_engineering_graphrag.domain.errors import UnauthorizedError
from context_engineering_graphrag.domain.models import Collection, Community, Document, Entity
from context_engineering_graphrag.evaluation.groundedness import GroundednessReport
from context_engineering_graphrag.evaluation.retrieval_metrics import RetrievalReport
from context_engineering_graphrag.infrastructure.database import ErasureReport
from context_engineering_graphrag.ingestion.service import IngestDocument, IngestResult
from context_engineering_graphrag.orchestration.workflow import (
    ContextRequest,
    ContextResult,
    QueryRequest,
    QueryResult,
)
from context_engineering_graphrag.runtime.container import Container

router = APIRouter()


class CollectionList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[Collection, ...]


class DocumentList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[Document, ...]
    limit: int
    offset: int


class EntityList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[Entity, ...]


class CommunityList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[Community, ...]


def container(request: Request) -> Container:
    value: Container = request.app.state.container
    return value


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthorizedError("bearer credential required")
    credential = authorization[7:].strip()
    if not credential or len(credential) > 8192:
        raise UnauthorizedError("bearer credential required")
    return credential


def caller(
    authorization: Annotated[str | None, Header()] = None,
    x_purpose_of_use: Annotated[str | None, Header(max_length=64)] = None,
) -> Caller:
    return Caller(token=bearer_token(authorization), purpose_of_use=x_purpose_of_use or None)


Deps = Annotated[Container, Depends(container)]
Who = Annotated[Caller, Depends(caller)]
CollectionPath = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")]
DocumentPath = Annotated[str, Path(min_length=1, max_length=200)]


# ----------------------------------------------------------------- collections


@router.post("/v1/collections", status_code=201, tags=["collections"])
async def create_collection(body: CreateCollection, c: Deps, who: Who) -> Collection:
    return await c.platform.create_collection(who, body)


@router.get("/v1/collections", tags=["collections"])
async def list_collections(c: Deps, who: Who) -> CollectionList:
    return CollectionList(items=tuple(await c.platform.list_collections(who)))


@router.get("/v1/collections/{collection}", tags=["collections"])
async def get_collection(collection: CollectionPath, c: Deps, who: Who) -> Collection:
    return await c.platform.get_collection(who, collection)


# ------------------------------------------------------------------- documents


@router.post("/v1/collections/{collection}/documents", tags=["documents"])
async def ingest(
    collection: CollectionPath, body: IngestDocument, c: Deps, who: Who
) -> IngestResult:
    return await c.platform.ingest(who, collection, body)


@router.get("/v1/collections/{collection}/documents", tags=["documents"])
async def list_documents(
    collection: CollectionPath,
    c: Deps,
    who: Who,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> DocumentList:
    items = await c.platform.list_documents(who, collection, limit=limit, offset=offset)
    return DocumentList(items=tuple(items), limit=limit, offset=offset)


@router.get("/v1/collections/{collection}/documents/{document_id:path}", tags=["documents"])
async def get_document(
    collection: CollectionPath, document_id: DocumentPath, c: Deps, who: Who
) -> DocumentView:
    return await c.platform.get_document(who, collection, document_id)


@router.delete("/v1/collections/{collection}/documents/{document_id:path}", tags=["documents"])
async def erase_document(
    collection: CollectionPath, document_id: DocumentPath, c: Deps, who: Who
) -> ErasureReport:
    """Right-to-erasure: removes the document, its chunks, vectors, mentions, assertions,
    orphaned entities and every community summary of the collection."""
    return await c.platform.erase_document(who, collection, document_id)


# ------------------------------------------------------------------- retrieval


@router.post("/v1/collections/{collection}/search", tags=["retrieval"])
async def search(
    collection: CollectionPath, body: SearchRequest, c: Deps, who: Who
) -> SearchResult:
    return await c.platform.search(who, collection, body)


@router.post("/v1/collections/{collection}/context", tags=["retrieval"])
async def build_context(
    collection: CollectionPath, body: ContextRequest, c: Deps, who: Who
) -> ContextResult:
    """Budgeted, attributed context pack for an external agent's own model call."""
    return await c.platform.context(who, collection, body)


@router.post("/v1/collections/{collection}/query", tags=["retrieval"])
async def query(collection: CollectionPath, body: QueryRequest, c: Deps, who: Who) -> QueryResult:
    """Agentic RAG: plan, retrieve, budget, answer with citations, check groundedness,
    and release the answer only if the governance policy allows it."""
    return await c.platform.query(who, collection, body)


# ----------------------------------------------------------------------- graph


@router.post("/v1/collections/{collection}/graph/build", tags=["graph"])
async def build_graph(
    collection: CollectionPath, c: Deps, who: Who, body: GraphBuildOptions | None = None
) -> GraphBuildReport:
    return await c.platform.build_graph(who, collection, body or GraphBuildOptions())


@router.get("/v1/collections/{collection}/entities", tags=["graph"])
async def entities(
    collection: CollectionPath,
    c: Deps,
    who: Who,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> EntityList:
    return EntityList(items=tuple(await c.platform.entities(who, collection, query=q, limit=limit)))


@router.get("/v1/collections/{collection}/entities/{entity_id}/neighbourhood", tags=["graph"])
async def neighbourhood(
    *,
    collection: CollectionPath,
    entity_id: Annotated[str, Path(pattern=r"^ent_[0-9a-f]{32}$")],
    c: Deps,
    who: Who,
    hops: Annotated[int, Query(ge=1, le=3)] = 1,
    as_of: datetime | None = None,
) -> Neighbourhood:
    return await c.platform.neighbourhood(who, collection, entity_id, hops=hops, as_of=as_of)


@router.get("/v1/collections/{collection}/communities", tags=["graph"])
async def communities(
    collection: CollectionPath,
    c: Deps,
    who: Who,
    level: Annotated[int | None, Query(ge=0, le=8)] = None,
) -> CommunityList:
    return CommunityList(items=tuple(await c.platform.communities(who, collection, level=level)))


# ------------------------------------------------------------------ evaluation


@router.post("/v1/evaluations/groundedness", tags=["evaluation"])
async def evaluate_groundedness(body: GroundednessRequest, c: Deps, who: Who) -> GroundednessReport:
    return await c.platform.evaluate_groundedness(who, body)


@router.post("/v1/collections/{collection}/evaluations/retrieval", tags=["evaluation"])
async def evaluate_retrieval(
    collection: CollectionPath, body: RetrievalEvaluationRequest, c: Deps, who: Who
) -> RetrievalReport:
    return await c.platform.evaluate_retrieval(who, collection, body)


# ------------------------------------------------------------------ operations


@router.get("/healthz", tags=["operations"], include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["operations"], include_in_schema=False)
async def readyz(c: Deps, response: Response) -> dict[str, str]:
    try:
        await c.store.ping()
    except Exception:
        response.status_code = 503
        return {"status": "database unavailable"}
    return {"status": "ready"}


@router.get("/metrics", tags=["operations"], include_in_schema=False)
async def metrics(c: Deps) -> Response:
    return Response(generate_latest(c.metrics.registry), media_type=CONTENT_TYPE_LATEST)
