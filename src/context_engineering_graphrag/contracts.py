"""Export published contracts: JSON Schemas for inputs/outputs and the OpenAPI document.

python -m context_engineering_graphrag.contracts            # rewrite schemas/ and docs/api/
python -m context_engineering_graphrag.contracts --check    # exit 1 if committed files are stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from context_engineering_graphrag.api.app import create_app
from context_engineering_graphrag.application.knowledge_graph import (
    GraphBuildOptions,
    GraphBuildReport,
    Neighbourhood,
)
from context_engineering_graphrag.application.service import (
    CreateCollection,
    GroundednessRequest,
    RetrievalEvaluationRequest,
    SearchRequest,
    SearchResult,
)
from context_engineering_graphrag.evaluation.groundedness import GroundednessReport
from context_engineering_graphrag.evaluation.retrieval_metrics import RetrievalReport
from context_engineering_graphrag.ingestion.service import IngestDocument, IngestResult
from context_engineering_graphrag.orchestration.workflow import (
    ContextRequest,
    ContextResult,
    QueryRequest,
    QueryResult,
)

MODELS: dict[str, type[BaseModel]] = {
    "create-collection.request": CreateCollection,
    "ingest-document.request": IngestDocument,
    "ingest-document.response": IngestResult,
    "search.request": SearchRequest,
    "search.response": SearchResult,
    "context.request": ContextRequest,
    "context.response": ContextResult,
    "query.request": QueryRequest,
    "query.response": QueryResult,
    "graph-build.request": GraphBuildOptions,
    "graph-build.response": GraphBuildReport,
    "neighbourhood.response": Neighbourhood,
    "groundedness.request": GroundednessRequest,
    "groundedness.response": GroundednessReport,
    "retrieval-evaluation.request": RetrievalEvaluationRequest,
    "retrieval-evaluation.response": RetrievalReport,
}


async def _no_container() -> Any:  # pragma: no cover - never started
    raise RuntimeError("contract export does not start the application")


def documents() -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {
        f"schemas/{name}.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/kogunlowo123/context-engineering-graphrag/schemas/"
            f"{name}.schema.json",
            **model.model_json_schema(
                mode="serialization" if name.endswith("response") else "validation"
            ),
        }
        for name, model in MODELS.items()
    }
    app: FastAPI = create_app(_no_container)
    docs["docs/api/openapi.json"] = app.openapi()
    return docs


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    base = root or Path.cwd()
    stale: list[str] = []
    for relative, document in documents().items():
        path = base / relative
        content = render(document)
        if "--check" in args:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        sys.stderr.write(
            "stale contracts (run python -m context_engineering_graphrag.contracts): "
            + ", ".join(stale)
            + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
