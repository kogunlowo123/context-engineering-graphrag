"""Settings validation, composition root wiring and model-backed features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from agentic_core.domain.classification import DataClassification
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from context_engineering_graphrag.application.knowledge_graph import GraphBuildOptions
from context_engineering_graphrag.application.service import Caller, CreateCollection
from context_engineering_graphrag.embeddings.providers import (
    BedrockTitanEmbedder,
    OpenAICompatibleEmbedder,
)
from context_engineering_graphrag.evaluation.groundedness import ModelEntailmentJudge
from context_engineering_graphrag.generation.answer import ModelAnswerGenerator
from context_engineering_graphrag.graph.extraction import ModelEntityExtractor
from context_engineering_graphrag.graph.summarize import ModelCommunitySummarizer
from context_engineering_graphrag.orchestration.workflow import Disposition, QueryRequest
from context_engineering_graphrag.retrieval.rerank import HttpCrossEncoder
from context_engineering_graphrag.runtime.container import (
    _audit_signer,
    build_container,
    build_embedder,
    build_reranker,
    build_router,
    load_gazetteer,
)
from context_engineering_graphrag.security.authorization import FabricAuthorizer, StaticJwks
from tests.conftest import ROOT, FakeClock, FakeIdp, corpus_document, make_settings
from tests.unit.test_context_generation_evaluation import model_router
from tests.unit.test_shipped_config import synthetic_env


def pem(key: Any) -> bytes:
    return bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


class TestSettings:
    def test_production_requirements(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as info:
            make_settings(tmp_path, environment="production", embedding_provider="hashing")
        message = str(info.value)
        assert "AUDIT_KEY_FILE is required" in message
        assert "hashing embedder" in message

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"local_issuer": None}, "GRAPHRAG_IDENTITY_FABRIC_URL"),
            ({"generator": "model"}, "PROVIDERS_FILE"),
            ({"vector_backend": "pgvector"}, "PostgreSQL"),
            ({"embedding_provider": "bedrock"}, "AWS_REGION"),
            ({"embedding_provider": "openai_compatible"}, "EMBEDDING_BASE_URL"),
            ({"reranker": "http"}, "RERANKER_URL"),
            ({"chunk_max_tokens": 64, "chunk_overlap_tokens": 40}, "overlap"),
        ],
    )
    def test_inconsistent_settings_fail_closed(
        self, tmp_path: Path, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            make_settings(tmp_path, **overrides)

    def test_allowed_hosts(self, tmp_path: Path) -> None:
        settings = make_settings(
            tmp_path, public_url="https://rag.corp.test/", additional_hosts=["graphrag.ns.svc:80"]
        )
        assert settings.public_url_str == "https://rag.corp.test"
        assert {"rag.corp.test", "rag.corp.test:*", "graphrag.ns.svc:80", "localhost:*"} <= set(
            settings.allowed_hosts
        )
        assert not settings.uses_models


class TestBuilders:
    def test_embedders_and_rerankers(self, tmp_path: Path) -> None:
        bedrock = build_embedder(
            make_settings(
                tmp_path,
                embedding_provider="bedrock",
                aws_region="us-east-1",
                embedding_dimension=512,
            )
        )
        assert isinstance(bedrock, BedrockTitanEmbedder)
        assert bedrock.dimension == 512
        compatible = build_embedder(
            make_settings(
                tmp_path,
                embedding_provider="openai_compatible",
                embedding_base_url="https://tei.corp.test",
                embedding_api_key_env="TEI_KEY",
            ),
            env={"TEI_KEY": "k"},
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )
        assert isinstance(compatible, OpenAICompatibleEmbedder)
        with pytest.raises(ValueError, match="TEI_KEY is not set"):
            build_embedder(
                make_settings(
                    tmp_path,
                    embedding_provider="openai_compatible",
                    embedding_base_url="https://tei.corp.test",
                    embedding_api_key_env="TEI_KEY",
                ),
                env={},
            )
        reranker = build_reranker(
            make_settings(
                tmp_path, reranker="http", reranker_url="http://rr.svc", reranker_api_key_env="RR"
            ),
            env={"RR": "x"},
        )
        assert isinstance(reranker, HttpCrossEncoder)

    def test_router_from_shipped_providers(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, providers_file=ROOT / "configs" / "providers.yaml")
        router = build_router(settings, env=synthetic_env())
        assert next(p.profile.name for p in router.plan(_request())) == "bedrock-claude"
        with pytest.raises(ValueError, match="not configured"):
            build_router(make_settings(tmp_path))

    def test_gazetteer_and_audit_key(self, tmp_path: Path) -> None:
        good = tmp_path / "gazetteer.yaml"
        good.write_text("Orion Analyzer: product\nHPLC: method\n", encoding="utf-8")
        assert load_gazetteer(make_settings(tmp_path, gazetteer_file=good)) == {
            "Orion Analyzer": "product",
            "HPLC": "method",
        }
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="gazetteer"):
            load_gazetteer(make_settings(tmp_path, gazetteer_file=bad))
        assert load_gazetteer(make_settings(tmp_path)) == {}
        wrong = tmp_path / "ec.pem"
        wrong.write_bytes(pem(ec.generate_private_key(ec.SECP256R1())))
        with pytest.raises(ValueError, match="Ed25519"):
            _audit_signer(make_settings(tmp_path, audit_key_file=wrong))


def _request() -> Any:
    from agentic_core.models.types import Message, ModelRequest, Role

    return ModelRequest(messages=(Message(role=Role.USER, content="hi"),), task="answer")


async def test_production_wiring_uses_fabric_and_persistent_audit_key(tmp_path: Path) -> None:
    key_file = tmp_path / "audit-2026-10.pem"
    key_file.write_bytes(pem(Ed25519PrivateKey.generate()))
    settings = make_settings(
        tmp_path,
        environment="production",
        audit_key_file=key_file,
        identity_fabric_url="https://identity.corp.test",
        local_issuer=None,
        embedding_provider="openai_compatible",
        embedding_base_url="https://tei.corp.test",
        embedding_auth="none",
        gazetteer_file=None,
    )
    container = await build_container(settings)
    try:
        assert isinstance(container.platform.authorizer, FabricAuthorizer)
        assert container.platform.audit._key_id == "audit-2026-10"
        assert container.router is None
    finally:
        await container.aclose()


def scripted_model(payload: dict[str, Any]) -> str:
    """Answers like a well-behaved model, keyed on the system prompt of each task."""
    system = payload["messages"][0]["content"]
    content = "\n".join(m["content"] for m in payload["messages"][1:])
    if "extract a knowledge graph" in system:
        entities = [
            n
            for n in ("Acme Diagnostics", "Beta Reagents", "Northwind Holdings", "Lyon")
            if n in content
        ]
        relations = []
        if "Northwind Holdings acquired Beta Reagents in March 2024." in content:
            relations.append(
                {
                    "source": "Northwind Holdings",
                    "target": "Beta Reagents",
                    "type": "acquired",
                    "evidence": "Northwind Holdings acquired Beta Reagents in March 2024.",
                }
            )
        if "Acme Diagnostics depends on Beta Reagents for assay reagents." in content:
            relations.append(
                {
                    "source": "Acme Diagnostics",
                    "target": "Beta Reagents",
                    "type": "depends_on",
                    "evidence": "Acme Diagnostics depends on Beta Reagents for assay reagents.",
                }
            )
        return json.dumps(
            {
                "entities": [{"name": n, "type": "organization"} for n in entities],
                "relations": relations,
            }
        )
    if "Summarise the knowledge-graph community" in system:
        return "Beta Reagents sits between Acme Diagnostics and its acquirer Northwind Holdings."
    if "Decide whether the evidence entails the claim" in system:
        return '{"label": "supported", "score": 0.9}'
    return "Northwind Holdings acquired Beta Reagents in March 2024 [S1]."


async def test_all_model_backed_features(tmp_path: Path, idp: FakeIdp, clock: FakeClock) -> None:
    settings = make_settings(
        tmp_path,
        generator="model",
        extractor="model",
        summarizer="model",
        judge="model",
        providers_file=ROOT / "configs" / "providers.yaml",
    )
    container = await build_container(
        settings,
        clock=clock,
        jwks_resolver=StaticJwks(idp.jwks()),
        router=model_router(scripted_model),
    )
    platform = container.platform
    try:
        assert isinstance(platform.pipeline._extractor, ModelEntityExtractor)
        assert isinstance(platform.graph.summarizer, ModelCommunitySummarizer)
        assert isinstance(platform.workflow.generator, ModelAnswerGenerator)
        assert isinstance(platform.evaluator.judge, ModelEntailmentJudge)
        caller = Caller(token=idp.mint(), purpose_of_use="research")
        await platform.create_collection(
            caller,
            CreateCollection(
                name="models", classification=DataClassification.CONFIDENTIAL, owner="t"
            ),
        )
        for doc in ("supplier-overview", "acquisition-2024"):
            await platform.ingest(caller, "models", corpus_document(doc))
        report = await platform.build_graph(caller, "models", GraphBuildOptions())
        assert report.summarizer == "model"
        assert report.communities >= 1
        communities = await platform.communities(caller, "models", level=None)
        assert communities[0].summary.startswith("Beta Reagents sits between")
        result = await platform.query(
            caller, "models", QueryRequest(query="Who acquired Beta Reagents?")
        )
        assert result.disposition is Disposition.ANSWERED
        assert result.generator == "model"
        assert result.groundedness.judge == "model-entailment"
    finally:
        await container.aclose()


async def test_model_summarizer_falls_back(tmp_path: Path, idp: FakeIdp, clock: FakeClock) -> None:
    settings = make_settings(
        tmp_path, summarizer="model", providers_file=ROOT / "configs" / "providers.yaml"
    )
    container = await build_container(
        settings,
        clock=clock,
        jwks_resolver=StaticJwks(idp.jwks()),
        router=model_router("x", status=503),
    )
    try:
        caller = Caller(token=idp.mint(), purpose_of_use="research")
        await container.platform.create_collection(
            caller,
            CreateCollection(
                name="fallback", classification=DataClassification.CONFIDENTIAL, owner="t"
            ),
        )
        await container.platform.ingest(caller, "fallback", corpus_document("acquisition-2024"))
        await container.platform.build_graph(caller, "fallback", GraphBuildOptions())
        summarizer = container.platform.graph.summarizer
        assert isinstance(summarizer, ModelCommunitySummarizer)
        assert summarizer.fallbacks >= 1
        summaries = await container.platform.communities(caller, "fallback", level=None)
        assert all(c.summary.startswith("Community of") for c in summaries)
    finally:
        await container.aclose()
