"""Embedders, BM25, rank fusion, MMR and rerankers."""

from __future__ import annotations

import io
import json
import math
from typing import Any

import httpx
import numpy as np
import pytest

from context_engineering_graphrag.domain.errors import DependencyUnavailableError
from context_engineering_graphrag.domain.text import terms
from context_engineering_graphrag.embeddings.hashing import HashingEmbedder
from context_engineering_graphrag.embeddings.providers import (
    BedrockTitanEmbedder,
    OpenAICompatibleEmbedder,
)
from context_engineering_graphrag.retrieval.bm25 import BM25Index
from context_engineering_graphrag.retrieval.fusion import (
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from context_engineering_graphrag.retrieval.rerank import (
    HttpCrossEncoder,
    LexicalCrossEncoder,
    _proximity,
)


def cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b))


class TestHashingEmbedder:
    async def test_deterministic_normalised_and_semantic(self) -> None:
        embedder = HashingEmbedder(256)
        a, b, c, empty = await embedder.embed(
            [
                "Beta Reagents supplies assay reagents",
                "assay reagent supplier Beta Reagents",
                "port congestion in Rotterdam",
                "",
            ]
        )
        assert embedder.model == "feature-hashing-v1-256"
        assert len(a) == embedder.dimension == 256
        assert math.isclose(float(np.linalg.norm(a)), 1.0, rel_tol=1e-9)
        assert a == (await HashingEmbedder(256).embed(["Beta Reagents supplies assay reagents"]))[0]
        assert cosine(a, b) > cosine(a, c) + 0.3
        assert not any(empty)
        await embedder.aclose()

    def test_dimension_floor(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            HashingEmbedder(8)


def embeddings_transport(
    *, dims: int = 4, status: int = 200, body: Any = None, seen: list[dict[str, Any]] | None = None
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if seen is not None:
            seen.append(
                {"payload": payload, "headers": dict(request.headers), "url": str(request.url)}
            )
        if body is not None:
            return httpx.Response(status, json=body)
        data = [
            {"index": i, "embedding": [float(i + 1)] + [1.0] * (dims - 1), "object": "embedding"}
            for i, _ in enumerate(payload["input"])
        ]
        return httpx.Response(
            status, json={"data": list(reversed(data)), "model": payload["model"]}
        )

    return httpx.MockTransport(handler)


class TestOpenAICompatibleEmbedder:
    async def test_batches_orders_and_normalises(self) -> None:
        seen: list[dict[str, Any]] = []
        embedder = OpenAICompatibleEmbedder(
            base_url="https://aoai.corp.test",
            path="/openai/deployments/emb/embeddings",
            query={"api-version": "2024-10-21"},
            model="text-embedding-3-large",
            dimension=4,
            api_key="k",
            auth="api-key",
            send_dimensions=True,
            batch_size=2,
            transport=embeddings_transport(seen=seen),
        )
        vectors = await embedder.embed(["a", "b", "c"])
        assert len(vectors) == 3
        assert len(seen) == 2
        assert seen[0]["headers"]["api-key"] == "k"
        assert seen[0]["payload"]["dimensions"] == 4
        assert "api-version=2024-10-21" in seen[0]["url"]
        assert vectors[0][0] < vectors[1][0]  # ordered by index despite reversed response
        assert all(math.isclose(sum(v * v for v in vec), 1.0) for vec in vectors)
        assert embedder.model == "text-embedding-3-large"
        assert embedder.dimension == 4
        await embedder.aclose()

    async def test_bearer_and_none_auth(self) -> None:
        seen: list[dict[str, Any]] = []
        bearer = OpenAICompatibleEmbedder(
            base_url="https://tei.corp.test",
            model="bge",
            dimension=4,
            api_key="t",
            transport=embeddings_transport(seen=seen),
        )
        await bearer.embed(["x"])
        assert seen[0]["headers"]["authorization"] == "Bearer t"
        assert "dimensions" not in seen[0]["payload"]
        anonymous = OpenAICompatibleEmbedder(
            base_url="http://tei.svc",
            model="bge",
            dimension=4,
            auth="none",
            transport=embeddings_transport(),
        )
        assert len(await anonymous.embed(["x"])) == 1
        with pytest.raises(ValueError, match="api key"):
            OpenAICompatibleEmbedder(base_url="https://x", model="m", dimension=4)

    @pytest.mark.parametrize(
        ("transport", "message"),
        [
            (embeddings_transport(status=503, body={"error": "busy"}), "returned 503"),
            (embeddings_transport(body={"nope": []}), "malformed"),
            (embeddings_transport(body={"data": []}), "wrong vector count"),
            (embeddings_transport(dims=3), "expected 4-dimensional"),
            (
                embeddings_transport(body={"data": [{"index": 0, "embedding": [0, 0, 0, 0]}]}),
                "degenerate",
            ),
        ],
    )
    async def test_provider_failures_fail_closed(
        self, transport: httpx.MockTransport, message: str
    ) -> None:
        embedder = OpenAICompatibleEmbedder(
            base_url="https://x.test", model="m", dimension=4, auth="none", transport=transport
        )
        with pytest.raises(DependencyUnavailableError, match=message):
            await embedder.embed(["x"])

    async def test_transport_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        embedder = OpenAICompatibleEmbedder(
            base_url="https://x.test",
            model="m",
            dimension=4,
            auth="none",
            transport=httpx.MockTransport(boom),
        )
        with pytest.raises(DependencyUnavailableError, match="ConnectError"):
            await embedder.embed(["x"])


class FakeBedrock:
    """Records InvokeModel calls; returns the Titan V2 response shape."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("ThrottlingException")
        body = json.loads(kwargs["body"])
        vector = [0.5] * body["dimensions"]
        return {
            "body": io.BytesIO(json.dumps({"embedding": vector, "inputTextTokenCount": 3}).encode())
        }


class TestBedrockTitanEmbedder:
    async def test_invoke_model_wire_shape(self) -> None:
        client = FakeBedrock()
        embedder = BedrockTitanEmbedder(client=client, dimension=256)
        vectors = await embedder.embed(["alpha", "beta"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 256
        call = client.calls[0]
        assert call["modelId"] == "amazon.titan-embed-text-v2:0"
        assert call["contentType"] == "application/json"
        assert json.loads(call["body"]) == {
            "inputText": "alpha",
            "dimensions": 256,
            "normalize": True,
        }
        assert embedder.model == "amazon.titan-embed-text-v2:0"
        assert embedder.dimension == 256
        await embedder.aclose()

    async def test_errors_and_validation(self) -> None:
        with pytest.raises(DependencyUnavailableError, match="RuntimeError"):
            await BedrockTitanEmbedder(client=FakeBedrock(fail=True), dimension=256).embed(["x"])
        with pytest.raises(ValueError, match="dimensions"):
            BedrockTitanEmbedder(client=FakeBedrock(), dimension=300)

    def test_from_region_builds_boto_client(self) -> None:
        embedder = BedrockTitanEmbedder.from_region(
            "us-east-1", model="amazon.titan-embed-text-v2:0", dimension=512
        )
        assert embedder.dimension == 512


DOCS = {
    "d1": "Beta Reagents supplies assay reagents to Acme Diagnostics",
    "d2": "Globex Logistics ships reagents across Europe",
    "d3": "Port congestion in Rotterdam delayed shipments",
    "d4": "Acme Diagnostics uses the Orion Analyzer",
}


class TestBM25:
    def test_ranking_and_idf(self) -> None:
        index = BM25Index((k, terms(v)) for k, v in DOCS.items())
        assert len(index) == 4
        hits = index.search(terms("assay reagents supplier"), 3)
        assert hits[0][0] == "d1"
        assert index.idf("assay") > index.idf("reagent")
        assert index.search(terms("unknown words"), 3) == []
        assert index.search([], 3) == []
        assert BM25Index([]).search(["x"], 3) == []

    def test_allow_filter(self) -> None:
        index = BM25Index((k, terms(v)) for k, v in DOCS.items())
        hits = index.search(terms("reagents"), 5, allow=lambda d: d != "d1")
        assert [h for h, _ in hits] == ["d2"]


class TestFusionAndDiversity:
    def test_rrf_matches_formula(self) -> None:
        fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]}, k=60)
        scores = dict(fused)
        assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["d"] == pytest.approx(1 / 63)
        assert [i for i, _ in fused][:2] == ["a", "b"]  # tie broken by best rank then id

    def test_rrf_weights_and_validation(self) -> None:
        fused = reciprocal_rank_fusion({"x": ["a"], "y": ["b"]}, weights={"x": 0.5, "y": 2.0})
        assert fused[0][0] == "b"
        with pytest.raises(ValueError, match="k must be"):
            reciprocal_rank_fusion({"x": ["a"]}, k=0)

    def test_mmr_prefers_diverse_results(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        assert maximal_marginal_relevance([1.0, 0.99, 0.6], vectors, k=2, lambda_=0.5) == [0, 2]
        assert maximal_marginal_relevance([1.0, 0.99, 0.6], vectors, k=2, lambda_=1.0) == [0, 1]
        assert maximal_marginal_relevance([], vectors, k=2) == []
        assert maximal_marginal_relevance([0.5, 0.5], vectors[:2], k=5) == [0, 1]
        with pytest.raises(ValueError, match="lambda_"):
            maximal_marginal_relevance([1.0], vectors[:1], k=1, lambda_=2.0)


class TestRerankers:
    async def test_lexical_cross_encoder_orders_by_relevance(self) -> None:
        encoder = LexicalCrossEncoder()
        scores = await encoder.score(
            "beta reagents assay supplier",
            [
                "Beta Reagents is the assay supplier for Acme",
                "Beta Reagents is based in Lyon",
                "Port congestion in Rotterdam",
            ],
        )
        assert scores[0] > scores[1] > scores[2] == 0.0
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert await encoder.score("the and of", ["x"]) == [0.0]
        assert encoder.name == "lexical-cross-encoder-v1"
        assert await encoder.score("reagents", ["reagents here"]) == [pytest.approx(1.0)]
        await encoder.aclose()

    def test_proximity(self) -> None:
        assert _proximity(["a", "x", "b"], {"a", "b"}) == pytest.approx(2 / 3)
        assert _proximity(["a", "b"], {"a", "b"}) == 1.0
        assert _proximity(["a"], {"a"}) == 1.0

    async def test_http_cross_encoder(self) -> None:
        seen: list[Any] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]
            )

        encoder = HttpCrossEncoder(
            base_url="http://reranker.svc",
            model="bge-reranker",
            api_key="t",
            transport=httpx.MockTransport(handler),
        )
        assert await encoder.score("q", ["a", "b"]) == [0.2, 0.9]
        assert seen[0] == {"query": "q", "texts": ["a", "b"], "raw_scores": False}
        assert await encoder.score("q", []) == []
        assert encoder.name == "bge-reranker"
        await encoder.aclose()

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (httpx.Response(500), "returned 500"),
            (httpx.Response(200, json=[{"index": 5, "score": 1}]), "malformed"),
            (httpx.Response(200, json=[{"index": 0, "score": 1}]), "every passage"),
            (httpx.Response(200, json={"bad": True}), "malformed"),
        ],
    )
    async def test_http_cross_encoder_failures(
        self, response: httpx.Response, message: str
    ) -> None:
        encoder = HttpCrossEncoder(
            base_url="http://r.svc", model="m", transport=httpx.MockTransport(lambda _: response)
        )
        with pytest.raises(DependencyUnavailableError, match=message):
            await encoder.score("q", ["a", "b"])

    async def test_http_cross_encoder_transport_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        encoder = HttpCrossEncoder(
            base_url="http://r.svc", model="m", transport=httpx.MockTransport(boom)
        )
        with pytest.raises(DependencyUnavailableError, match="ReadTimeout"):
            await encoder.score("q", ["a"])
