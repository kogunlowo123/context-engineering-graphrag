"""Provider-backed embedders used in production.

* ``BedrockTitanEmbedder``: Amazon Titan Text Embeddings V2 through the
  Bedrock runtime (primary cloud). Credentials come from the AWS default
  chain, i.e. IRSA / EKS Pod Identity, never from configuration.
* ``OpenAICompatibleEmbedder``: any endpoint speaking the OpenAI
  ``/v1/embeddings`` wire format: Azure OpenAI, OpenAI, and self-hosted
  servers such as Text Embeddings Inference or vLLM.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol, cast

import httpx

from context_engineering_graphrag.domain.errors import DependencyUnavailableError

AuthStyle = Literal["bearer", "api-key", "none"]


def _normalise(vector: Sequence[float], dimension: int, provider: str) -> list[float]:
    if len(vector) != dimension:
        raise DependencyUnavailableError(
            f"{provider}: expected {dimension}-dimensional embeddings, got {len(vector)}"
        )
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise DependencyUnavailableError(f"{provider}: returned a degenerate embedding")
    return [float(v) / norm for v in vector]


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        auth: AuthStyle = "bearer",
        path: str = "/v1/embeddings",
        query: dict[str, str] | None = None,
        send_dimensions: bool = False,
        batch_size: int = 64,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"content-type": "application/json"}
        if auth != "none":
            if not api_key:
                raise ValueError(f"embeddings: api key required for auth={auth}")
            if auth == "bearer":
                headers["authorization"] = f"Bearer {api_key}"
            else:
                headers["api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
        )
        self._path = path if not query else f"{path}?{httpx.QueryParams(query)}"
        self._model = model
        self._dimension = dimension
        self._send_dimensions = send_dimensions
        self._batch = batch_size

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for offset in range(0, len(texts), self._batch):
            out.extend(await self._embed_batch(list(texts[offset : offset + self._batch])))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"input": batch, "model": self._model}
        if self._send_dimensions:
            payload["dimensions"] = self._dimension
        try:
            response = await self._client.post(self._path, json=payload)
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError(
                f"embeddings: {type(exc).__name__} calling provider"
            ) from exc
        if response.status_code >= 400:
            raise DependencyUnavailableError(
                f"embeddings: provider returned {response.status_code}"
            )
        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = [cast(list[float], item["embedding"]) for item in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            raise DependencyUnavailableError("embeddings: malformed provider response") from exc
        if len(vectors) != len(batch):
            raise DependencyUnavailableError("embeddings: provider returned wrong vector count")
        return [_normalise(v, self._dimension, "embeddings") for v in vectors]


class InvokeModelClient(Protocol):
    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


class BedrockTitanEmbedder:
    """Titan Text Embeddings V2 (``amazon.titan-embed-text-v2:0``): 256, 512 or 1024 dims."""

    SUPPORTED_DIMENSIONS = frozenset({256, 512, 1024})

    def __init__(
        self,
        *,
        client: InvokeModelClient,
        model: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
        concurrency: int = 8,
    ) -> None:
        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(f"titan v2 supports dimensions {sorted(self.SUPPORTED_DIMENSIONS)}")
        self._client = client
        self._model = model
        self._dimension = dimension
        self._limit = asyncio.Semaphore(concurrency)

    @classmethod
    def from_region(cls, region: str, *, model: str, dimension: int) -> BedrockTitanEmbedder:
        import boto3  # noqa: PLC0415 - optional dependency shipped with agentic-core[bedrock]

        client = boto3.client("bedrock-runtime", region_name=region)
        return cls(client=cast(InvokeModelClient, client), model=model, dimension=dimension)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def aclose(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return list(await asyncio.gather(*(self._one(t) for t in texts)))

    async def _one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self._dimension, "normalize": True})
        async with self._limit:
            try:
                response = await asyncio.to_thread(
                    self._client.invoke_model,
                    modelId=self._model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                payload = json.loads(response["body"].read())
                vector = cast(list[float], payload["embedding"])
            except DependencyUnavailableError:
                raise
            except Exception as exc:
                raise DependencyUnavailableError(
                    f"bedrock embeddings: {type(exc).__name__}"
                ) from exc
        return _normalise(vector, self._dimension, "bedrock embeddings")
