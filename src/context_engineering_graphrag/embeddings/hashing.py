"""Deterministic local embedder based on signed feature hashing.

Each text is mapped to stemmed word unigrams, word bigrams and character
trigrams; every feature is hashed (BLAKE2b over a family-prefixed feature name) into a
bucket with a sign bit, weighted by sub-linear term frequency, and the vector
is L2-normalised. Similar wording yields high cosine similarity, morphology
is partly captured by character n-grams, and the output is identical on every
machine. It needs no network access, so it powers tests and air-gapped
development; production deployments use a provider-backed embedder.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from context_engineering_graphrag.domain.text import STOPWORDS, stem, words


class HashingEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self._dimension = dimension

    @property
    def model(self) -> str:
        return f"feature-hashing-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    async def aclose(self) -> None:
        return None

    def embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self._dimension, dtype=np.float64)
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return [float(v) for v in vector]
        return [float(v) for v in vector / norm]

    @staticmethod
    def _features(text: str) -> Counter[str]:
        tokens = [w.casefold() for w in words(text)]
        content = [stem(t) for t in tokens if t not in STOPWORDS]
        features: Counter[str] = Counter()
        for term in content:
            features[f"w:{term}"] += 2
            padded = f"#{term}#"
            for i in range(len(padded) - 2):
                features[f"c:{padded[i:i + 3]}"] += 1
        for left, right in itertools.pairwise(content):
            features[f"b:{left}_{right}"] += 1
        return features
