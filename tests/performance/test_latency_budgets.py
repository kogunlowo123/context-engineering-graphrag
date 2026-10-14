"""Latency budgets for hot paths. Loose enough for shared CI runners."""

from __future__ import annotations

import random
import time

import pytest

from context_engineering_graphrag.application.service import Caller, ContextPlatform, SearchRequest
from context_engineering_graphrag.context.budget import (
    BudgetPolicy,
    Candidate,
    ContextBudgeter,
    SourceKind,
)
from context_engineering_graphrag.domain.text import HeuristicTokenizer, terms
from context_engineering_graphrag.embeddings.hashing import HashingEmbedder
from context_engineering_graphrag.graph.community import build_graph, louvain
from context_engineering_graphrag.orchestration.workflow import QueryRequest
from context_engineering_graphrag.retrieval.bm25 import BM25Index
from tests.conftest import corpus_document

pytestmark = pytest.mark.performance

VOCAB = [f"term{i}" for i in range(3000)]


def synthetic_passages(n: int, seed: int = 7) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(f"c{i}", " ".join(rng.choices(VOCAB, k=120))) for i in range(n)]


def test_bm25_search_budget() -> None:
    # Terms are computed once at ingestion and stored, so a revision reload only indexes them.
    tokenized = [(cid, terms(text)) for cid, text in synthetic_passages(5000)]
    started = time.perf_counter()
    index = BM25Index(tokenized)
    build_s = time.perf_counter() - started
    rng = random.Random(3)
    queries = [" ".join(rng.choices(VOCAB, k=6)) for _ in range(100)]
    started = time.perf_counter()
    for query in queries:
        index.search(terms(query), 50)
    per_query_ms = (time.perf_counter() - started) * 1000 / len(queries)
    assert build_s < 10
    assert per_query_ms < 50


def test_embedding_throughput() -> None:
    embedder = HashingEmbedder(384)
    texts = [text for _, text in synthetic_passages(300)]
    started = time.perf_counter()
    for text in texts:
        embedder.embed_one(text)
    per_text_ms = (time.perf_counter() - started) * 1000 / len(texts)
    assert per_text_ms < 15


def test_louvain_budget() -> None:
    rng = random.Random(11)
    edges = []
    for block in range(40):
        members = [f"n{block}_{i}" for i in range(50)]
        edges += [(rng.choice(members), rng.choice(members), 1.0) for _ in range(300)]
    edges += [
        (
            f"n{rng.randrange(40)}_{rng.randrange(50)}",
            f"n{rng.randrange(40)}_{rng.randrange(50)}",
            0.2,
        )
        for _ in range(200)
    ]
    graph = build_graph(edges)
    started = time.perf_counter()
    levels = louvain(graph)
    elapsed = time.perf_counter() - started
    assert len(set(levels[-1].values())) >= 20
    assert elapsed < 20


def test_context_budgeting_budget() -> None:
    tokenizer = HeuristicTokenizer()
    candidates = [
        Candidate(key=cid, kind=SourceKind.PASSAGE, text=text, score=1.0 / (i + 1), title=cid)
        for i, (cid, text) in enumerate(synthetic_passages(100))
    ]
    started = time.perf_counter()
    for _ in range(10):
        pack = ContextBudgeter(tokenizer).build(
            "term1 term2 term3", candidates, BudgetPolicy(max_tokens=4000)
        )
    per_pack_ms = (time.perf_counter() - started) * 1000 / 10
    assert pack.used_tokens <= 4000
    assert per_pack_ms < 500


async def test_end_to_end_query_budget(seeded: ContextPlatform, caller: Caller) -> None:
    for i in range(60):
        await seeded.ingest(
            caller,
            "supplier-risk",
            corpus_document(
                f"filler-{i}",
                title=f"Filler {i}",
                text=" ".join(random.Random(i).choices(VOCAB, k=200)),
            ),
        )
    await seeded.search(caller, "supplier-risk", SearchRequest(query="warm the index"))
    started = time.perf_counter()
    for _ in range(10):
        await seeded.search(
            caller, "supplier-risk", SearchRequest(query="reagent supplier concentration risk")
        )
    search_ms = (time.perf_counter() - started) * 1000 / 10
    started = time.perf_counter()
    for _ in range(5):
        result = await seeded.query(
            caller, "supplier-risk", QueryRequest(query="Which company acquired Beta Reagents?")
        )
    query_ms = (time.perf_counter() - started) * 1000 / 5
    assert result.answer
    assert search_ms < 400
    assert query_ms < 1500
