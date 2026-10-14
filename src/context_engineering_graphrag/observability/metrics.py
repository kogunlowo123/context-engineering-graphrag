"""Service metrics exposed on /metrics in Prometheus text format."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

_LATENCY = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class PlatformMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Histogram(
            "graphrag_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route", "status"],
            registry=self.registry,
            buckets=_LATENCY,
        )
        self.stage = Histogram(
            "graphrag_stage_duration_seconds",
            "Pipeline stage latency (retrieve, rerank, graph, budget, generate, evaluate)",
            ["stage"],
            registry=self.registry,
            buckets=_LATENCY,
        )
        self.answers = Counter(
            "graphrag_answers_total",
            "Answers by disposition and retrieval mode",
            ["disposition", "mode"],
            registry=self.registry,
        )
        self.groundedness = Histogram(
            "graphrag_answer_groundedness",
            "Groundedness score of released and withheld answers",
            registry=self.registry,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.context_tokens = Histogram(
            "graphrag_context_tokens",
            "Tokens placed into context packs",
            registry=self.registry,
            buckets=(100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000),
        )
        self.ingested_chunks = Counter(
            "graphrag_ingested_chunks_total",
            "Chunks written by ingestion",
            ["collection"],
            registry=self.registry,
        )
        self.quarantined_chunks = Counter(
            "graphrag_quarantined_chunks_total",
            "Chunks quarantined at ingestion (prompt injection)",
            ["collection"],
            registry=self.registry,
        )
        self.authz = Counter(
            "graphrag_authorization_decisions_total",
            "Inbound authorization decisions",
            ["decision", "source"],
            registry=self.registry,
        )
        self.index_reloads = Histogram(
            "graphrag_index_reload_seconds",
            "Time to rebuild a collection index after a revision change",
            registry=self.registry,
            buckets=_LATENCY,
        )
        self.graph_builds = Counter(
            "graphrag_graph_builds_total",
            "Knowledge-graph builds (resolution, communities, summaries)",
            ["outcome"],
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "graphrag_agent_calls_total",
            "MCP tool and A2A calls",
            ["protocol", "operation", "outcome"],
            registry=self.registry,
        )
