"""Load profile for a deployed platform.

Run against a staging deployment with a seeded collection, never production:

    GRAPHRAG_LOAD_TOKEN=<agent credential with rag:search rag:query rag:context> \
    GRAPHRAG_LOAD_COLLECTION=<collection name> \
    locust -f tests/performance/locustfile.py --host https://<staging host> \
           --users 100 --spawn-rate 10 --run-time 10m --headless

Agents call retrieval and context packing far more often than full answers,
so the mix weights search and context over query.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

TOKEN = os.environ.get("GRAPHRAG_LOAD_TOKEN", "")
COLLECTION = os.environ.get("GRAPHRAG_LOAD_COLLECTION", "")
QUESTIONS = (
    "Which suppliers does the analyzer platform depend on?",
    "What changed after the acquisition?",
    "Summarize the main logistics risks across the portfolio.",
    "Who leads the holding company?",
    "Which audits found deviations?",
)


class AgentTraffic(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self.client.headers.update(
            {"Authorization": f"Bearer {TOKEN}", "X-Purpose-Of-Use": "load-test"}
        )

    @task(6)
    def search(self) -> None:
        self.client.post(
            f"/v1/collections/{COLLECTION}/search",
            json={"query": random.choice(QUESTIONS), "k": 8},
            name="/v1/collections/[c]/search",
        )

    @task(3)
    def context(self) -> None:
        self.client.post(
            f"/v1/collections/{COLLECTION}/context",
            json={"query": random.choice(QUESTIONS), "budget": {"max_tokens": 2000}},
            name="/v1/collections/[c]/context",
        )

    @task(1)
    def query(self) -> None:
        self.client.post(
            f"/v1/collections/{COLLECTION}/query",
            json={"query": random.choice(QUESTIONS)},
            name="/v1/collections/[c]/query",
        )
