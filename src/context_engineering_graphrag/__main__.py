"""Process entry point: ``python -m context_engineering_graphrag``."""

from __future__ import annotations

import os

import uvicorn
from agentic_core.observability import configure_logging, configure_telemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from context_engineering_graphrag import __version__
from context_engineering_graphrag.api.app import create_app
from context_engineering_graphrag.config import Settings
from context_engineering_graphrag.runtime.container import build_container

SERVICE = "context-engineering-graphrag"


def main() -> None:
    settings = Settings()  # populated from GRAPHRAG_* environment variables
    configure_logging(service=SERVICE, environment=settings.environment, level=settings.log_level)
    exporter = None
    if settings.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    configure_telemetry(
        service_name=SERVICE,
        service_version=__version__,
        environment=settings.environment,
        span_exporter=exporter,
    )
    app = create_app(lambda: build_container(settings), allowed_hosts=settings.allowed_hosts)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    uvicorn.run(
        app,
        host=os.environ.get("GRAPHRAG_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRAPHRAG_PORT", "8080")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("GRAPHRAG_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
