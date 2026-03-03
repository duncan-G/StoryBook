"""
OpenTelemetry setup for the screenplay-parser FastAPI backend.

Sends traces and logs to the Aspire dashboard via OTLP HTTP (default endpoint
http://localhost:4318). Set OTEL_EXPORTER_OTLP_ENDPOINT or
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT / OTEL_EXPORTER_OTLP_LOGS_ENDPOINT to override.

If no OTLP endpoint is configured, instrumentation is skipped (no-op).
"""

from __future__ import annotations

import logging
import os

# Base OTLP endpoint (e.g. http://localhost:4318). When set, we configure traces and logs.
OTEL_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
OTEL_PROTOCOL_ENV = "OTEL_EXPORTER_OTLP_PROTOCOL"


def _get_otlp_base() -> str | None:
    """Base OTLP endpoint; None disables telemetry."""
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base:
        return None
    return base.rstrip("/")


def _traces_endpoint(base: str) -> str:
    return f"{base}/v1/traces"


def _logs_endpoint(base: str) -> str:
    return f"{base}/v1/logs"


def instrument_app(app, engine=None):
    """
    Instrument the FastAPI app with OpenTelemetry (traces + logs) and optionally
    configure OTLP export to the Aspire dashboard.

    If `engine` is provided (e.g. SQLAlchemy AsyncEngine), DB queries are
    instrumented so Postgres/SQLAlchemy spans appear in traces. Pass the engine
    from your database module: instrument_app(app, engine=engine).
    """
    base = _get_otlp_base()
    if not base:
        logging.getLogger(__name__).info(
            "OTEL disabled: set OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318) "
            "to send traces and logs to the dashboard."
        )
    if base:
        # Ensure HTTP/protobuf so the proto-http exporters work when env is read elsewhere
        os.environ.setdefault(OTEL_PROTOCOL_ENV, "http/protobuf")
        if OTEL_ENDPOINT_ENV not in os.environ:
            os.environ[OTEL_ENDPOINT_ENV] = base

    # Inject trace context into log records first (span id, trace id)
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        LoggingInstrumentor().instrument(set_logging_format=True)
    except ImportError:
        pass

    # Traces
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        pass
    else:
        if base:
            resource = Resource(attributes={"service.name": "screenplay-parser"})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=_traces_endpoint(base)))
            )
            trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,/openapi.json,/docs,/redoc",
        )

    # SQLAlchemy: trace DB queries (Postgres) so they show as spans
    if base and engine is not None:
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            # AsyncEngine wraps a sync engine; the instrumentor hooks into the sync engine
            sync_engine = getattr(engine, "sync_engine", engine)
            SQLAlchemyInstrumentor().instrument(engine=sync_engine)
            logging.getLogger(__name__).info(
                "OTEL SQLAlchemy instrumentation enabled (DB spans will appear in traces)"
            )
        except ImportError as e:
            logging.getLogger(__name__).warning(
                "OTEL SQLAlchemy instrumentation skipped (missing deps): %s. "
                "Install opentelemetry-instrumentation-sqlalchemy.",
                e,
            )

    # Logs: send Python logging to OTLP so logs show in Aspire
    if base:
        try:
            from opentelemetry._logs import set_logger_provider
            try:
                from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                    OTLPLogExporter,
                )
            except ImportError:
                from opentelemetry.exporter.otlp.proto.http.log_exporter import (
                    OTLPLogExporter,
                )
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.resources import Resource as LogResource
        except ImportError as e:
            logging.getLogger(__name__).warning(
                "OTEL log export skipped (missing deps): %s. "
                "Install opentelemetry-exporter-otlp-proto-http and opentelemetry-sdk.",
                e,
            )
        else:
            log_resource = LogResource(attributes={"service.name": "screenplay-parser"})
            log_provider = LoggerProvider(resource=log_resource)
            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=_logs_endpoint(base)))
            )
            set_logger_provider(log_provider)
            handler = LoggingHandler(
                level=logging.DEBUG, logger_provider=log_provider
            )
            logging.getLogger().addHandler(handler)
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger(__name__).info(
                "OTEL logs export enabled → %s (logs will appear in dashboard)",
                _logs_endpoint(base),
            )
