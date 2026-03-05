import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_TELEMETRY_INITIALIZED = False
_LOGGER = logging.getLogger(__name__)


def setup_telemetry(service_name: str = "higgs-tts") -> None:
    """Configure tracing/logging exporters once per process."""
    global _TELEMETRY_INITIALIZED
    if _TELEMETRY_INITIALIZED:
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "CogniVault"),
            "deployment.environment": os.getenv("ENVIRONMENT", "local"),
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    exporter = _build_otlp_exporter()
    if exporter is None:
        _LOGGER.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set, using console span exporter"
        )
        exporter = ConsoleSpanExporter()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    # Inject trace context into standard logging so logs/spans correlate in backends.
    LoggingInstrumentor().instrument(
        set_logging_format=True,
        logging_format=(
            "%(asctime)s %(levelname)s "
            "[%(name)s] trace_id=%(otelTraceID)s span_id=%(otelSpanID)s - %(message)s"
        ),
    )
    _TELEMETRY_INITIALIZED = True


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer that is safe to call before setup happens."""
    if not _TELEMETRY_INITIALIZED:
        setup_telemetry()
    return trace.get_tracer(name)


def _build_otlp_exporter() -> Optional[OTLPSpanExporter]:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return None
    # gRPC exporter accepts headers (list of (key, value) or env OTEL_EXPORTER_OTLP_HEADERS)
    headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    headers: Optional[list[tuple[str, str]]] = None
    if headers_str:
        headers = [
            tuple(item.split("=", 1))  # type: ignore[misc]
            for item in headers_str.split(",")
            if "=" in item
        ]
    return OTLPSpanExporter(endpoint=endpoint, headers=headers or None)

