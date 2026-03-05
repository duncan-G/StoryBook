"use client";

import { WebTracerProvider } from "@opentelemetry/sdk-trace-web";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { FetchInstrumentation } from "@opentelemetry/instrumentation-fetch";
import { XMLHttpRequestInstrumentation } from "@opentelemetry/instrumentation-xml-http-request";
import { DocumentLoadInstrumentation } from "@opentelemetry/instrumentation-document-load";
import { ZoneContextManager } from "@opentelemetry/context-zone";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { B3Propagator } from "@opentelemetry/propagator-b3";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import { CompositePropagator } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { trace, SpanStatusCode } from "@opentelemetry/api";
import { telemetryConfig } from "@/lib/telemetry-config";
import { isBrowser } from "@/lib/utils";

let initialized = false;

/**
 * Initialize client-side OpenTelemetry: web tracer, OTLP exporter to Aspire dashboard,
 * document load + fetch + XHR instrumentations, and global error handlers.
 */
export function initWebTelemetry(serviceName: string = telemetryConfig.serviceName) {
  if (!isBrowser() || initialized) return;

  const endpoint = telemetryConfig.otlpHttpEndpoint;
  if (!endpoint) {
    return;
  }

  const exporter = new OTLPTraceExporter({
    url: `${endpoint}/traces`,
    headers: {},
  });

  const provider = new WebTracerProvider({
    resource: resourceFromAttributes({
      "service.name": serviceName,
      "service.instance.id": serviceName,
    }),
    spanProcessors: [new BatchSpanProcessor(exporter)],
  });

  provider.register({
    contextManager: new ZoneContextManager(),
    propagator: new CompositePropagator({
      propagators: [new W3CTraceContextPropagator(), new B3Propagator()],
    }),
  });

  // Capture unhandled errors and promise rejections as error spans
  try {
    const errorTracer = trace.getTracer("errors");
    if (typeof window !== "undefined") {
      window.addEventListener("error", (event: ErrorEvent) => {
        try {
          const span = errorTracer.startSpan("unhandled_error");
          span.recordException(event.error ?? event.message);
          span.setStatus({ code: SpanStatusCode.ERROR, message: String(event.message) });
          span.end();
        } catch {
          // ignore
        }
      });
      window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
        try {
          const span = errorTracer.startSpan("unhandled_promise_rejection");
          const reason = event.reason as unknown;
          const message = reason instanceof Error ? reason.message : String(reason);
          const exception = reason instanceof Error ? reason : message;
          span.recordException(exception);
          span.setStatus({ code: SpanStatusCode.ERROR, message });
          span.end();
        } catch {
          // ignore
        }
      });
    }
  } catch {
    // ignore
  }

  registerInstrumentations({
    instrumentations: [
      new DocumentLoadInstrumentation(),
      new FetchInstrumentation({
        propagateTraceHeaderCorsUrls: [/.*/],
        ignoreUrls: [/otlp/, /localhost:4318/],
        clearTimingResources: true,
      }),
      new XMLHttpRequestInstrumentation({
        propagateTraceHeaderCorsUrls: [/.*/],
        ignoreUrls: [/otlp/, /localhost:4318/],
        clearTimingResources: true,
      }),
    ],
  });

  initialized = true;
}

export function getTracer(name: string = "web") {
  return trace.getTracer(name);
}
