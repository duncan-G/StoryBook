import { type NextRequest, NextResponse } from "next/server";
import { trace } from "@opentelemetry/api";

export const config = {
  matcher:
    "/((?!_next/static|_next/image|favicon.ico|error|.*\\.(?:svg|png|jpg|jpeg|gif|webp|woff|woff2|ttf|eot|css|js)$|.well-known/).*)",
};

/**
 * Middleware delegates all tracing to @vercel/otel. Its only job is to emit a
 * Server-Timing response header so the browser's DocumentLoadInstrumentation
 * can parent its documentLoad span under the server trace.
 */
export default function middleware(_request: NextRequest) {
  const response = NextResponse.next();

  const span = trace.getActiveSpan();
  if (span) {
    const { traceId, spanId, traceFlags } = span.spanContext();
    const sampled = (traceFlags & 0x01) === 0x01 ? "01" : "00";
    response.headers.set(
      "Server-Timing",
      `traceparent;desc="00-${traceId}-${spanId}-${sampled}"`,
    );
  }

  return response;
}
