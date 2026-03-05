/**
 * Telemetry configuration. OTLP endpoint for Aspire dashboard (HTTP on 4318).
 * Set NEXT_PUBLIC_OTLP_HTTP_ENDPOINT in .env.local (e.g. http://localhost:4318/v1).
 */
export const telemetryConfig = {
  /** Base URL for OTLP HTTP (e.g. http://localhost:4318/v1). Used by client to send traces. */
  otlpHttpEndpoint:
    process.env.NEXT_PUBLIC_OTLP_HTTP_ENDPOINT ?? "http://localhost:4318/v1",
  /** Service name for resource attributes */
  serviceName: "screenplay-viewer",
} as const;
