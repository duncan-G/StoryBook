/**
 * Next.js instrumentation: runs on server startup.
 * Initializes server-side OpenTelemetry and sends traces to the Aspire dashboard.
 * @see https://nextjs.org/docs/app/building-your-application/optimizing/instrumentation
 */
import type { Configuration } from "@vercel/otel";
import { registerOTel } from "@vercel/otel";

const serviceName = "screenplay-viewer-server";

function initServerTelemetry() {
  const config: Configuration = {
    serviceName,
    instrumentationConfig: {
      fetch: {
        ignoreUrls: [
          /^https:\/\/telemetry\.nextjs\.org/,
          /\.(png|jpg|jpeg|gif|svg|css|js|map|woff|woff2|eot|ttf|json|txt|webmanifest)$/,
          /\/(_next)\//,
          /favicon\.ico$/,
        ],
        propagateContextUrls: [/^https?:\/\/localhost(:\d+)?/],
        dontPropagateContextUrls: [/no-propagation\=1/],
      },
    },
  };

  registerOTel(config);
}

export function register() {
  const origWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    if (
      typeof args[0] === "string" &&
      args[0].includes("Unexpected root span type")
    ) {
      return;
    }
    origWarn.apply(console, args);
  };

  initServerTelemetry();
}
