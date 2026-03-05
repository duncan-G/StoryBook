"use client";

import { useEffect } from "react";
import { initWebTelemetry } from "@/lib/telemetry";
import { telemetryConfig } from "@/lib/telemetry-config";

export default function TelemetryProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    initWebTelemetry(telemetryConfig.serviceName);
  }, []);

  return <>{children}</>;
}
