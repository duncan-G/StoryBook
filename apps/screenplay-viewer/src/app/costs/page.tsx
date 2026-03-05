import { context, propagation } from "@opentelemetry/api";
import type { LLMCostsResponse } from "@/types/screenplay";
import CostsClient from "./CostsClient";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default async function CostsPage() {
  const headers: Record<string, string> = {};
  propagation.inject(context.active(), headers);

  let costs: LLMCostsResponse | null = null;
  try {
    const res = await fetch(`${API_BASE}/costs`, {
      headers,
      cache: "no-store",
    });
    if (res.ok) {
      costs = await res.json();
    }
  } catch {
    // show error state
  }

  return <CostsClient costs={costs} />;
}
