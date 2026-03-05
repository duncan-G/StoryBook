import { notFound } from "next/navigation";
import { context, propagation } from "@opentelemetry/api";
import type { Screenplay } from "@/types/screenplay";
import ScreenplayPageClient from "./ScreenplayPageClient";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Server Component: fetches screenplay data during SSR so the fetch to
 * screenplay-parser shares the same trace as the incoming page request.
 */
export default async function ScreenplayPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  if (!id) notFound();

  const headers: Record<string, string> = {};
  propagation.inject(context.active(), headers);

  const res = await fetch(`${API_BASE}/screenplays/${id}`, { headers });
  if (!res.ok) {
    if (res.status === 404) notFound();
    throw new Error("Failed to load screenplay");
  }
  const screenplay: Screenplay = await res.json();

  return <ScreenplayPageClient screenplay={screenplay} screenplayId={id} />;
}
