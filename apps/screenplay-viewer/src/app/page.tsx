import { context, propagation } from "@opentelemetry/api";
import type { ScreenplayListItem } from "@/types/screenplay";
import HomeClient from "./HomeClient";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default async function Home() {
  const headers: Record<string, string> = {};
  propagation.inject(context.active(), headers);

  let screenplays: ScreenplayListItem[] = [];
  try {
    const res = await fetch(`${API_BASE}/screenplays`, { headers });
    if (res.ok) {
      screenplays = await res.json();
    }
  } catch {
    // show empty list on failure
  }

  return <HomeClient screenplays={screenplays} />;
}
