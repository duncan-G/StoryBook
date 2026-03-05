"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, RefreshCw, Zap, MessageSquare, Database, DollarSign } from "lucide-react";
import { cn } from "@/lib/utils";
import ThemeToggle from "@/components/ThemeToggle";
import type { LLMCostsResponse, LLMCostEntry } from "@/types/screenplay";

const REASON_LABELS: Record<string, string> = {
  qa_answer: "Q&A Answers",
  embed_ingest: "Embedding (Ingest)",
  embed_query: "Embedding (Query)",
};

const REASON_ICONS: Record<string, typeof Zap> = {
  qa_answer: MessageSquare,
  embed_ingest: Database,
  embed_query: Zap,
};

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function formatCost(n: number): string {
  if (n === 0) return "$0.00";
  if (n < 0.01) return `$${n.toFixed(6)}`;
  return `$${n.toFixed(4)}`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function StatCard({
  label,
  value,
  sub,
  icon: Icon,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: typeof Zap;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-xl border border-border bg-card p-4 sm:p-5",
        "transition-colors hover:border-primary/30",
      )}
    >
      <div className="flex items-center gap-2 text-muted-foreground">
        <Icon className="h-4 w-4" />
        <span className="font-mono text-xs uppercase tracking-wider">{label}</span>
      </div>
      <span className="font-mono text-2xl font-bold text-foreground sm:text-3xl">
        {value}
      </span>
      {sub && (
        <span className="font-mono text-xs text-muted-foreground">{sub}</span>
      )}
    </div>
  );
}

function ReasonBreakdown({ costs }: { costs: LLMCostsResponse }) {
  if (costs.by_reason.length === 0) return null;

  const maxCost = Math.max(...costs.by_reason.map((r) => r.cost), 0.001);

  return (
    <section>
      <h2 className="mb-3 font-mono text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        By Category
      </h2>
      <div className="space-y-3">
        {costs.by_reason.map((r) => {
          const Icon = REASON_ICONS[r.reason] ?? Zap;
          const pct = (r.cost / maxCost) * 100;
          return (
            <div
              key={r.reason}
              className="rounded-lg border border-border bg-card p-4"
            >
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-primary" />
                  <span className="font-mono text-sm font-medium text-foreground">
                    {REASON_LABELS[r.reason] ?? r.reason}
                  </span>
                </div>
                <span className="font-mono text-sm font-bold text-foreground">
                  {formatCost(r.cost)}
                </span>
              </div>
              <div className="mb-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-muted-foreground">
                <span>{r.request_count} requests</span>
                <span>{formatTokens(r.input_tokens)} in</span>
                <span>{formatTokens(r.output_tokens)} out</span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ScreenplayBreakdown({ costs }: { costs: LLMCostsResponse }) {
  if (costs.by_screenplay.length === 0) return null;

  return (
    <section>
      <h2 className="mb-3 font-mono text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        By Screenplay
      </h2>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full font-mono text-sm">
          <thead>
            <tr className="border-b border-border bg-card text-left">
              <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Title
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Requests
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Input
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Output
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Cost
              </th>
            </tr>
          </thead>
          <tbody>
            {costs.by_screenplay.map((sp) => (
              <tr
                key={sp.screenplay_id}
                className="border-b border-border/50 transition-colors hover:bg-accent/50"
              >
                <td className="max-w-[200px] truncate px-4 py-2.5 text-foreground">
                  {sp.title}
                </td>
                <td className="px-4 py-2.5 text-right text-muted-foreground">
                  {sp.request_count}
                </td>
                <td className="px-4 py-2.5 text-right text-muted-foreground">
                  {formatTokens(sp.input_tokens)}
                </td>
                <td className="px-4 py-2.5 text-right text-muted-foreground">
                  {formatTokens(sp.output_tokens)}
                </td>
                <td className="px-4 py-2.5 text-right font-medium text-foreground">
                  {formatCost(sp.cost)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RecentEntries({ entries }: { entries: LLMCostEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <section>
      <h2 className="mb-3 font-mono text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        Recent Requests
      </h2>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full font-mono text-sm">
          <thead>
            <tr className="border-b border-border bg-card text-left">
              <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Time
              </th>
              <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Category
              </th>
              <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Screenplay
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Tokens
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Cost
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr
                key={e.id}
                className="border-b border-border/50 transition-colors hover:bg-accent/50"
              >
                <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">
                  {formatDate(e.created_at)}
                </td>
                <td className="px-4 py-2.5 text-foreground">
                  {REASON_LABELS[e.reason] ?? e.reason}
                </td>
                <td className="max-w-[160px] truncate px-4 py-2.5 text-muted-foreground">
                  {e.screenplay_title ?? "—"}
                </td>
                <td className="whitespace-nowrap px-4 py-2.5 text-right text-muted-foreground">
                  {formatTokens(e.input_tokens + e.output_tokens)}
                </td>
                <td className="px-4 py-2.5 text-right font-medium text-foreground">
                  {e.cost != null ? formatCost(e.cost) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function CostsClient({
  costs,
}: {
  costs: LLMCostsResponse | null;
}) {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    router.refresh();
    setTimeout(() => setRefreshing(false), 600);
  }, [router]);

  return (
    <div className="min-h-screen bg-background">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-background px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm",
              "text-foreground hover:border-primary/50 hover:bg-accent/50",
            )}
          >
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Back</span>
          </button>
          <h1 className="font-mono text-lg font-bold tracking-tight text-foreground">
            LLM Costs
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm",
              "text-foreground hover:border-primary/50 hover:bg-accent/50",
            )}
          >
            <RefreshCw
              className={cn("h-4 w-4", refreshing && "animate-spin")}
            />
            <span className="hidden sm:inline">Refresh</span>
          </button>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 p-4 sm:p-6">
        {!costs ? (
          <div className="flex min-h-[40vh] items-center justify-center">
            <p className="font-mono text-sm text-muted-foreground">
              Unable to load cost data. Is the API running?
            </p>
          </div>
        ) : costs.totals.request_count === 0 ? (
          <div className="flex min-h-[40vh] flex-col items-center justify-center gap-3">
            <DollarSign className="h-12 w-12 text-muted" />
            <p className="font-mono text-sm text-muted-foreground">
              No LLM usage recorded yet.
            </p>
            <p className="font-mono text-xs text-muted-foreground/70">
              Costs are tracked when you ingest screenplays or ask questions.
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4">
              <StatCard
                label="Total Cost"
                value={formatCost(costs.totals.cost)}
                icon={DollarSign}
              />
              <StatCard
                label="Requests"
                value={costs.totals.request_count.toLocaleString()}
                icon={Zap}
              />
              <StatCard
                label="Input Tokens"
                value={formatTokens(costs.totals.input_tokens)}
                icon={Database}
              />
              <StatCard
                label="Output Tokens"
                value={formatTokens(costs.totals.output_tokens)}
                icon={MessageSquare}
              />
            </div>

            <ReasonBreakdown costs={costs} />
            <ScreenplayBreakdown costs={costs} />
            <RecentEntries entries={costs.recent} />
          </>
        )}
      </main>
    </div>
  );
}
