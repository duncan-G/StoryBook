"use client";

import { useCallback, useState } from "react";
import { DollarSign, FileUp } from "lucide-react";
import { useRouter } from "next/navigation";
import type { IngestResponse, ScreenplayListItem } from "@/types/screenplay";
import UploadForm from "@/components/UploadForm";
import ScreenplayList from "@/components/ScreenplayList";
import ThemeToggle from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";
import { deleteScreenplay } from "@/lib/api";

export default function HomeClient({
  screenplays,
}: {
  screenplays: ScreenplayListItem[];
}) {
  const router = useRouter();
  const [showUploadForm, setShowUploadForm] = useState(false);

  const handleUpload = useCallback(
    (data: IngestResponse) => {
      setShowUploadForm(false);
      if (data.screenplay_id != null) {
        router.push(`/screenplay/${data.screenplay_id}`);
      } else {
        router.refresh();
      }
    },
    [router],
  );

  const handleSelectItem = useCallback(
    (id: string) => {
      router.push(`/screenplay/${id}`);
    },
    [router],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteScreenplay(id);
        router.refresh();
      } catch {
        // Could add toast; for now list will not change
      }
    },
    [router],
  );

  const hasItems = screenplays.length > 0;
  const showUploadOnly = !hasItems;

  return (
    <div className="min-h-screen bg-background">
      {hasItems && !showUploadForm && (
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-background px-4 py-3 sm:px-6">
          <h1 className="font-mono text-lg font-bold tracking-tight text-foreground">
            Screenplays
          </h1>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <button
              type="button"
              onClick={() => router.push("/costs")}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm",
                "text-foreground hover:border-primary/50 hover:bg-accent/50",
              )}
            >
              <DollarSign className="h-4 w-4" />
              <span className="hidden sm:inline">Costs</span>
            </button>
            <button
              type="button"
              onClick={() => setShowUploadForm(true)}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm",
                "text-foreground hover:border-primary/50 hover:bg-accent/50",
              )}
            >
              <FileUp className="h-4 w-4" />
              Upload PDF
            </button>
          </div>
        </header>
      )}

      {showUploadOnly && <UploadForm onUploadComplete={handleUpload} />}

      {hasItems && showUploadForm && (
        <div className="p-4 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <button
              type="button"
              onClick={() => setShowUploadForm(false)}
              className="font-mono text-sm text-muted-foreground hover:text-foreground"
            >
              ← Back to list
            </button>
            <ThemeToggle />
          </div>
          <UploadForm onUploadComplete={handleUpload} />
        </div>
      )}

      {hasItems && !showUploadForm && (
        <main className="p-4 sm:p-6">
          <ScreenplayList
            items={screenplays}
            onSelect={handleSelectItem}
            onDelete={handleDelete}
            loading={false}
          />
        </main>
      )}
    </div>
  );
}
