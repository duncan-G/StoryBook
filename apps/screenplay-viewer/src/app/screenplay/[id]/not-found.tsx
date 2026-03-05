"use client";

import { useRouter } from "next/navigation";

export default function NotFound() {
  const router = useRouter();

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background px-4">
      <p className="font-mono text-sm text-destructive">Screenplay not found</p>
      <button
        type="button"
        onClick={() => router.push("/")}
        className="font-mono text-sm text-muted-foreground underline hover:text-foreground"
      >
        Back to list
      </button>
    </div>
  );
}
