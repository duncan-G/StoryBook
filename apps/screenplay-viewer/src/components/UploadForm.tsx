"use client";

import { useCallback, useRef, useState } from "react";
import { FileUp } from "lucide-react";
import { cn } from "@/lib/utils";
import ThemeToggle from "./ThemeToggle";

import { IngestResponse } from "@/types/screenplay";

interface Props {
  onUploadComplete: (data: IngestResponse) => void;
}

const ACCEPTED_TYPE = "application/pdf";
const MAX_SIZE_MB = 50;

export default function UploadForm({ onUploadComplete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const validate = useCallback((file: File): string | null => {
    if (file.type !== ACCEPTED_TYPE) {
      return "Only PDF files are accepted. Please upload a .pdf file.";
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File exceeds ${MAX_SIZE_MB} MB limit.`;
    }
    return null;
  }, []);

  const upload = useCallback(
    async (file: File) => {
      const err = validate(file);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      setFileName(file.name);
      setUploading(true);

      try {
        const form = new FormData();
        form.append("file", file);

        const apiBase =
          process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
        const res = await fetch(`${apiBase}/ingest`, {
          method: "POST",
          body: form,
        });

        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `Server error ${res.status}`);
        }

        const data: IngestResponse = await res.json();
        onUploadComplete(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setUploading(false);
      }
    },
    [validate, onUploadComplete],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) upload(file);
    },
    [upload],
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) upload(file);
    },
    [upload],
  );

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-6">
      {/* Theme toggle in corner */}
      <div className="fixed right-4 top-4">
        <ThemeToggle />
      </div>

      <div className="w-full max-w-lg">
        <div className="mb-10 text-center">
          <h1 className="font-mono text-3xl font-bold tracking-tight text-foreground">
            Screenplay Parser
          </h1>
          <p className="mt-2 font-mono text-sm text-muted-foreground">
            Upload a screenplay PDF to view it in a structured reader.
          </p>
        </div>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          disabled={uploading}
          className={cn(
            "group flex w-full flex-col items-center justify-center gap-4",
            "rounded-2xl border-2 border-dashed px-8 py-16 transition-all",
            dragOver
              ? "border-primary bg-primary/5"
              : "border-border bg-card/50 hover:border-muted-foreground hover:bg-card",
            uploading && "cursor-wait opacity-60",
            !uploading && "cursor-pointer",
          )}
        >
          {uploading ? (
            <>
              <div className="h-10 w-10 animate-spin rounded-full border-4 border-muted border-t-primary" />
              <span className="font-mono text-sm text-muted-foreground">
                Parsing {fileName}...
              </span>
            </>
          ) : (
            <>
              <FileUp className="h-10 w-10 text-muted-foreground/50 transition-colors group-hover:text-muted-foreground" />
              <div className="text-center">
                <span className="font-mono text-sm text-foreground/80">
                  Drop a PDF here or click to browse
                </span>
                <br />
                <span className="font-mono text-xs text-muted-foreground/60">
                  PDF only &middot; up to {MAX_SIZE_MB} MB
                </span>
              </div>
            </>
          )}
        </button>

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={onFileChange}
          className="hidden"
        />

        {error && (
          <div
            className={cn(
              "mt-4 rounded-lg border px-4 py-3 font-mono text-sm",
              "border-destructive/50 bg-destructive/10 text-destructive",
            )}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
