"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MessageCircle, Send, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { askQuestion } from "@/lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: boolean;
}

interface Props {
  screenplayId: string;
  /** Optional title for panel header */
  screenplayTitle?: string;
}

const PANEL_WIDTH = "min(420px, 100vw - 2rem)";

export default function ScreenplayChat({
  screenplayId,
  screenplayTitle,
}: Props) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (open) {
      const t = requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
      return () => cancelAnimationFrame(t);
    }
  }, [open]);

  const scrollToBottom = useCallback(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    const assistantId = `assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    scrollToBottom();

    try {
      const { answer } = await askQuestion(text, screenplayId);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: answer } : m,
        ),
      );
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: "Sorry, I couldn’t get an answer. Please try again.",
                error: true,
              }
            : m,
        ),
      );
    } finally {
      setLoading(false);
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [input, loading, screenplayId, scrollToBottom]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    },
    [send],
  );

  return (
    <>
      {/* FAB: bottom-right, above footer area */}
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          "fixed bottom-20 right-6 z-40 flex h-12 w-12 items-center justify-center rounded-full shadow-lg",
          "bg-primary text-primary-foreground",
          "transition-transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background",
        )}
        aria-label="Ask a question about this screenplay"
      >
        <MessageCircle className="h-5 w-5" />
      </button>

      {/* Overlay backdrop when panel is open */}
      {open && (
        <button
          type="button"
          aria-label="Close chat"
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[2px]"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Right-side panel (overlay, does not resize main content) */}
      <aside
        role="dialog"
        aria-label="Ask about this screenplay"
        className={cn(
          "fixed top-0 right-0 z-50 flex h-full w-[min(420px,100vw-2rem)] max-w-[calc(100vw-2rem)] flex-col",
          "border-l border-border bg-card shadow-xl",
          "transition-transform duration-200 ease-out",
          open ? "translate-x-0" : "translate-x-full",
        )}
        style={{ width: PANEL_WIDTH }}
      >
        <header className="flex shrink-0 flex-col gap-0.5 border-b border-border px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="truncate font-mono text-sm font-semibold text-card-foreground">
              Ask about this screenplay
            </h2>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
                "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          {screenplayTitle && (
            <p className="truncate font-mono text-xs text-muted-foreground">
              {screenplayTitle}
            </p>
          )}
        </header>

        {/* Message list */}
        <div className="flex-1 overflow-y-auto px-4 py-3">
          {messages.length === 0 && (
            <p className="font-mono text-xs text-muted-foreground">
              Ask anything about this screenplay — characters, scenes, plot, or
              structure.
            </p>
          )}
          <ul className="flex flex-col gap-3">
            {messages.map((m, i) => {
              const isPending =
                loading &&
                m.role === "assistant" &&
                !m.content &&
                i === messages.length - 1;
              return (
                <li
                  key={m.id}
                  className={cn(
                    "rounded-lg px-3 py-2 font-mono text-sm",
                    m.role === "user"
                      ? "ml-8 bg-primary/10 text-foreground"
                      : "mr-4 bg-muted/80 text-foreground",
                    m.error && "text-destructive",
                  )}
                >
                  {m.content || (isPending ? "…" : null)}
                </li>
              );
            })}
          </ul>
          <div ref={listEndRef} />
        </div>

        {/* Input area */}
        <div className="shrink-0 border-t border-border p-3">
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question…"
              rows={2}
              disabled={loading}
              className={cn(
                "min-h-[2.5rem] flex-1 resize-none rounded-lg border border-input bg-background px-3 py-2 font-mono text-sm",
                "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring",
                "disabled:opacity-60",
              )}
            />
            <button
              type="button"
              onClick={send}
              disabled={!input.trim() || loading}
              className={cn(
                "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground",
                "disabled:opacity-40 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-ring",
              )}
              aria-label="Send"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
