"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MessageCircle, Send, PanelRightClose, ExternalLink } from "lucide-react";
import Markdown from "react-markdown";
import { cn } from "@/lib/utils";
import { askQuestion } from "@/lib/api";
import type { Scene } from "@/types/screenplay";

// ---------------------------------------------------------------------------
// Citation preprocessing + injection into react-markdown output
// ---------------------------------------------------------------------------

const CITATION_RE = /\[\[ref:(\d+)(?:\s+"([^"]*)")?\]\]/g;
const CITE_SPLIT_RE = /(%%CITE_\d+%%)/;
const CITE_MATCH_RE = /^%%CITE_(\d+)%%$/;

interface Citation {
  sceneIndex: number;
  quote: string;
}

/** Replace [[ref:N "quote"]] with %%CITE_N%% placeholders and collect citations. */
function preprocessCitations(text: string): { markdown: string; citations: Citation[] } {
  const citations: Citation[] = [];
  // Normalize grouped refs: [[ref:1 "a"], [ref:2 "b"]] → [[ref:1 "a"]] [[ref:2 "b"]]
  const normalized = text.replace(/\],\s*\[ref:/g, "]] [[ref:");
  const markdown = normalized.replace(CITATION_RE, (_, sceneIdx, quote) => {
    const idx = citations.length;
    citations.push({ sceneIndex: parseInt(sceneIdx, 10), quote: quote || "" });
    return `%%CITE_${idx}%%`;
  });
  return { markdown, citations };
}

/** Recursively walk React children, replacing %%CITE_N%% strings with buttons. */
function processChildren(
  children: React.ReactNode,
  citations: Citation[],
  onNav?: (sceneIndex: number, quote?: string) => void,
  scenes?: Scene[],
): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (typeof child === "string") {
      const parts = child.split(CITE_SPLIT_RE);
      if (parts.length <= 1) return child;
      return parts.map((part, i) => {
        const m = part.match(CITE_MATCH_RE);
        if (!m) return part || null;
        const cite = citations[parseInt(m[1], 10)];
        if (!cite) return null;
        return (
          <button
            key={`c${m[1]}-${i}`}
            type="button"
            onClick={() => onNav?.(cite.sceneIndex, cite.quote || undefined)}
            title={
              cite.quote
                ? `Scene ${cite.sceneIndex + 1}: "${cite.quote}"`
                : `Go to scene ${cite.sceneIndex + 1}`
            }
            className={cn(
              "mx-0.5 inline-flex items-center gap-0.5 rounded px-1.5 py-0.5",
              "bg-primary/15 text-primary hover:bg-primary/25",
              "font-mono text-[11px] font-medium transition-colors",
              "cursor-pointer border-0",
            )}
          >
            <ExternalLink className="inline h-2.5 w-2.5 shrink-0" />
            <span>
              Scene {cite.sceneIndex + 1}
              {scenes?.[cite.sceneIndex]
                ? `: ${scenes[cite.sceneIndex].location || scenes[cite.sceneIndex].heading}`
                : ""}
            </span>
          </button>
        );
      });
    }
    if (React.isValidElement(child)) {
      const props = child.props as Record<string, unknown>;
      if (props.children != null) {
        return React.cloneElement(
          child,
          {},
          processChildren(props.children as React.ReactNode, citations, onNav, scenes),
        );
      }
    }
    return child;
  });
}

function AssistantMessage({
  content,
  onNavigateToScene,
  scenes,
}: {
  content: string;
  onNavigateToScene?: (sceneIndex: number, quote?: string) => void;
  scenes?: Scene[];
}) {
  const { markdown, citations } = useMemo(() => preprocessCitations(content), [content]);

  const components = useMemo(() => {
    if (citations.length === 0) return undefined;
    const process = (children: React.ReactNode) =>
      processChildren(children, citations, onNavigateToScene, scenes);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wrap = (Tag: keyof React.JSX.IntrinsicElements): any =>
      function Wrapped({ node, children, ...props }: any) {
        return React.createElement(Tag, props, process(children));
      };
    return {
      p: wrap("p"),
      li: wrap("li"),
      strong: wrap("strong"),
      em: wrap("em"),
      td: wrap("td"),
      blockquote: wrap("blockquote"),
    };
  }, [citations, onNavigateToScene, scenes]);

  return (
    <div
      className={cn(
        "[&_p]:my-1 [&_p]:leading-relaxed",
        "[&_ul]:my-1 [&_ol]:my-1 [&_ul]:pl-4 [&_ol]:pl-4",
        "[&_ul]:list-disc [&_ol]:list-decimal",
        "[&_li]:my-0.5 [&_li]:leading-relaxed",
        "[&_blockquote]:border-l-2 [&_blockquote]:border-muted-foreground/30 [&_blockquote]:pl-2 [&_blockquote]:italic",
        "[&_strong]:font-semibold",
      )}
    >
      <Markdown components={components}>{markdown}</Markdown>
    </div>
  );
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reference?: string;
  error?: boolean;
}

interface Props {
  screenplayId: string;
  screenplayTitle?: string;
  scenes?: Scene[];
  open: boolean;
  onClose: () => void;
  /** When opened from highlight, the selected text to show as reference */
  initialReference?: string | null;
  onInitialReferenceConsumed?: () => void;
  /** Notify parent of the reference currently being "discussed" (for glow in script) */
  onActiveReferenceChange?: (text: string | null) => void;
  /** Navigate the viewer to a specific scene; optionally highlight a quote */
  onNavigateToScene?: (sceneIndex: number, quote?: string) => void;
  /** 'panel' = side panel (default), 'popover' = overlay/sheet on small screens */
  variant?: "panel" | "popover";
}

const MARGIN_WIDTH = 380;

export default function MarginAssistant({
  screenplayId,
  screenplayTitle,
  scenes,
  open,
  onClose,
  initialReference,
  onInitialReferenceConsumed,
  onActiveReferenceChange,
  onNavigateToScene,
  variant = "panel",
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [reference, setReference] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const listEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Consume initial reference when opened from highlight
  useEffect(() => {
    if (open && initialReference?.trim()) {
      setReference(initialReference.trim());
      onInitialReferenceConsumed?.();
    }
  }, [open, initialReference, onInitialReferenceConsumed]);

  useEffect(() => {
    if (open) {
      const t = requestAnimationFrame(() => inputRef.current?.focus());
      return () => cancelAnimationFrame(t);
    }
  }, [open]);

  const scrollToBottom = useCallback(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const send = useCallback(
    async (questionText?: string) => {
      const text = (questionText ?? input.trim()).trim();
      if (!text || loading) return;

      setInput("");
      const refToSend = reference?.trim() || undefined;
      if (refToSend) {
        onActiveReferenceChange?.(refToSend);
      } else {
        onActiveReferenceChange?.(null);
      }

      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: text,
        reference: refToSend,
      };
      setMessages((prev) => [...prev, userMsg]);
      setReference(null);
      setLoading(true);

      const fullQuestion = refToSend
        ? `Regarding the following passage:\n"""\n${refToSend}\n"""\n\nQuestion: ${text}`
        : text;

      const assistantId = `assistant-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      scrollToBottom();

      try {
        const { answer } = await askQuestion(fullQuestion, screenplayId);
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
                  content:
                    "Sorry, I couldn't get an answer. Please try again.",
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
    },
    [
      input,
      loading,
      reference,
      screenplayId,
      scrollToBottom,
      onActiveReferenceChange,
    ],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    },
    [send],
  );

  const clearReference = useCallback(() => setReference(null), []);

  if (!open) return null;

  const isPopover = variant === "popover";

  return (
    <aside
      role="complementary"
      aria-label="Context margin — ask about this screenplay"
      className={cn(
        "flex h-full w-full flex-col bg-card/95 backdrop-blur-sm",
        isPopover
          ? "border-0 rounded-t-xl"
          : "border-l border-border",
      )}
      style={isPopover ? undefined : { minWidth: MARGIN_WIDTH, maxWidth: MARGIN_WIDTH }}
    >
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-border/70 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <MessageCircle className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <h2 className="truncate font-mono text-xs font-semibold text-card-foreground">
              Context margin
            </h2>
            {screenplayTitle && (
              <p className="truncate font-mono text-[11px] text-muted-foreground">
                {screenplayTitle}
              </p>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className={cn(
            "flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
            "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
          )}
          aria-label="Close margin"
        >
          <PanelRightClose className="h-4 w-4" />
        </button>
      </header>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-3 py-2">
        {messages.length === 0 && !reference && (
          <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">
            Highlight text in the script and click “Ask” to ask about it, or type
            a question below.
          </p>
        )}
        <ul className="flex flex-col gap-2">
          {messages.map((m, i) => {
            const isPending =
              loading &&
              m.role === "assistant" &&
              !m.content &&
              i === messages.length - 1;
            const isAssistantWithContent = m.role === "assistant" && m.content;
            return (
              <li key={m.id} className="flex flex-col gap-1">
                {m.reference && (
                  <blockquote className="border-l-2 border-primary/40 bg-muted/50 px-2 py-1 font-mono text-[11px] italic text-muted-foreground">
                    {m.reference}
                  </blockquote>
                )}
                <div
                  className={cn(
                    "rounded-md px-2 py-1.5 font-mono text-xs",
                    m.role === "user"
                      ? "ml-4 bg-primary/10 text-foreground"
                      : "mr-1 bg-muted/80 text-foreground",
                    m.error && "text-destructive",
                  )}
                >
                  {isAssistantWithContent ? (
                    <AssistantMessage
                      content={m.content}
                      onNavigateToScene={onNavigateToScene}
                      scenes={scenes}
                    />
                  ) : (
                    m.content || (isPending ? "…" : null)
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        <div ref={listEndRef} />
      </div>

      {/* Pending reference (from highlight) */}
      {reference && (
        <div className="shrink-0 border-t border-border/70 px-3 py-2">
          <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            Referenced passage
          </p>
          <div className="flex items-start justify-between gap-2">
            <p className="min-w-0 flex-1 font-mono text-[11px] italic text-foreground/90 line-clamp-3">
              {reference}
            </p>
            <button
              type="button"
              onClick={clearReference}
              className="shrink-0 font-mono text-[10px] text-muted-foreground hover:text-foreground"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Sticky translucent input bar */}
      <div className="sticky bottom-0 shrink-0 border-t border-border/70 bg-card/80 px-3 py-2 backdrop-blur-sm">
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
              "min-h-[2.25rem] flex-1 resize-none rounded-md border border-input/80 bg-background/90 px-2.5 py-1.5 font-mono text-xs",
              "placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring",
              "disabled:opacity-60",
            )}
          />
          <button
            type="button"
            onClick={() => send()}
            disabled={!input.trim() || loading}
            className={cn(
              "flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground",
              "disabled:opacity-40 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-ring",
            )}
            aria-label="Send"
          >
            <Send className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </aside>
  );
}
