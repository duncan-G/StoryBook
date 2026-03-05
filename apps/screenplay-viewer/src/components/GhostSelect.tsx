"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface GhostSelectProps {
  value: string;
  options: string[];
  placeholder?: string;
  /** When true, shows a text input at the bottom for free-form entry. */
  allowCustom?: boolean;
  customPlaceholder?: string;
  onCommit: (newValue: string) => void;
  className?: string;
  /** Classes applied to the trigger in its resting state. */
  triggerClassName?: string;
  /** Classes applied to each option row. */
  optionClassName?: string;
  /** Align dropdown to left (default) or right of trigger. Use "right" when trigger is on the right edge to avoid overflow. */
  align?: "left" | "right";
}

/**
 * Ghost-styled inline select. Looks like static text at rest.
 * On click, opens a dropdown with preset options (and optional custom input).
 * Selecting an option or submitting custom text commits immediately.
 */
export default function GhostSelect({
  value,
  options,
  placeholder = "Select",
  allowCustom = false,
  customPlaceholder = "Custom…",
  onCommit,
  className,
  triggerClassName,
  optionClassName,
  align = "left",
}: GhostSelectProps) {
  const [open, setOpen] = useState(false);
  const [customText, setCustomText] = useState("");
  const wrapperRef = useRef<HTMLDivElement>(null);
  const customInputRef = useRef<HTMLInputElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [open]);

  const select = useCallback(
    (val: string) => {
      setOpen(false);
      setCustomText("");
      if (val !== value) onCommit(val);
    },
    [value, onCommit],
  );

  const submitCustom = useCallback(() => {
    const trimmed = customText.trim();
    if (!trimmed) return;
    select(trimmed);
  }, [customText, select]);

  const display = value || placeholder;
  const isEmpty = !value;

  return (
    <div ref={wrapperRef} className={cn("relative inline-block", className)}>
      {/* Trigger */}
      <button
        type="button"
        onClick={() => {
          setOpen((o) => !o);
          setCustomText("");
        }}
        className={cn(
          "inline-flex items-center gap-1 transition-all duration-200 cursor-pointer",
          "rounded px-2 py-0.5 font-mono text-xs uppercase",
          isEmpty && "opacity-40 italic",
          !open && "hover:opacity-80",
          open && "opacity-100",
          triggerClassName,
        )}
      >
        <span>{display}</span>
        <ChevronDown
          className={cn(
            "h-3 w-3 shrink-0 transition-transform duration-200",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className={cn(
            "absolute top-full z-30 mt-1 min-w-[120px] overflow-hidden",
            align === "right" ? "right-0 left-auto" : "left-0",
            "rounded-lg border border-border bg-card shadow-lg",
          )}
        >
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => select(opt)}
              className={cn(
                "flex w-full items-center px-3 py-1.5 text-left font-mono text-xs uppercase",
                "transition-colors hover:bg-accent hover:text-accent-foreground",
                opt === value && "bg-accent/50 font-semibold",
                optionClassName,
              )}
            >
              {opt}
            </button>
          ))}

          {allowCustom && (
            <div className="border-t border-border px-2 py-1.5">
              <input
                ref={customInputRef}
                type="text"
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    submitCustom();
                  }
                }}
                placeholder={customPlaceholder}
                className={cn(
                  "w-full rounded border border-input bg-background px-2 py-1",
                  "font-mono text-xs uppercase text-foreground placeholder:normal-case placeholder:text-muted-foreground/60",
                  "outline-none focus:border-ring",
                )}
                autoFocus
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
