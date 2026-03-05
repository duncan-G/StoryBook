"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Pencil } from "lucide-react";
import { cn } from "@/lib/utils";

interface GhostInputProps {
  value: string;
  placeholder?: string;
  onCommit: (newValue: string) => void;
  className?: string;
  /** Extra classes applied only when the field is NOT focused (resting state). */
  restingClassName?: string;
  /** Extra classes applied only on hover (not focused). */
  hoverClassName?: string;
  /** Show the one-time pencil hint on first mount (only when value is non-empty). */
  showPencilHint?: boolean;
  /** Tag to render — defaults to "span". */
  as?: "span" | "h1" | "p";
  /** Called when the field transitions into editing mode (e.g. to clear external highlight state). */
  onEditingStart?: () => void;
}

const EDITING_STYLE: React.CSSProperties = {
  boxShadow: "0 0 0 1.5px var(--border), 0 1px 4px 0 rgba(0,0,0,0.12)",
  padding: "2px 6px",
  margin: "-2px -6px",
  borderRadius: "4px",
};

/**
 * Single-element inline-editable text. One `<Tag>` always rendered — no swapping,
 * no size jumps. Width is snapshotted on activate so the box never shrinks.
 *
 * Empty: shows placeholder text + dashed border + persistent pencil icon.
 * Non-empty: ghost text — hover for underline, click to edit.
 * Editing: box-shadow ring, caret at click position.
 */
export default function GhostInput({
  value,
  placeholder = "Untitled",
  onCommit,
  className,
  restingClassName,
  hoverClassName,
  showPencilHint = false,
  as: Tag = "span",
  onEditingStart,
}: GhostInputProps) {
  const ref = useRef<HTMLElement>(null);
  const [editing, setEditing] = useState(false);
  const editingRef = useRef(false);
  const [hovered, setHovered] = useState(false);
  const [pencilVisible, setPencilVisible] = useState(showPencilHint && !!value);
  const committedValue = useRef(value);
  const clickPos = useRef<{ x: number; y: number } | null>(null);

  const isEmpty = !value;

  // One-time pencil fade for non-empty fields
  useEffect(() => {
    if (!showPencilHint || !value) return;
    setPencilVisible(true);
    const t = setTimeout(() => setPencilVisible(false), 1500);
    return () => clearTimeout(t);
  }, [showPencilHint, value]);

  // Sync DOM text when not editing. Shows placeholder when value is empty.
  useLayoutEffect(() => {
    if (!editing && ref.current) {
      committedValue.current = value;
      ref.current.textContent = value || placeholder;
    }
  }, [value, editing, placeholder]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    clickPos.current = { x: e.clientX, y: e.clientY };
    // Only prevent default when not editing so that text selection (click-drag) works in edit mode
    if (!editingRef.current) {
      e.preventDefault();
    }
  }, []);

  const activate = useCallback(() => {
    if (editingRef.current) return;
    const el = ref.current;
    if (!el) return;

    const pos = clickPos.current;
    clickPos.current = null;

    // Snapshot width so the box never shrinks when we clear placeholder
    el.style.minWidth = `${el.offsetWidth}px`;

    // Clear placeholder — set to real value (empty string if none)
    el.textContent = committedValue.current || "";

    editingRef.current = true;
    setEditing(true);
    onEditingStart?.();
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.focus();
        const sel = window.getSelection();
        if (!sel) return;

        // Click activation — caret at click position (Chrome/Safari use caretRangeFromPoint; Firefox uses caretPositionFromPoint)
        if (pos && committedValue.current) {
          let range: Range | null = null;
          if (document.caretRangeFromPoint) {
            range = document.caretRangeFromPoint(pos.x, pos.y);
          } else if (document.caretPositionFromPoint) {
            const posObj = document.caretPositionFromPoint(pos.x, pos.y);
            if (posObj?.offsetNode && el.contains(posObj.offsetNode)) {
              range = document.createRange();
              range.setStart(posObj.offsetNode, posObj.offset);
              range.collapse(true);
            }
          }
          if (range && el.contains(range.startContainer)) {
            sel.removeAllRanges();
            sel.addRange(range);
            return;
          }
        }

        // Keyboard / empty / fallback — caret at end
        const range = document.createRange();
        range.selectNodeContents(el);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      });
    });
  }, []);

  const endEditing = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    editingRef.current = false;
    el.style.minWidth = "";
    setEditing(false);
  }, []);

  const commit = useCallback(() => {
    if (!editingRef.current) return;
    const el = ref.current;
    if (!el) return;
    const raw = (el.textContent ?? "").replace(/\n/g, " ").trim();
    endEditing();
    if (raw !== committedValue.current) {
      committedValue.current = raw;
      onCommit(raw);
    }
  }, [onCommit, endEditing]);

  const revert = useCallback(() => {
    endEditing();
    ref.current?.blur();
  }, [endEditing]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        e.preventDefault();
        revert();
      }
    },
    [commit, revert],
  );

  // Pencil: always visible when empty, fades when non-empty
  const showPencil = isEmpty || pencilVisible;

  return (
    <span
      className={cn(
        "relative inline-flex items-center gap-1.5 rounded transition-colors",
        isEmpty && !editing && "border border-dashed border-muted-foreground/40 px-1.5 py-0.5 hover:border-foreground/50",
      )}
    >
      <Tag
        ref={ref as React.Ref<HTMLElement>}
        contentEditable={editing}
        suppressContentEditableWarning
        role="textbox"
        aria-label={placeholder}
        tabIndex={0}
        onMouseDown={handleMouseDown}
        onClick={activate}
        onFocus={activate}
        onBlur={commit}
        onKeyDown={handleKeyDown}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={editing ? EDITING_STYLE : undefined}
        className={cn(
          "outline-none transition-all duration-200 cursor-text",
          // Non-empty resting
          !editing && !isEmpty && !hovered && restingClassName,
          // Hover (non-empty only)
          !editing &&
            !isEmpty &&
            hovered && [
              "underline decoration-1 underline-offset-4 decoration-foreground/30 opacity-80",
              hoverClassName,
            ],
          // Editing
          editing && "caret-foreground",
          // Empty + not editing: placeholder appearance
          isEmpty && !editing && "opacity-40 italic",
          className,
        )}
      />

      <Pencil
        className={cn(
          "h-3 w-3 shrink-0 text-muted-foreground pointer-events-none transition-opacity duration-500",
          showPencil ? "opacity-60" : "opacity-0",
        )}
        aria-hidden
      />
    </span>
  );
}
