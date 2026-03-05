"use client";

import { MessageCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const BUTTON_OFFSET = 6;
const APPROX_BUTTON_HEIGHT = 40;
const BUTTON_WIDTH = 140;

interface Props {
  rect: DOMRect;
  /** Ref to scene content container; button is clamped so it stays inside this area (not in header/footer) */
  contentBoundsRef?: React.RefObject<HTMLElement | null>;
  onClick: () => void;
}

function getPosition(rect: DOMRect, contentBoundsRef?: React.RefObject<HTMLElement | null>) {
  if (typeof window === "undefined") {
    return { top: rect.bottom + BUTTON_OFFSET, left: rect.right + BUTTON_OFFSET, transform: undefined };
  }
  const contentRect = contentBoundsRef?.current?.getBoundingClientRect();
  const minTop = contentRect?.top ?? 0;
  const maxTop = contentRect ? contentRect.bottom - APPROX_BUTTON_HEIGHT : window.innerHeight - APPROX_BUTTON_HEIGHT;
  const minLeft = contentRect?.left ?? 0;
  const maxLeft = contentRect ? contentRect.right - BUTTON_WIDTH : window.innerWidth - BUTTON_WIDTH;

  const spaceBelow = (contentRect ? contentRect.bottom - rect.bottom : window.innerHeight - rect.bottom);
  const spaceAbove = (contentRect ? rect.top - contentRect.top : rect.top);
  const showAbove = spaceBelow < APPROX_BUTTON_HEIGHT + BUTTON_OFFSET && spaceAbove >= spaceBelow;

  let top = showAbove ? rect.top - BUTTON_OFFSET : rect.bottom + BUTTON_OFFSET;
  let left = Math.min(rect.right + BUTTON_OFFSET, maxLeft);
  left = Math.max(minLeft, left);

  if (showAbove) {
    const maxTopWhenAbove = contentRect ? contentRect.bottom : window.innerHeight;
    top = Math.max(minTop + APPROX_BUTTON_HEIGHT, Math.min(maxTopWhenAbove, top));
  } else {
    top = Math.max(minTop, Math.min(maxTop, top));
  }

  return {
    top,
    left,
    transform: showAbove ? "translateY(-100%)" : undefined,
  };
}

export default function HighlightAskButton({ rect, contentBoundsRef, onClick }: Props) {
  const { top, left, transform } = getPosition(rect, contentBoundsRef);
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "fixed z-50 flex items-center gap-1.5 rounded-md px-2.5 py-1.5 shadow-md",
        "bg-primary text-primary-foreground font-mono text-xs",
        "hover:opacity-95 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background",
      )}
      style={{ top, left, transform }}
      aria-label="Ask about this passage"
    >
      <MessageCircle className="h-3.5 w-3.5" />
      Ask
    </button>
  );
}
