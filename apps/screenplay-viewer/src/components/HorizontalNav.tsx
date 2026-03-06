"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Scene } from "@/types/screenplay";
import SceneCard from "./SceneCard";

export interface SceneNavigation {
  sceneIndex: number;
  key: number;
}

interface Props {
  scenes: Scene[];
  activeReferenceText?: string | null;
  /** Called when the user starts editing the highlighted reference (clears orange highlight). */
  onClearActiveReference?: () => void;
  /** Ref for the scrollable scene body (used to constrain selection and highlight button to scene content only) */
  sceneContentRef?: React.RefObject<HTMLDivElement | null>;
  /** When set, navigates to the given scene index. Change `key` to re-trigger same index. */
  navigateToScene?: SceneNavigation | null;
  /** Scene index to start on (used to restore position on mode switch). */
  initialSceneIndex?: number;
  /** Called whenever the visible scene changes. */
  onSceneChange?: (index: number) => void;
  screenplayId?: string;
  /** Called when user clicks generate-audio on the focused scene. */
  onGenerateAudio?: (sceneIndex: number) => void | Promise<void>;
}

const btnClass = cn(
  "flex items-center gap-1.5 rounded-lg px-3 py-1.5 sm:gap-2 sm:px-4 sm:py-2",
  "border border-border bg-card font-mono text-xs text-foreground sm:text-sm",
  "transition-colors hover:bg-accent hover:text-accent-foreground",
  "disabled:cursor-not-allowed disabled:opacity-30",
);

export default function HorizontalNav({
  scenes,
  activeReferenceText,
  onClearActiveReference,
  sceneContentRef,
  navigateToScene,
  initialSceneIndex = 0,
  onSceneChange,
  screenplayId,
  onGenerateAudio,
}: Props) {
  const [current, setCurrent] = useState(initialSceneIndex);
  const scrollRef = useRef<HTMLDivElement>(null);
  const skipScrollToTop = useRef(false);

  const setScrollRef = useCallback(
    (el: HTMLDivElement | null) => {
      (scrollRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
      if (sceneContentRef) (sceneContentRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
    },
    [sceneContentRef],
  );
  const total = scenes.length;

  const go = useCallback(
    (delta: number) => {
      setCurrent((prev) => Math.max(0, Math.min(total - 1, prev + delta)));
    },
    [total],
  );

  useEffect(() => {
    onSceneChange?.(current);
    if (skipScrollToTop.current) {
      skipScrollToTop.current = false;
      return;
    }
    scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [current]);

  useEffect(() => {
    if (!navigateToScene || navigateToScene.sceneIndex < 0 || navigateToScene.sceneIndex >= total) return;
    const isSameScene = navigateToScene.sceneIndex === current;
    if (!isSameScene) {
      skipScrollToTop.current = true;
      setCurrent(navigateToScene.sceneIndex);
    }
    const timer = setTimeout(() => {
      const highlighted = scrollRef.current?.querySelector("[data-active-ref]");
      if (highlighted) {
        highlighted.scrollIntoView({ behavior: "smooth", block: "center" });
      } else if (!isSameScene) {
        scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
      }
    }, 80);
    return () => clearTimeout(timer);
  }, [navigateToScene?.key, navigateToScene?.sceneIndex, total]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.role === "textbox")
      ) {
        return;
      }
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        go(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  if (total === 0) return null;

  return (
    <div className="flex h-full flex-col">
      <div ref={setScrollRef} className="flex-1 cursor-text overflow-y-auto">
        <SceneCard
          scene={scenes[current]}
          index={current}
          total={total}
          activeReferenceText={activeReferenceText}
          onClearActiveReference={onClearActiveReference}
          screenplayId={screenplayId}
          isFocused={true}
          onGenerateAudio={onGenerateAudio}
        />
      </div>

      <nav
        className={cn(
          "grid grid-cols-[auto_1fr_auto] items-center gap-2 px-3 py-2 sm:flex sm:justify-between sm:px-6 sm:py-3",
          "border-t border-border bg-background",
        )}
      >
        <button onClick={() => go(-1)} disabled={current === 0} className={btnClass}>
          <ChevronLeft className="h-4 w-4" />
          <span className="hidden sm:inline">Previous</span>
        </button>

        <div className="flex min-w-0 items-center justify-center gap-2 sm:gap-3">
          <span className="font-mono text-xs text-muted-foreground">Scene</span>
          <input
            type="number"
            min={1}
            max={total}
            value={current + 1}
            onChange={(e) => {
              const n = parseInt(e.target.value, 10);
              if (!isNaN(n) && n >= 1 && n <= total) setCurrent(n - 1);
            }}
            className={cn(
              "w-12 rounded border border-input bg-card px-2 py-1 sm:w-14",
              "text-center font-mono text-sm text-foreground",
              "outline-none focus:border-ring",
            )}
          />
          <span className="font-mono text-xs text-muted-foreground">of {total}</span>
        </div>

        <button onClick={() => go(1)} disabled={current === total - 1} className={btnClass}>
          <span className="hidden sm:inline">Next</span>
          <ChevronRight className="h-4 w-4" />
        </button>
      </nav>
    </div>
  );
}
