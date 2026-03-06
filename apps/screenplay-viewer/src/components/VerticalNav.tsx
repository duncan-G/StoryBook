"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { Scene } from "@/types/screenplay";
import SceneCard from "./SceneCard";
import type { SceneNavigation } from "./HorizontalNav";

interface Props {
  scenes: Scene[];
  activeReferenceText?: string | null;
  /** Called when the user starts editing the highlighted reference (clears orange highlight). */
  onClearActiveReference?: () => void;
  /** Ref for the scrollable scene body (used to constrain selection and highlight button to scene content only) */
  sceneContentRef?: React.RefObject<HTMLDivElement | null>;
  /** When set, scrolls to the given scene index. Change `key` to re-trigger same index. */
  navigateToScene?: SceneNavigation | null;
  /** Scene index to start on (used to restore position on mode switch). */
  initialSceneIndex?: number;
  /** Called whenever the visible scene changes. */
  onSceneChange?: (index: number) => void;
  screenplayId?: string;
  /** Called when user clicks generate-audio on a focused scene. */
  onGenerateAudio?: (sceneIndex: number) => void | Promise<void>;
}

export default function VerticalNav({
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
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRefs = useRef<(HTMLDivElement | null)[]>([]);

  const setContainerRef = useCallback(
    (el: HTMLDivElement | null) => {
      (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
      if (sceneContentRef) (sceneContentRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
    },
    [sceneContentRef],
  );
  const [activeIdx, setActiveIdx] = useState(initialSceneIndex);
  const lastScrollTop = useRef(0);
  const total = scenes.length;

  useEffect(() => {
    if (initialSceneIndex > 0) {
      const el = sceneRefs.current[initialSceneIndex];
      if (el) el.scrollIntoView({ behavior: "instant", block: "start" });
    }
  }, []);

  const setRef = useCallback((el: HTMLDivElement | null, idx: number) => {
    sceneRefs.current[idx] = el;
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const scrollDown = (container.scrollTop - lastScrollTop.current) >= 0;
        lastScrollTop.current = container.scrollTop;

        let best: { idx: number; ratio: number } | null = null;
        for (const entry of entries) {
          const idx = Number(entry.target.getAttribute("data-scene-idx"));
          if (isNaN(idx)) continue;
          if (!best || entry.intersectionRatio > best.ratio) {
            best = { idx, ratio: entry.intersectionRatio };
          } else if (
            entry.intersectionRatio === best.ratio &&
            best.ratio > 0 &&
            (scrollDown ? idx > best.idx : idx < best.idx)
          ) {
            // Tie-break by scroll direction so we don't flip backward
            best = { idx, ratio: entry.intersectionRatio };
          }
        }

        if (best && best.ratio > 0) {
          setActiveIdx((prev) => {
            if (best!.idx === prev) return prev;
            // When scrolling down, don't flip back to a lower index (prevents jitter)
            if (scrollDown && best!.idx < prev) return prev;
            if (!scrollDown && best!.idx > prev) return prev;
            // Accept when clearly in view or when moving in scroll direction
            if (best!.ratio >= 0.5) return best!.idx;
            if (scrollDown && best!.idx > prev) return best!.idx;
            if (!scrollDown && best!.idx < prev) return best!.idx;
            return prev;
          });
        }
      },
      { root: container, threshold: [0, 0.25, 0.5, 0.75, 1] },
    );

    sceneRefs.current.forEach((el) => {
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [scenes]);

  useEffect(() => {
    onSceneChange?.(activeIdx);
  }, [activeIdx]);

  useEffect(() => {
    if (!navigateToScene || navigateToScene.sceneIndex < 0 || navigateToScene.sceneIndex >= total) return;
    const sceneEl = sceneRefs.current[navigateToScene.sceneIndex];
    if (!sceneEl) return;
    sceneEl.scrollIntoView({ behavior: "smooth", block: "start" });
    const timer = setTimeout(() => {
      const highlighted = sceneEl.querySelector("[data-active-ref]");
      if (highlighted) {
        highlighted.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 80);
    return () => clearTimeout(timer);
  }, [navigateToScene?.key, navigateToScene?.sceneIndex, total]);

  if (total === 0) return null;

  return (
    <div className="flex h-full flex-col">
      {/* Scrollable scene list */}
      <div ref={setContainerRef} className="vertical-scroll flex-1 cursor-text overflow-y-auto">
        {scenes.map((scene, i) => (
          <div
            key={i}
            ref={(el) => setRef(el, i)}
            data-scene-idx={i}
            className="min-h-[55vh] border-b border-border/50 sm:min-h-[60vh]"
          >
            <SceneCard
              scene={scene}
              index={i}
              total={total}
              activeReferenceText={activeReferenceText}
              onClearActiveReference={onClearActiveReference}
              screenplayId={screenplayId}
              isFocused={activeIdx === i}
              onGenerateAudio={onGenerateAudio}
            />
          </div>
        ))}
        <div className="h-[40vh]" />
      </div>

      {/* Bottom bar */}
      <nav
        className={cn(
          "grid grid-cols-[auto_1fr_auto] items-center gap-2 px-3 py-2 sm:flex sm:justify-between sm:px-6 sm:py-3",
          "border-t border-border bg-background",
        )}
      >
        <span className="font-mono text-xs text-muted-foreground">
          Scene {activeIdx + 1} of {total}
        </span>
        <span className="truncate text-center font-mono text-xs text-foreground sm:text-sm">
          {scenes[activeIdx]?.location || scenes[activeIdx]?.heading}
        </span>
        <span className="font-mono text-xs text-muted-foreground/60">
          p.{scenes[activeIdx]?.page}
        </span>
      </nav>
    </div>
  );
}
